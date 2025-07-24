# planner/views.py
import json
import logging
from io import BytesIO

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseForbidden, JsonResponse
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

import pandas as pd

from .models import (
    Trip, TripDay, DailyActivityItem, ActivityRating, Voyage, JournalEntry,
    VoyageMedia, Like, Comment, ChatMessage
)
from .forms import (
    TripPlannerForm, UserUpdateForm, ProfileUpdateForm, RatingForm, SignUpForm,
    VoyageForm, JournalEntryForm, VoyageMediaForm, CommentForm
)
from .utils import plan_trip_django, load_and_preprocess_data, OSMNX_AVAILABLE 
from .reportlab_utils import generate_trip_pdf_django, generate_voyage_pdf_django, generate_schedule_content_objects_django
from .chatbot_logic import process_user_query # Assurez-vous que cette ligne est correcte

NUM_FEATURED_CITIES = 3
NUM_ACTIVITIES_PER_CITY = 3

logger = logging.getLogger(__name__)


def home_showcase_view(request):
    """
    Charge et affiche les activités sur la page d'accueil de manière dynamique.
    """
    showcase_context = {}
    try:
        # Ici, on passe target_cities_for_api=[] pour que load_and_preprocess_data charge toutes les données statiques
        activities_df, _, _, _ = load_and_preprocess_data(use_realtime_api=False, target_cities_for_api=[])
        if not activities_df.empty and 'ville_normalisee' in activities_df.columns:

            featured_cities = []
            if 'rating' in activities_df.columns:
                city_scores = activities_df.groupby('ville_normalisee')['rating'].mean()
                featured_cities = city_scores.sort_values(ascending=False).head(NUM_FEATURED_CITIES).index.tolist()

            if not featured_cities:
                logger.info("Fallback : Impossible de trier les villes par note, sélection des premières villes disponibles.")
                unique_cities = activities_df['ville_normalisee'].dropna().unique()
                featured_cities = list(unique_cities[:NUM_FEATURED_CITIES])

            showcase_activities = {}
            for city in featured_cities:
                city_activities_df = activities_df[activities_df['ville_normalisee'] == city]
                if not city_activities_df.empty:
                    sorted_activities = city_activities_df.sort_values(by='rating', ascending=False)
                    showcase_activities[city] = sorted_activities.head(NUM_ACTIVITIES_PER_CITY).to_dict('records')

            showcase_context = {'showcase_activities_by_city': showcase_activities}
            if not showcase_activities:
                messages.info(request, "Aucune activité à afficher pour les villes vedettes. Vérifiez vos fichiers de données.")
        else:
            messages.warning(request, "Les données des activités n'ont pas pu être chargées ou sont mal formatées.")

    except Exception as e:
        logger.error(f"Erreur lors de la préparation de l'aperçu : {e}", exc_info=True)
        messages.error(request, "Une erreur est survenue lors du chargement des activités.")

    return render(request, 'planner/home_showcase.html', showcase_context)

@login_required
def plan_trip_view(request):
    """Gère la planification et affiche les résultats sur la même page."""
    # Initialiser les variables pour le contexte de rendu, même si le formulaire n'est pas encore soumis
    folium_map_html = None 
    schedule_md = None 
    trip_plan_result = None
    trip_params = {}
    
    if request.method == 'POST':
        form = TripPlannerForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            
            use_realtime = data.get('use_realtime_api', False)
            target_cities = data['target_cities']

            location_status_choice = data.get('location_status_choice')
            start_city_choice = data.get('start_city_choice')
            start_gps_coords = data.get('start_gps_coords')

            final_start_location_type = None
            final_start_location_value = None

            if location_status_choice == 'in_morocco':
                final_start_location_type = 'current_gps'
                final_start_location_value = start_gps_coords
                if not final_start_location_value:
                    messages.error(request, "Veuillez autoriser la géolocalisation ou entrer vos coordonnées GPS si vous êtes au Maroc.")
                    return render(request, 'planner/plan_trip.html', {
                        'form': form, 
                        'folium_map_html': folium_map_html, 
                        'schedule_md': schedule_md, 
                        'trip_plan_result': trip_plan_result, 
                        'trip_params': trip_params
                    })
            elif location_status_choice == 'not_in_morocco':
                final_start_location_type = 'choose_city'
                final_start_location_value = start_city_choice
                if not final_start_location_value:
                    messages.error(request, "Veuillez sélectionner votre aéroport/ville d'arrivée si vous n'êtes pas encore au Maroc.")
                    return render(request, 'planner/plan_trip.html', {
                        'form': form, 
                        'folium_map_html': folium_map_html, 
                        'schedule_md': schedule_md, 
                        'trip_plan_result': trip_plan_result, 
                        'trip_params': trip_params
                    })
            
            activities_df, hotels_df, api_restaurants_cafes_df, city_coords_map = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
            
            logger.debug(f"plan_trip_view: Tentative de chargement des données. use_realtime={use_realtime}, target_cities={target_cities}")

            if use_realtime:
                try:
                    activities_df, hotels_df, api_restaurants_cafes_df, city_coords_map = load_and_preprocess_data(
                        use_realtime_api=True,
                        target_cities_for_api=target_cities
                    )
                    logger.debug(f"plan_trip_view: Après tentative API. activities_df.shape={activities_df.shape}, hotels_df.shape={hotels_df.shape}, api_restaurants_cafes_df.shape={api_restaurants_cafes_df.shape}")
                    
                    if activities_df.empty or hotels_df.empty:
                        messages.warning(request, "L'API n'a pas pu récupérer suffisamment de données en temps réel pour les villes sélectionnées. Essai avec les données statiques.")
                        logger.warning("plan_trip_view: API a retourné des DataFrames vides (ou le fallback statique initial n'a rien donné).")
                        activities_df, hotels_df, api_restaurants_cafes_df, city_coords_map = load_and_preprocess_data(use_realtime_api=False, target_cities_for_api=[])
                except Exception as e:
                    logger.error(f"plan_trip_view: Erreur lors de la récupération des données en temps réel: {e}", exc_info=True)
                    messages.error(request, "Une erreur est survenue lors de la tentative de récupération des données en temps réel. Essai avec les données statiques.")
                    activities_df, hotels_df, api_restaurants_cafes_df, city_coords_map = load_and_preprocess_data(use_realtime_api=False, target_cities_for_api=[])
            else:
                activities_df, hotels_df, api_restaurants_cafes_df, city_coords_map = load_and_preprocess_data(use_realtime_api=False, target_cities_for_api=[])


            if activities_df.empty or hotels_df.empty:
                logger.error(f"plan_trip_view: DataFrames d'activités ou d'hôtels sont vides après toutes les tentatives. Activités: {activities_df.shape}, Hôtels: {hotels_df.shape}")
                messages.error(request, "Impossible de charger les données des activités ou des hôtels. Les fichiers statiques sont peut-être vides, mal formatés ou l'API n'a rien renvoyé de valable.")
                return render(request, 'planner/plan_trip.html', {
                    'form': form, 
                    'folium_map_html': folium_map_html, 
                    'schedule_md': schedule_md, 
                    'trip_plan_result': trip_plan_result, 
                    'trip_params': trip_params
                })

            logger.debug("plan_trip_view: Données activités et hôtels chargées avec succès (non vides).")

            use_astar = data.get('use_astar_routes_planning', False)

            trip_data_dict = plan_trip_django(
                target_cities_list=data['target_cities'],
                total_budget_str=str(data['total_budget']),
                num_days_str=str(data['num_days']),
                num_persons=data['num_persons'],
                min_hotel_rating_str=str(data['min_hotel_rating']),
                activity_preferences_str=",".join(data['activity_prefs']),
                activity_intensity=data['activity_intensity'],
                activities_df_global=activities_df,
                hotels_df_global=hotels_df,
                api_restaurants_cafes_df_global=api_restaurants_cafes_df, 
                city_coords_map_global=city_coords_map,
                use_astar_for_planning=use_astar,
                start_location_type=final_start_location_type, 
                start_location_value=final_start_location_value
            )
            
            trip_plan_result = trip_data_dict.get('trip_plan_result')
            trip_params = trip_data_dict.get('params')
            cities = trip_data_dict.get('ordered_cities_with_start')
            folium_map_html = trip_data_dict.get('folium_map_html')
            schedule_md = trip_data_dict.get('schedule_md')


            if trip_plan_result:
                new_trip = Trip.objects.create(
                    user=request.user, 
                    name=f"Plan pour {', '.join(cities) if cities else 'voyage sans ville'}",
                    num_persons=data['num_persons'],
                    target_cities_input_str=", ".join(data['target_cities']),
                    num_days_str=str(data['num_days'])
                )
                day_counter = 0
                for city_data in trip_plan_result:
                    for daily_plan in city_data.get('activites_par_jour_optimisees', []):
                        day_counter += 1
                        trip_day = TripDay.objects.create(trip=new_trip, day_number=day_counter, city_name=city_data.get('ville', 'Ville inconnue'))
                        for i, item in enumerate(daily_plan):
                            item_name = item.get('nom') or item.get('name', 'Activité non nommée')
                            activity_type_name = item.get('type', 'N/A')
                            
                            item_type_for_db = 'activity'
                            if item.get('type') == 'hotel' or item.get('is_hotel_stop'): 
                                item_type_for_db = 'hotel'
                            elif item.get('type') == 'Gastronomique':
                                item_type_for_db = 'restaurant' 
                            elif item.get('type') == 'Gastronomique/Café':
                                item_type_for_db = 'cafe' 
                            
                            DailyActivityItem.objects.create(
                                trip_day=trip_day,
                                order_in_day=i,
                                item_type=item_type_for_db, 
                                name=item_name,
                                activity_type_name=activity_type_name
                            )
                messages.success(request, "Voyage planifié et sauvegardé !")
                request.session['trip_plan_result_for_pdf'] = trip_plan_result
                request.session['trip_params_for_pdf'] = trip_params
                
                return render(request, 'planner/plan_trip.html', {
                    'form': form,
                    'trip_plan_result': trip_plan_result,
                    'trip_params': trip_params,
                    'folium_map_html': folium_map_html, 
                    'schedule_md': schedule_md
                })
            else:
                messages.warning(request, "Impossible de générer un plan avec les critères fournis. Essayez d'ajuster vos préférences ou votre budget.")

    form = TripPlannerForm()
    return render(request, 'planner/plan_trip.html', {
        'form': form, 
        'folium_map_html': folium_map_html, 
        'schedule_md': schedule_md, 
        'trip_plan_result': trip_plan_result, 
        'trip_params': trip_params
    })


@login_required
def profile_view(request):
    """Gère la mise à jour du profil utilisateur."""
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, 'Votre profil a été mis à jour avec succès !')
            return redirect('planner:profile')
    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)
    context = {'u_form': u_form, 'p_form': p_form, 'page_title': 'Mon Profil'}
    return render(request, 'planner/profile.html', context)


def signup_view(request):
    """Gère l'inscription de nouveaux utilisateurs."""
    if request.method == 'POST':
        form = SignUpForm(request.POST) 
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Inscription réussie ! Vous êtes maintenant connecté.")
            return redirect('planner:home_showcase')
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', {'form': form})


@login_required
def my_trips_view(request):
    """Affiche les plans de voyage de l'utilisateur."""
    user_trips = Trip.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'planner/my_trips.html', {'user_trips': user_trips})


@login_required
def trip_detail_view(request, trip_id):
    """Affiche les détails d'un plan de voyage non publié."""
    trip = get_object_or_404(Trip, id=trip_id, user=request.user)
    days_with_items = []
    for day in trip.days.all():
        items_for_day = []
        for item in day.activity_items.all():
            rating_obj = item.user_ratings.filter(user=request.user).first()
            items_for_day.append({
                'db_item': item,
                'user_has_rated': bool(rating_obj),
                'user_rating': rating_obj.rating if rating_obj else 0
            })
        days_with_items.append({'day_obj': day, 'items': items_for_day})
    return render(request, 'planner/trip_detail.html', {'trip': trip, 'days_with_items': days_with_items})


@login_required
def publish_trip_as_voyage_view(request, trip_id):
    """Crée un 'Voyage' (carnet) à partir d'un 'Trip' (plan)."""
    trip = get_object_or_404(Trip, id=trip_id, user=request.user)
    voyage, created = Voyage.objects.get_or_create(
        author=request.user, 
        source_trip=trip, 
        defaults={'title': f"Carnet pour : {trip.name}"}
    )
    if created and trip.days.exists():
        for trip_day in trip.days.all():
            JournalEntry.objects.get_or_create(
                voyage=voyage, 
                day_number=trip_day.day_number, 
                defaults={'title': f"Jour à {trip_day.city_name}"}
            )
    messages.success(request, "Votre carnet de voyage est prêt à être modifié.")
    return redirect('planner:voyage_edit_dashboard', voyage_id=voyage.id)


@login_required
def voyage_detail_view(request, voyage_id):
    """Affiche un carnet de voyage."""
    voyage = get_object_or_404(Voyage, id=voyage_id)
    comment_form = CommentForm(request.POST or None)
    if request.method == 'POST' and comment_form.is_valid():
        comment = comment_form.save(commit=False)
        comment.voyage = voyage
        comment.author = request.user
        comment.save()
        return redirect('planner:voyage_detail', voyage_id=voyage.id)
    context = {
        'voyage': voyage,
        'journal_entries': voyage.journal_entries.all().prefetch_related('media'),
        'comments': voyage.comments.all().order_by('-created_at'),
        'comment_form': comment_form,
        'is_liked': voyage.likes.filter(user=request.user).exists() if request.user.is_authenticated else False,
        'like_count': voyage.likes.count()
    }
    return render(request, 'planner/voyage_detail.html', context)


@login_required
def voyage_edit_dashboard_view(request, voyage_id):
    """Tableau de bord pour éditer les infos générales d'un carnet."""
    voyage = get_object_or_404(Voyage, id=voyage_id, author=request.user)
    form = VoyageForm(request.POST or None, request.FILES or None, instance=voyage)
    if request.method == 'POST' and form.is_valid():
        form.save()
        messages.success(request, 'Informations mises à jour.')
        return redirect(request.path)
    return render(request, 'planner/voyage_edit_dashboard.html', {'voyage': voyage, 'form': form, 'journal_entries': voyage.journal_entries.all()})


@login_required
def journal_entry_edit_view(request, voyage_id, day_number):
    """Vue pour éditer le récit et les médias d'une seule journée."""
    voyage = get_object_or_404(Voyage, id=voyage_id, author=request.user)
    entry, _ = JournalEntry.objects.get_or_create(voyage=voyage, day_number=day_number)
    form = JournalEntryForm(request.POST or None, instance=entry)
    media_form = VoyageMediaForm(request.POST or None, request.FILES or None)

    if request.method == 'POST':
        if 'save_story' in request.POST and form.is_valid():
            form.save()
            messages.success(request, f'Récit du jour {day_number} sauvegardé.')
            return redirect(request.path)
        
        if 'add_media' in request.POST and media_form.is_valid():
            media = media_form.save(commit=False)
            media.journal_entry = entry
            file = request.FILES.get('media_file')
            if file and file.content_type in ['audio/mpeg', 'audio/wav', 'audio/ogg']:
                media.media_type = 'AUDIO'
            else:
                media.media_type = 'IMAGE'
            media.save()
            messages.success(request, 'Média ajouté.')
            return redirect(request.path)
            
    return render(request, 'planner/journal_entry_form.html', {'form': form, 'media_form': media_form, 'journal_entry': entry, 'voyage': voyage})


@login_required
def like_voyage_view(request, voyage_id):
    """Gère le like/unlike d'un carnet."""
    voyage = get_object_or_404(Voyage, id=voyage_id)
    like, created = Like.objects.get_or_create(voyage=voyage, user=request.user)
    if not created:
        like.delete()
    return redirect('planner:voyage_detail', voyage_id=voyage.id)


@login_required
def download_plan_pdf_view(request):
    """Génère le PDF du plan de voyage initial depuis la session."""
    trip_plan = request.session.get('trip_plan_result_for_pdf')
    trip_params = request.session.get('trip_params_for_pdf')
    if not trip_plan or not trip_params:
        messages.error(request, "Aucun plan de voyage à télécharger.")
        return redirect('planner:plan_trip')
    
    buffer = BytesIO()
    num_days_str = trip_params.get("Durée du voyage", "0").split()[0]
    num_days = int(num_days_str) if num_days_str.isdigit() else 0

    _, schedule = generate_schedule_content_objects_django(trip_plan, num_days)
    generate_trip_pdf_django(buffer, trip_plan, trip_params, schedule)
    
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="plan_de_voyage.pdf"'
    return response


@login_required
def download_voyage_pdf_view(request, voyage_id):
    """Génère le PDF d'un carnet de voyage publié."""
    voyage = get_object_or_404(Voyage, id=voyage_id, author=request.user)
    buffer = BytesIO()
    generate_voyage_pdf_django(buffer, voyage)
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="carnet_{voyage.id}.pdf"'
    return response


@login_required
def rate_item_view(request, item_id):
    """Gère la notation d'une activité."""
    item_to_rate = get_object_or_404(DailyActivityItem, id=item_id)
    if item_to_rate.trip_day.trip.user != request.user:
        return HttpResponseForbidden("Vous n'êtes pas autorisé à noter cet élément.")

    rating_instance = ActivityRating.objects.filter(daily_activity_item=item_to_rate, user=request.user).first()
    
    if request.method == 'POST':
        form = RatingForm(request.POST, instance=rating_instance)
        if form.is_valid():
            new_rating = form.save(commit=False)
            new_rating.daily_activity_item = item_to_rate
            new_rating.user = request.user
            new_rating.save()
            messages.success(request, f'Votre note pour "{item_to_rate.name}" a été enregistrée.')
            return redirect('planner:trip_detail', trip_id=item_to_rate.trip_day.trip.id)
    else:
        form = RatingForm(instance=rating_instance)
        
    context = {
        'form': form,
        'item_to_rate': item_to_rate,
        'page_title': f'Noter : {item_to_rate.name}'
    }
    return render(request, 'planner/rate_item.html', context)


# --- VUES DU CHATBOT ---
@login_required
def chat_interface_view(request):
    chat_history = ChatMessage.objects.filter(user=request.user)
    context = {'chat_history': chat_history}
    return render(request, 'planner/chatbot.html', context)


@login_required
@csrf_exempt
@require_POST
def chatbot_api_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({'reply': 'Veuillez vous connecter pour utiliser le chatbot.'}, status=403)

    try:
        data = json.loads(request.body)
        user_message_text = data.get('message', '')
        image_base64 = data.get('image_base64', None)
        gps_coords = data.get('gps_coords', None)

        # Sauvegarder le message de l'utilisateur
        if user_message_text:
            ChatMessage.objects.create(
                user=request.user,
                message=user_message_text,
                is_from_user=True
            )
        elif image_base64: # Si seulement une image est envoyée, enregistrer un message générique
             ChatMessage.objects.create(
                user=request.user,
                message="[Image envoyée]",
                is_from_user=True
            )

        # Appeler la logique du chatbot
        ai_reply = process_user_query(
            user_message=user_message_text,
            user_id=request.user.id,
            image_base64=image_base64,
            gps_coords=gps_coords
        )

        # Sauvegarder la réponse de l'IA
        ChatMessage.objects.create(
            user=request.user,
            message=ai_reply,
            is_from_user=False
        )

        return JsonResponse({'reply': ai_reply})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Erreur dans chatbot_api_view: {e}", exc_info=True)
        return JsonResponse({'error': f'Internal server error: {e}'}, status=500)