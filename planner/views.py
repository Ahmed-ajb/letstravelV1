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

from .models import *
from .forms import *
from .utils import plan_trip_django, load_and_preprocess_data, OSMNX_AVAILABLE
from .folium_utils import generate_trip_map_folium_django
from .reportlab_utils import generate_trip_pdf_django, generate_voyage_pdf_django, generate_schedule_content_objects_django
from .chatbot_logic import process_user_query

logger = logging.getLogger(__name__)


def home_showcase_view(request):
    """
    Charge et affiche les activités sur la page d'accueil.
    """
    showcase_context = {}
    try:
        activities_df, _, _ = load_and_preprocess_data()
        if not activities_df.empty and 'ville_normalisee' in activities_df.columns:
            featured_cities = ['Marrakech', 'Fès', 'Essaouira']
            available_cities = [c for c in featured_cities if c in activities_df['ville_normalisee'].unique()]
            showcase_activities = {}
            for city in available_cities:
                city_activities_df = activities_df[activities_df['ville_normalisee'] == city]
                if not city_activities_df.empty:
                    showcase_activities[city] = city_activities_df.sort_values(by='rating', ascending=False).head(3).to_dict('records')
            
            showcase_context = {'showcase_activities_by_city': showcase_activities}
            
            if not showcase_activities:
                 messages.info(request, "Aucune activité à afficher pour les villes vedettes (Marrakech, Fès, Essaouira). Vérifiez vos fichiers de données.")
        else:
            messages.warning(request, "Les données des activités n'ont pas pu être chargées ou sont mal formatées.")

    except Exception as e:
        print(f"Erreur lors de la préparation de l'aperçu : {e}")
        messages.error(request, "Une erreur est survenue lors du chargement des activités.")
        
    return render(request, 'planner/home_showcase.html', showcase_context)

@login_required
def plan_trip_view(request):
    """Gère la planification et affiche les résultats sur la même page."""
    if request.method == 'POST':
        form = TripPlannerForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            activities_df, hotels_df, city_coords_map = load_and_preprocess_data()
            
            if activities_df.empty or 'ville_normalisee' not in activities_df.columns:
                messages.error(request, "Impossible de charger les données des activités. Le fichier est peut-être vide ou mal formaté.")
                return render(request, 'planner/plan_trip.html', {'form': form})

            use_astar = data.get('use_astar_routes_planning', False)
            
            # --- MODIFICATION : Passer le nouveau paramètre d'intensité ---
            trip_plan, params, cities = plan_trip_django(
                target_cities_list=data['target_cities'], 
                total_budget_str=str(data['total_budget']), 
                num_days_str=str(data['num_days']), 
                num_persons=data['num_persons'],
                min_hotel_rating_str=str(data['min_hotel_rating']), 
                activity_preferences_str=",".join(data['activity_prefs']),
                activity_intensity=data['activity_intensity'],  # Ajout du paramètre
                activities_df_global=activities_df, 
                hotels_df_global=hotels_df, 
                city_coords_map_global=city_coords_map, 
                use_astar_for_planning=use_astar
            )
            
            if trip_plan:
                new_trip = Trip.objects.create(user=request.user, name=f"Plan pour {', '.join(cities)}", num_persons=data['num_persons'])
                day_counter = 0
                for city_data in trip_plan:
                    for daily_plan in city_data.get('activites_par_jour_optimisees', []):
                        day_counter += 1
                        trip_day = TripDay.objects.create(trip=new_trip, day_number=day_counter, city_name=city_data['ville'])
                        for i, item in enumerate(daily_plan):
                            DailyActivityItem.objects.create(trip_day=trip_day, order_in_day=i, item_type='hotel' if item.get('type') == 'hotel' else 'activity', name=item.get('nom') or item.get('name'))
                messages.success(request, "Voyage planifié et sauvegardé !")
                request.session['trip_plan_result_for_pdf'] = trip_plan
                request.session['trip_params_for_pdf'] = params
                folium_map = generate_trip_map_folium_django(cities, trip_plan, city_coords_map, int(data['num_days']), use_astar, OSMNX_AVAILABLE)
                schedule_md, _ = generate_schedule_content_objects_django(trip_plan, int(data['num_days']))
                return render(request, 'planner/plan_trip.html', {'form': form, 'trip_plan_result': trip_plan, 'trip_params': params, 'folium_map_html': folium_map._repr_html_(), 'schedule_md': schedule_md})
            else:
                messages.warning(request, "Impossible de générer un plan avec les critères fournis. Essayez d'ajuster vos préférences ou votre budget.")

    form = TripPlannerForm()
    return render(request, 'planner/plan_trip.html', {'form': form})

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
        form = UserCreationForm(request.POST) 
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Inscription réussie ! Vous êtes maintenant connecté.")
            return redirect('planner:home_showcase')
    else:
        form = UserCreationForm()
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
            rating = item.user_ratings.filter(user=request.user).first()
            items_for_day.append({'db_item': item, 'user_has_rated': bool(rating), 'user_rating': rating.rating if rating else 0})
        days_with_items.append({'day_obj': day, 'items': items_for_day})
    return render(request, 'planner/trip_detail.html', {'trip': trip, 'days_with_items': days_with_items})

@login_required
def publish_trip_as_voyage_view(request, trip_id):
    """Crée un 'Voyage' (carnet) à partir d'un 'Trip' (plan)."""
    trip = get_object_or_404(Trip, id=trip_id, user=request.user)
    voyage, created = Voyage.objects.get_or_create(author=request.user, source_trip=trip, defaults={'title': f"Carnet pour : {trip.name}"})
    if created and trip.days.exists():
        for trip_day in trip.days.all():
            JournalEntry.objects.get_or_create(voyage=voyage, day_number=trip_day.day_number, defaults={'title': f"Jour à {trip_day.city_name}"})
    messages.success(request, "Votre carnet de voyage est prêt à être modifié.")
    return redirect('planner:voyage_edit_dashboard', voyage_id=voyage.id)

@login_required
def voyage_detail_view(request, voyage_id):
    """Affiche un carnet de voyage privé."""
    voyage = get_object_or_404(Voyage, id=voyage_id, author=request.user)
    comment_form = CommentForm(request.POST or None)
    if request.method == 'POST' and comment_form.is_valid():
        comment = comment_form.save(commit=False); comment.voyage, comment.author = voyage, request.user; comment.save()
        return redirect('planner:voyage_detail', voyage_id=voyage.id)
    context = {'voyage': voyage, 'journal_entries': voyage.journal_entries.all().prefetch_related('media'), 'comments': voyage.comments.all().order_by('-created_at'), 'comment_form': comment_form, 'is_liked': voyage.likes.filter(user=request.user).exists(), 'like_count': voyage.likes.count()}
    return render(request, 'planner/voyage_detail.html', context)

@login_required
def voyage_edit_dashboard_view(request, voyage_id):
    """Tableau de bord pour éditer les infos générales d'un carnet."""
    voyage = get_object_or_404(Voyage, id=voyage_id, author=request.user)
    form = VoyageForm(request.POST or None, request.FILES or None, instance=voyage)
    if request.method == 'POST' and form.is_valid():
        form.save(); messages.success(request, 'Informations mises à jour.'); return redirect(request.path)
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
            form.save(); messages.success(request, f'Récit du jour {day_number} sauvegardé.'); return redirect(request.path)
        if 'add_media' in request.POST and media_form.is_valid():
            media = media_form.save(commit=False); media.journal_entry = entry
            file = request.FILES.get('media_file')
            if file and 'audio' in file.content_type: media.media_type = 'AUDIO'
            else: media.media_type = 'IMAGE'
            media.save(); messages.success(request, 'Média ajouté.'); return redirect(request.path)
    return render(request, 'planner/journal_entry_form.html', {'form': form, 'media_form': media_form, 'journal_entry': entry, 'voyage': voyage})

@login_required
def like_voyage_view(request, voyage_id):
    """Gère le like/unlike d'un carnet."""
    voyage = get_object_or_404(Voyage, id=voyage_id, author=request.user)
    like, created = Like.objects.get_or_create(voyage=voyage, user=request.user)
    if not created: like.delete()
    return redirect('planner:voyage_detail', voyage_id=voyage.id)

@login_required
def download_plan_pdf_view(request):
    """Génère le PDF du plan de voyage initial depuis la session."""
    trip_plan, trip_params = request.session.get('trip_plan_result_for_pdf'), request.session.get('trip_params_for_pdf')
    if not trip_plan or not trip_params: return redirect('planner:plan_trip')
    buffer = BytesIO()
    num_days = int(trip_params.get("Durée du voyage", "0").split()[0])
    _, schedule = generate_schedule_content_objects_django(trip_plan, num_days)
    generate_trip_pdf_django(buffer, trip_plan, trip_params, schedule)
    buffer.seek(0); response = HttpResponse(buffer, content_type='application/pdf'); response['Content-Disposition'] = 'attachment; filename="plan_de_voyage.pdf"'; return response

@login_required
def download_voyage_pdf_view(request, voyage_id):
    """Génère le PDF d'un carnet de voyage publié."""
    voyage = get_object_or_404(Voyage, id=voyage_id, author=request.user)
    buffer = BytesIO(); generate_voyage_pdf_django(buffer, voyage); buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf'); response['Content-Disposition'] = f'attachment; filename="carnet_{voyage.id}.pdf"'; return response

@login_required
def rate_item_view(request, item_id):
    """Gère la notation d'une activité."""
    item_to_rate = get_object_or_404(DailyActivityItem, id=item_id)
    if item_to_rate.trip_day.trip.user != request.user:
        return HttpResponseForbidden()
    try:
        rating_instance = ActivityRating.objects.get(daily_activity_item=item_to_rate, user=request.user)
    except ActivityRating.DoesNotExist:
        rating_instance = None
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
    context = { 'chat_history': chat_history }
    return render(request, 'planner/chatbot.html', context)

@login_required
@csrf_exempt
@require_POST
def chatbot_api_view(request):
    try:
        data = json.loads(request.body)
        user_message_text = data.get('message')
        if not user_message_text:
            return JsonResponse({'error': 'Message manquant.'}, status=400)
        ChatMessage.objects.create(user=request.user, message=user_message_text, is_from_user=True)
        history = ChatMessage.objects.filter(user=request.user).order_by('timestamp')
        user_name = request.user.first_name or request.user.username
        ai_response_text = process_user_query(user_message_text, user_name, list(history))
        ChatMessage.objects.create(user=request.user, message=ai_response_text, is_from_user=False)
        return JsonResponse({'reply': ai_response_text})
    except Exception as e:
        logger.error(f"Erreur dans l'API du chatbot: {e}", exc_info=True)
        return JsonResponse({'error': 'Erreur interne du serveur.'}, status=500)
