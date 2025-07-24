# planner/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
import logging 

from .models import (
    Profile, ActivityRating, Voyage, VoyageMedia, JournalEntry, Comment
)
# Assurez-vous que le chemin est correct si utils.py est dans le même dossier
from .utils import load_and_preprocess_data, MANUAL_CITY_COORDINATES 

logger = logging.getLogger(__name__)

# Logique pour charger les choix de villes et d'activités
try:
    # Pour le chargement initial du formulaire, nous ne ciblons pas de villes spécifiques pour l'API
    # ni n'activons le mode temps réel.
    activities_df_form, _, _, _ = load_and_preprocess_data(use_realtime_api=False, target_cities_for_api=[]) 
    
    if activities_df_form is not None and not activities_df_form.empty:
        logger.debug(f"forms.py: activities_df_form chargé avec {activities_df_form.shape[0]} lignes.")
        
        unique_cities = sorted(list(set(c for c in activities_df_form["ville_normalisee"].unique() if c)))
        ALL_AVAILABLE_CITIES_CHOICES = [(city, city) for city in unique_cities]
        activity_types = sorted(list(set(t for t in activities_df_form["type"].unique() if t)) if "type" in activities_df_form.columns else [])
        ACTIVITY_TYPE_CHOICES = [(atype, atype) for atype in activity_types]
        
        logger.info(f"forms.py: {len(ALL_AVAILABLE_CITIES_CHOICES)} villes chargées pour le formulaire.")
        logger.info(f"forms.py: Villes disponibles: {', '.join([c[0] for c in ALL_AVAILABLE_CITIES_CHOICES])}")
        logger.info(f"forms.py: {len(ACTIVITY_TYPE_CHOICES)} types d'activités chargés pour le formulaire.")
        logger.info(f"forms.py: Types d'activités disponibles: {', '.join([c[0] for c in ACTIVITY_TYPE_CHOICES])}")
    else:
        logger.warning("forms.py: activities_df_form est vide lors du chargement de forms.py. Les choix de villes/activités seront vides.")
        ALL_AVAILABLE_CITIES_CHOICES, ACTIVITY_TYPE_CHOICES = [], []
except Exception as e:
    logger.error(f"forms.py: Erreur CRITIQUE lors du chargement des données initiales pour les choix de formulaire: {e}", exc_info=True)
    ALL_AVAILABLE_CITIES_CHOICES, ACTIVITY_TYPE_CHOICES = [], []

logger.debug(f"forms.py: MANUAL_CITY_COORDINATES (keys): {list(MANUAL_CITY_COORDINATES.keys())}")
logger.debug(f"forms.py: MANUAL_CITY_COORDINATES (full): {MANUAL_CITY_COORDINATES}")


AIRPORT_CITY_CHOICES = [
    ('Casablanca (CMN)', 'Casablanca (Aéroport Mohammed V)'),
    ('Marrakech (RAK)', 'Marrakech (Aéroport Menara)'),
    ('Agadir (AGA)', 'Agadir (Aéroport Al Massira)'),
    ('Tangier (TNG)', 'Tanger (Aéroport Ibn Battouta)'),
    ('Fès (FEZ)', 'Fès (Aéroport Fès–Saïs)'),
    ('Rabat (RBA)', 'Rabat (Aéroport Rabat–Salé)'),
    ('Oujda (OUD)', 'Oujda (Aéroport Angads)'),
    ('Essaouira (ESU)', 'Essaouira (Aéroport Essaouira-Mogador)'),
]

temp_airport_city_choices = list(AIRPORT_CITY_CHOICES) 
# Ajouter les villes normales aussi comme options de départ
for city, coords in MANUAL_CITY_COORDINATES.items():
    is_airport_already_listed = False
    for airport_tuple in temp_airport_city_choices:
        if airport_tuple[0] == city:
            is_airport_already_listed = True
            break
    
    if not is_airport_already_listed: 
        temp_airport_city_choices.append((city, city)) 
        logger.debug(f"forms.py: Ajout de la ville '{city}' aux choix de départ.")
    else:
        logger.debug(f"forms.py: La ville '{city}' est déjà listée comme aéroport, pas d'ajout en double.")

AIRPORT_CITY_CHOICES = sorted(temp_airport_city_choices, key=lambda x: x[1]) 

logger.info(f"forms.py: {len(AIRPORT_CITY_CHOICES)} aéroports/villes de départ chargés au total.")
logger.debug(f"forms.py: Aéroports/Villes de départ disponibles (final): {AIRPORT_CITY_CHOICES}")


class TripPlannerForm(forms.Form):
    target_cities = forms.MultipleChoiceField(
        label="Villes à visiter ?", 
        choices=ALL_AVAILABLE_CITIES_CHOICES, 
        widget=forms.CheckboxSelectMultiple, 
        required=True
    )
    num_days = forms.IntegerField(label="Nombre de jours ?", min_value=1, initial=3)
    
    activity_intensity = forms.ChoiceField(
        label="Quel rythme pour vos journées ?",
        choices=[
            ('relaxed', 'Détendu (peu d\'activités, ~4h)'),
            ('moderate', 'Modéré (rythme équilibré, ~6h)'),
            ('intense', 'Intense (journées bien remplies, ~8h)')
        ],
        initial='moderate',
        widget=forms.RadioSelect,
        required=True
    )
    
    num_persons = forms.IntegerField(label="Nombre de personnes ?", min_value=1, initial=2)
    total_budget = forms.FloatField(label="Budget total (MAD) ?", min_value=100.0, initial=5000.0)
    min_hotel_rating = forms.FloatField(label="Note minimale de l'hôtel ?", min_value=0.0, max_value=10.0, initial=7.5)
    activity_prefs = forms.MultipleChoiceField(
        label="Préférences d'activités ?", 
        choices=ACTIVITY_TYPE_CHOICES, 
        widget=forms.CheckboxSelectMultiple, 
        required=False
    )
    # use_astar_routes_planning reste pour activer/désactiver les calculs OSMnx
    use_astar_routes_planning = forms.BooleanField(
        label="Calculer les itinéraires détaillés (A*)", 
        required=False, 
        initial=True
    )

    # REMOVED: calculate_driving_routes et calculate_walking_routes sont gérés en interne par la carte maintenant.
    # Ils n'apparaissent plus dans le formulaire.

    use_realtime_api = forms.BooleanField(
        label="Rechercher des données en temps réel (API externe)",
        required=False,
        initial=False,
        help_text="Activez pour obtenir les hôtels, restaurants et activités les plus récents via une API de recherche (ex: Google Places via SerpAPI). Note: Cette option peut être plus lente et dépend de la disponibilité de l'API."
    )

    location_status_choice = forms.ChoiceField(
        label="Où vous situez-vous pour le départ de votre voyage ?",
        choices=[
            ('not_in_morocco', 'Je ne suis pas encore au Maroc (je vais arriver par avion)'),
            ('in_morocco', 'Je suis déjà au Maroc'),
        ],
        initial='not_in_morocco', 
        widget=forms.RadioSelect,
        required=True
    )

    start_city_choice = forms.ChoiceField(
        label="Sélectionner un aéroport/ville de départ :",
        choices=[('', '---------')] + AIRPORT_CITY_CHOICES, 
        required=False 
    )

    start_gps_coords = forms.CharField(
        label="Coordonnées GPS (Latitude,Longitude)",
        required=False, 
        help_text="Sera rempli automatiquement si GPS activé."
    )


# --- Autres formulaires (inchangés) ---
class UserUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['bio', 'travel_style', 'profile_picture']

class SignUpForm(UserCreationForm):
    email = forms.EmailField(max_length=254, required=True)
    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('first_name', 'last_name', 'email',)

class RatingForm(forms.ModelForm):
    class Meta:
        model = ActivityRating
        fields = ['rating', 'comment']
        widgets = {'rating': forms.RadioSelect}

class VoyageForm(forms.ModelForm):
    class Meta:
        model = Voyage
        fields = ['title', 'cover_image']
        labels = {
            'title': "Titre du carnet de voyage",
            'cover_image': "Image de couverture",
        }

class JournalEntryForm(forms.ModelForm):
    class Meta:
        model = JournalEntry
        fields = ['title', 'story']
        widgets = {
            'story': forms.Textarea(attrs={'rows': 12, 'placeholder': 'Racontez les moments forts de votre journée ici...'}),
        }
        labels = {
            'title': "Titre de la journée (ex: Découverte de la Médina)",
            'story': "Votre récit",
        }

class VoyageMediaForm(forms.ModelForm):
    class Meta:
        model = VoyageMedia
        fields = ['media_file', 'caption']
        labels = {
            'media_file': "Fichier (Image ou Audio)",
            'caption': "Légende (optionnel)",
        }

class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['body']
        widgets = {
            'body': forms.Textarea(attrs={'rows': 2, 'placeholder': 'Laisser un commentaire...'})
        }
        labels = {
            'body': ""
        }