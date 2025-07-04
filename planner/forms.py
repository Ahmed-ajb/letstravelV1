# planner/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import (
    Profile, ActivityRating, Voyage, VoyageMedia, JournalEntry, Comment
)
from .utils import load_and_preprocess_data

# Logique pour charger les choix de villes et d'activités
try:
    activities_df_form, _, _ = load_and_preprocess_data()
    if activities_df_form is not None and not activities_df_form.empty:
        unique_cities = sorted(list(set(c for c in activities_df_form["ville_normalisee"].unique() if c)))
        ALL_AVAILABLE_CITIES_CHOICES = [(city, city) for city in unique_cities]
        activity_types = sorted(list(set(t for t in activities_df_form["type"].unique() if t)))
        ACTIVITY_TYPE_CHOICES = [(atype, atype) for atype in activity_types]
    else:
        ALL_AVAILABLE_CITIES_CHOICES, ACTIVITY_TYPE_CHOICES = [], []
except Exception:
    ALL_AVAILABLE_CITIES_CHOICES, ACTIVITY_TYPE_CHOICES = [], []

class TripPlannerForm(forms.Form):
    target_cities = forms.MultipleChoiceField(
        label="Villes à visiter ?", 
        choices=ALL_AVAILABLE_CITIES_CHOICES, 
        widget=forms.CheckboxSelectMultiple, 
        required=True
    )
    num_days = forms.IntegerField(label="Nombre de jours ?", min_value=1, initial=3)
    
    # Champ pour que l'utilisateur choisisse le rythme de son voyage
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
    use_astar_routes_planning = forms.BooleanField(
        label="Calculer les itinéraires réels (plus lent)", 
        required=False, 
        initial=False
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
