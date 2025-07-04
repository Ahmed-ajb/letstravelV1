# planner/admin.py
from django.contrib import admin
from .models import (
    Profile, 
    Trip, 
    TripDay, 
    DailyActivityItem, 
    ActivityRating,
    Voyage,
    JournalEntry,
    VoyageMedia,
    Comment,
    Like,
    ChatMessage # --- CORRECTION : Importer le nouveau modèle ici ---
)

# Pour une meilleure visualisation des carnets dans l'admin
class VoyageMediaInline(admin.TabularInline):
    """Permet d'ajouter des médias directement depuis une entrée de journal."""
    model = VoyageMedia
    extra = 1

class JournalEntryInline(admin.StackedInline):
    """Permet de voir les entrées de journal directement depuis un carnet."""
    model = JournalEntry
    extra = 0 # N'affiche pas de jour vide par défaut

class CommentInline(admin.TabularInline):
    """Permet de voir les commentaires directement depuis un carnet."""
    model = Comment
    extra = 0
    readonly_fields = ('author', 'body', 'created_at')

@admin.register(Voyage)
class VoyageAdmin(admin.ModelAdmin):
    """Configuration de l'affichage pour les Carnets de Voyage."""
    list_display = ('title', 'author', 'created_at')
    list_filter = ('author',)
    search_fields = ('title', 'author__username')
    inlines = [JournalEntryInline, CommentInline]

@admin.register(JournalEntry)
class JournalEntryAdmin(admin.ModelAdmin):
    """Configuration de l'affichage pour les entrées de journal."""
    list_display = ('__str__', 'voyage', 'day_number')
    inlines = [VoyageMediaInline]

# --- CLASSE AJOUTÉE POUR LE CHATBOT ---
@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    """Configuration de l'affichage pour les messages du chatbot."""
    list_display = ('user', 'is_from_user', 'timestamp', 'message')
    list_filter = ('user', 'is_from_user', 'timestamp')
    search_fields = ('user__username', 'message')
    readonly_fields = ('user', 'message', 'is_from_user', 'timestamp')

# Enregistrement des modèles existants
admin.site.register(Profile)
admin.site.register(Trip)
admin.site.register(TripDay)
admin.site.register(DailyActivityItem)
admin.site.register(ActivityRating)
admin.site.register(Like)
