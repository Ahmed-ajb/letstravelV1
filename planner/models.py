# planner/models.py
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.urls import reverse

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(blank=True, help_text="Une petite biographie.")
    profile_picture = models.ImageField(upload_to='profile_pics/', default='profile_pics/default.jpg')
    TRAVEL_STYLE_CHOICES = [('ADVENTURE', 'Aventure & Plein Air'), ('RELAXATION', 'Détente & Plage'), ('CULTURE', 'Culture & Histoire'), ('BUDGET', 'Budget & Backpacking'), ('LUXURY', 'Luxe & Gastronomie'), ('FAMILY', 'En Famille')]
    travel_style = models.CharField(max_length=20, choices=TRAVEL_STYLE_CHOICES, blank=True)

@receiver(post_save, sender=User)
def create_or_save_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    instance.profile.save()

class Trip(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='generated_trips')
    name = models.CharField(max_length=200, default="Mon Voyage Planifié")
    created_at = models.DateTimeField(auto_now_add=True)
    target_cities_input_str = models.TextField(blank=True, null=True)
    num_days_str = models.CharField(max_length=10, blank=True, null=True)
    num_persons = models.PositiveIntegerField(default=1)

class TripDay(models.Model):
    trip = models.ForeignKey(Trip, on_delete=models.CASCADE, related_name='days')
    day_number = models.PositiveIntegerField()
    city_name = models.CharField(max_length=100, blank=True, null=True)

class DailyActivityItem(models.Model):
    trip_day = models.ForeignKey(TripDay, on_delete=models.CASCADE, related_name='activity_items')
    order_in_day = models.PositiveIntegerField()
    item_type = models.CharField(max_length=20, choices=[('hotel', 'Hôtel'), ('activity', 'Activité')])
    name = models.CharField(max_length=255)
    activity_type_name = models.CharField(max_length=100, blank=True, null=True)

class ActivityRating(models.Model):
    RATING_CHOICES = [(i, str(i)) for i in range(1, 6)]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='activity_ratings')
    daily_activity_item = models.ForeignKey(DailyActivityItem, on_delete=models.CASCADE, related_name="user_ratings")
    rating = models.PositiveSmallIntegerField(choices=RATING_CHOICES)
    comment = models.TextField(blank=True, null=True)

class Voyage(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='voyages')
    title = models.CharField(max_length=200)
    cover_image = models.ImageField(upload_to='voyage_covers/', blank=True, null=True)
    source_trip = models.OneToOneField('Trip', on_delete=models.SET_NULL, null=True, blank=True, related_name='published_voyage')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    def get_absolute_url(self): return reverse('planner:voyage_detail', kwargs={'voyage_id': self.pk})

class JournalEntry(models.Model):
    voyage = models.ForeignKey(Voyage, on_delete=models.CASCADE, related_name='journal_entries')
    day_number = models.PositiveIntegerField()
    title = models.CharField(max_length=200, blank=True)
    story = models.TextField(blank=True)

class VoyageMedia(models.Model):
    journal_entry = models.ForeignKey(JournalEntry, on_delete=models.CASCADE, related_name='media', null=True, blank=True)
    media_file = models.FileField(upload_to='voyage_media/')
    caption = models.CharField(max_length=255, blank=True)
    media_type = models.CharField(max_length=5, choices=[('IMAGE', 'Image'), ('AUDIO', 'Audio')], default='IMAGE')

class Comment(models.Model):
    voyage = models.ForeignKey(Voyage, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='voyage_comments')
    body = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

class Like(models.Model):
    voyage = models.ForeignKey(Voyage, on_delete=models.CASCADE, related_name='likes')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='voyage_likes')

# --- MODÈLE AJOUTÉ POUR L'HISTORIQUE DU CHAT ---
class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_messages')
    message = models.TextField()
    is_from_user = models.BooleanField(default=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        role = "User" if self.is_from_user else "AI"
        return f"{role} message from {self.user.username} at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        ordering = ['timestamp']