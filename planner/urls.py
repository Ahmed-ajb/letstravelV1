# planner/urls.py
from django.urls import path
from . import views

app_name = 'planner'

urlpatterns = [
    # URLs existantes
    path('', views.home_showcase_view, name='home_showcase'),
    path('profile/', views.profile_view, name='profile'),
    path('plan/', views.plan_trip_view, name='plan_trip'),
    path('signup/', views.signup_view, name='signup'),
    path('download-plan-pdf/', views.download_plan_pdf_view, name='download_pdf'),
    path('my-trips/', views.my_trips_view, name='my_trips'),
    path('trip/<int:trip_id>/', views.trip_detail_view, name='trip_detail'),
    path('rate-item/<int:item_id>/', views.rate_item_view, name='rate_item'),
    path('trip/<int:trip_id>/publish/', views.publish_trip_as_voyage_view, name='publish_trip'),
    path('voyage/<int:voyage_id>/', views.voyage_detail_view, name='voyage_detail'),
    path('voyage/<int:voyage_id>/like/', views.like_voyage_view, name='like_voyage'),
    path('voyage/<int:voyage_id>/edit/', views.voyage_edit_dashboard_view, name='voyage_edit_dashboard'),
    path('voyage/<int:voyage_id>/edit-day/<int:day_number>/', views.journal_entry_edit_view, name='journal_entry_edit'),
    path('voyage/<int:voyage_id>/download-pdf/', views.download_voyage_pdf_view, name='download_voyage_pdf'),

    # --- URLs POUR LE CHATBOT ---
    path('chatbot/', views.chat_interface_view, name='chat_interface'),
    path('chatbot-api/', views.chatbot_api_view, name='chatbot_api'),
]
