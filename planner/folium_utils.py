# planner/folium_utils.py

import folium
from folium import plugins
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

DAILY_ROUTE_COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'lightgray', 'black', 'pink']
ACTIVITY_TYPE_ICONS_MAP = {"Culturelle": "landmark", "Aventure": "mountain-sun", "Gastronomique": "utensils", "Historique": "monument", "Loisir": "umbrella-beach", "Bien-être": "spa", "Marché": "store", "Nature": "tree", "Artisanat": "palette", "Sport/Loisir": "person-biking", "Sport/Aventure": "person-skiing", "Shopping": "shopping-bag", "Religieux": "place-of-worship", "Défaut": "map-pin", "hotel": "bed"}

def generate_trip_map_folium_django(ordered_cities, trip_plan_data, city_coords_map_global, num_days_total_trip, use_astar_on_map=True, osmnx_available_flag=False):
    if not ordered_cities:
        logger.warning("generate_trip_map_folium_django: Pas de villes ordonnées fournies.")
        return None

    # Centrage de la carte
    lats = [city_coords_map_global[c]["latitude"] for c in ordered_cities if c in city_coords_map_global and pd.notna(city_coords_map_global[c].get("latitude"))]
    lons = [city_coords_map_global[c]["longitude"] for c in ordered_cities if c in city_coords_map_global and pd.notna(city_coords_map_global[c].get("longitude"))]
    map_center = [np.mean(lats), np.mean(lons)] if lats and lons else [31.7917, -7.0926]
    zoom_start = 6 

    # --- LIGNE MANQUANTE CORRIGÉE ---
    # Création de l'objet carte principal
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="OpenStreetMap")
    # --- FIN DE LA CORRECTION ---


    # Trajet principal inter-villes
    inter_city_coords = [(city_coords_map_global[c]["latitude"], city_coords_map_global[c]["longitude"]) for c in ordered_cities if c in city_coords_map_global]
    if len(inter_city_coords) > 1:
        folium.PolyLine(inter_city_coords, color="#007bff", weight=3, opacity=0.7, dash_array='8, 4', tooltip="Trajet Principal").add_to(m)
    for i, city_name in enumerate(ordered_cities):
        if city_name in city_coords_map_global:
            coords = (city_coords_map_global[city_name]["latitude"], city_coords_map_global[city_name]["longitude"])
            folium.Marker(location=coords, popup=f"<b>Étape {i+1}: {city_name}</b>", icon=folium.Icon(color="darkblue", icon="map-marker", prefix="fa")).add_to(m)


    # Création des groupes de calques
    hotels_fg = folium.FeatureGroup(name="🏨 Hôtels", show=True)
    activities_fg = folium.FeatureGroup(name="📍 Activités", show=True)
    show_astar = use_astar_on_map and osmnx_available_flag
    drive_fg = folium.FeatureGroup(name="🚗 Itinéraires Voiture (A*)", show=show_astar)
    walk_fg = folium.FeatureGroup(name="🚶 Itinéraires Piéton (A*)", show=False)
    geodesic_fg = folium.FeatureGroup(name="📏 Itinéraires Directs", show=not show_astar)

    current_day = 0
    placed_markers = set()

    for city_plan in trip_plan_data:
        daily_plans = city_plan.get('activites_par_jour_optimisees', [])
        daily_drive_segments = city_plan.get('itineraire_voiture_segments_par_jour', [])
        daily_walk_segments = city_plan.get('itineraire_pieton_segments_par_jour', [])
        
        for i, day_plan in enumerate(daily_plans):
            current_day += 1
            if not day_plan: continue

            # Marqueurs
            for point in day_plan:
                if not pd.notna(point.get("latitude")): continue
                coords = (point["latitude"], point["longitude"])
                marker_key = (round(coords[0], 5), round(coords[1], 5))
                if marker_key in placed_markers: continue
                
                is_hotel = point.get('type') == 'hotel'
                fg = hotels_fg if is_hotel else activities_fg
                icon_name = "bed" if is_hotel else ACTIVITY_TYPE_ICONS_MAP.get(point.get('type'), "map-pin")
                color = "darkgreen" if is_hotel else "purple"
                point_name = point.get('name') or point.get('nom')
                popup_text = f"<b>{point_name}</b>" + (f" (Jour {current_day})" if not is_hotel else "")

                folium.Marker(location=coords, popup=popup_text, tooltip=point_name, icon=folium.Icon(color=color, icon=icon_name, prefix="fa")).add_to(fg)
                placed_markers.add(marker_key)
            
            # Itinéraires
            route_color = DAILY_ROUTE_COLORS[(current_day - 1) % len(DAILY_ROUTE_COLORS)]
            
            # Itinéraire A* Voiture
            if show_astar and i < len(daily_drive_segments) and daily_drive_segments[i]:
                for segment in daily_drive_segments[i]:
                    if segment: folium.PolyLine(segment, color=route_color, weight=4, opacity=0.8, tooltip=f"Itinéraire Voiture Jour {current_day}").add_to(drive_fg)
            # Itinéraire A* Piéton
            if show_astar and i < len(daily_walk_segments) and daily_walk_segments[i]:
                for segment in daily_walk_segments[i]:
                    if segment: folium.PolyLine(segment, color=route_color, weight=2, opacity=0.9, dash_array='5,5', tooltip=f"Itinéraire Piéton Jour {current_day}").add_to(walk_fg)
            # Itinéraire Géodésique (fallback)
            elif not show_astar:
                geodesic_path = [ (p['latitude'], p['longitude']) for p in day_plan if pd.notna(p.get('latitude')) ]
                if len(geodesic_path) > 1:
                    folium.PolyLine(geodesic_path, color=route_color, weight=3, opacity=0.7, dash_array='5,5', tooltip=f"Trajet Jour {current_day}").add_to(geodesic_fg)

    # Ajout des calques
    hotels_fg.add_to(m)
    activities_fg.add_to(m)
    drive_fg.add_to(m)
    walk_fg.add_to(m)
    geodesic_fg.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MiniMap().add_to(m)
    
    return m