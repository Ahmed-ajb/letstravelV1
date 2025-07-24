import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itertools import permutations
from geopy.distance import geodesic
import requests
from serpapi import GoogleSearch
import os
import pickle
import logging
from django.conf import settings
import re
import math
from datetime import datetime, timedelta, date
import folium
from folium.plugins import MarkerCluster

logger = logging.getLogger(__name__)

# --- Constantes et Configuration ---
ACTIVITIES_JSON_PATH = settings.DATA_DIR / 'activities.json'
HOTELS_JSON_PATH = settings.DATA_DIR / 'hotels_with_real_coordinates_vf_v2.json'
GRAPHS_CACHE_DIR_DJANGO = settings.GRAPHS_CACHE_DIR_DJANGO

OSMNX_AVAILABLE = False
try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
    logger.info("Bibliothèques osmnx et networkx chargées avec succès.")
except ImportError:
    logger.warning("Osmnx ou Networkx non trouvés. Le calcul d'itinéraire A* sera désactivé.")
    pass

if OSMNX_AVAILABLE and not os.path.exists(GRAPHS_CACHE_DIR_DJANGO):
    try:
        os.makedirs(GRAPHS_CACHE_DIR_DJANGO)
    except OSError as e:
        logger.error(f"Impossible de créer le dossier de cache {GRAPHS_CACHE_DIR_DJANGO}: {e}")

CITY_NAME_MAPPING = {
    "Marrakech": ["marakech", "marrakesh"],
    "Fès": ["fez", "fes", "fes el bali", "fès el bali"],
    "Casablanca": ["casa"],
    "Meknès": ["meknes", "meknès el bali"],
    "Rabat": [], "Agadir": [], "Chefchaouen": ["chefchaouene", "chaouen"],
    "Essaouira": ["mogador"], "Ouarzazate": [], "Tangier": ["tanger", "tanger-assilah"],
    "Merzouga (Erg Chebbi)": ["merzouga", "erg chebbi"],
    "Casablanca (CMN)": ["cmn", "mohammed v international airport"],
    "Marrakech (RAK)": ["rak", "marrakech menara airport"],
    "Agadir (AGA)": ["aga", "al massira airport"],
    "Tangier (TNG)": ["tng", "tangier ibn battouta airport"],
    "Fès (FEZ)": ["fez", "fes saïs airport"],
    "Oujda (OUD)": ["oud", "angads airport"],
    "Rabat (RBA)": ["rba", "rabat salé airport"],
    "Essaouira (ESU)": ["esu", "essaouira-mogador airport"]
}
MANUAL_CITY_COORDINATES = {
    "Marrakech": {"latitude": 31.6295, "longitude": -7.9811},
    "Fès": {"latitude": 34.0181, "longitude": -5.0078},
    "Casablanca": {"latitude": 33.5731, "longitude": -7.5898},
    "Meknès": {"latitude": 33.8935, "longitude": -5.5473},
    "Rabat": {"latitude": 34.0209, "longitude": -6.8417},
    "Agadir": {"latitude": 30.4202, "longitude": -9.5981},
    "Chefchaouen": {"latitude": 35.1688, "longitude": -5.2636},
    "Essaouira": {"latitude": 31.5085, "longitude": -9.7595},
    "Ouarzazate": {"latitude": 30.9189, "longitude": -6.8934},
    "Tangier": {"latitude": 35.7595, "longitude": -5.8330},
    "Merzouga (Erg Chebbi)": {"latitude": 31.0983, "longitude": -4.0119},
    "Casablanca (CMN)": {"latitude": 33.3675, "longitude": -7.5899},
    "Marrakech (RAK)": {"latitude": 31.6069, "longitude": -8.0363},
    "Agadir (AGA)": {"latitude": 30.3250, "longitude": -9.4130},
    "Tangier (TNG)": {"latitude": 35.7260, "longitude": -9.2890},
    "Fès (FEZ)": {"latitude": 33.9306, "longitude": -4.9774},
    "Oujda (OUD)": {"latitude": 34.7892, "longitude": -1.9234},
    "Rabat (RBA)": {"latitude": 34.0514, "longitude": -6.7515},
    "Essaouira (ESU)": {"latitude": 31.3952, "longitude": -9.6816}
}

TAVILY_API_KEY = settings.TAVILY_API_KEY if hasattr(settings, 'TAVILY_API_KEY') else None
SERPAPI_API_KEY = settings.SERPAPI_API_KEY if hasattr(settings, 'SERPAPI_API_KEY') else None
MAX_API_RESULTS_PER_CATEGORY = 20
DEFAULT_RADIUS_KM = 15

# --- Fonctions Utilitaires ---
def normalize_city_name(city_name, city_mapping):
    if not city_name or pd.isna(city_name): return None
    city_name_lower = str(city_name).strip().lower()
    for canonical, variations in city_mapping.items():
        if city_name_lower == canonical.lower() or city_name_lower in [v.lower() for v in variations]: return canonical
    return str(city_name).strip().capitalize()

def extract_city_from_hotel_location(location_str, canonical_activity_cities_list, city_name_mapping):
    if not location_str or pd.isna(location_str): return None
    location_lower = str(location_str).lower().replace(',', ' ')
    for city_canonical in canonical_activity_cities_list:
        if f" {city_canonical.lower()} " in f" {location_lower} ": return city_canonical
        if city_canonical in city_name_mapping:
            for variation in city_name_mapping[city_canonical]:
                if f" {variation.lower()} " in f" {location_lower} ": return city_canonical
    return None

def parse_price_string(price_str):
    if not isinstance(price_str, str): return None
    numeric_price = re.sub(r'[^\d\.,]+', '', price_str).replace(',', '.')
    try: return float(numeric_price)
    except ValueError: return None

# --- Fonctions API ---
def _call_serpapi(query, engine, api_key=SERPAPI_API_KEY, **kwargs):
    if not api_key:
        logger.error("SERPAPI_API_KEY non configurée.")
        return {"error": "API key is missing"}
    params = {"api_key": api_key, "engine": engine, "q": query, "hl": "fr", "num": MAX_API_RESULTS_PER_CATEGORY}
    params.update(kwargs)
    try: return GoogleSearch(params).get_dict()
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à SerpApi pour '{query}': {e}", exc_info=True)
        return {"error": str(e)}

def _process_serpapi_places_results(results_json, item_type, city_name):
    processed_items = []
    if results_json is None or results_json.get("error"): return processed_items
    results_list = results_json.get("hotels_results") or results_json.get("local_results", [])
    for item in results_list:
        name = item.get("title") or item.get("name")
        if not name: continue
        gps_coords = item.get("gps_coordinates", {})
        latitude, longitude = gps_coords.get("latitude"), gps_coords.get("longitude")
        if not latitude or not longitude: continue
        rating = item.get("rating")
        data = {
            "nom": name, "latitude": latitude, "longitude": longitude, "coordonnees": (latitude, longitude),
            "ville_normalisee": city_name, "booking_link": item.get("link") or item.get("website"),
            "description": item.get("description") or item.get("address"), "rating": (rating * 2 if rating else 7.0)
        }
        if "hotel" in item_type:
            data.update({"type": "hotel", "duree_estimee": "24h", "budget_estime": parse_price_string(item.get("price")) or 150.0})
        else:
            price_range = item.get("price", "")
            budget_map = {"$": 20, "$$": 50, "$$$": 100, "$$$$": 200}
            data.update({"duree_estimee": "1.5h", "budget_estime": budget_map.get(price_range, 50.0)})
            category = item.get("type", "")
            if "Restaurant" in category: data["type"] = "Gastronomique"
            elif "Cafe" in category: data["type"] = "Gastronomique/Café"
            else: data["type"] = item_type
        processed_items.append(data)
    return processed_items

def get_realtime_data_for_city(city_name, city_coords_map):
    check_in_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    check_out_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
    hotel_results = _call_serpapi(f"hotels in {city_name}, Morocco", "google_hotels", check_in_date=check_in_date, check_out_date=check_out_date)
    restaurant_results = _call_serpapi(f"restaurants in {city_name}, Morocco", "google_maps")
    cafe_results = _call_serpapi(f"cafes in {city_name}, Morocco", "google_maps")
    hotels_list = _process_serpapi_places_results(hotel_results, "hotel", city_name)
    restaurants_cafes_list = _process_serpapi_places_results(restaurant_results, "Gastronomique", city_name)
    restaurants_cafes_list.extend(_process_serpapi_places_results(cafe_results, "Gastronomique/Café", city_name))
    return pd.DataFrame(hotels_list), pd.DataFrame(restaurants_cafes_list)

# --- Chargement et Cache des Données ---
_CACHED_STATIC_DATA, _CACHED_API_DATA = {}, {}
def load_and_preprocess_data(use_realtime_api=False, target_cities_for_api=None):
    global _CACHED_STATIC_DATA, _CACHED_API_DATA
    if not _CACHED_STATIC_DATA:
        try:
            with open(ACTIVITIES_JSON_PATH, 'r', encoding='utf-8') as f:
                activities_list = [{**act, "ville_normalisee": normalize_city_name(city.get("ville"), CITY_NAME_MAPPING)} for city in json.load(f) for act in city.get("activites", [])]
            static_activities_df = pd.DataFrame(activities_list)
            with open(HOTELS_JSON_PATH, 'r', encoding='utf-8') as f:
                hotels_list = json.load(f).get("hotels", [])
            static_hotels_df = pd.DataFrame(hotels_list)
            if not static_hotels_df.empty:
                if 'name' in static_hotels_df.columns: static_hotels_df.rename(columns={'name': 'nom'}, inplace=True)
                cities = sorted(static_activities_df["ville_normalisee"].dropna().unique()) if not static_activities_df.empty else []
                static_hotels_df['ville_normalisee'] = static_hotels_df['location'].apply(lambda x: extract_city_from_hotel_location(x, cities, CITY_NAME_MAPPING))
            _CACHED_STATIC_DATA = {'activities_df': static_activities_df, 'hotels_df': static_hotels_df, 'city_coords_map_global': MANUAL_CITY_COORDINATES}
        except Exception as e:
            logger.error(f"Erreur CRITIQUE lors du chargement des fichiers JSON: {e}", exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    final_activities_df = _CACHED_STATIC_DATA['activities_df'].copy()
    final_hotels_df = _CACHED_STATIC_DATA['hotels_df'].copy()
    api_rc_df_combined = pd.DataFrame()

    if use_realtime_api and target_cities_for_api:
        api_hotels_dfs, api_rc_dfs = [], []
        for city in target_cities_for_api:
            if city in _CACHED_API_DATA and _CACHED_API_DATA[city]['date'] == date.today():
                hotels_df, rc_df = _CACHED_API_DATA[city]['data']
            else:
                hotels_df, rc_df = get_realtime_data_for_city(city, _CACHED_STATIC_DATA['city_coords_map_global'])
                _CACHED_API_DATA[city] = {'date': date.today(), 'data': (hotels_df, rc_df)}
            if not hotels_df.empty: api_hotels_dfs.append(hotels_df)
            if not rc_df.empty: api_rc_dfs.append(rc_df)
        if api_hotels_dfs: final_hotels_df = pd.concat(api_hotels_dfs, ignore_index=True)
        if api_rc_dfs: api_rc_df_combined = pd.concat(api_rc_dfs, ignore_index=True)

    for df in [final_activities_df, final_hotels_df, api_rc_df_combined]:
        if not df.empty:
            df['coordonnees'] = df.apply(lambda r: (r.get('latitude'), r.get('longitude')) if pd.notna(r.get('latitude')) else None, axis=1)
            df.dropna(subset=['coordonnees'], inplace=True)
    return final_activities_df, final_hotels_df, api_rc_df_combined, _CACHED_STATIC_DATA['city_coords_map_global']

# --- Fonctions de Calcul d'Itinéraire ---
def parse_duration_to_hours(duration_str):
    if pd.isna(duration_str): return 1.5
    try:
        duration_str_lower = str(duration_str).lower()
        if "journée" in duration_str_lower and "demi" not in duration_str_lower: return 6.0
        if "demi-journée" in duration_str_lower: return 3.5
        hours_match = re.search(r'(\d+\.?\d*)\s?h', duration_str_lower)
        if hours_match: return float(hours_match.group(1))
        return 1.5
    except: return 1.5

def calculate_daily_routes_osmnx(daily_points, city_name, network_type='drive'):
    valid_points = [p for p in daily_points if p.get('coordonnees') and all(pd.notna(c) for c in p['coordonnees'])]
    if not OSMNX_AVAILABLE or len(valid_points) < 2: return []
    try:
        graph_path = GRAPHS_CACHE_DIR_DJANGO / f"{city_name.replace(' ', '_')}_{network_type}.graphml"
        G = ox.load_graphml(graph_path) if os.path.exists(graph_path) else ox.graph_from_place(f"{city_name}, Morocco", network_type=network_type, buffer=2000)
        if not os.path.exists(graph_path): ox.save_graphml(G, graph_path)
    except Exception as e:
        logger.error(f"OSMNX: Erreur de graphe pour {city_name}: {e}")
        return []
    segments = []
    for i in range(len(valid_points) - 1):
        try:
            start_c, end_c = valid_points[i]['coordonnees'], valid_points[i+1]['coordonnees']
            start_node, end_node = ox.distance.nearest_nodes(G, start_c[1], start_c[0]), ox.distance.nearest_nodes(G, end_c[1], end_c[0])
            route = nx.shortest_path(G, start_node, end_node, weight='length')
            segments.append([(G.nodes[n]['y'], G.nodes[n]['x']) for n in route])
        except Exception:
            logger.warning(f"OSMNX: Itinéraire impossible. Ligne directe.")
            segments.append([start_c, end_c])
    return segments

def find_optimal_path_permutations(cities_to_visit_list, city_coordinates_map, start_point_name=None, start_point_coords=None):
    if start_point_name and start_point_coords:
        city_coordinates_map[start_point_name] = start_point_coords
    locations = [start_point_name] + list(cities_to_visit_list) if start_point_name else list(cities_to_visit_list)
    if len(locations) <= 1: return locations
    start_city = locations[0]
    other_cities = locations[1:]
    if len(other_cities) > 8:
        path, remaining = [start_city], set(other_cities)
        while remaining:
            last_city = path[-1]
            next_city = min(remaining, key=lambda city: geodesic((city_coordinates_map[last_city]['latitude'], city_coordinates_map[last_city]['longitude']), (city_coordinates_map[city]['latitude'], city_coordinates_map[city]['longitude'])).km)
            path.append(next_city)
            remaining.remove(next_city)
        return path
    else:
        min_dist, best_path = float('inf'), []
        for p in permutations(other_cities):
            current_path = [start_city] + list(p)
            dist = sum(geodesic((city_coordinates_map[current_path[i]]['latitude'], city_coordinates_map[current_path[i]]['longitude']), (city_coordinates_map[current_path[i+1]]['latitude'], city_coordinates_map[current_path[i+1]]['longitude'])).km for i in range(len(current_path) - 1))
            if dist < min_dist:
                min_dist, best_path = dist, current_path
        return best_path

def optimize_daily_path(points):
    if len(points) <= 1: return points
    start_point = points[0]
    other_points = points[1:]
    path = [start_point]
    remaining = other_points
    while remaining:
        last_point_coords = path[-1]['coordonnees']
        next_point = min(remaining, key=lambda p: geodesic(last_point_coords, p['coordonnees']).km)
        path.append(next_point)
        remaining.remove(next_point)
    return path

# --- Logique de Recommandation ---
def recommend_for_city_django(city_name, hotels_df_global, activities_df_global, api_restaurants_cafes_df_global, budget_activities_for_stay_in_city, min_hotel_rating, activity_preferences, activity_intensity, num_persons=1, num_days_in_city=1, use_astar_for_planning=True, start_point_coords_for_day_1=None):
    recommendations = {"ville": city_name, "jours_alloues": num_days_in_city}
    default_day_plan = [{"nom": "Repos / Exploration libre", "type": "Loisir"}]

    city_hotels = hotels_df_global[(hotels_df_global["ville_normalisee"] == city_name) & (hotels_df_global["rating"] >= min_hotel_rating)]
    recommendations["top_5_hotels"] = city_hotels.sort_values(by="rating", ascending=False).head(5).to_dict('records')
    
    top_restaurants, top_cafes = [], []
    if not api_restaurants_cafes_df_global.empty:
        city_rc = api_restaurants_cafes_df_global[api_restaurants_cafes_df_global["ville_normalisee"] == city_name]
        top_restaurants = city_rc[city_rc['type'] == 'Gastronomique'].sort_values(by='rating', ascending=False).head(num_days_in_city).to_dict('records')
        top_cafes = city_rc[city_rc['type'] == 'Gastronomique/Café'].sort_values(by='rating', ascending=False).head(num_days_in_city).to_dict('records')
    recommendations["top_2_restaurants"], recommendations["top_2_cafes"] = top_restaurants, top_cafes

    candidate_activities = activities_df_global[activities_df_global["ville_normalisee"] == city_name].copy()
    if activity_preferences and any(p.strip() for p in activity_preferences):
        candidate_activities = candidate_activities[candidate_activities['type'].isin(activity_preferences)]
    
    if candidate_activities.empty:
        recommendations["activites_par_jour_optimisees"] = [default_day_plan] * num_days_in_city
        recommendations.update({'budget_activites_depense': 0, 'itineraire_voiture_segments_par_jour': [], 'itineraire_pieton_segments_par_jour': []})
        return recommendations

    candidate_activities['duration_hours'] = candidate_activities['duree_estimee'].apply(parse_duration_to_hours)
    for col in ['rating', 'budget_estime']:
        candidate_activities[col] = pd.to_numeric(candidate_activities[col], errors='coerce').fillna(candidate_activities[col].median())
    scaler = MinMaxScaler()
    candidate_activities['score'] = scaler.fit_transform(candidate_activities[['rating']]) - scaler.fit_transform(candidate_activities[['budget_estime']])
    candidate_activities['value_score'] = candidate_activities['score'] / candidate_activities['duration_hours'].replace(0, 1)
    candidate_activities = candidate_activities.sort_values(by="value_score", ascending=False)
    
    intensity_hours = {'relaxed': 4.0, 'moderate': 6.0, 'intense': 8.0}.get(activity_intensity, 6.0)
    daily_plans, total_spent, used_indices = [], 0, set()
    used_restaurants, used_cafes = [], []

    for day_num in range(num_days_in_city):
        points_to_visit_today, day_hours = [], 0.0
        
        if day_num == 0 and start_point_coords_for_day_1:
            points_to_visit_today.append({"nom": "Votre Point de Départ", "type": "Point de départ", "coordonnees": start_point_coords_for_day_1, "duree_estimee": "0h"})
        elif recommendations.get("top_5_hotels"):
            points_to_visit_today.append(recommendations["top_5_hotels"][0])
        
        selected_activities = []
        for index, activity in candidate_activities.iterrows():
            if index in used_indices: continue
            duration, cost = activity['duration_hours'], activity['budget_estime'] * num_persons
            if day_hours + duration <= intensity_hours and total_spent + cost <= budget_activities_for_stay_in_city:
                selected_activities.append(activity.to_dict())
                day_hours += duration
                total_spent += cost
                used_indices.add(index)
        points_to_visit_today.extend(selected_activities)
        
        if top_restaurants and len(used_restaurants) < len(top_restaurants):
            restaurant = top_restaurants[len(used_restaurants)]
            points_to_visit_today.append(restaurant)
            used_restaurants.append(restaurant)
        if top_cafes and len(used_cafes) < len(top_cafes):
            cafe = top_cafes[len(used_cafes)]
            points_to_visit_today.append(cafe)
            used_cafes.append(cafe)
        if recommendations.get("top_5_hotels"):
            points_to_visit_today.extend(recommendations["top_5_hotels"][1:])

        optimized_plan = optimize_daily_path(points_to_visit_today)
        daily_plans.append(optimized_plan if len(optimized_plan) > 1 else (optimized_plan + default_day_plan))
        
    recommendations['activites_par_jour_optimisees'] = daily_plans
    recommendations['budget_activites_depense'] = total_spent
    
    if use_astar_for_planning:
        all_driving, all_walking = [], []
        for plan in daily_plans:
            all_driving.extend(calculate_daily_routes_osmnx(plan, city_name, 'drive'))
            all_walking.extend(calculate_daily_routes_osmnx(plan, city_name, 'walk'))
        recommendations['itineraire_voiture_segments_par_jour'] = all_driving
        recommendations['itineraire_pieton_segments_par_jour'] = all_walking
    else:
        recommendations['itineraire_voiture_segments_par_jour'], recommendations['itineraire_pieton_segments_par_jour'] = [], []
    
    return recommendations

# --- Fonction Principale d'Orchestration ---
def plan_trip_django(target_cities_list, total_budget_str, num_days_str, num_persons, min_hotel_rating_str, activity_preferences_str, activity_intensity, activities_df_global, hotels_df_global, api_restaurants_cafes_df_global, city_coords_map_global, use_astar_for_planning, start_location_type=None, start_location_value=None):
    try:
        total_budget, num_days, min_rating = float(total_budget_str), int(num_days_str), float(min_hotel_rating_str)
        persons = int(num_persons)
    except (ValueError, TypeError):
        return {"trip_plan_result": None, "params": {"error": "Paramètres numériques invalides."}}

    start_point_name, start_point_coords = None, None
    if start_location_type == 'current_gps' and start_location_value:
        try:
            lat, lon = map(float, start_location_value.split(','))
            start_point_name, start_point_coords = "Ma position actuelle", {"latitude": lat, "longitude": lon}
        except (ValueError, IndexError): pass
    elif start_location_type == 'choose_city' and start_location_value:
        start_point_name, start_point_coords = start_location_value, city_coords_map_global.get(start_location_value)

    ordered_cities = find_optimal_path_permutations(target_cities_list, city_coords_map_global, start_point_name, start_point_coords)
    actual_cities_to_visit = [c for c in ordered_cities if c != start_point_name] if start_point_name else ordered_cities

    if not actual_cities_to_visit:
        return {"trip_plan_result": None, "params": {"error": "Aucune ville de destination valide trouvée."}}

    days_per_city = [num_days // len(actual_cities_to_visit)] * len(actual_cities_to_visit)
    for i in range(num_days % len(actual_cities_to_visit)): days_per_city[i] += 1
    
    budget_for_activities = total_budget * 0.5

    trip_plan_final = []
    for i, city in enumerate(actual_cities_to_visit):
        start_coords_day1 = (start_point_coords['latitude'], start_point_coords['longitude']) if i == 0 and start_point_coords else None
        city_rec = recommend_for_city_django(
            city, hotels_df_global, activities_df_global, api_restaurants_cafes_df_global,
            budget_for_activities, min_rating, activity_preferences_str.split(','),
            activity_intensity, persons, days_per_city[i], use_astar_for_planning, start_coords_day1
        )
        trip_plan_final.append(city_rec)

    params = {"Villes demandées": ", ".join(target_cities_list), "Ordre de visite suggéré": ", ".join(ordered_cities), "Durée du voyage": f"{num_days} jours"}
    
    folium_map_html = generate_trip_map_folium_v2(trip_plan_final, ordered_cities, city_coords_map_global, start_point_coords, start_point_name)
    
    schedule_md = ""
    try:
        from .reportlab_utils import generate_schedule_content_objects_django
        schedule_md, _ = generate_schedule_content_objects_django(trip_plan_final, num_days)
    except ImportError:
        schedule_md = "Génération de l'emploi du temps non disponible."

    return {"trip_plan_result": trip_plan_final, "params": params, "ordered_cities_with_start": ordered_cities, "folium_map_html": folium_map_html, "schedule_md": schedule_md}


# --- CARTE AMÉLIORÉE AVEC PLUSIEURS THÈMES CLAIRS ---
def generate_trip_map_folium_v2(trip_plan_result, ordered_cities_list, city_coords_map_global, start_point_coords=None, start_point_name="Point de départ"):
    all_coords = []
    def get_coords(item):
        if item and 'coordonnees' in item and all(pd.notna(c) for c in item['coordonnees']):
            return item['coordonnees']
        return None

    if start_point_coords: all_coords.append((start_point_coords['latitude'], start_point_coords['longitude']))
    for plan in trip_plan_result:
        for cat in ['top_5_hotels', 'top_2_restaurants', 'top_2_cafes']:
            for item in plan.get(cat, []):
                coords = get_coords(item)
                if coords: all_coords.append(coords)
        for day in plan.get('activites_par_jour_optimisees', []):
            for item in day:
                coords = get_coords(item)
                if coords: all_coords.append(coords)
    
    if not all_coords:
        return folium.Map(location=[32, -5], zoom_start=6, tiles="CartoDB Positron")._repr_html_()

    center_lat = sum(p[0] for p in all_coords) / len(all_coords)
    center_lon = sum(p[1] for p in all_coords) / len(all_coords)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap", name="Vue Standard")
    folium.TileLayer("CartoDB Positron", name="Vue Minimaliste", attr="© <a href='https://carto.com/'>CartoDB</a>").add_to(m)
    folium.TileLayer("Stamen Terrain", name="Vue Terrain", attr="Map tiles by <a href='http://stamen.com'>Stamen Design</a>, under <a href='http://creativecommons.org/licenses/by/3.0'>CC BY 3.0</a>. Data by <a href='http://openstreetmap.org'>OpenStreetMap</a>, under <a href='http://www.openstreetmap.org/copyright'>ODbL</a>.").add_to(m)
    
    fg_hotels = folium.FeatureGroup(name="🏨 Hôtels", show=True).add_to(m)
    fg_food = folium.FeatureGroup(name="🍽️ Restaurants & Cafés", show=True).add_to(m)
    fg_activities = folium.FeatureGroup(name="🎯 Activités", show=True).add_to(m)
    fg_driving = folium.FeatureGroup(name="🚗 Itinéraires (Voiture)", show=True).add_to(m)
    fg_walking = folium.FeatureGroup(name="🚶 Itinéraires (Piéton)", show=False).add_to(m)
    
    icon_map = {
        'Culturel': {'icon': 'landmark', 'color': 'blue'}, 'Nature': {'icon': 'seedling', 'color': 'green'},
        'Aventure': {'icon': 'mountain', 'color': 'red'}, 'Plage': {'icon': 'umbrella-beach', 'color': 'purple'},
        'Musée': {'icon': 'palette', 'color': 'orange'}, 'Loisir': {'icon': 'martini-glass', 'color': 'pink'},
        'default': {'icon': 'info-sign', 'color': 'gray'}
    }

    if start_point_coords:
        folium.Marker((start_point_coords['latitude'], start_point_coords['longitude']),
            popup=f"📍 <b>{start_point_name}</b>",
            icon=folium.Icon(color='black', icon='plane', prefix='fa')).add_to(m)

    for plan in trip_plan_result:
        for hotel in plan.get('top_5_hotels', []):
            coords = get_coords(hotel)
            if coords: folium.Marker(coords, popup=f"🏨 <b>{hotel.get('nom')}</b><br>Note: {hotel.get('rating', 'N/A')}/10", icon=folium.Icon(color='red', icon='bed', prefix='fa')).add_to(fg_hotels)
        for restaurant in plan.get('top_2_restaurants', []):
            coords = get_coords(restaurant)
            if coords: folium.Marker(coords, popup=f"🍽️ <b>{restaurant.get('nom')}</b><br>Note: {restaurant.get('rating', 'N/A')}/10", icon=folium.Icon(color='green', icon='utensils', prefix='fa')).add_to(fg_food)
        for cafe in plan.get('top_2_cafes', []):
            coords = get_coords(cafe)
            if coords: folium.Marker(coords, popup=f"☕ <b>{cafe.get('nom')}</b><br>Note: {cafe.get('rating', 'N/A')}/10", icon=folium.Icon(color='purple', icon='coffee', prefix='fa')).add_to(fg_food)
        
        for day in plan.get('activites_par_jour_optimisees', []):
            for activity in day:
                coords, act_type = get_coords(activity), activity.get('type')
                if coords and act_type not in ['hotel', 'Point de départ', 'Gastronomique', 'Gastronomique/Café']:
                    icon_style = icon_map.get(act_type, icon_map['default'])
                    folium.Marker(coords, popup=f"<b>{activity.get('nom')}</b><br>Type: {act_type}", icon=folium.Icon(color=icon_style['color'], icon=icon_style['icon'], prefix='fa')).add_to(fg_activities)

    inter_city_coords = [list(city_coords_map_global[c].values()) for c in ordered_cities_list if c in city_coords_map_global]
    if start_point_coords: inter_city_coords.insert(0, list(start_point_coords.values()))
    if len(inter_city_coords) > 1:
        folium.PolyLine(inter_city_coords, color='#8B0000', weight=2.5, opacity=0.8, dash_array='10, 5', tooltip="Trajet Inter-Villes").add_to(m)

    for plan in trip_plan_result:
        for seg in plan.get('itineraire_voiture_segments_par_jour', []):
            folium.PolyLine(seg, color='#007BFF', weight=4, opacity=0.8, tooltip="Itinéraire voiture").add_to(fg_driving)
        for seg in plan.get('itineraire_pieton_segments_par_jour', []):
            folium.PolyLine(seg, color='#28A745', weight=4, opacity=0.8, tooltip="Itinéraire piéton").add_to(fg_walking)
            
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(m.get_bounds(), padding=(20, 20))
    return m._repr_html_()