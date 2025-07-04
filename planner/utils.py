import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itertools import permutations
from geopy.distance import geodesic
import os
import pickle
import logging
from django.conf import settings
import re
import math

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
except ImportError:
    pass

if OSMNX_AVAILABLE and not os.path.exists(GRAPHS_CACHE_DIR_DJANGO):
    try: os.makedirs(GRAPHS_CACHE_DIR_DJANGO)
    except OSError as e: logger.error(f"Impossible de créer le dossier de cache {GRAPHS_CACHE_DIR_DJANGO}: {e}")

CITY_NAME_MAPPING = { "Marrakech": ["marakech", "marrakesh"], "Fès": ["fez", "fes", "fes el bali", "fès el bali"], "Casablanca": ["casa"], "Meknès": ["meknes", "meknès el bali"], "Rabat": [], "Agadir": [], "Chefchaouen": ["chefchaouene", "chaouen"], "Essaouira": ["mogador"], "Ouarzazate": [], "Tangier": ["tanger", "tanger-assilah"], "Merzouga (Erg Chebbi)": ["merzouga", "erg chebbi"] }
MANUAL_CITY_COORDINATES = { "Marrakech": {"latitude": 31.6295, "longitude": -7.9811}, "Fès": {"latitude": 34.0181, "longitude": -5.0078}, "Casablanca": {"latitude": 33.5731, "longitude": -7.5898}, "Meknès": {"latitude": 33.8935, "longitude": -5.5473}, "Rabat": {"latitude": 34.0209, "longitude": -6.8417}, "Agadir": {"latitude": 30.4202, "longitude": -9.5981}, "Chefchaouen": {"latitude": 35.1688, "longitude": -5.2636}, "Essaouira": {"latitude": 31.5085, "longitude": -9.7595}, "Ouarzazate": {"latitude": 30.9189, "longitude": -6.8934}, "Tangier": {"latitude": 35.7595, "longitude": -5.8330}, "Merzouga (Erg Chebbi)": {"latitude": 31.0983, "longitude": -4.0119} }

# --- Fonctions Utilitaires de Prétraitement ---
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
    for city_canonical in canonical_activity_cities_list:
        if city_canonical.lower() in location_lower: return city_canonical
        if city_canonical in city_name_mapping:
            for variation in city_name_mapping[city_canonical]:
                if variation.lower() in location_lower: return city_canonical
    return None

_CACHED_PREPROCESSED_DATA = None
def load_and_preprocess_data():
    global _CACHED_PREPROCESSED_DATA
    if _CACHED_PREPROCESSED_DATA is not None: return _CACHED_PREPROCESSED_DATA
    try:
        with open(ACTIVITIES_JSON_PATH, 'r', encoding='utf-8') as f: activities_data_raw = json.load(f)
        with open(HOTELS_JSON_PATH, 'r', encoding='utf-8') as f: hotels_data_raw_list = json.load(f).get("hotels", [])
    except Exception as e:
        logger.error(f"Erreur chargement des fichiers JSON: {e}", exc_info=True); return pd.DataFrame(), pd.DataFrame(), {}
    
    activities_list = [ {**activity, "ville_normalisee": normalize_city_name(city_data_act.get("ville"), CITY_NAME_MAPPING)} for city_data_act in activities_data_raw if isinstance(city_data_act, dict) for activity in city_data_act.get("activites", []) if isinstance(activity, dict) ]
    
    activities_df = pd.DataFrame(activities_list)
    if not activities_df.empty:
        for col in ['budget_estime', 'rating']:
            if col in activities_df.columns: activities_df[col] = pd.to_numeric(activities_df[col], errors='coerce')
    
    hotels_df = pd.DataFrame(hotels_data_raw_list)
    if not hotels_df.empty:
        canonical_cities_with_activities = sorted(activities_df["ville_normalisee"].dropna().unique().tolist()) if not activities_df.empty else []
        hotels_df['ville_normalisee'] = hotels_df['location'].apply(lambda x: extract_city_from_hotel_location(x, canonical_cities_with_activities, CITY_NAME_MAPPING))
        for col in ["rating", "price_per_night", "latitude", "longitude"]:
            if col in hotels_df.columns: hotels_df[col] = pd.to_numeric(hotels_df[col], errors='coerce')
    
    _CACHED_PREPROCESSED_DATA = (activities_df, hotels_df, MANUAL_CITY_COORDINATES)
    return activities_df, hotels_df, MANUAL_CITY_COORDINATES

def parse_duration_to_hours(duration_str):
    if pd.isna(duration_str) or not isinstance(duration_str, str): return 1.5
    duration_str_lower = str(duration_str).lower()
    if "journée" in duration_str_lower and "demi" not in duration_str_lower: return 6.0
    if "demi-journée" in duration_str_lower: return 3.5
    try:
        if '-' in duration_str_lower:
            parts = re.findall(r'\d+\.?\d*', duration_str_lower)
            return (float(parts[0]) + float(parts[1])) / 2.0 if len(parts) >= 2 else 1.5
        hours_match = re.search(r'(\d+\.?\d*)\s?h', duration_str_lower)
        minutes_match = re.search(r'(\d+)\s?min', duration_str_lower)
        hours = float(hours_match.group(1)) if hours_match else 0
        minutes = int(minutes_match.group(1)) if minutes_match else 0
        total_hours = hours + (minutes / 60.0)
        return total_hours if total_hours > 0 else 1.5
    except Exception: return 1.5

# --- Fonctions de Calcul d'Itinéraire ---
def find_optimal_path_permutations(cities_to_visit_list, city_coordinates_map):
    valid_cities = [c for c in cities_to_visit_list if c in city_coordinates_map]
    if len(valid_cities) > 8:
        start_city = valid_cities[0]; other_cities = valid_cities[1:]
        return find_optimal_path_greedy(start_city, other_cities, city_coordinates_map)
    best_path, min_dist = list(valid_cities), float('inf')
    for p in permutations(valid_cities):
        dist = calculate_total_distance(p, city_coordinates_map)
        if dist < min_dist: min_dist, best_path = dist, list(p)
    return best_path

def calculate_total_distance(path, city_coordinates_map):
    total_dist = 0
    for i in range(len(path) - 1):
        coord1 = city_coordinates_map.get(path[i])
        coord2 = city_coordinates_map.get(path[i+1])
        if coord1 and coord2:
            total_dist += geodesic((coord1['latitude'], coord1['longitude']), (coord2['latitude'], coord2['longitude'])).km
    return total_dist

def find_optimal_path_greedy(start_city, other_cities_list, city_coordinates_map):
    path, remaining_cities, current_city = [start_city], list(other_cities_list), start_city
    while remaining_cities:
        current_coords = city_coordinates_map.get(current_city)
        if not current_coords: break
        next_city = min(remaining_cities, key=lambda city: geodesic(current_coords, city_coordinates_map.get(city, (0,0))).km)
        path.append(next_city); remaining_cities.remove(next_city); current_city = next_city
    return path

# --- Fonctions de Recommandation ---
def recommend_for_city_django(city_name, hotels_df_global, activities_df_global, budget_activities_for_stay_in_city, min_hotel_rating, activity_preferences, activity_intensity, num_persons=1, num_days_in_city=1, use_astar_for_planning=True):
    recommendations = {"ville": city_name, "hotel": [], "jours_alloues": num_days_in_city, "activites_par_jour_optimisees": [], "budget_activites_depense": 0}
    
    default_day_plan = [{"nom": "Repos / Exploration libre", "type": "Loisir"}]
    
    if activities_df_global.empty or 'ville_normalisee' not in activities_df_global.columns:
        recommendations["activites_par_jour_optimisees"] = [default_day_plan] * num_days_in_city
        return recommendations
        
    candidate_activities = activities_df_global[activities_df_global["ville_normalisee"] == city_name].copy()
    
    if candidate_activities.empty:
        recommendations["activites_par_jour_optimisees"] = [default_day_plan] * num_days_in_city
        return recommendations
    
    if activity_preferences:
        pref_mask = candidate_activities['type'].isin(activity_preferences)
        if pref_mask.any():
            candidate_activities = candidate_activities[pref_mask]

    if candidate_activities.empty:
        recommendations["activites_par_jour_optimisees"] = [default_day_plan] * num_days_in_city
        return recommendations

    for col, default in [('rating', 3.5), ('budget_estime', 50.0)]:
        if col not in candidate_activities.columns:
            candidate_activities[col] = default
        else:
            candidate_activities.loc[:, col] = candidate_activities[col].fillna(default)

    candidate_activities['duration_hours'] = candidate_activities['duree_estimee'].apply(parse_duration_to_hours)
    candidate_activities = candidate_activities[candidate_activities['duration_hours'] > 0].copy()

    if candidate_activities.empty:
        recommendations["activites_par_jour_optimisees"] = [default_day_plan] * num_days_in_city
        return recommendations

    scaler = MinMaxScaler()
    candidate_activities['score'] = scaler.fit_transform(candidate_activities[['rating']]) - scaler.fit_transform(candidate_activities[['budget_estime']])
    candidate_activities['value_score'] = candidate_activities['score'] / candidate_activities['duration_hours']
    
    candidate_activities = candidate_activities.sort_values(by="value_score", ascending=False)
    
    intensity_hours_map = {'relaxed': 4.0, 'moderate': 6.0, 'intense': 8.0}
    daily_time_budget = intensity_hours_map.get(activity_intensity, 6.0)
    
    total_spent, daily_plans = 0, []
    for _ in range(num_days_in_city):
        activities_for_this_day, time_spent_on_day = [], 0.0
        for idx, activity in candidate_activities.iterrows():
            duration, cost = activity['duration_hours'], activity['budget_estime'] * num_persons
            if (time_spent_on_day + duration) <= daily_time_budget and (total_spent + cost) <= budget_activities_for_stay_in_city:
                activities_for_this_day.append(activity.to_dict())
                time_spent_on_day += duration
                total_spent += cost
                candidate_activities = candidate_activities.drop(idx)
        daily_plans.append(activities_for_this_day if activities_for_this_day else default_day_plan)
        
    recommendations["activites_par_jour_optimisees"] = daily_plans
    recommendations["budget_activites_depense"] = total_spent
    
    return recommendations

# --- Planificateur Principal ---
def plan_trip_django(target_cities_list, total_budget_str, num_days_str, num_persons, min_hotel_rating_str, activity_preferences_str, activity_intensity, activities_df_global, hotels_df_global, city_coords_map_global, use_astar_for_planning):
    try:
        total_budget, num_days, min_rating = float(total_budget_str), int(num_days_str), float(min_hotel_rating_str)
        num_persons = int(num_persons)
    except (ValueError, TypeError):
        return None, {}, []

    ordered_cities = find_optimal_path_permutations(target_cities_list, city_coords_map_global)
    if not ordered_cities:
        return None, {"error": "Impossible de déterminer un ordre de visite pour les villes sélectionnées."}, []
        
    days_per_city = [num_days // len(ordered_cities)] * len(ordered_cities) if ordered_cities else []
    for i in range(num_days % len(ordered_cities) if ordered_cities else 0): days_per_city[i] += 1

    num_rooms_needed = math.ceil(num_persons / 2.0)
    
    total_hotel_cost, preliminary_trip_plan = 0, []
    temp_daily_hotel_budget_per_room = ((total_budget / num_days) * 0.40) / num_rooms_needed if num_days > 0 else 0
    
    for i, city in enumerate(ordered_cities):
        days_in_city = days_per_city[i]
        if days_in_city == 0: continue
        
        city_plan = {"ville": city, "hotel": [], "jours_alloues": days_in_city}
        if not hotels_df_global.empty and 'ville_normalisee' in hotels_df_global.columns:
            hotels_in_city_df = hotels_df_global[(hotels_df_global["ville_normalisee"] == city) & (hotels_df_global["price_per_night"] <= temp_daily_hotel_budget_per_room) & (hotels_df_global["rating"] >= min_rating)]
            if not hotels_in_city_df.empty:
                best_hotel = hotels_in_city_df.sort_values(by="rating", ascending=False).head(1).to_dict('records')[0]
                city_plan["hotel"] = [best_hotel]
                total_hotel_cost += best_hotel.get('price_per_night', 0) * num_rooms_needed * days_in_city
        preliminary_trip_plan.append(city_plan)
        
    budget_remaining_for_activities = max(0, total_budget - total_hotel_cost)
    total_days_for_activity_budgetting = sum(p['jours_alloues'] for p in preliminary_trip_plan)

    trip_plan_final = []
    for city_plan in preliminary_trip_plan:
        city, days_in_city = city_plan['ville'], city_plan['jours_alloues']
        budget_for_this_city_activities = (budget_remaining_for_activities / total_days_for_activity_budgetting) * days_in_city if total_days_for_activity_budgetting > 0 else 0
        
        prefs = [p.strip() for p in activity_preferences_str.split(',') if p.strip()]
        
        city_rec = recommend_for_city_django(city, hotels_df_global, activities_df_global, budget_for_this_city_activities, min_rating, prefs, activity_intensity, num_persons, days_in_city, use_astar_for_planning)
        city_rec['hotel'] = city_plan['hotel']
        trip_plan_final.append(city_rec)
        
    params = {"Villes demandées": ", ".join(target_cities_list), "Ordre de visite suggéré": ", ".join(ordered_cities), "Durée du voyage": f"{num_days} jours"}
    return trip_plan_final, params, ordered_cities
