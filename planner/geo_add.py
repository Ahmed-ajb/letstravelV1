import os
import json
import time
import re
from opencage.geocoder import OpenCageGeocode
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("OPENCAGE_API_KEY") or "21fae24d64ea4ac0b1122cae14a27463"
ACTIVITIES_FILE_PATH = "E:/PFE MASTER/application streamlit/trip_planner_project/planner/data/activities.json"

if not API_KEY:
    print("❌ ERREUR : Clé API manquante.")
    exit()

geocoder = OpenCageGeocode(API_KEY)
city_coords_cache = {}

# --- FONCTION POUR EXTRAIRE LE NOM D'ACTIVITÉ CIBLE ---
def extraire_mot_cle(nom):
    # Expressions communes à supprimer
    stopwords = [
        r"visite de la ", r"excursion (dans|à) ", r"cours de ", r"atelier de ", r"balade (en|autour).*", 
        r"expérience ", r"shopping dans les ", r"découverte de la ", r"exploration de .*", r"randonnée .*",
        r"dans le désert .*", r"lever ou coucher de soleil .*", r"trekking .*", r"bivouac .*"
    ]
    clean = nom.lower()
    for pattern in stopwords:
        clean = re.sub(pattern, "", clean)
    return clean.strip()

# --- CHARGEMENT DU FICHIER JSON ---
try:
    with open(ACTIVITIES_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"❌ Erreur : {e}")
    exit()

print("🔍 Mise à jour des coordonnées...\n")

for city_block in data:
    city_name = city_block.get("ville")
    if not city_name or 'activites' not in city_block:
        continue

    for activity in city_block['activites']:
        if activity.get('latitude') is not None and activity.get('longitude') is not None:
            continue

        activity_name = activity.get("nom")
        mot_cle = extraire_mot_cle(activity_name)
        print(f"🔎 Recherche : '{mot_cle}, {city_name}'")

        found = False
        try:
            # Recherche ciblée dans la ville
            query = f"{mot_cle}, {city_name}, Maroc"
            results = geocoder.geocode(query, language='fr', no_annotations=1)

            if results:
                lat = results[0]['geometry']['lat']
                lng = results[0]['geometry']['lng']
                activity['latitude'] = lat
                activity['longitude'] = lng
                print(f"✅ Coordonnées précises : ({lat:.4f}, {lng:.4f})")
                found = True
        except Exception as e:
            print(f"❗ Erreur : {e}")

        # Fallback à la ville si non trouvé
        if not found:
            if city_name not in city_coords_cache:
                print(f"📍 Recherche fallback pour la ville : {city_name}")
                city_results = geocoder.geocode(f"{city_name}, Maroc", language='fr', no_annotations=1)
                if city_results:
                    city_lat = city_results[0]['geometry']['lat']
                    city_lng = city_results[0]['geometry']['lng']
                    city_coords_cache[city_name] = (city_lat, city_lng)
                    print(f"🏙️  Coordonnées ville : ({city_lat:.4f}, {city_lng:.4f})")
                else:
                    city_coords_cache[city_name] = (None, None)
            lat, lng = city_coords_cache[city_name]
            activity['latitude'] = lat
            activity['longitude'] = lng
            if lat:
                print(f"➕ Coordonnées fallback ville utilisées : ({lat:.4f}, {lng:.4f})")
            else:
                print("🚫 Aucun point trouvé.")

        time.sleep(1.1)

# --- SAUVEGARDE DU FICHIER ---
with open(ACTIVITIES_FILE_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("\n✅ Mise à jour terminée.")
