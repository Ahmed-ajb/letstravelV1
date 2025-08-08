# planner/chatbot_logic.py
"""
Ce fichier contient toute la logique du chatbot MarocGuide.
Il gère le traitement du langage naturel, la reconnaissance d'images,
la récupération d'informations depuis des sources locales (Pandas DF),
externes (API météo, recherche web) et la base de données Django (voyages planifiés).
Il inclut un garde-fou strict, une gestion de session intelligente et une logique de suggestion conditionnelle.
"""

# --- 1. Importations ---
# Importations de la bibliothèque standard Python
import os
import json
import re
import logging
from datetime import datetime
from io import BytesIO
import base64
from typing import Optional
from math import radians, sin, cos, sqrt, atan2

# Importations des bibliothèques tierces (ML/Data)
import pandas as pd
import numpy as np
import faiss
import tensorflow as tf
from PIL import Image as PILImage
import requests

# Importations Langchain / Ollama
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains import LLMChain

# Importations spécifiques au projet Django
from dotenv import load_dotenv
from .utils import load_and_preprocess_data
from planner.models import Trip, ChatMessage, User
from django.db.models import Q

# Importations spécifiques à Keras/TensorFlow pour le modèle InceptionV3
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image_preprocessing


# --- 2. Configuration Initiale et Constantes ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration du modèle de langage principal
MODEL_MISTRAL = "mistral"
llm_mistral = Ollama(model=MODEL_MISTRAL, temperature=0.3)

# Chemins vers les fichiers de données
DATA_RAG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_data')
FAISS_IMAGE_INDEX_FILE = os.path.join(DATA_RAG_FOLDER, "faiss_image_index_inception.bin")
FAISS_IMAGE_IDS_FILE = os.path.join(DATA_RAG_FOLDER, "faiss_image_ids_inception.json")
MAROC_LIEUX_CORPUS_FILE = os.path.join(DATA_RAG_FOLDER, "maroc_lieux_corpus.json")
STATIC_IMAGES_SERVED_PATH = "planner/rag_data/images/"

# Récupération des clés API et initialisation des outils
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY")
tavily_search_tool = TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# Constantes pour la logique métier
CITY_NAME_MAPPING = {
    "Marrakech": ["marakech", "marrakesh"], "Fès": ["fez", "fes"], "Casablanca": ["casa"], "Meknès": ["meknes"],
    "Rabat": [], "Agadir": [], "Chefchaouen": ["chaouen"], "Essaouira": ["mogador"], "Ouarzazate": [],
    "Tangier": ["tanger"], "Merzouga (Erg Chebbi)": ["merzouga", "erg chebbi"]
}
HOTEL_KEYWORDS = ['hôtel', 'hotel', 'hébergement', 'dormir', 'logement', 'riad', 'auberge']
FORBIDDEN_LOCATIONS = [
    'algérie', 'algerie', 'alger', 'oran', 'constantine', 'tunisie', 'tunis',
    'france', 'paris', 'marseille', 'lyon', 'espagne', 'madrid', 'barcelone',
    'égypte', 'egypte', 'caire', 'turquie', 'istanbul', 'australie'
]
GREETING_WORDS = ['bonjour', 'salut', 'salam', 'hello', 'yo', 'hi', 'hey', 'salutations']


# --- 3. Variables Globales et Fonctions d'Initialisation ---
inception_feature_extractor_model, faiss_image_index, faiss_image_ids, maroc_lieux_data = None, None, [], {}
ACTIVITIES_DF, HOTELS_DF = pd.DataFrame(), pd.DataFrame()

def initialize_image_comparison_components():
    """Charge le modèle InceptionV3, l'index FAISS et les métadonnées pour la recherche d'images."""
    global inception_feature_extractor_model, faiss_image_index, faiss_image_ids, maroc_lieux_data
    try:
        logger.info("Chargement des composants pour la comparaison d'images...")
        base_model_inception = InceptionV3(weights='imagenet', include_top=False)
        x = base_model_inception.output
        output = tf.keras.layers.GlobalAveragePooling2D()(x)
        inception_feature_extractor_model = Model(inputs=base_model_inception.input, outputs=output)
        inception_feature_extractor_model.trainable = False

        if os.path.exists(FAISS_IMAGE_INDEX_FILE):
            faiss_image_index = faiss.read_index(FAISS_IMAGE_INDEX_FILE)
        if os.path.exists(FAISS_IMAGE_IDS_FILE):
            with open(FAISS_IMAGE_IDS_FILE, "r", encoding="utf-8") as f: faiss_image_ids = json.load(f)
        if os.path.exists(MAROC_LIEUX_CORPUS_FILE):
            with open(MAROC_LIEUX_CORPUS_FILE, "r", encoding="utf-8") as f:
                maroc_lieux_data = {doc['id']: doc for doc in json.load(f)}
        logger.info("Composants de comparaison d'images chargés avec succès.")
    except Exception as e:
        logger.error(f"Erreur critique lors du chargement des composants de comparaison d'images: {e}", exc_info=True)
        inception_feature_extractor_model, faiss_image_index = None, None

def initialize_data():
    """Charge les données touristiques (activités, hôtels) dans des DataFrames Pandas."""
    global ACTIVITIES_DF, HOTELS_DF
    try:
        logger.info("Chargement des données statiques (activités, hôtels)...")
        dfs = load_and_preprocess_data()
        ACTIVITIES_DF = dfs[0] if len(dfs) > 0 else pd.DataFrame()
        HOTELS_DF = dfs[1] if len(dfs) > 1 else pd.DataFrame()
        
        for df in [ACTIVITIES_DF, HOTELS_DF]:
            if not df.empty:
                for col in ['latitude', 'longitude']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.info(f"Données statiques chargées: {len(ACTIVITIES_DF)} activités, {len(HOTELS_DF)} hôtels.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données statiques: {e}", exc_info=True)

# Exécution des initialisations au démarrage du module
initialize_image_comparison_components()
initialize_data()
if not tavily_search_tool: logger.warning("TAVILY_API_KEY non configurée. La recherche web est désactivée.")
if not OPENWEATHERMAP_API_KEY: logger.warning("OPENWEATHERMAP_API_KEY non configurée. La météo via API est désactivée.")


# --- 4. Fonctions Utilitaires Générales ---
def normalize_city_name(city_name: str, city_mapping: dict) -> Optional[str]:
    """Standardise un nom de ville."""
    if not city_name or pd.isna(city_name): return None
    city_name_lower = str(city_name).strip().lower()
    for canonical, variations in city_mapping.items():
        if city_name_lower == canonical.lower() or city_name_lower in [v.lower() for v in variations]: return canonical
    return str(city_name).strip().capitalize()

def clean_json_from_llm(raw_text: str) -> dict:
    """Extrait et parse un bloc JSON d'une chaîne de texte."""
    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if not json_match: return {}
    try:
        cleaned_str = re.sub(r',\s*([}\]])', r'\1', json_match.group(0))
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        return {}

def clean_final_llm_response(response_text: str) -> str:
    """Nettoie la réponse finale du LLM."""
    cleaned = re.sub(r'\[/?INST\]', '', response_text.strip()).strip()
    signal = "MarocGuide:"
    if signal in cleaned: cleaned = cleaned.split(signal)[-1].strip()
    return cleaned or "Désolé, une erreur est survenue lors de la formulation de la réponse."

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcule la distance en kilomètres entre deux points GPS."""
    R = 6371.0
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

def is_query_off_topic_by_rules(user_message: str) -> bool:
    """Garde-fou par règles pour bloquer les questions hors sujet évidentes."""
    return any(loc in user_message.lower() for loc in FORBIDDEN_LOCATIONS)

def is_greeting(user_message: str) -> bool:
    """Détecte si un message est une salutation pour forcer une nouvelle session."""
    message_start = user_message.strip().lower()
    return any(message_start.startswith(word) for word in GREETING_WORDS)


# --- 5. Définition des Chaînes Langchain ---
def get_entity_extraction_chain_revised():
    """Crée la chaîne LLM pour l'extraction d'entités, avec une règle stricte pour le hors-sujet."""
    template = """[INST] Tu es un expert en extraction d'entités pour un chatbot spécialisé **uniquement** dans le tourisme au Maroc.
Ta tâche est de classifier l'intention et d'extraire les entités.
Historique: {chat_history_string}
Question: "{input}"

Règle stricte: Si la question ne concerne pas le voyage, la culture ou la géographie du Maroc, l'intention doit être "hors_sujet_tourisme".
Réponds UNIQUEMENT avec un objet JSON valide.
Entités possibles: "ville", "type_activite", "mots_cles", "trip_identifier", "day_number", "intention_utilisateur" (choisis parmi: "chercher_activites_hotels", "info_generale_tourisme", "demande_meteo", "consult_trip", "salutation", "remerciement", "hors_sujet_tourisme").

Exemple 1: "Hôtels à Rabat avec piscine" -> {{"ville": "Rabat", "mots_cles": "piscine", "intention_utilisateur": "chercher_activites_hotels"}}
Exemple 2: "Quelle est la capitale de l'Australie ?" -> {{"intention_utilisateur": "hors_sujet_tourisme"}}
Exemple 3: "je veux planifier un voyage en algerie" -> {{"intention_utilisateur": "hors_sujet_tourisme"}}

Si rien n'est trouvé, retourne {{}}[/INST]
JSON Extrait:"""
    prompt = PromptTemplate(template=template, input_variables=["chat_history_string", "input"])
    return LLMChain(llm=llm_mistral, prompt=prompt, output_key="text")

def get_answer_generation_chain_revised():
    """Crée la chaîne LLM pour générer la réponse finale, avec un prompt système renforcé."""
    formatted_date = datetime.now().strftime('%d %B %Y')
    system_prompt_str = f"""Tu es MarocGuide, un assistant de voyage virtuel expert **uniquement sur le Maroc**.
Ta réponse doit être en français, amicale, concise et DIRECTEMENT utile.
La date d'aujourd'hui est le {formatted_date}. Utilise cette information si besoin.
Utilise IMPÉRATIVEMENT les informations du "Contexte Fourni".
Si le contexte contient des données Météo de l'API, base ta réponse **exclusivement** sur ces données sans inventer.
**Règle absolue: NE JAMAIS répondre à une question qui ne concerne pas le voyage ou la culture au Maroc.** Si tu es forcé de le faire, décline poliment en rappelant ta spécialisation marocaine.
NE mentionne PAS l'existence du "Contexte Fourni". Agis comme un chatbot naturel."""
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt_str),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="Contexte Fourni:\n{retrieved_context}\n\nQuestion actuelle: {input}\n\nMarocGuide:")
    ])
    return LLMChain(llm=llm_mistral, prompt=prompt, output_key="text")


# --- 6. Fonctions de Récupération de Données (Le "R" de RAG) ---
def get_weather_data(city_name: str) -> str:
    """Récupère les données météo actuelles via l'API OpenWeatherMap."""
    if not OPENWEATHERMAP_API_KEY: return "Info Météo: Le service est indisponible."
    params = {"q": city_name, "appid": OPENWEATHERMAP_API_KEY, "units": "metric", "lang": "fr"}
    try:
        response = requests.get("http://api.openweathermap.org/data/2.5/weather?", params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("cod") != 200: return f"Info Météo: Impossible de trouver la météo pour {city_name}."
        main, desc = data["main"], data["weather"][0]["description"]
        return (f"Météo actuelle à {data['name']}:\n- Ciel: {desc.capitalize()}\n"
                f"- Température: {main['temp']:.0f}°C (Ressentie: {main['feels_like']:.0f}°C)\n"
                f"- Humidité: {main['humidity']}%")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur API OpenWeatherMap pour {city_name}: {e}")
        return "Info Météo: Erreur de connexion au service météo."

def get_city_from_gps(lat: float, lon: float) -> Optional[str]:
    """Utilise le géocodage inversé pour trouver le nom d'une ville depuis des coordonnées."""
    if not OPENWEATHERMAP_API_KEY: return None
    params = {"lat": lat, "lon": lon, "limit": 1, "appid": OPENWEATHERMAP_API_KEY}
    try:
        response = requests.get("http://api.openweathermap.org/geo/1.0/reverse", params=params)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list): return data[0].get("name")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur API Reverse Geocoding: {e}")
    return None

def retrieve_activities_for_city(city_name: str) -> str:
    """Récupère les activités les mieux notées pour une ville donnée."""
    if ACTIVITIES_DF.empty: return ""
    normalized_city = normalize_city_name(city_name, CITY_NAME_MAPPING)
    if not normalized_city: return ""
    retrieved = ACTIVITIES_DF[ACTIVITIES_DF['ville_normalisee'] == normalized_city].sort_values(by='rating', ascending=False).head(3)
    if retrieved.empty: return ""
    return "Activités suggérées:\n" + "".join([f"- {row.get('nom', 'N/A')} ({row.get('type', 'N/A')})\n" for _, row in retrieved.iterrows()])

def retrieve_hotels_for_city(city_name: str) -> str:
    """Récupère les hôtels les mieux notés pour une ville donnée."""
    if HOTELS_DF.empty: return ""
    normalized_city = normalize_city_name(city_name, CITY_NAME_MAPPING)
    if not normalized_city: return ""
    retrieved = HOTELS_DF[HOTELS_DF['ville_normalisee'] == normalized_city].sort_values(by='rating', ascending=False).head(2)
    if retrieved.empty: return ""
    parts = ["Hôtels suggérés:\n"]
    for _, row in retrieved.iterrows():
        prix = f"{row.get('price_per_night', 0):.0f} MAD/nuit" if pd.notna(row.get('price_per_night')) else 'Non spécifié'
        parts.append(f"- {row.get('name', 'N/A')} (Rating: {row.get('rating', 'N/A')}/10, Prix: {prix})\n")
    return "".join(parts)

def get_web_search_context(query: str) -> str:
    """Effectue une recherche web via Tavily comme filet de sécurité."""
    if not tavily_search_tool: return ""
    try:
        results = tavily_search_tool.invoke(query)
        if not results: return ""
        return f"Infos Web sur '{query}':\n" + "\n".join([f"- {res.get('title', '')}: {res.get('content', '')}" for res in results])
    except Exception as e:
        logger.error(f"Erreur lors de la recherche web: {e}")
        return "Erreur durant la recherche web."

def get_user_trips_summary_db(user_id: int) -> str:
    """Récupère un résumé des voyages planifiés de l'utilisateur."""
    user = User.objects.filter(id=user_id).first()
    if not user: return "Utilisateur non trouvé."
    trips = Trip.objects.filter(user=user).order_by('-created_at')[:5]
    if not trips.exists(): return "Vous n'avez pas de voyages planifiés."
    return "Voici un résumé de vos voyages:\n" + "".join([f"- **{t.name}** ({t.num_days_str} jours)\n" for t in trips])

def get_trip_details_db(user_id: int, trip_identifier: str, day_number: Optional[int] = None) -> str:
    """Récupère les détails d'un voyage spécifique de l'utilisateur."""
    user = User.objects.filter(id=user_id).first()
    if not user: return "Utilisateur non trouvé."
    trip = Trip.objects.filter(user=user, name__icontains=trip_identifier).first()
    if not trip: return f"Désolé, je n'ai pas trouvé de voyage correspondant à '{trip_identifier}'."
    details = f"Détails pour le voyage '{trip.name}':\n"
    days = trip.days.all() if not day_number else trip.days.filter(day_number=day_number)
    if not days.exists(): return f"Le jour {day_number} n'a pas été trouvé."
    for day in days.order_by('day_number'):
        details += f"\n--- Jour {day.day_number} à {day.city_name} ---\n"
        items = day.activity_items.all().order_by('order_in_day')
        if not items.exists(): details += "  Aucune activité détaillée.\n"
        else: details += "".join([f"  - {item.name}\n" for item in items])
    return details


# --- 7. Fonctions de Traitement des Requêtes Spécifiques ---
def find_similar_place_by_image(image_base64: str, user_message: str) -> str:
    """Identifie un lieu depuis une image, affiche l'image de l'utilisateur, et suggère des infos."""
    if not all([inception_feature_extractor_model, faiss_image_index]):
        return "Désolé, le service d'identification d'image est indisponible."
    
    try:
        image_bytes = base64.b64decode(image_base64)
        image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
        img_array = keras_image_preprocessing.img_to_array(image.resize((299, 299)))
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)

        query_embedding = inception_feature_extractor_model.predict(img_preprocessed, verbose=0)
        distances, indices = faiss_image_index.search(query_embedding.astype(np.float32), 1)

        if not indices.size or indices[0][0] == -1:
            return "Aucun lieu similaire trouvé dans ma base de données."

        lieu_info = maroc_lieux_data.get(faiss_image_ids[indices[0][0]])
        if not lieu_info:
            return "Lieu similaire trouvé, mais ses informations détaillées sont indisponibles."

        user_image_html = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Votre image" style="max-width:250px; border-radius: 8px; margin-bottom: 10px;"><br>'
        place_name, place_city = lieu_info['name'], lieu_info['city']
        reply_parts = [user_image_html, f"Ce lieu ressemble beaucoup à **{place_name}** à **{place_city}**."]

        activities_info = retrieve_activities_for_city(place_city)
        if activities_info:
            reply_parts.append(f"\nVoici des suggestions d'activités à proximité :\n{activities_info}")
        
        if any(keyword in user_message.lower() for keyword in HOTEL_KEYWORDS):
            hotel_info = retrieve_hotels_for_city(place_city)
            if hotel_info:
                reply_parts.append(f"\nComme demandé, voici des suggestions d'hébergements :\n{hotel_info}")
            
        return "\n\n".join(reply_parts)

    except Exception as e:
        logger.error(f"Erreur grave lors de la comparaison d'image: {e}", exc_info=True)
        return "Désolé, une erreur interne est survenue lors de l'identification de l'image."

def identify_place_from_gps_via_web(gps_coords: str) -> str:
    """Identifie un lieu à partir de coordonnées GPS en utilisant une recherche web."""
    logger.info(f"Identification de lieu par GPS via recherche web pour les coordonnées: {gps_coords}")
    try:
        lat, lon = map(float, gps_coords.split(','))
        query = f"Quel est le monument ou lieu d'intérêt aux coordonnées latitude {lat}, longitude {lon}?"
        
        web_context = get_web_search_context(query)
        if "Erreur" in web_context or not web_context:
            return "Je n'ai pas pu identifier de lieu spécifique à partir de vos coordonnées via une recherche web."
        
        prompt = f"""Basé sur l'information suivante issue d'une recherche web, décris brièvement le lieu identifié.
        Information: "{web_context}"
        
        Description du lieu:"""
        
        response = llm_mistral.invoke(prompt)
        return f"D'après une recherche web, votre position semble correspondre au lieu suivant :\n\n{response}"
        
    except (ValueError, TypeError):
        return "Les coordonnées GPS fournies pour l'identification sont invalides."
    except Exception as e:
        logger.error(f"Erreur dans identify_place_from_gps_via_web: {e}", exc_info=True)
        return "Une erreur est survenue lors de l'identification du lieu par GPS."

def process_gps_query(gps_coords: str, user_message: str) -> str:
    """Gère les requêtes GPS, avec fallback sur le web si la recherche interne échoue."""
    try:
        user_lat, user_lon = map(float, gps_coords.split(','))
    except (ValueError, TypeError):
        return "Les coordonnées GPS sont invalides. Format attendu: 'latitude,longitude'."

    internal_results = []
    all_dfs = [(ACTIVITIES_DF, 'activité'), (HOTELS_DF, 'hôtel')] if any(kw in user_message.lower() for kw in HOTEL_KEYWORDS) else [(ACTIVITIES_DF, 'activité')]
    
    for df, df_type in all_dfs:
        if df.empty: continue
        
        temp_df = df.copy().dropna(subset=['latitude', 'longitude'])
        
        if user_message.strip() and df_type == 'activité':
            keywords = [word for word in user_message.lower().split() if len(word) > 3]
            if keywords:
                mask = temp_df.apply(lambda row: any(kw in str(row.get('nom', '')).lower() or kw in str(row.get('type', '')).lower() for kw in keywords), axis=1)
                temp_df = temp_df[mask]

        if not temp_df.empty:
            temp_df['distance'] = temp_df.apply(lambda row: haversine_distance(user_lat, user_lon, row['latitude'], row['longitude']), axis=1)
            nearby = temp_df.sort_values('distance').head(3)
            if not nearby.empty:
                name_col = 'nom' if 'nom' in nearby.columns else 'name'
                intro = "Activités à proximité:\n" if df_type == 'activité' else "Hôtels à proximité:\n"
                results_str = intro + "".join([f"- **{row[name_col]}** ({row['distance']:.1f} km)\n" for _, row in nearby.iterrows()])
                internal_results.append(results_str)
    
    if internal_results:
        return "\n".join(internal_results)
        
    logger.info(f"Aucun résultat interne pour GPS {gps_coords}, fallback sur la recherche web.")
    city_name = get_city_from_gps(user_lat, user_lon)
    search_location = f"près de {city_name}" if city_name else f"autour des coordonnées {gps_coords}"
    web_query = f"{user_message} {search_location}?" if user_message.strip() else f"Quelles sont les activités touristiques {search_location}?"
    
    web_context = get_web_search_context(web_query)
    if "Erreur" in web_context or not web_context:
        return f"Désolé, je n'ai trouvé aucun lieu correspondant à votre demande à proximité."
    return f"Je n'ai pas trouvé de correspondances dans ma base de données, mais voici ce que j'ai trouvé sur le web {search_location}:\n\n{web_context}"


# --- 8. Orchestrateur Principal ---
def process_user_query(
    user_message: str, 
    user_id: int, 
    image_base64: Optional[str] = None, 
    gps_coords: Optional[str] = None,
    start_new_session: bool = False
) -> str:
    """
    Fonction principale qui orchestre le traitement de la requête de l'utilisateur.
    """
    logger.info(f"Requête user {user_id}: Msg='{user_message[:50]}...', Img={'Oui' if image_base64 else 'Non'}, GPS={'Oui' if gps_coords else 'Non'}, NewSession={start_new_session}")

    # --- ROUTAGE INITIAL (NON-TEXTUEL) ---
    if image_base64 and gps_coords: return identify_place_from_gps_via_web(gps_coords)
    if image_base64: return find_similar_place_by_image(image_base64, user_message)
    if gps_coords: return process_gps_query(gps_coords, user_message)

    # --- TRAITEMENT TEXTUEL ---
    try:
        # ETAPE 1: GARDE-FOU PAR RÈGLES (NON-NÉGOCIABLE)
        if is_query_off_topic_by_rules(user_message):
            logger.info(f"Requête bloquée par garde-fou: '{user_message}'")
            return "Je suis désolé, mais je suis un assistant spécialisé uniquement dans le tourisme au Maroc. Comment puis-je vous aider avec votre voyage au Maroc ?"

        # ETAPE 2: GESTION DE LA SESSION
        force_new_session = start_new_session or is_greeting(user_message)
        history_for_prompt, history_str = [], ""
        if not force_new_session:
            recent_history = ChatMessage.objects.filter(user_id=user_id).order_by('-timestamp')[:6]
            history_for_prompt = [HumanMessage(content=m.message) if m.is_from_user else AIMessage(content=m.message) for m in reversed(recent_history)]
            history_str = "\n".join([f"{'User' if m.is_from_user else 'AI'}: {m.message}" for m in reversed(recent_history)])
        else:
            logger.info(f"Nouvelle session forcée pour user {user_id}.")

        # ETAPE 3: Extraction des entités par LLM (deuxième garde-fou)
        entity_chain = get_entity_extraction_chain_revised()
        entities = clean_json_from_llm(entity_chain.invoke({"chat_history_string": history_str, "input": user_message}).get("text", "{}"))
        user_intent = entities.get("intention_utilisateur", "info_generale_tourisme")
        
        # ETAPE 4: Routage par intention et collecte de contexte
        if user_intent == "salutation": return "Bonjour ! Je suis MarocGuide. Comment puis-je vous aider ?"
        if user_intent == "remerciement": return "Avec plaisir !"
        if user_intent == "hors_sujet_tourisme": 
            return "Je suis désolé, mais je suis un assistant spécialisé uniquement dans le tourisme au Maroc. Comment puis-je vous aider avec votre voyage au Maroc ?"

        context_parts = []
        if user_intent == "demande_meteo":
            context_parts.append(get_weather_data(entities.get("ville", "Maroc")))
        elif user_intent == "consult_trip":
            trip_id = entities.get("trip_identifier")
            context_parts.append(get_trip_details_db(user_id, trip_id, entities.get("day_number")) if trip_id else get_user_trips_summary_db(user_id))
        else:
            city = entities.get("ville")
            if city:
                context_parts.append(retrieve_activities_for_city(city))
                if any(keyword in user_message.lower() for keyword in HOTEL_KEYWORDS):
                    context_parts.append(retrieve_hotels_for_city(city))
            if user_intent == "info_generale_tourisme" or not any(filter(None, context_parts)):
                context_parts.append(get_web_search_context(user_message))

        # ETAPE 5: Génération de la réponse finale
        final_context = "\n\n".join(filter(None, context_parts)).strip() or "Aucune information spécifique trouvée."
        answer_chain = get_answer_generation_chain_revised()
        response = answer_chain.invoke({"chat_history": history_for_prompt, "input": user_message, "retrieved_context": final_context})
        return clean_final_llm_response(response.get("text", ""))

    except Exception as e:
        logger.error(f"Erreur majeure dans process_user_query: {e}", exc_info=True)
        return "Oups, une erreur technique est survenue. Veuillez réessayer."