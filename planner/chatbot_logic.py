# planner/chatbot_logic.py

import os
import json
import re
import logging
from datetime import datetime
import pandas as pd
from io import BytesIO
import base64
from PIL import Image as PILImage 

# Importations Langchain / Ollama
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

import requests
from dotenv import load_dotenv

# Importations Django (vérifiez que ces modèles existent dans planner/models.py)
from .utils import load_and_preprocess_data
from planner.models import Trip, TripDay, DailyActivityItem, ChatMessage, User
from django.db.models import Q

# --- IMPORTATIONS POUR COMPARAISON D'IMAGES (InceptionV3) ---
import faiss
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input 
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing import image as keras_image_preprocessing 


logger = logging.getLogger(__name__)

# --- Charger les variables d'environnement du fichier .env ---
load_dotenv()

# --- Configuration des modèles Ollama ---
MODEL_PHI3 = "phi3"
MODEL_MISTRAL = "mistral"
MODEL_LLAVA_DESCRIPTION_GENERAL = "llava" 
llm_phi3 = Ollama(model=MODEL_PHI3, temperature=0.3)
llm_mistral = Ollama(model=MODEL_MISTRAL, temperature=0.3)
llm_llava_description = Ollama(model=MODEL_LLAVA_DESCRIPTION_GENERAL, temperature=0.1)


# --- CONFIGURATION POUR LA COMPARAISON D'IMAGES (InceptionV3/FAISS) ---
# ADAPTEZ CE CHEMIN ABSOLU pour qu'il corresponde à l'emplacement réel de votre dossier 'rag_data'
DATA_RAG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_data')
FAISS_IMAGE_INDEX_FILE = os.path.join(DATA_RAG_FOLDER, "faiss_image_index_inception.bin") 
FAISS_IMAGE_IDS_FILE = os.path.join(DATA_RAG_FOLDER, "faiss_image_ids_inception.json") 
MAROC_LIEUX_CORPUS_FILE = os.path.join(DATA_RAG_FOLDER, "maroc_lieux_corpus.json")

# Chemin de base pour les images qui seront servies par Django (relative à STATIC_URL).
# Assurez-vous que votre settings.py est configuré pour servir
# 'votre_projet_django/planner/rag_data/images/' COMME FICHIERS STATIQUES.
# Exemple dans settings.py:
# STATIC_URL = '/static/'
# STATICFILES_DIRS = [
#     os.path.join(BASE_DIR, 'static'),
#     os.path.join(BASE_DIR, 'planner', 'rag_data', 'images'), # AJOUTEZ CETTE LIGNE
# ]
STATIC_IMAGES_SERVED_PATH = "planner/rag_data/images/" # C'est le chemin qui sera ajouté à STATIC_URL


# --- Chargement des Composants de Comparaison d'Images (une seule fois au démarrage de Django) ---
global inception_feature_extractor_model, faiss_image_index, faiss_image_ids, maroc_lieux_data

inception_feature_extractor_model = None
faiss_image_index = None
faiss_image_ids = []
maroc_lieux_data = {} 

def initialize_image_comparison_components():
    global inception_feature_extractor_model, faiss_image_index, faiss_image_ids, maroc_lieux_data
    try:
        logger.info("Chargement du modèle InceptionV3 pour l'extraction de features...")
        base_model_inception = InceptionV3(weights='imagenet', include_top=False)
        
        x = base_model_inception.output 
        output = tf.keras.layers.GlobalAveragePooling2D()(x) 
        inception_feature_extractor_model = Model(inputs=base_model_inception.input, outputs=output) 
        inception_feature_extractor_model.trainable = False 
        
        logger.info("Modèle InceptionV3 chargé.")

        if not os.path.exists(FAISS_IMAGE_INDEX_FILE):
            raise FileNotFoundError(f"Fichier d'index FAISS manquant: {FAISS_IMAGE_INDEX_FILE}")
        faiss_image_index = faiss.read_index(FAISS_IMAGE_INDEX_FILE)
        logger.info(f"Index FAISS des images chargé depuis : {FAISS_IMAGE_INDEX_FILE}")

        if not os.path.exists(FAISS_IMAGE_IDS_FILE):
            raise FileNotFoundError(f"Fichier des IDs FAISS manquant: {FAISS_IMAGE_IDS_FILE}")
        with open(FAISS_IMAGE_IDS_FILE, "r", encoding="utf-8") as f:
            faiss_image_ids = json.load(f)
        logger.info(f"IDs d'images FAISS chargés depuis : {FAISS_IMAGE_IDS_FILE}")

        if not os.path.exists(MAROC_LIEUX_CORPUS_FILE):
            raise FileNotFoundError(f"Fichier de corpus de lieux manquant: {MAROC_LIEUX_CORPUS_FILE}")
        with open(MAROC_LIEUX_CORPUS_FILE, "r", encoding="utf-8") as f:
            corpus_list = json.load(f)
            maroc_lieux_data = {doc['id']: doc for doc in corpus_list} 
        logger.info(f"Base de données de lieux chargée depuis : {MAROC_LIEUX_CORPUS_FILE}")

        logger.info("Composants de comparaison d'images (InceptionV3/FAISS) chargés avec succès.")
    except FileNotFoundError as fnfe:
        logger.error(f"Erreur (FileNotFoundError) lors du chargement des composants de comparaison d'images: {fnfe}", exc_info=True)
        logger.error("Veuillez vous assurer que tous les fichiers de données FAISS (.bin, .json) et le corpus de lieux sont présents aux chemins spécifiés.")
        inception_feature_extractor_model = None
        faiss_image_index = None
        faiss_image_ids = []
        maroc_lieux_data = {}
        logger.error("La fonctionnalité d'identification d'image par similarité visuelle sera désactivée.")
    except Exception as e:
        logger.error(f"Erreur GÉNÉRALE lors du chargement des composants de comparaison d'images: {e}", exc_info=True)
        inception_feature_extractor_model = None
        faiss_image_index = None
        faiss_image_ids = []
        maroc_lieux_data = {}
        logger.error("La fonctionnalité d'identification d'image par similarité visuelle sera désactivée.")

initialize_image_comparison_components()


# --- Outil de recherche web (Tavily) ---
tavily_tool = TavilySearchResults(max_results=3)

# --- Clé API OpenWeatherMap ---
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", None)
if not OPENWEATHERMAP_API_KEY:
    logger.warning("OPENWEATHERMAP_API_KEY n'est pas configurée dans les variables d'environnement. La fonction météo sera limitée ou désactivée.")


# --- Chargement des données statiques (activités, hôtels, etc.) ---
ACTIVITIES_DF, HOTELS_DF, RESTAURANTS_CAFES_DF, CITY_COORDS_MAP = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

def initialize_data():
    global ACTIVITIES_DF, HOTELS_DF, RESTAURANTS_CAFES_DF, CITY_COORDS_MAP
    try:
        ACTIVITIES_DF, HOTELS_DF, RESTAURANTS_CAFES_DF, CITY_COORDS_MAP = load_and_preprocess_data(use_realtime_api=False, target_cities_for_api=[])
        logger.info("Données statiques chargées avec succès pour le chatbot.")
        if ACTIVITIES_DF.empty:
            logger.warning("ACTIVITIES_DF est vide après le chargement initial.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement initial des données statiques pour le chatbot: {e}", exc_info=True)

initialize_data()

# --- Fonctions utilitaires ---
CITY_NAME_MAPPING = {
    "Marrakech": ["marakech", "marrakesh"], "Fès": ["fez", "fes"], "Casablanca": ["casa"],
    "Meknès": ["meknes"], "Rabat": [], "Agadir": [], "Chefchaouen": ["chaouen"],
    "Essaouira": ["mogador"], "Ouarzazate": [], "Tangier": ["tanger"],
    "Merzouga (Erg Chebbi)": ["merzouga", "erg chebbi"]
}

def normalize_city_name(city_name, city_mapping):
    if not city_name or pd.isna(city_name): return None
    city_name_lower = str(city_name).strip().lower()
    for canonical, variations in city_mapping.items():
        if city_name_lower == canonical.lower() or city_name_lower in [v.lower() for v in variations]: return canonical
    return str(city_name).strip().capitalize()

def retrieve_touristic_info_internal(user_query: str, city_name: Optional[str]) -> str:
    """
    Récupère des informations touristiques (activités, hôtels, restaurants)
    pour une ville donnée en utilisant les DataFrames chargés globalement.
    """
    db_context_parts = []
    
    normalized_city = None
    if city_name:
        normalized_city = normalize_city_name(city_name, CITY_NAME_MAPPING)
        if normalized_city:
            db_context_parts.append(f"Informations pour {normalized_city}:")
        else:
            db_context_parts.append(f"Informations générales (ville '{city_name}' non reconnue dans nos données):")
            
    if not ACTIVITIES_DF.empty:
        activities_filtered = ACTIVITIES_DF.copy()
        if normalized_city:
            activities_filtered = activities_filtered[activities_filtered['ville_normalisee'] == normalized_city]
        
        if not activities_filtered.empty:
            if not normalized_city and user_query: # Si pas de ville normalisée, essaie avec mots-clés dans la requête
                keywords = user_query.lower().split()
                activities_filtered = activities_filtered[
                    activities_filtered['nom'].str.lower().apply(lambda x: any(k in x for k in keywords)) |
                    activities_filtered['type'].str.lower().apply(lambda x: any(k in x for k in keywords))
                ]
            
            top_activities = activities_filtered.sort_values(by='rating', ascending=False).head(3)
            if not top_activities.empty:
                db_context_parts.append("Activités populaires : " + 
                                       ", ".join(top_activities['nom'].tolist()) + ".")

    if not HOTELS_DF.empty:
        hotels_filtered = HOTELS_DF.copy()
        if normalized_city:
            hotels_filtered = hotels_filtered[hotels_filtered['ville_normalisee'] == normalized_city]
        
        if not hotels_filtered.empty:
            top_hotels = hotels_filtered.sort_values(by='rating', ascending=False).head(2)
            if not top_hotels.empty:
                db_context_parts.append("Hébergements suggérés : " + 
                                       ", ".join(top_hotels['nom'].tolist()) + ".")
    
    if not RESTAURANTS_CAFES_DF.empty:
        rc_filtered = RESTAURANTS_CAFES_DF.copy()
        if normalized_city:
            rc_filtered = rc_filtered[rc_filtered['ville_normalisee'] == normalized_city]
        
        if not rc_filtered.empty:
            top_rc = rc_filtered.sort_values(by='rating', ascending=False).head(2)
            if not top_rc.empty:
                db_context_parts.append("Restaurants/Cafés recommandés : " + 
                                       ", ".join(top_rc['nom'].tolist()) + ".")

    if not db_context_parts:
        return "Aucune information spécifique n'a été trouvée dans nos données pour votre requête."
    
    return "\n".join(db_context_parts)

def get_weather_data(city_name: str) -> str:
    """
    Appelle l'API OpenWeatherMap pour obtenir les données météo actuelles.
    """
    if not OPENWEATHERMAP_API_KEY:
        return "Désolé, la clé API OpenWeatherMap n'est pas configurée. Je ne peux pas obtenir la météo pour le moment."

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    params = {
        "q": city_name,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric", # Pour obtenir la température en Celsius
        "lang": "fr" # Pour obtenir la description en français
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Lève une exception pour les codes d'état HTTP d'erreur
        weather_data = response.json()

        if weather_data.get("cod") == 200:
            main_data = weather_data.get("main", {})
            weather_desc = weather_data.get("weather", [{}])[0].get("description", "non disponible")
            temp = main_data.get("temp")
            feels_like = main_data.get("feels_like")
            humidity = main_data.get("humidity")
            wind_speed = weather_data.get("wind", {}).get("speed")

            return (
                f"La météo actuelle à {city_name.capitalize()} est : {weather_desc}.\n"
                f"Température : {temp}°C (Ressentie : {feels_like}°C).\n"
                f"Humidité : {humidity}%.\n"
                f"Vitesse du vent : {wind_speed} m/s."
            )
        else:
            return f"Désolé, je n'ai pas pu obtenir la météo pour {city_name}. Erreur : {weather_data.get('message', 'inconnue')}."
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de l'appel à l'API OpenWeatherMap pour {city_name}: {e}", exc_info=True)
        return "Désolé, une erreur de connexion est survenue lors de la récupération de la météo."
    except Exception as e:
        logger.error(f"Erreur inattendue lors du traitement de la météo pour {city_name}: {e}", exc_info=True)
        return "Désolé, une erreur est survenue lors du traitement des données météorologiques."


# --- Fonctions d'Interaction avec la Base de Données (Lecture seule) ---

def get_user_trips_summary_db(user_id: int) -> str:
    """Récupère un résumé des voyages planifiés de l'utilisateur."""
    user_obj = User.objects.filter(id=user_id).first()
    if not user_obj:
        return "Utilisateur non trouvé."

    trips = Trip.objects.filter(user=user_obj).order_by('-created_at')
    if not trips.exists():
        return "Vous n'avez pas encore de voyages planifiés. Utilisez la section 'Planifier un Voyage' pour en créer un !"

    summary = "Voici un résumé de vos voyages planifiés :\n"
    for i, trip in enumerate(trips[:5]): # Afficher les 5 derniers voyages
        summary += f"- **{trip.name}** ({trip.num_days_str} jours à {trip.target_cities_input_str})\n"
    if trips.count() > 5:
        summary += f"\n... et {trips.count() - 5} autres voyages. Demandez-moi si vous voulez la liste complète !"
    return summary

def get_trip_details_db(user_id: int, trip_name_or_city: str, day_number: Optional[int] = None) -> str:
    """Récupère les détails d'un voyage spécifique ou d'un jour spécifique de ce voyage."""
    user_obj = User.objects.filter(id=user_id).first()
    if not user_obj:
        return "Utilisateur non trouvé."

    trip = Trip.objects.filter(user=user_obj).filter(
        Q(name__icontains=trip_name_or_city) |
        Q(target_cities_input_str__icontains=trip_name_or_city)
    ).first()

    if not trip:
        return f"Désolé, je n'ai pas trouvé de voyage correspondant à '{trip_name_or_city}' dans vos plans. Essayez de mentionner le nom exact du voyage ou une ville cible (ex: 'mon voyage à Marrakech')."

    details = f"Détails pour le voyage **'{trip.name}'** ({trip.num_days_str} jours à {trip.target_cities_input_str}):\n"

    if day_number:
        day_plan = trip.days.filter(day_number=day_number).first()
        if day_plan:
            details += f"\n--- Jour {day_plan.day_number} à {day_plan.city_name} ---\n"
            if not day_plan.activity_items.exists():
                details += "  Aucune activité détaillée prévue pour ce jour.\n"
            else:
                for item in day_plan.activity_items.all().order_by('order_in_day'):
                    item_type_display = item.item_type
                    if item.activity_type_name and item.activity_type_name != 'N/A':
                        item_type_display = item.activity_type_name
                    details += f"  - {item.name} (Type: {item_type_display})\n"
        else:
            details += f"\nLe jour {day_number} n'existe pas dans ce voyage ou n'a pas été trouvé."
    else:
        for day_plan in trip.days.all().order_by('day_number'):
            details += f"\n--- Jour {day_plan.day_number} à {day_plan.city_name} ---\n"
            if not day_plan.activity_items.exists():
                details += "  Aucune activité détaillée pour ce jour.\n"
            else:
                for item in day_plan.activity_items.all().order_by('order_in_day')[:3]:
                    item_type_display = item.item_type
                    if item.activity_type_name and item.activity_type_name != 'N/A':
                        item_type_display = item.activity_type_name
                    details += f"  - {item.name} (Type: {item_type_display})\n"
                if day_plan.activity_items.count() > 3:
                    details += f"  ... et {day_plan.activity_items.count() - 3} autres activités. Demandez-moi des détails sur le 'Jour {day_plan.day_number}' pour tout voir !\n"
    return details


# --- Chaînes Langchain ---

def get_language_detection_chain():
    prompt = PromptTemplate(
        template="Detect the language of the following text and respond with only the language name (e.g., 'French', 'English', 'Arabic').\nText: {text}",
        input_variables=["text"]
    )
    return prompt | llm_phi3 | StrOutputParser()

def get_intention_chain():
    prompt = PromptTemplate(
        template="""Based on the user's message, classify the intent. Choose one of the following:
        - 'plan_trip': User wants to plan a trip.
        - 'get_info': User wants general information about Morocco, a city, an activity, etc. (Includes general recommendations, weather, news)
        - 'consult_trip': User wants to consult their saved trip plans or details about a specific trip/day.
        - 'multimodal_query': User has provided an image and might be asking a question about it or seeking suggestions based on it.
        - 'greeting': User is greeting.
        - 'other': Any other intent not covered above.

        Respond with only the intent name.

        Message: {message}""",
        input_variables=["message"]
    )
    return prompt | llm_phi3 | StrOutputParser()

class EntityExtraction(BaseModel):
    city: Optional[str] = Field(description="The city mentioned by the user.")
    budget: Optional[str] = Field(description="The budget mentioned by the user (e.g., 'low', 'medium', 'high', or a specific amount).")
    travel_style: Optional[str] = Field(description="The travel style mentioned by the user (e.g., 'adventurous', 'relaxing', 'cultural').")
    start_date: Optional[str] = Field(description="The start date of the trip in YYYY-MM-DD format.")
    end_date: Optional[str] = Field(description="The end date of the trip in YYYY-MM-DD format.")
    object_in_image: Optional[str] = Field(description="The main object or scene described in the image, if applicable.")
    gps_coords: Optional[str] = Field(description="The GPS coordinates provided by the user (latitude,longitude).")
    trip_identifier: Optional[str] = Field(description="A name or city that identifies a specific trip (e.g., 'my Marrakech trip', 'the trip to Fes').")
    day_number: Optional[int] = Field(description="The specific day number of a trip the user is asking about (e.g., 'day 2', 'jour 3').")
    info_type: Optional[str] = Field(description="The specific type of information requested, e.g., 'weather', 'news', 'activities', 'hotels'.")


parser = JsonOutputParser(pydantic_object=EntityExtraction)

def get_entity_extraction_chain():
    prompt = PromptTemplate(
        template="""Extract the following entities from the user's message as a JSON object.
        {format_instructions}
        If an entity is not mentioned, its value should be null.
        For dates, infer the year if not provided, assuming the current year or next year if the date is in the past.
        For object_in_image, focus on the most prominent object or scene.
        For gps_coords, extract if explicitly mentioned as 'latitude,longitude'.
        For trip_identifier, try to capture the name or a key city that identifies a saved trip.
        For day_number, extract the specific day number mentioned (e.g., 'jour 1' -> 1).
        For info_type, determine if the user is asking specifically for 'weather', 'news', 'activities', 'hotels', or null if general.

        Message: {message}
        JSON: """,
        input_variables=["message"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt | llm_phi3 | parser

def get_history_summarizer_chain():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant that summarizes chat history concisely."),
        HumanMessage(content="Summarize the following chat history for context in a new conversation. Focus on the main topic and any relevant details about user's preferences or prior questions:\n{chat_history_string}\n\nSummary:")
    ])
    return prompt | llm_phi3 | StrOutputParser()


def get_general_response_chain():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Tu es MarocGuide, un guide de voyage IA spécialisé dans le tourisme au Maroc.
        Réponds à l'utilisateur de manière concise et amicale en utilisant les informations fournies.
        Base-toi *uniquement* sur les 'Infos Locales', 'Infos Web' et 'Historique Précédent' si elles sont pertinentes.
        If the question is about trip planning, advise the user to use the dedicated section.
        If you don't have the information, state it clearly without inventing.
        Do not hallucinate and do not fabricate facts.
        
        Local Info (internal): {db_context}
        Web Info: {web_context}
        Previous History: {history_summary}
        """),
        HumanMessage(content="Question: {question}")
    ])
    return prompt | llm_mistral | StrOutputParser()

# --- get_multimodal_chain et get_activity_suggestion_chain sont conservées pour référence si besoin ---
# Elles utilisent le modèle LLaVA via Ollama pour des descriptions générales.
def get_multimodal_chain():
    prompt = PromptTemplate(
        template="""You are an expert image analysis assistant.
        Describe the image in detail, focusing on any prominent objects, landmarks, or scenes.
        Then, specifically list 2-3 key objects or elements you clearly identify in the image.
        If the image seems related to tourism or travel, mention that.
        
        Description and Key Objects: """,
        input_variables=[]
    )
    
    def _run_llava(input_data):
        image_base64 = input_data.get("image_base64")
        user_message_text = input_data.get("user_message_text", "")

        if not image_base64:
            return "Aucune image fournie pour l'analyse."

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt.format()},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ]
            )
        ]
        
        try:
            response = llm_llava_description.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à LLaVA: {e}", exc_info=True)
            return "Désolé, je n'ai pas pu analyser l'image."

    return RunnableLambda(_run_llava) | StrOutputParser()

def get_activity_suggestion_chain():
    prompt = PromptTemplate(
        template="""You are MarocGuide, an AI assistant specialized in Moroccan tourism.
        Based on the following image analysis and GPS coordinates, suggest 3-5 relevant and interesting activities or places to visit.
        Prioritize activities that are related to the objects identified in the image and are geographically close to the provided GPS coordinates.
        If no GPS coordinates are given, assume the location is Fès, Morocco, and suggest activities relevant to Fès and the image content.
        For each suggestion, provide a brief reason why it's relevant.
        *Do not invent activities or places.* Only suggest what is truly relevant and potentially searchable.
        
        Image Analysis: {image_analysis}
        GPS Coordinates: {gps_coords} (Format: latitude,longitude. If empty, assume Fès, Morocco)
        
        Suggestions: """,
        input_variables=["image_analysis", "gps_coords"]
    )
    return prompt | llm_mistral | StrOutputParser()


# --- FONCTION DE COMPARAISON D'IMAGE PAR SIMILARITÉ VISUELLE (Cœur du RAG visuel) ---
def find_similar_place_by_image(image_base64: str) -> str:
    # Vérifie si les composants d'identification d'image sont chargés au démarrage de l'appli
    if inception_feature_extractor_model is None or faiss_image_index is None or maroc_lieux_data is None:
        logger.error("Composants d'identification d'image (InceptionV3/FAISS) non initialisés.")
        return "Désolé, la fonctionnalité d'identification d'image est actuellement indisponible. Veuillez contacter l'administrateur."

    try:
        # Décode l'image Base64 en un objet PIL Image
        image_bytes = base64.b64decode(image_base64)
        image = PILImage.open(BytesIO(image_bytes)).convert("RGB")

        # Prétraiter l'image pour InceptionV3
        # Redimensionne l'image à la taille attendue par InceptionV3 (299x299)
        img_array = keras_image_preprocessing.img_to_array(image.resize((299, 299)))
        # Ajoute une dimension de batch (TensorFlow s'attend à [batch_size, height, width, channels])
        img_array_expanded = np.expand_dims(img_array, axis=0) 
        # Applique le prétraitement spécifique à InceptionV3 (normalisation des pixels)
        img_preprocessed = preprocess_input(img_array_expanded) 
        
        # Obtenir les features (embeddings) du modèle InceptionV3
        # verbose=0 pour ne pas afficher la barre de progression de Keras
        query_embedding = inception_feature_extractor_model.predict(img_preprocessed, verbose=0).flatten().astype(np.float32)
        
        # Recherche dans l'index FAISS des images
        k = 3 # Nombre de résultats similaires à récupérer
        # np.expand_dims car faiss.search s'attend à un tableau 2D pour la requête (batch_size, embedding_dim)
        distances, indices = faiss_image_index.search(np.expand_dims(query_embedding, axis=0), k)

        if len(indices[0]) == 0:
            return "Aucun lieu similaire trouvé dans ma base de données d'images."

        # Initialisation pour éviter NameError si aucun match n'est jugé "concluant" par les seuils initiaux
        identified_lieu_info = None 
        
        # Le lieu le plus similaire trouvé par FAISS
        best_match_id = faiss_image_ids[indices[0][0]]
        best_match_distance = distances[0][0]
        best_match_info = maroc_lieux_data.get(best_match_id) # Informations complètes du lieu le plus similaire

        # --- SEUILS DE CONFIANCE (À AJUSTER APRÈS TESTS AVEC VOS DONNÉES) ---
        # Plus la distance L2 est PETITE, plus la similarité est GRANDE.
        SEUIL_CONFIANCE_HAUTE = 100.0  # Exemple: Si distance < 100, très confiant
        SEUIL_CONFIANCE_MOYENNE = 150.0 # Exemple: Si distance < 150, assez confiant
        
        # Le deuxième terme (différence avec le 2nd meilleur score) peut être utilisé si vous avez besoin
        # d'une grande distinction entre le 1er et le 2ème match pour être sûr.
        # SEUIL_DIFFERENCE_AVEC_SECOND_MATCH = 20.0 # Exemple: Le 1er doit être 20 unités meilleur que le 2ème

        confidence_message = ""
        if best_match_info: # Si un lieu a été trouvé
            if best_match_distance < SEUIL_CONFIANCE_HAUTE:
                confidence_message = "J'ai identifié ce lieu avec une grande confiance :"
            elif best_match_distance < SEUIL_CONFIANCE_MOYENNE:
                confidence_message = "J'ai trouvé un lieu très similaire à celui de votre image :"
            else: # Si la distance est au-delà du seuil moyen, l'identification est faible
                confidence_message = "Je n'ai pas pu identifier le lieu de manière concluante, mais voici le lieu le plus similaire que j'ai trouvé :"
            
            identified_lieu_info = best_match_info # On utilise toujours le meilleur match trouvé.
        else: # Cas où best_match_info est None (devrait pas arriver si len(indices[0]) > 0)
            return "Aucun lieu correspondant n'a pu être récupéré de la base de données malgré une recherche FAISS."


        reply_parts = []

        if identified_lieu_info: # Si nous avons des informations sur le lieu le plus similaire
            place_name = identified_lieu_info['name']
            place_city = identified_lieu_info['city']
            place_description = identified_lieu_info['description']
            place_image_path = identified_lieu_info.get('representative_image_path') # Récupérer le chemin d'image

            # --- Ajout de l'image de l'utilisateur d'abord ---
            # Si l'image téléchargée par l'utilisateur doit aussi apparaître
            user_uploaded_image_html = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Votre image" style="max-width:250px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #ccc;"><br>'
            reply_parts.append(f"Votre image a été reçue :")
            reply_parts.append(user_uploaded_image_html)


            image_url_html = ""
            if place_image_path:
                # Construire l'URL statique de l'image pour le frontend
                # Cela dépend de comment Django sert vos static files.
                # 'STATIC_IMAGES_SERVED_PATH' doit correspondre au chemin que Django sert pour ce dossier.
                static_image_full_path = f"/{STATIC_IMAGES_SERVED_PATH}{place_image_path}" # Commence par / pour URL absolue
                image_url_html = f'<img src="{static_image_full_path}" alt="Image de {place_name}" style="max-width:250px; border-radius: 8px; margin-top: 10px;"><br>'
            
            reply_parts.append(f"{confidence_message} **{place_name}** à **{place_city}**.")
            if image_url_html:
                reply_parts.append(image_url_html) # Ajouter l'image du lieu de la base
            reply_parts.append(f"Description : {place_description}")
            
            # Obtenir des activités/infos pour la ville du lieu identifié
            # user_query est vide ici si seulement une image est fournie, donc on passe place_name comme query context.
            activities_info = retrieve_touristic_info_internal(place_name, place_city)

            if activities_info and "Aucune information spécifique" not in activities_info:
                reply_parts.append(f"\nVoici aussi quelques informations utiles sur {place_city} ou des activités connexes :\n{activities_info}")
            
            return "\n\n".join(reply_parts)
        else:
            return "Désolé, je n'ai pas trouvé d'informations détaillées pour le lieu identifié dans ma base de connaissances."

    except Exception as e:
        logger.error(f"Erreur grave lors de la comparaison d'image: {e}", exc_info=True)
        return f"Désolé, une erreur interne est survenue lors de l'identification du lieu à partir de l'image. Erreur: {e}"

# --- Fonction principale de traitement des requêtes utilisateur (process_user_query) ---
def process_user_query(user_message: str, user_id: int, image_base64: Optional[str] = None, gps_coords: Optional[str] = None) -> str:
    logger.info(f"Traitement de la requête pour l'utilisateur {user_id}: Message='{user_message[:50]}...', Image_present={bool(image_base64)}, GPS_present={bool(gps_coords)}")

    recent_history = ChatMessage.objects.filter(user_id=user_id).order_by('-timestamp')[:10]
    formatted_history = []
    for msg in reversed(recent_history):
        role = "User" if msg.is_from_user else "AI"
        formatted_history.append(f"{role}: {msg.message}")
    chat_history_string = "\n".join(formatted_history)
    
    history_summary = "Aucun historique pertinent."
    if len(formatted_history) > 2:
        try:
            summarizer_chain = get_history_summarizer_chain()
            history_summary = summarizer_chain.invoke({"chat_history_string": chat_history_string})
            logger.debug(f"Résumé historique: {history_summary}")
        except Exception as e:
            logger.warning(f"Impossible de résumer l'historique: {e}. Fallback à l'historique complet.", exc_info=True)
            history_summary = "Historique précédent: " + chat_history_string


    # 1. Déterminer l'intention principale et extraire les entités
    intent_chain = get_intention_chain()
    # ------ LIGNE CORRIGÉE ------
    entity_extraction_chain = get_entity_extraction_chain()

    if image_base64 and not user_message.strip():
        intent = "multimodal_query"
        entities = EntityExtraction() 
        logger.debug("Intention forcée à multimodal_query, entités textuelles vides.")
    else:
        try:
            intent = intent_chain.invoke({"message": user_message}).strip().lower()
            entities = entity_extraction_chain.invoke({"message": user_message})
            logger.debug(f"Intention détectée: {intent}, Entités extraites: {entities}")
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'intention/extraction d'entités: {e}. Fallback à 'other'.", exc_info=True)
            intent = "other"
            entities = EntityExtraction() 


    # Extraire les entités de manière sûre avec getattr
    city = getattr(entities, 'city', None)
    budget = getattr(entities, 'budget', None)
    travel_style = getattr(entities, 'travel_style', None)
    start_date = getattr(entities, 'start_date', None)
    end_date = getattr(entities, 'end_date', None)
    object_in_image = getattr(entities, 'object_in_image', None)
    gps_coords_extracted = getattr(entities, 'gps_coords', None)
    trip_identifier = getattr(entities, 'trip_identifier', None)
    day_number = getattr(entities, 'day_number', None)
    info_type = getattr(entities, 'info_type', None)

    # 2. Traitement basé sur l'intention (ordre de priorité des fonctionnalités)

    # PRIORITY 1: Requête Multimodale avec Image (Comparaison Visuelle Directe via InceptionV3/FAISS)
    if intent == "multimodal_query" and image_base64:
        logger.info("Exécution de l'identification d'image par similarité visuelle.")
        
        identification_reply = find_similar_place_by_image(image_base64)
        
        return identification_reply

    # PRIORITY 2: Consultation des voyages planifiés
    elif intent == "consult_trip":
        logger.info("Exécution de la logique de consultation de voyage.")
        if trip_identifier:
            return get_trip_details_db(user_id, trip_identifier, day_number)
        else:
            return get_user_trips_summary_db(user_id)

    # PRIORITY 3: Recherche d'informations générales (inclut recommandations d'activités, météo, actualités)
    elif intent == "get_info":
        logger.info("Exécution de la logique de recherche d'informations générales.")
        
        response_content = ""
        
        if info_type == "weather":
            target_city = city if city else "Fès" 
            weather_info = get_weather_data(target_city) 
            response_content += f"{weather_info}\n\n"
            if "actualités" in user_message.lower() or "nouvelles" in user_message.lower():
                pass 
            else:
                return response_content.strip() 


        if info_type == "news" or ("actualités" in user_message.lower() or "nouvelles" in user_message.lower()):
            query_for_web = user_message
            if "actualités" in query_for_web.lower() and not city:
                 query_for_web = f"actualités Maroc" 
            
            try:
                web_search_results = tavily_tool.invoke({"query": query_for_web})
                if web_search_results:
                    formatted_results = "\n".join([f"- {res['title']}: {res['content']}" for res in web_search_results])
                    response_content += f"Voici ce que j'ai trouvé sur le web concernant votre question :\n{formatted_results}\n\n"
                else:
                    response_content += "Désolé, je n'ai pas trouvé d'actualités pertinentes sur le web.\n\n"
            except Exception as e:
                logger.error(f"Erreur lors de la recherche web pour actualités: {e}", exc_info=True)
                response_content += "Désolé, une erreur est survenue lors de la recherche d'actualités.\n\n"
        
        if not info_type in ["weather", "news"] or (info_type is None and not response_content): 
            info_result_db = retrieve_touristic_info_internal(user_message, city)
            if "Aucune information" not in info_result_db:
                response_content += f"Voici quelques informations basées sur nos données internes :\n{info_result_db}\n\n"
            elif not response_content: 
                response_content = "Désolé, je n'ai pas trouvé d'informations pertinentes pour cette requête ni en interne ni sur le web."
        
        return response_content.strip() if response_content else "Désolé, je n'ai pas trouvé d'informations pertinentes pour cette requête."


    # Autres intentions
    elif intent == "plan_trip":
        return "Pour planifier un voyage détaillé, veuillez utiliser la section 'Planifier un Voyage' dédiée. Je peux vous aider avec des informations générales sur les villes et activités."

    elif intent == "greeting":
        return "Bonjour ! Je suis MarocGuide, votre assistant personnel pour explorer le Maroc. Comment puis-je vous aider aujourd'hui ?"

    else: # 'other' ou intention non reconnue
        logger.info("Exécution de la chaîne de réponse générale pour une intention 'autre'.")
        general_response_chain = get_general_response_chain()
        return general_response_chain.invoke({
            "question": user_message,
            "db_context": "", 
            "web_context": "", 
            "history_summary": history_summary
        })