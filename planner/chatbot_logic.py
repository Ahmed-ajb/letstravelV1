# planner/chatbot_logic.py

import os
import json
import re
import logging
from datetime import datetime
import pandas as pd
# from django.conf import settings # Pas nécessaire pour os.environ.get directement

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

import requests # NOUVEL IMPORT pour les requêtes HTTP
from dotenv import load_dotenv # NOUVEL IMPORT pour charger les variables d'environnement

from .utils import load_and_preprocess_data
from planner.models import Trip, TripDay, DailyActivityItem, ChatMessage, User
from django.db.models import Q # Import pour les requêtes OR


logger = logging.getLogger(__name__)

# --- Charger les variables d'environnement du fichier .env ---
load_dotenv()

# --- Configuration des modèles Ollama ---
MODEL_PHI3 = "phi3"
MODEL_MISTRAL = "mistral"
MODEL_LLAVA = "llava"

llm_phi3 = Ollama(model=MODEL_PHI3, temperature=0.3)
llm_mistral = Ollama(model=MODEL_MISTRAL, temperature=0.3)
llm_llava = Ollama(model=MODEL_LLAVA, temperature=0.3)

# --- Outil de recherche web ---
tavily_tool = TavilySearchResults(max_results=3)

# --- Clé API OpenWeatherMap ---
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", None)
if not OPENWEATHERMAP_API_KEY:
    logger.warning("OPENWEATHERMAP_API_KEY n'est pas configurée dans les variables d'environnement. La fonction météo sera limitée ou désactivée.")


# --- Chargement des données statiques (une seule fois au démarrage de l'application) ---
ACTIVITIES_DF, HOTELS_DF, RESTAURANTS_CAFES_DF, CITY_COORDS_MAP = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

def initialize_data():
    global ACTIVITIES_DF, HOTELS_DF, RESTAURANTS_CAFES_DF, CITY_COORDS_MAP
    try:
        ACTIVITIES_DF, HOTELS_DF, RESTAURANTS_CAFES_DF, CITY_COORDS_MAP = load_and_preprocess_data(use_realtime_api=False, target_cities_for_api=[])
        logger.info("Données statiques chargées avec succès pour le chatbot.")
        if ACTIVITIES_DF.empty:
            logger.warning("ACTIVITIES_DF est vide après le chargement initial.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement initial des données statiques pour le chatbot: {e}")

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

# --- NOUVELLE FONCTION POUR L'API OPENWEATHERMAP ---
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


# --- FONCTIONS D'INTERACTION AVEC LA BASE DE DONNÉES (LECTURE SEULE) ---

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
        Si la question est sur la planification de voyage, conseille l'utilisateur d'utiliser la section dédiée.
        Si tu n'as pas l'information, dis-le clairement sans inventer.
        Ne fais pas d'hallucinations et ne fabrique pas de faits.
        
        Infos Locales (internes): {db_context}
        Infos Web: {web_context}
        Historique Précédent: {history_summary}
        """),
        HumanMessage(content="Question: {question}")
    ])
    return prompt | llm_mistral | StrOutputParser()

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
            response = llm_llava.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à LLaVA: {e}")
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


# --- Fonction principale de traitement des requêtes utilisateur ---
def process_user_query(user_message: str, user_id: int, image_base64: Optional[str] = None, gps_coords: Optional[str] = None) -> str:
    logger.info(f"Traitement de la requête pour l'utilisateur {user_id}: Message='{user_message[:50]}...', Image_present={bool(image_base64)}, GPS_present={bool(gps_coords)}")

    # Récupérer l'historique complet pour la contextualisation
    recent_history = ChatMessage.objects.filter(user_id=user_id).order_by('-timestamp')[:10] # 10 derniers messages
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
            logger.warning(f"Impossible de résumer l'historique: {e}. Fallback à l'historique complet.")
            history_summary = "Historique précédent: " + chat_history_string


    # 1. Déterminer l'intention principale et extraire les entités
    intent_chain = get_intention_chain()
    entity_extraction_chain = get_entity_extraction_chain()

    if image_base64 and not user_message.strip():
        intent = "multimodal_query"
        entities = EntityExtraction() # Créer une instance vide pour l'interface
        logger.debug("Intention forcée à multimodal_query, entités textuelles vides.")
    else:
        intent = intent_chain.invoke({"message": user_message}).strip().lower()
        entities = entity_extraction_chain.invoke({"message": user_message})
        logger.debug(f"Intention détectée: {intent}, Entités extraites: {entities}")

    # Extraire les entités de manière sûre avec .get()
    city = entities.get('city')
    info_type = entities.get('info_type')
    trip_identifier = entities.get('trip_identifier')
    day_number = entities.get('day_number')

    # 2. Traitement basé sur l'intention (ordre de priorité des fonctionnalités)

    # PRIORITY 1: Requête Multimodale avec Image
    if intent == "multimodal_query" and image_base64:
        logger.info("Exécution de la chaîne multimodale pour l'analyse d'image.")
        
        llava_chain = get_multimodal_chain()
        image_analysis_result = llava_chain.invoke({
            "image_base64": image_base64,
            "user_message_text": user_message
        })
        logger.info(f"Résultat de l'analyse d'image LLaVA: {image_analysis_result[:200]}...")

        if not gps_coords:
            reply = f"J'ai analysé votre image :\n\n{image_analysis_result}\n\nPour des suggestions d'activités basées sur la localisation, veuillez partager vos coordonnées GPS."
        else:
            activity_suggestion_chain = get_activity_suggestion_chain()
            final_gps_coords = gps_coords
            
            suggestions = activity_suggestion_chain.invoke({
                "image_analysis": image_analysis_result,
                "gps_coords": final_gps_coords
            })
            reply = f"Voici ce que j'ai compris de votre image et de votre position :\n\n{image_analysis_result}\n\nVoici quelques suggestions d'activités basées sur cela :\n{suggestions}"
        return reply

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
        
        # Gérer la météo en premier si info_type est "weather"
        if info_type == "weather":
            target_city = city if city else "Fès" # Ville par défaut si non spécifiée
            weather_info = get_weather_data(target_city) # Utilise la nouvelle fonction météo
            response_content += f"{weather_info}\n\n"
            # Si le message contient aussi des mots clés d'actualités, continuer vers tavily
            if "actualités" in user_message.lower() or "nouvelles" in user_message.lower():
                pass # Continue pour la recherche web après la météo
            else:
                return response_content.strip() # Retourne juste la météo si pas d'autres demandes explicites


        # Gérer les actualités ou d'autres recherches web si info_type est "news" ou si la météo n'était pas la seule demande
        if info_type == "news" or ("actualités" in user_message.lower() or "nouvelles" in user_message.lower()):
            query_for_web = user_message
            if "actualités" in query_for_web.lower() and not city:
                 query_for_web = f"actualités Maroc" # Actualités du Maroc par défaut
            
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
        
        # Gérer les recommandations d'activités/hôtels basée sur les données internes
        # Ceci sera appelé si ce n'est PAS une requête météo/actualités PUR
        if not info_type in ["weather", "news"] or (info_type is None and not response_content): # Si l'info_type est générique ou non spécifié, ou si aucune réponse n'a encore été générée
            info_result_db = retrieve_touristic_info_internal(user_message, city)
            if "Aucune information" not in info_result_db:
                response_content += f"Voici quelques informations basées sur nos données internes :\n{info_result_db}\n\n"
            elif not response_content: # Si aucune réponse n'a encore été générée par météo/actualités
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