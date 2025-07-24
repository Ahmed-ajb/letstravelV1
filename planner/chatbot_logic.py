# planner/chatbot_logic.py

import os
import json
import logging
import pandas as pd
import hashlib 
from django.conf import settings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Import pour la gestion des messages multimodaux
from langchain_core.messages import HumanMessage

# --- Configuration ---
logger = logging.getLogger(__name__)
load_dotenv()

# --- Constantes et Chemins ---
ACTIVITIES_JSON_PATH = settings.DATA_DIR / 'activities.json'
HOTELS_JSON_PATH = settings.DATA_DIR / 'hotels_with_real_coordinates_vf_v2.json'
CITY_NAME_MAPPING = {
    "Marrakech": ["marakech", "marrakesh"], "Fès": ["fez", "fes"], "Casablanca": ["casa"],
    "Meknès": ["meknes"], "Rabat": [], "Agadir": [], "Chefchaouen": ["chaouen"],
    "Essaouira": ["mogador"], "Ouarzazate": [], "Tangier": ["tanger"],
    "Merzouga (Erg Chebbi)": ["merzouga", "erg chebbi"]
}

# --- Outils & Services ---
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
tavily_search_tool = None
if TAVILY_API_KEY:
    try:
        tavily_search_tool = TavilySearch(k=3, tavily_api_key=TAVILY_API_KEY)
        logger.info("Outil de recherche Tavily initialisé.")
    except Exception as e:
        logger.warning(f"Erreur initialisation Tavily: {e}")

_ai_response_cache = {}
_cache_duration = timedelta(minutes=30) 


class LLMService:
    _instance = None
    _llm_fast_instance = None
    _llm_main_instance = None
    _llm_multimodal_instance = None # Pour LLaVA

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance.llm_fast = None
            cls._instance.llm_main = None
            cls._instance.llm_multimodal = None
        return cls._instance

    def _load_llm_fast(self):
        if self._llm_fast_instance is None:
            logger.info("Chargement du LLM 'phi3' (rapide)...")
            try:
                self._llm_fast_instance = OllamaLLM(model="phi3", temperature=0)
                self._llm_fast_instance.invoke("Test") 
                logger.info("Connexion au LLM 'phi3' réussie.")
            except Exception as e:
                logger.error(f"ERREUR CONNEXION OLLAMA 'phi3': {e}.")
                self._llm_fast_instance = None
        return self._llm_fast_instance

    def _load_llm_main(self):
        if self._llm_main_instance is None:
            logger.info("Chargement du LLM 'mistral' (principal)...")
            try:
                self._llm_main_instance = OllamaLLM(model="mistral", temperature=0.6)
                self._llm_main_instance.invoke("Test")
                logger.info("Connexion au LLM 'mistral' réussie.")
            except Exception as e:
                logger.error(f"ERREUR CONNEXION OLLAMA 'mistral': {e}.")
                self._llm_main_instance = None
        return self._llm_main_instance

    def _load_llm_multimodal(self):
        """Charge le modèle LLM multimodal (llava) si ce n'est pas déjà fait."""
        if self._llm_multimodal_instance is None:
            logger.info("Chargement du LLM 'llava' (multimodal)...")
            try:
                # Utilisation du bind pour s'assurer que le format de réponse est JSON
                self._llm_multimodal_instance = OllamaLLM(model="llava", temperature=0.2)
                self._llm_multimodal_instance.invoke("Test multimodal")
                logger.info("Connexion au LLM 'llava' réussie.")
            except Exception as e:
                logger.error(f"ERREUR CONNEXION OLLAMA 'llava': {e}. Assurez-vous d'avoir fait 'ollama pull llava'.")
                self._llm_multimodal_instance = None
        return self._llm_multimodal_instance

    def get_llm_fast(self): return self._load_llm_fast()
    def get_llm_main(self): return self._load_llm_main()
    def get_llm_multimodal(self): return self._load_llm_multimodal()


llm_service = LLMService()
_CACHED_DATA = None 

def load_touristic_data():
    global _CACHED_DATA
    if _CACHED_DATA: return _CACHED_DATA
    try:
        from .utils import load_and_preprocess_data
        activities_df, hotels_df, _, _ = load_and_preprocess_data()
        _CACHED_DATA = (activities_df, hotels_df)
        return activities_df, hotels_df
    except Exception as e:
        logger.error(f"Erreur chargement données touristiques : {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()

def retrieve_touristic_info(entities: dict, activities_df: pd.DataFrame, hotels_df: pd.DataFrame):
    act_df_filtered, hot_df_filtered = activities_df.copy(), hotels_df.copy()
    city_found = None
    if city := entities.get("ville"):
        from .utils import normalize_city_name
        normalized_city = normalize_city_name(city, CITY_NAME_MAPPING)
        if normalized_city:
            city_found = normalized_city
            if not act_df_filtered.empty:
                act_df_filtered = act_df_filtered[act_df_filtered['ville_normalisee'] == normalized_city]
            if not hot_df_filtered.empty:
                hot_df_filtered = hot_df_filtered[hot_df_filtered['ville_normalisee'] == normalized_city]
    return {
        "city_found": city_found,
        "activities": act_df_filtered.head(3).to_dict(orient="records"), 
        "hotels": hot_df_filtered.head(2).to_dict(orient="records")
    }

def format_retrieved_data_for_prompt(retrieved_data: dict) -> str:
    city, activities, hotels = retrieved_data.get("city_found"), retrieved_data.get("activities", []), retrieved_data.get("hotels", [])
    if not city and not activities and not hotels:
        return "Aucune information spécifique n'a été trouvée."
    parts = [f"Infos pour {city if city else 'Non spécifiée'}:"]
    if activities: parts.append(f"- Activités: {', '.join([a.get('nom', 'N/A') for a in activities])}.")
    if hotels: parts.append(f"- Hôtels: {', '.join([h.get('name', 'N/A') for h in hotels])}.")
    return "\n".join(parts)

def search_web(query: str):
    if not tavily_search_tool: return "La recherche web n'est pas disponible."
    try: return tavily_search_tool.invoke(query)
    except Exception as e:
        logger.error(f"Erreur de recherche web Tavily : {e}")
        return "Une erreur s'est produite lors de la recherche web."

def get_chatbot_chains():
    llm_fast, llm_main = llm_service.get_llm_fast(), llm_service.get_llm_main()
    if not llm_fast or not llm_main: raise ConnectionError("Modèles LLM texte non disponibles.")
    
    lang_detect_chain = PromptTemplate.from_template("Langue de '{input}'? Réponds 'french', 'english', ou 'arabic'.") | llm_fast | StrOutputParser()
    intent_chain = PromptTemplate.from_template("Intention de '{input}'? Réponds 'salutation', 'remerciement', 'recherche_info_voyage', 'recherche_web', 'meteo', 'hors_sujet'.") | llm_fast | StrOutputParser()
    entity_chain = PromptTemplate.from_template("Extrais entités de '{input}'. Réponds en JSON: 'ville', 'budget'. Sinon {{}}.") | llm_fast | StrOutputParser()
    history_summarizer_chain = ChatPromptTemplate.from_messages([("user", "Résume l'historique:\n{chat_history}")]) | llm_fast | StrOutputParser()
    answer_chain = ChatPromptTemplate.from_messages([
        ("system", "Tu es FLOUKY, un guide de voyage au Maroc. Réponds à {user_name} en {language}, de façon courte et amicale. Utilise les infos locales, puis tes connaissances, puis le web. Résumé: {history_summary}"),
        ("user", "Infos locales:\n{db_context}\n\nInfos web:\n{web_context}\n\nQuestion de {user_name}:\n{input}")
    ]) | llm_main | StrOutputParser()
    
    return lang_detect_chain, intent_chain, entity_chain, answer_chain, history_summarizer_chain

def get_multimodal_chain():
    """Crée et retourne une chaîne simple pour les requêtes multimodales."""
    llm_multimodal = llm_service.get_llm_multimodal()
    if not llm_multimodal: raise ConnectionError("Modèle multimodal LLaVA non disponible.")
    return llm_multimodal

def process_user_query(user_query: str, user_name: str, chat_history: list = None, image_base64: str = None):
    """Traite la requête de l'utilisateur, qui peut être textuelle ou multimodale."""
    
    # CAS 1: Il y a une image -> Logique multimodale
    if image_base64:
        logger.info(f"Traitement d'une requête multimodale de {user_name}")
        try:
            multimodal_chain = get_multimodal_chain()
            prompt = user_query or "Décris cette image et explique son lien avec le Maroc."
            message_content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}]
            msg = HumanMessage(content=message_content)
            response = multimodal_chain.invoke([msg])
            return response
        except ConnectionError as e:
            return f"Désolé, le service d'analyse d'images est indisponible: {e}"
        except Exception as e:
            logger.error(f"Erreur traitement multimodal: {e}", exc_info=True)
            return "Oups, je n'ai pas réussi à analyser cette image."

    # CAS 2: Pas d'image -> Logique textuelle existante
    logger.info(f"Traitement d'une requête textuelle de {user_name}")
    try:
        lang_detect_chain, intent_chain, entity_chain, answer_chain, history_summarizer_chain = get_chatbot_chains()
    except ConnectionError as e:
        return f"Désolé, le service de chatbot est indisponible: {e}"

    simplified_history = [f"{'u' if m.is_from_user else 'a'}:{m.message}" for m in (chat_history or [])[-5:]]
    cache_key_data = user_query + "|||" + "|".join(simplified_history)
    cache_key = hashlib.md5(cache_key_data.encode('utf-8')).hexdigest()

    if cache_key in _ai_response_cache and (datetime.now() - _ai_response_cache[cache_key]['timestamp']) < _cache_duration:
        return _ai_response_cache[cache_key]['response']

    language = lang_detect_chain.invoke({"input": user_query}).strip().lower()
    if language not in ['french', 'english', 'arabic']: language = 'french'
    
    intent = intent_chain.invoke({"input": user_query}).strip().lower()

    if intent == "salutation": return f"Bonjour {user_name} ! Je suis FLOUKY, votre guide. Comment puis-je vous aider ?"
    if intent == "remerciement": return "Avec plaisir !"
    if intent == "hors_sujet": return "Je suis spécialisé dans les voyages au Maroc."
    
    try:
        history_str = "\n".join([f"{'User' if m.is_from_user else 'AI'}: {m.message}" for m in chat_history or []])
        history_summary = history_summarizer_chain.invoke({"chat_history": history_str or "Aucun"})
        
        entities_text = entity_chain.invoke({"input": user_query})
        entities = json.loads(entities_text[entities_text.find('{'):entities_text.rfind('}')+1]) if '{' in entities_text else {}
        
        activities_df, hotels_df = load_touristic_data()
        db_data = retrieve_touristic_info(entities, activities_df, hotels_df)
        db_context = format_retrieved_data_for_prompt(db_data)
        
        web_context = "Non utilisé."
        if intent in ["recherche_web", "meteo"] or (intent == "recherche_info_voyage" and "Aucune information" in db_context):
            web_context = search_web(user_query)

        final_response = answer_chain.invoke({
            "language": language, "user_name": user_name, "input": user_query,
            "db_context": db_context, "web_context": web_context, "history_summary": history_summary
        })

        _ai_response_cache[cache_key] = {'response': final_response, 'timestamp': datetime.now()}
        return final_response

    except Exception as e:
        logger.error(f"Erreur traitement de la requête textuelle: {e}", exc_info=True)
        return "Oups! Un problème technique est survenu."