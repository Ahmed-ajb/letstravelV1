# planner/chatbot_logic.py

import os
import json
import logging
import pandas as pd
from django.conf import settings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# --- Configuration ---
logger = logging.getLogger(__name__)
load_dotenv()

# --- Constantes et Chemins (inchangés) ---
ACTIVITIES_JSON_PATH = settings.DATA_DIR / 'activities.json'
HOTELS_JSON_PATH = settings.DATA_DIR / 'hotels_with_real_coordinates_vf_v2.json'
CITY_NAME_MAPPING = {
    "Marrakech": ["marakech", "marrakesh"], "Fès": ["fez", "fes"], "Casablanca": ["casa"],
    "Meknès": ["meknes"], "Rabat": [], "Agadir": [], "Chefchaouen": ["chaouen"],
    "Essaouira": ["mogador"], "Ouarzazate": [], "Tangier": ["tanger"],
    "Merzouga (Erg Chebbi)": ["merzouga", "erg chebbi"]
}

# --- Outils & Services (inchangés) ---
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
tavily_search_tool = None
if TAVILY_API_KEY:
    try:
        tavily_search_tool = TavilySearch(k=3, tavily_api_key=TAVILY_API_KEY)
        logger.info("Outil de recherche Tavily initialisé.")
    except Exception as e:
        logger.warning(f"Erreur initialisation Tavily: {e}")

class LLMService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance.llm_fast = None
            cls._instance.llm_main = None
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        try:
            self.llm_fast = OllamaLLM(model="phi3", temperature=0)
            self.llm_fast.invoke("Test")
            logger.info("Connexion au LLM 'phi3' réussie.")
            self.llm_main = OllamaLLM(model="mistral", temperature=0.6)
            self.llm_main.invoke("Test")
            logger.info("Connexion au LLM 'mistral' réussie.")
        except Exception as e:
            logger.error(f"ERREUR CONNEXION OLLAMA: {e}. Vérifiez que Ollama est lancé.")

llm_service = LLMService()
_CACHED_DATA = None

# --- Fonctions de traitement de données (inchangées) ---
def load_touristic_data():
    global _CACHED_DATA
    if _CACHED_DATA: return _CACHED_DATA
    try:
        from .utils import load_and_preprocess_data
        activities_df, hotels_df, _ = load_and_preprocess_data()
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
    city = retrieved_data.get("city_found")
    activities = retrieved_data.get("activities", [])
    hotels = retrieved_data.get("hotels", [])

    if not city and not activities and not hotels:
        return "Aucune information spécifique n'a été trouvée dans la base de données locale."

    parts = [f"Informations trouvées pour la ville de {city if city else 'Non spécifiée'}:"]
    if activities:
        parts.append(f"- Activités: {', '.join([a.get('nom', 'N/A') for a in activities])}.")
    if hotels:
        parts.append(f"- Hôtels: {', '.join([h.get('name', 'N/A') for h in hotels])}.")
        
    return "\n".join(parts)

def search_web(query: str):
    if not tavily_search_tool: return "La recherche web n'est pas disponible."
    try: return tavily_search_tool.invoke(query)
    except Exception as e:
        logger.error(f"Erreur de recherche web Tavily : {e}")
        return "Une erreur s'est produite lors de la recherche web."

# --- MODIFICATION MAJEURE : NOUVEAUX PROMPTS ---
def get_chatbot_chains():
    llm_fast, llm_main = llm_service.llm_fast, llm_service.llm_main
    if not llm_fast or not llm_main: raise ConnectionError("Modèles LLM non disponibles.")
    
    lang_detect_chain = PromptTemplate.from_template(
        "<|user|>Quelle est la langue de cette question : '{input}'? Réponds avec 'french', 'english', ou 'arabic'.<|end|><|assistant|>"
    ) | llm_fast | StrOutputParser()

    intent_chain = PromptTemplate.from_template(
        """<|user|>Analyse l'intention. Réponds par UN mot: 'salutation', 'remerciement', 'recherche_info_voyage', 'recherche_web', 'meteo', 'hors_sujet'.
        - "cherche sur le web", "sur internet" -> recherche_web
        - "météo", "quel temps" -> meteo
        - infos sur voyage, hôtel, activité -> recherche_info_voyage
        Question: '{input}'<|end|><|assistant|>"""
    ) | llm_fast | StrOutputParser()

    entity_chain = PromptTemplate.from_template(
        "<|user|>Extrais les entités de '{input}'. Réponds UNIQUEMENT en JSON. Entités: 'ville', 'budget'. Si rien, réponds {{}}.<|end|><|assistant|>"
    ) | llm_fast | StrOutputParser()
    
    history_summarizer_chain = ChatPromptTemplate.from_messages([
        ("user", "Résume cet historique de conversation en une phrase. Historique:\n{chat_history}"),
    ]) | llm_fast | StrOutputParser()

    answer_chain = ChatPromptTemplate.from_messages([
        ("system", """Tu es FLOUKY, un guide de voyage au Maroc, amical, chaleureux et expert. Tu parles avec {user_name}.

        **Ta mission :** Répondre de manière naturelle et utile en suivant une hiérarchie d'information stricte.

        **HIÉRARCHIE DES CONNAISSANCES (à suivre dans l'ordre) :**
        1.  **Données Locales :** C'est ta source principale. Si le "Contexte des données locales" contient des informations, base ta réponse dessus.
        2.  **Connaissances Générales :** Si les données locales sont vides ou ne répondent pas à la question, utilise tes propres connaissances sur le Maroc pour aider {user_name}.
        3.  **Recherche Web :** N'utilise le "Contexte de la recherche web" qu'en dernier recours, ou si {user_name} le demande explicitement (météo, recherche web).

        **RÈGLES DE CONVERSATION :**
        - **Langue :** Réponds TOUJOURS dans la langue demandée : {language}.
        - **Ton :** Sois amical et engageant. Adresse-toi directement à {user_name}.
        - **Format :** Formule une réponse courte et conversationnelle. Pas de listes à puces.
        - **Transparence :** Ne dis JAMAIS "d'après mes données" ou "d'après le contexte". Intègre l'information naturellement.

        Résumé de votre conversation : {history_summary}"""),
        ("user", "Contexte des données locales:\n{db_context}\n\nContexte de la recherche web:\n{web_context}\n\nQuestion de {user_name}:\n{input}")
    ]) | llm_main | StrOutputParser()
    
    return lang_detect_chain, intent_chain, entity_chain, answer_chain, history_summarizer_chain

# --- MODIFICATION MAJEURE : NOUVELLE LOGIQUE DE TRAITEMENT ---
def process_user_query(user_query: str, user_name: str, chat_history: list = None):
    if not llm_service.llm_fast:
        return "Désolé, le service de chatbot est actuellement indisponible."

    lang_detect_chain, intent_chain, entity_chain, answer_chain, history_summarizer_chain = get_chatbot_chains()
    
    language = lang_detect_chain.invoke({"input": user_query}).strip().lower()
    if language not in ['french', 'english', 'arabic']: language = 'french'
    
    intent = intent_chain.invoke({"input": user_query}).strip().lower()

    if intent == "salutation": return f"Bonjour {user_name} ! Je suis FLOUKY, votre guide pour le Maroc. Comment puis-je vous aider aujourd'hui ?"
    if intent == "remerciement": return "Avec plaisir !"
    if intent == "hors_sujet": return "Je suis spécialisé dans les voyages au Maroc, mais n'hésitez pas si vous avez une question sur ce sujet !"
    
    try:
        history_summary = history_summarizer_chain.invoke({"chat_history": "\n".join([f"{'H' if m.is_from_user else 'IA'}: {m.message}" for m in chat_history]) if chat_history else "Aucun"})
        
        entities = {}
        if "recherche" in intent or "meteo" in intent:
            entities_text = entity_chain.invoke({"input": user_query})
            try:
                start = entities_text.find('{')
                end = entities_text.rfind('}') + 1
                if start != -1 and end != 0:
                    entities = json.loads(entities_text[start:end])
            except json.JSONDecodeError:
                logger.warning(f"Impossible de parser le JSON d'entités: {entities_text}")
        
        activities_df, hotels_df = load_touristic_data()
        db_data = retrieve_touristic_info(entities, activities_df, hotels_df)
        db_context = format_retrieved_data_for_prompt(db_data)
        
        web_context = "Non utilisé."
        # On considère les données locales "vides" si on n'a trouvé ni activité, ni hôtel pour une ville spécifique
        no_local_data = not db_data.get("activities") and not db_data.get("hotels")
        if intent in ["recherche_web", "meteo"] or (intent == "recherche_info_voyage" and no_local_data):
            logger.info(f"Déclenchement de la recherche web. Raison: {intent}, Données locales trouvées: {not no_local_data}")
            web_context = search_web(user_query)

        return answer_chain.invoke({
            "language": language,
            "user_name": user_name,
            "input": user_query,
            "db_context": db_context,
            "web_context": web_context,
            "history_summary": history_summary
        })
    except Exception as e:
        logger.error(f"Erreur traitement de la requête: {e}", exc_info=True)
        return "Oups! Un problème technique est survenu."
