import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Configurez le logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration des chemins de fichiers ---
# Le dossier 'rag_data' doit être dans le même répertoire que ce script ou dans un répertoire parent
# pour que le chemin soit correct.
DATA_RAG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_data')
MAROC_LIEUX_CORPUS_FILE = os.path.join(DATA_RAG_FOLDER, "maroc_lieux_corpus.json")
FAISS_TEXT_INDEX_FILE = os.path.join(DATA_RAG_FOLDER, "faiss_text_index.bin")
FAISS_TEXT_IDS_FILE = os.path.join(DATA_RAG_FOLDER, "faiss_text_ids.json")

def build_faiss_text_index():
    """
    Construit et sauvegarde un index FAISS à partir d'un corpus de texte.
    Cet index sera utilisé par le chatbot pour la recherche sémantique.
    """
    logging.info("Démarrage de la construction de l'index FAISS textuel.")

    # Vérification de l'existence du fichier corpus
    if not os.path.exists(MAROC_LIEUX_CORPUS_FILE):
        logging.error(f"Fichier corpus introuvable : {MAROC_LIEUX_CORPUS_FILE}")
        return

    # --- 1. Chargement des données ---
    logging.info("Chargement du corpus de lieux depuis le fichier JSON...")
    try:
        with open(MAROC_LIEUX_CORPUS_FILE, "r", encoding="utf-8") as f:
            corpus = json.load(f)
    except Exception as e:
        logging.error(f"Erreur lors de la lecture du fichier JSON : {e}")
        return

    # Assurez-vous que chaque document a un champ 'id' unique
    doc_ids = [doc.get('id') for doc in corpus]
    if None in doc_ids or len(doc_ids) != len(set(doc_ids)):
        logging.error("Chaque document du corpus doit avoir un 'id' unique et non nul.")
        return
    
    # --- 2. Préparation du texte et création des embeddings ---
    # Utilisez SentenceTransformer pour transformer le texte en vecteurs.
    # On combine les champs 'name' et 'description' pour un embedding plus riche.
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    logging.info(f"Chargement du modèle d'embeddings : {model_name}...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logging.error(f"Impossible de charger le modèle SentenceTransformer : {e}")
        return

    texts_to_embed = [
        f"Nom: {doc.get('name', '')}. Description: {doc.get('description', '')}"
        for doc in corpus
    ]

    logging.info(f"Création des embeddings pour {len(texts_to_embed)} documents...")
    try:
        embeddings = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype('float32') # FAISS a besoin de float32
    except Exception as e:
        logging.error(f"Erreur lors de la création des embeddings : {e}")
        return

    # --- 3. Construction et sauvegarde de l'index FAISS ---
    logging.info("Construction de l'index FAISS...")
    dimension = embeddings.shape[1]
    
    # Normalisation des vecteurs pour une meilleure recherche de similarité cosinus
    faiss.normalize_L2(embeddings)
    
    # IndexFlatIP utilise le produit scalaire, équivalent à la similarité cosinus pour des vecteurs normalisés.
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)

    # Création du répertoire s'il n'existe pas
    os.makedirs(DATA_RAG_FOLDER, exist_ok=True)
    
    logging.info("Sauvegarde de l'index FAISS et des IDs...")
    faiss.write_index(index, FAISS_TEXT_INDEX_FILE)
    
    # Sauvegarde des IDs dans un fichier JSON pour la récupération des documents
    with open(FAISS_TEXT_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    logging.info("Index FAISS textuel créé et sauvegardé avec succès.")
    logging.info(f"Fichiers créés : {FAISS_TEXT_INDEX_FILE} et {FAISS_TEXT_IDS_FILE}")

if __name__ == "__main__":
    build_faiss_text_index()