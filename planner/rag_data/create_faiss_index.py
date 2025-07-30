import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration des Chemins ---
# Adaptez ce chemin pour qu'il corresponde à l'emplacement réel de votre dossier 'rag_data'.
# Si ce script est dans 'votre_projet_django/planner/rag_data/',
# alors les fichiers sont dans le même répertoire que le script.
BASE_DATA_DIR = os.path.dirname(os.path.abspath(__file__)) # Chemin du dossier 'rag_data'

CORPUS_FILE = os.path.join(BASE_DATA_DIR, "maroc_lieux_corpus.json")
FAISS_INDEX_FILE = os.path.join(BASE_DATA_DIR, "faiss_index_maroc.bin")
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' # Modèle d'embeddings textuels

# --- Chargement du Corpus ---
print(f"Chargement du corpus depuis : {CORPUS_FILE}")
try:
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus_documents = json.load(f)
except FileNotFoundError:
    print(f"Erreur: Le fichier {CORPUS_FILE} est introuvable. Assurez-vous de l'avoir créé et nommé correctement.")
    exit()
except json.JSONDecodeError as e:
    print(f"Erreur de format JSON dans {CORPUS_FILE}: {e}. Assurez-vous que le fichier est un JSON valide (une liste d'objets JSON).")
    exit()

# --- Préparation des Textes pour l'Embedding ---
# Nous allons embédder la description et les mots-clés pour chaque document
corpus_texts = [doc["description"] + " " + " ".join(doc.get("keywords", [])) for doc in corpus_documents]
corpus_ids = [doc["id"] for doc in corpus_documents] # Garder les IDs pour la référence

# --- Chargement du Modèle d'Embeddings ---
print(f"Chargement du modèle d'embeddings : {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- Génération des Embeddings ---
print(f"Génération de {len(corpus_texts)} embeddings...")
corpus_embeddings = embedding_model.encode(corpus_texts, convert_to_tensor=False)
print(f"Embeddings générés. Dimension : {corpus_embeddings.shape[1]}")

# --- Création et Indexation FAISS ---
embedding_dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension) # Index FlatL2 (Euclidean distance)

print("Ajout des embeddings à l'index FAISS...")
index.add(corpus_embeddings.astype(np.float32)) # FAISS attend des float32

# --- Sauvegarde de l'Index FAISS ---
# L'index est sauvegardé. Les documents originaux sont déjà dans maroc_lieux_corpus.json
faiss.write_index(index, FAISS_INDEX_FILE)

print(f"Index FAISS sauvegardé sous : {FAISS_INDEX_FILE}")
print("\nPréparation de la base de connaissances FAISS terminée avec succès !")