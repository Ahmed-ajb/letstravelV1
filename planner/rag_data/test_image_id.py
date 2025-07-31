import os
import json
import faiss
import numpy as np
from PIL import Image as PILImage
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing import image as keras_image_preprocessing 
import base64
from io import BytesIO

# --- CONFIGURATION DES CHEMINS (Adaptez-les !) ---
# Remplacez par le chemin ABSOLU de votre dossier 'rag_data'
DATA_RAG_FOLDER = "C:\\Users\\legionPC7\\Desktop\\AhmedProject\\letstravelV1\\planner\\rag_data" 

FAISS_IMAGE_INDEX_FILE = os.path.join(DATA_RAG_FOLDER, "faiss_image_index_inception.bin")
FAISS_IMAGE_IDS_FILE = os.path.join(DATA_RAG_FOLDER, "faiss_image_ids_inception.json")
MAROC_LIEUX_CORPUS_FILE = os.path.join(DATA_RAG_FOLDER, "maroc_lieux_corpus.json")

# --- Chargement du modèle InceptionV3 (comme dans chatbot_logic.py) ---
try:
    base_model_inception = InceptionV3(weights='imagenet', include_top=False)
    inception_feature_extractor_model = Model(inputs=base_model_inception.input, outputs=base_model_inception.get_layer('avg_pool').output)
    inception_feature_extractor_model.trainable = False 
    print("Modèle InceptionV3 chargé avec succès pour le test.")
except Exception as e:
    print(f"ERREUR LORS DU TEST DE CHARGEMENT INCEPTIONV3: {e}")
    exit()

# --- Chargement de l'index FAISS et des données ---
try:
    faiss_image_index = faiss.read_index(FAISS_IMAGE_INDEX_FILE)
    with open(FAISS_IMAGE_IDS_FILE, "r", encoding="utf-8") as f:
        faiss_image_ids = json.load(f)
    with open(MAROC_LIEUX_CORPUS_FILE, "r", encoding="utf-8") as f:
        maroc_lieux_data = {doc['id']: doc for doc in json.load(f)}
    print("Index FAISS et données de lieux chargés avec succès pour le test.")
except Exception as e:
    print(f"ERREUR LORS DU TEST DE CHARGEMENT FAISS/DONNÉES: {e}")
    exit()

# --- Fonction de test de similarité ---
def test_find_similar_place_by_image(image_base64_string: str) -> str:
    try:
        image_bytes = base64.b64decode(image_base64_string)
        image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((299, 299)) # Redimensionner avant conversion en array
        img_array = keras_image_preprocessing.img_to_array(image)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)

        query_embedding = inception_feature_extractor_model.predict(img_preprocessed, verbose=0).flatten().astype(np.float32)

        k = 3
        distances, indices = faiss_image_index.search(np.expand_dims(query_embedding, axis=0), k)

        if len(indices[0]) == 0:
            return "Aucun lieu similaire trouvé."

        results_details = []
        for i, idx in enumerate(indices[0]):
            lieu_id = faiss_image_ids[idx]
            lieu_info = maroc_lieux_data.get(lieu_id)
            if lieu_info:
                results_details.append(
                    f"- Nom: {lieu_info['name']}, Ville: {lieu_info['city']}. "
                    f"Description: {lieu_info['description'][:50]}... (Sim: {distances[0][i]:.2f})"
                )
        return "Résultats de la recherche FAISS:\n" + "\n".join(results_details)

    except Exception as e:
        return f"ERREUR LORS DE LA RECHERCHE SIMILARITÉ: {e}"

# --- Exécuter le test ---
# REMPLACEZ CE CHEMIN par le chemin ABSOLU d'une de vos images de test
TEST_IMAGE_PATH = "C:/Users/legionPC7/Downloads/bab-bou-jeloud.jpg"

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"ATTENTION: L'image de test '{TEST_IMAGE_PATH}' n'existe pas. Veuillez adapter le chemin.")
    exit()

with open(TEST_IMAGE_PATH, "rb") as image_file:
    test_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

print("\n--- TEST DE SIMILARITÉ D'IMAGE ---")
result = test_find_similar_place_by_image(test_image_base64)
print(result)