import json
import os
import faiss
import numpy as np
from PIL import Image as PILImage 
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input 
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing import image as keras_image_preprocessing 

# --- Configuration des Chemins (relatifs au script) ---
BASE_DATA_DIR = os.path.dirname(os.path.abspath(__file__)) 

IMAGE_METADATA_FILE = os.path.join(BASE_DATA_DIR, "image_corpus_metadata.json")
IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images") 
FAISS_IMAGE_INDEX_FILE = os.path.join(BASE_DATA_DIR, "faiss_image_index_inception.bin") 
FAISS_IMAGE_IDS_FILE = os.path.join(BASE_DATA_DIR, "faiss_image_ids_inception.json") 
MAROC_LIEUX_CORPUS_FILE = os.path.join(BASE_DATA_DIR, "maroc_lieux_corpus.json") 

# --- Configuration du Modèle InceptionV3 pour l'Extraction de Features ---
print("Chargement et configuration du modèle InceptionV3 pour l'extraction de features...")
try:
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # Correction: Utiliser GlobalAveragePooling2D pour obtenir le vecteur de features.
    x = base_model.output 
    output = tf.keras.layers.GlobalAveragePooling2D()(x) 
    model = Model(inputs=base_model.input, outputs=output) 

    print("Modèle InceptionV3 chargé et configuré pour l'extraction de features via GlobalAveragePooling2D.")
except Exception as e:
    print(f"Erreur CRITIQUE lors du chargement du modèle InceptionV3: {e}")
    print("Assurez-vous que TensorFlow est correctement installé et que vous avez une connexion internet pour télécharger les poids 'imagenet'.")
    exit()

# --- Fonction pour charger et prétraiter une image ---
def get_img_embedding(img_path):
    img = PILImage.open(img_path).convert('RGB')
    img = img.resize((299, 299)) 

    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0) 

    img_preprocessed = preprocess_input(img_array_expanded) 
    
    features = model.predict(img_preprocessed, verbose=0) 

    return features.flatten() 

# --- Chargement des Métadonnées d'Images ---
print(f"Chargement des métadonnées d'images depuis : {IMAGE_METADATA_FILE}")
try:
    with open(IMAGE_METADATA_FILE, "r", encoding="utf-8") as f:
        image_metadata = json.load(f)
except FileNotFoundError:
    print(f"Erreur: Le fichier {IMAGE_METADATA_FILE} est introuvable. Créez-le selon le format spécifié (Partie 1.4).")
    exit()
except json.JSONDecodeError as e:
    print(f"Erreur de format JSON dans {IMAGE_METADATA_FILE}: {e}.")
    exit()

# --- Chargement de la base de données textuelle des lieux (pour les descriptions) ---
print(f"Chargement de la base de données de lieux pour les descriptions: {MAROC_LIEUX_CORPUS_FILE}")
try:
    with open(MAROC_LIEUX_CORPUS_FILE, "r", encoding="utf-8") as f:
        maroc_lieux_data_full = {doc['id']: doc for doc in json.load(f)}
except FileNotFoundError:
    print(f"Erreur: Le fichier {MAROC_LIEUX_CORPUS_FILE} est introuvable. Assurez-vous de l'avoir créé (Partie 1.2).")
    exit()
except json.JSONDecodeError as e:
    print(f"Erreur de format JSON dans {MAROC_LIEUX_CORPUS_FILE}: {e}.")
    exit()

# --- Génération des Embeddings Visuels ---
image_embeddings = []
image_ids_in_order = [] 

print(f"Génération des embeddings pour {len(image_metadata)} images...")
for i, item in enumerate(image_metadata):
    full_image_path = os.path.join(IMAGES_DIR, item['image_path_relative'])
    try:
        embedding = get_img_embedding(full_image_path)
        image_embeddings.append(embedding)
        image_ids_in_order.append(item['lieu_id']) 

    except FileNotFoundError:
        print(f"ATTENTION: Image non trouvée à {full_image_path}. Ignorée. Vérifiez le chemin et le nom du fichier.")
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {full_image_path}: {e}. Ignorée.")

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(image_metadata)} images traitées.")

# --- NOUVELLES LIGNES DE DEBUG : VÉRIFIER image_ids_in_order ---
print(f"\n--- Vérification des données avant sauvegarde ---")
print(f"Total d'images traitées avec succès: {len(image_embeddings)}")
print(f"Taille de la liste d'IDs (image_ids_in_order): {len(image_ids_in_order)}")
if len(image_ids_in_order) == 0:
    print("AVERTISSEMENT: La liste 'image_ids_in_order' est vide. Le fichier JSON des IDs sera vide.")
print(f"Exemple des 5 premiers IDs: {image_ids_in_order[:5]}")
# --- FIN NOUVELLES LIGNES DE DEBUG ---


if not image_embeddings: # Si aucune image n'a généré d'embedding
    print("Aucun embedding d'image généré. Assurez-vous que les images existent et sont accessibles et qu'elles n'ont pas causé d'erreur de traitement.")
    exit()

image_embeddings_array = np.array(image_embeddings).astype(np.float32)
embedding_dimension = image_embeddings_array.shape[1]

# --- Création et Indexation FAISS ---
print(f"Création de l'index FAISS avec dimension {embedding_dimension}...")
index = faiss.IndexFlatL2(embedding_dimension) 

print("Ajout des embeddings à l'index FAISS...")
index.add(image_embeddings_array)

# --- Sauvegarde de l'Index FAISS et des IDs ---
faiss.write_index(index, FAISS_IMAGE_INDEX_FILE) 

with open(FAISS_IMAGE_IDS_FILE, "w", encoding="utf-8") as f:
    json.dump(image_ids_in_order, f, ensure_ascii=False, indent=2)

print(f"\nIndex FAISS des images sauvegardé sous : {FAISS_IMAGE_INDEX_FILE}")
print(f"Liste des IDs d'images sauvegardée sous : {FAISS_IMAGE_IDS_FILE}")
print("\nPréparation de la base de données d'images FAISS terminée avec succès !")