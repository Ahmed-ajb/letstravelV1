import json
import os
import faiss
import numpy as np
from PIL import Image as PILImage # Renommé pour éviter conflit
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Configuration des Chemins (relatifs au script) ---
BASE_DATA_DIR = os.path.dirname(os.path.abspath(__file__)) # Chemin du dossier 'rag_data'

IMAGE_METADATA_FILE = os.path.join(BASE_DATA_DIR, "image_corpus_metadata.json")
IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images") # Dossier où sont stockées vos images
FAISS_IMAGE_INDEX_FILE = os.path.join(BASE_DATA_DIR, "faiss_image_index.bin") # Index pour les embeddings d'image
FAISS_IMAGE_IDS_FILE = os.path.join(BASE_DATA_DIR, "faiss_image_ids.json") # Liste des IDs des images dans l'ordre de l'index
MAROC_LIEUX_CORPUS_FILE = os.path.join(BASE_DATA_DIR, "maroc_lieux_corpus.json") # Base de descriptions

# --- Configuration du Modèle LLaVA pour l'Extraction de Features Visuelles ---
# C'est le modèle LLaVA de base que vous utiliseriez avec Ollama pour la description
LLAVA_MODEL_NAME_OR_PATH = "liuhaotian/llava-v1.6-mistral-7b" # Ou liuhaotian/llava-v1.5-7b

print(f"Chargement du modèle LLaVA pour l'extraction de features : {LLAVA_MODEL_NAME_OR_PATH}...")
try:
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_NAME_OR_PATH)

    model_full = AutoModelForCausalLM.from_pretrained(
        LLAVA_MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16, # Utiliser FP16 pour économiser la VRAM sur RTX 4090
        low_cpu_mem_usage=True # Optimiser l'utilisation CPU lors du chargement
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_full.to(device)

    if hasattr(model_full, 'get_model') and hasattr(model_full.get_model(), 'vision_tower'):
        vision_tower = model_full.get_model().vision_tower
    elif hasattr(model_full, 'vision_tower'):
        vision_tower = model_full.vision_tower
    else:
        raise AttributeError("Impossible de trouver le 'vision_tower' dans le modèle LLaVA chargé.")

    vision_tower.eval() # Mettre le vision_tower en mode évaluation
    vision_tower.to(device) # S'assurer qu'il est sur le bon appareil

    print(f"Vision Tower chargée sur : {vision_tower.device}")

except Exception as e:
    print(f"Erreur CRITIQUE lors du chargement du modèle LLaVA pour les features visuelles: {e}")
    print("Assurez-vous que les dépendances (transformers, torch) sont correctement installées et que le modèle est accessible.")
    exit()

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
image_ids_in_order = [] # Pour garder l'ordre des IDs d'images pour retrouver les métadonnées

print(f"Génération des embeddings pour {len(image_metadata)} images...")
for i, item in enumerate(image_metadata):
    full_image_path = os.path.join(IMAGES_DIR, item['image_path_relative'])
    try:
        image = PILImage.open(full_image_path).convert('RGB')

        inputs = processor(images=image, return_tensors="pt").to(vision_tower.device)

        with torch.no_grad():
            features_output = vision_tower(inputs.pixel_values.to(vision_tower.dtype))

            if hasattr(features_output, 'pooler_output') and features_output.pooler_output is not None:
                image_features_selected = features_output.pooler_output
            elif hasattr(features_output, 'last_hidden_state'):
                image_features_selected = features_output.last_hidden_state[:, 0, :]
            else:
                raise ValueError(f"Structure d'output du vision_tower inattendue: {type(features_output)}. Ne peut pas extraire les features globales.")

            image_embedding = image_features_selected.squeeze(0).cpu().numpy().flatten()

        image_embeddings.append(image_embedding)
        image_ids_in_order.append(item['lieu_id']) # ID du lieu pour retrouver les infos plus tard

    except FileNotFoundError:
        print(f"ATTENTION: Image non trouvée à {full_image_path}. Ignorée. Vérifiez le chemin et le nom du fichier.")
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {full_image_path}: {e}. Ignorée.")

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(image_metadata)} images traitées.")

if not image_embeddings:
    print("Aucun embedding d'image généré. Assurez-vous que les images existent et sont accessibles.")
    exit()

image_embeddings_array = np.array(image_embeddings).astype(np.float32)
embedding_dimension = image_embeddings_array.shape[1]

# --- Création et Indexation FAISS ---
print(f"Création de l'index FAISS avec dimension {embedding_dimension}...")
index = faiss.IndexFlatL2(embedding_dimension) # Index de distance L2 (euclidienne)

print("Ajout des embeddings à l'index FAISS...")
index.add(image_embeddings_array)

# --- Sauvegarde de l'Index FAISS et des IDs ---
faiss.write_index(index, FAISS_IMAGE_INDEX_FILE)

with open(FAISS_IMAGE_IDS_FILE, "w", encoding="utf-8") as f:
    json.dump(image_ids_in_order, f, ensure_ascii=False, indent=2)

print(f"Index FAISS des images sauvegardé sous : {FAISS_IMAGE_INDEX_FILE}")
print(f"Liste des IDs d'images sauvegardée sous : {FAISS_IMAGE_IDS_FILE}")
print("\nPréparation de la base de données d'images FAISS terminée avec succès !")