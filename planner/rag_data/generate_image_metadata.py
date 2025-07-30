import json
import os
import re

def slugify(text):
    """
    Convertit une chaîne de caractères en un format 'slug' pour les noms de fichiers.
    Ex: "Médina de Marrakech" -> "medina_de_marrakech"
    """
    text = text.lower()
    # Remplacer les caractères non alphanumériques (y compris les accents) par des espaces, puis slugifier
    text = re.sub(r'[^\w\s-]', '', text) 
    text = re.sub(r'[\s_]+', '_', text)
    return text.strip('_')

# --- Configuration des Chemins ---
# Adaptez ce chemin pour qu'il corresponde à l'emplacement réel de votre dossier 'rag_data'.
BASE_DATA_DIR = os.path.dirname(os.path.abspath(__file__)) # Chemin du dossier 'rag_data'

MAROC_LIEUX_CORPUS_FILE = os.path.join(BASE_DATA_DIR, "maroc_lieux_corpus.json")
IMAGES_BASE_DIR = os.path.join(BASE_DATA_DIR, "images") # Dossier où sont stockées vos images
OUTPUT_METADATA_FILE = os.path.join(BASE_DATA_DIR, "image_corpus_metadata.json")

# --- Chargement du Corpus des Lieux pour les IDs ---
print(f"Chargement du corpus de lieux depuis : {MAROC_LIEUX_CORPUS_FILE}")
try:
    with open(MAROC_LIEUX_CORPUS_FILE, "r", encoding="utf-8") as f:
        maroc_lieux_corpus = json.load(f)
    # Créer un mapping rapide des noms de lieux slugifiés aux IDs et noms complets
    lieu_info_map = {
        slugify(doc['name']): {'id': doc['id'], 'name': doc['name'], 'city': doc['city']}
        for doc in maroc_lieux_corpus
    }
    print(f"{len(maroc_lieux_corpus)} lieux chargés du corpus.")
except FileNotFoundError:
    print(f"Erreur: Le fichier {MAROC_LIEUX_CORPUS_FILE} est introuvable. Assurez-vous de l'avoir créé et nommé correctement.")
    exit()
except json.JSONDecodeError as e:
    print(f"Erreur de format JSON dans {MAROC_LIEUX_CORPUS_FILE}: {e}.")
    exit()

# --- Parcourir le dossier d'images et générer les métadonnées ---
image_metadata_list = []
processed_count = 0

print(f"Parcours du dossier d'images : {IMAGES_BASE_DIR}")
for root, dirs, files in os.walk(IMAGES_BASE_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            relative_path = os.path.relpath(os.path.join(root, file), IMAGES_BASE_DIR)

            # Extraire le nom de la ville et le nom du lieu slugifié du chemin
            # Exemple: marrakech/koutoubia_01.jpg -> ville='marrakech', lieu_slug='koutoubia'
            path_parts = relative_path.replace("\\", "/").split('/')

            if len(path_parts) >= 2: # Attendre au moins city_folder/image_name.jpg
                city_slug_from_path = path_parts[-2] # Le dossier parent de l'image est la ville slugifiée
                image_name_without_ext = os.path.splitext(path_parts[-1])[0] # ex: koutoubia_01

                # Tenter de trouver le lieu_slug dans le nom du fichier
                # Si le fichier est nommé 'koutoubia_01.jpg', on extrait 'koutoubia'
                # Cela nécessite une convention de nommage des fichiers: lieu_slug_numero.jpg
                lieu_slug_from_file = "_".join(image_name_without_ext.split('_')[:-1]) # Retire _01, _02, etc.

                # Chercher le lieu correspondant dans notre map basée sur le corpus
                # Nous essayons de matcher le lieu_slug du fichier avec les clés slugifiées dans lieu_info_map
                matched_lieu_info = lieu_info_map.get(lieu_slug_from_file)

                if matched_lieu_info:
                    image_metadata_list.append({
                        "image_path_relative": relative_path.replace("\\", "/"), # Assurer des slashs pour la compatibilité
                        "lieu_id": matched_lieu_info['id'],
                        "name": matched_lieu_info['name'],
                        "city": matched_lieu_info['city']
                    })
                    processed_count += 1
                else:
                    print(f"AVERTISSEMENT: Aucun lieu correspondant trouvé dans le corpus pour l'image: {relative_path}. Vérifiez le nom du fichier ou la description du lieu dans votre corpus.")
            else:
                print(f"AVERTISSEMENT: Chemin d'image inattendu: {relative_path}. Ignoré.")

# --- Sauvegarde du Fichier de Métadonnées ---
if image_metadata_list:
    with open(OUTPUT_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(image_metadata_list, f, ensure_ascii=False, indent=2)
    print(f"\nFichier de métadonnées d'images généré avec succès : {OUTPUT_METADATA_FILE}")
    print(f"Total de {processed_count} images avec métadonnées.")
else:
    print("\nAucun fichier d'image traité ou aucun lieu correspondant trouvé. Le fichier de métadonnées n'a pas été créé.")
    print("Vérifiez que les images sont dans le dossier 'images' et que leurs noms correspondent aux lieux de votre corpus.")