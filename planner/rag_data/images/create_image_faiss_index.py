import torch
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaProcessor

LLAVA_MODEL_NAME_OR_PATH = "liuhaotian/llava-v1.5-7b"

print("--- DÉBUT DU TEST DE CHARGEMENT MANUEL ---")
print(f"Tentative de chargement manuel des composants pour : {LLAVA_MODEL_NAME_OR_PATH}")

try:
    # Étape 1 : Charger le tokenizer seul
    print("1. Chargement du Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLAVA_MODEL_NAME_OR_PATH)
    print("   Tokenizer chargé avec succès.")

    # Étape 2 : Charger le processeur d'image seul
    print("\n2. Chargement de l'Image Processor...")
    image_processor = CLIPImageProcessor.from_pretrained(LLAVA_MODEL_NAME_OR_PATH)
    print("   Image Processor chargé avec succès.")

    # Étape 3 : Créer le processeur LLaVA en lui donnant les deux pièces
    print("\n3. Création du LlavaProcessor à partir des composants...")
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
    print("   LlavaProcessor assemblé avec succès.")

    print("\n\n>>> SUCCÈS ! Le processeur a été assemblé manuellement.")
    print(f">>> Type de processeur créé : {type(processor)}")

except Exception as e:
    print(f"\n\n>>> ERREUR : Le chargement a échoué.")
    print(f">>> Message d'erreur : {e}")

print("\n--- FIN DU TEST DE CHARGEMENT MANUEL ---")