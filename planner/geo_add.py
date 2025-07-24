import json
from serpapi import GoogleSearch
import os

# --- Remplacez par votre VRAIE clé API SerpAPI ---
# Assurez-vous que cette clé est la même que dans votre settings.py
SERPAPI_API_KEY = "bc8f7f649098b04ab862462cf737563c9acc62dd85553e4cd74107893ff6999c" 

# --- Requêtes de test ---
TEST_CITY = "Marrakech" # Ou toute autre ville que vous voulez tester

print(f"--- Test de l'API SerpAPI pour {TEST_CITY} ---")

# --- Test 1: Hôtels avec engine="Google Hotels" ---
print("\nTentative de récupération d'HÔTELS via engine='Google Hotels'...")
hotel_params = {
    "api_key": SERPAPI_API_KEY,
    "engine": "Google Hotels", # C'est le nom du moteur que vous devez utiliser
    "q": f"hotels in {TEST_CITY}, Morocco",
    "type": "search",
    "hl": "fr",
    "num": 5
}
try:
    Google Hotels = GoogleSearch(hotel_params)
    results_hotels = Google Hotels.get_dict()
    print("Réponse brute de l'API pour les HÔTELS :")
    print(json.dumps(results_hotels, indent=2))
    if "hotels_results" in results_hotels:
        print(f"--> Succès : {len(results_hotels['hotels_results'])} hôtels trouvés.")
        for i, hotel in enumerate(results_hotels["hotels_results"][:3]): # Afficher les 3 premiers
            name = hotel.get("name") or hotel.get("title", "Nom non trouvé")
            rating = hotel.get("rating")
            print(f"    {i+1}. Nom: {name}, Note: {rating}")
    elif "error" in results_hotels:
        print(f"--- ERREUR API pour les HÔTELS : {results_hotels['error']}")
    else:
        print("--- AVERTISSEMENT : La recherche d'hôtels n'a pas renvoyé 'hotels_results'.")
except Exception as e:
    print(f"--- ERREUR LORS DE L'APPEL API pour les HÔTELS : {e}")

# --- Test 2: Restaurants avec engine="Maps" ---
print("\nTentative de récupération de RESTAURANTS via engine='Maps'...")
restaurant_params = {
    "api_key": SERPAPI_API_KEY,
    "engine": "Maps", # C'est le nom du moteur que vous devez utiliser pour les lieux
    "q": f"restaurants in {TEST_CITY}, Morocco",
    "type": "search",
    "hl": "fr",
    "num": 5
}
try:
    search_restaurants = GoogleSearch(restaurant_params)
    results_restaurants = search_restaurants.get_dict()
    print("Réponse brute de l'API pour les RESTAURANTS :")
    print(json.dumps(results_restaurants, indent=2))
    if "local_results" in results_restaurants:
        print(f"--> Succès : {len(results_restaurants['local_results'])} restaurants trouvés.")
        for i, restaurant in enumerate(results_restaurants["local_results"][:3]):
            name = restaurant.get("title") or restaurant.get("name", "Nom non trouvé")
            rating = restaurant.get("rating")
            print(f"    {i+1}. Nom: {name}, Note: {rating}")
    elif "error" in results_restaurants:
        print(f"--- ERREUR API pour les RESTAURANTS : {results_restaurants['error']}")
    else:
        print("--- AVERTISSEMENT : La recherche de restaurants n'a pas renvoyé 'local_results'.")
except Exception as e:
    print(f"--- ERREUR LORS DE L'APPEL API pour les RESTAURANTS : {e}")

# --- Test 3: Cafés avec engine="Maps" ---
print("\nTentative de récupération de CAFÉS via engine='Maps'...")
cafe_params = {
    "api_key": SERPAPI_API_KEY,
    "engine": "Maps", # C'est le nom du moteur que vous devez utiliser pour les lieux
    "q": f"cafes in {TEST_CITY}, Morocco",
    "type": "search",
    "hl": "fr",
    "num": 5
}
try:
    search_cafes = GoogleSearch(cafe_params)
    results_cafes = search_cafes.get_dict()
    print("Réponse brute de l'API pour les CAFÉS :")
    print(json.dumps(results_cafes, indent=2))
    if "local_results" in results_cafes:
        print(f"--> Succès : {len(results_cafes['local_results'])} cafés trouvés.")
        for i, cafe in enumerate(results_cafes["local_results"][:3]):
            name = cafe.get("title") or cafe.get("name", "Nom non trouvé")
            rating = cafe.get("rating")
            print(f"    {i+1}. Nom: {name}, Note: {rating}")
    elif "error" in results_cafes:
        print(f"--- ERREUR API pour les CAFÉS : {results_cafes['error']}")
    else:
        print("--- AVERTISSEMENT : La recherche de cafés n'a pas renvoyé 'local_results'.")
except Exception as e:
    print(f"--- ERREUR LORS DE L'APPEL API pour les CAFÉS : {e}")