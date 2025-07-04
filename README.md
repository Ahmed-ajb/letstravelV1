# letstravelV1


MarocPlanner - Application de Planification de VoyageBienvenue sur MarocPlanner ! Cette application web, développée avec Django, permet de générer des itinéraires de voyage personnalisés au Maroc et intègre un chatbot intelligent pour assister les utilisateurs.Suivez ce guide pour installer et lancer le projet sur un nouvel ordinateur.1. PrérequisAvant de commencer, assurez-vous d'avoir installé les logiciels suivants sur votre machine :Python (version 3.10 ou supérieure) : Télécharger PythonGit : Télécharger GitOllama (pour le chatbot) : Télécharger Ollama2. InstallationÉtape A : Cloner le ProjetOuvrez un terminal ou une invite de commandes et clonez le projet depuis votre dépôt GitHub.git clone https://github.com/Ahmed-ajb/letstravelV1.git
cd letstravelV1
Étape B : Configurer l'Environnement VirtuelIl est fortement recommandé d'utiliser un environnement virtuel pour isoler les dépendances du projet.# Créer un environnement virtuel nommé "env"
python -m venv env

# Activer l'environnement virtuel
# Sur Windows :
.\env\Scripts\activate
# Sur macOS/Linux :
source env/bin/activate
Étape C : Installer les DépendancesUtilisez le fichier requirements.txt pour installer toutes les librairies Python nécessaires en une seule commande.pip install -r requirements.txt
3. ConfigurationÉtape A : Configurer les Variables d'EnvironnementLe chatbot utilise une clé API pour la recherche web.À la racine du projet, créez un fichier nommé .env.Ouvrez ce fichier et ajoutez la ligne suivante (si vous avez une clé API Tavily) :TAVILY_API_KEY="votre_cle_api_ici"
Étape B : Préparer la Base de DonnéesCes commandes vont créer la base de données locale (SQLite) et préparer les tables nécessaires.python manage.py makemigrations
python manage.py migrate
Étape C : Créer un Super-utilisateurPour accéder à l'interface d'administration de Django, créez un compte administrateur.python manage.py createsuperuser
Suivez les instructions pour choisir un nom d'utilisateur, une adresse e-mail et un mot de passe.4. Lancer l'ApplicationPour que l'application soit pleinement fonctionnelle, vous devez lancer deux services : le serveur Ollama pour l'IA et le serveur Django pour le site web.Étape A : Lancer le Serveur OllamaAssurez-vous que l'application Ollama est en cours d'exécution sur votre ordinateur.Ouvrez un nouveau terminal et téléchargez les modèles d'IA nécessaires (si ce n'est pas déjà fait) :ollama pull phi3
```bash
ollama pull mistral
Étape B : Lancer le Serveur DjangoRevenez au premier terminal (celui où votre environnement virtuel est activé).Lancez le serveur de développement Django :python manage.py runserver
Ouvrez votre navigateur web et allez à l'adresse suivante : http://127.0.0.1:8000/Votre application est maintenant lancée et prête à être utilisée !