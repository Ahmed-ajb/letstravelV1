# trip_planner_project/trip_planner_project/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv # <-- NOUVEAU : Importe load_dotenv

load_dotenv() # <-- NOUVEAU : Charge les variables d'environnement du fichier .env

BASE_DIR = Path(__file__).resolve().parent.parent

# Chemins pour les données et caches (utilisés par planner/utils.py)
# Utilisation de Path pour la robustesse et la clarté
DATA_DIR = BASE_DIR / 'planner' / 'data'
# Ces répertoires sont utilisés par utils.py et peuvent être différents de ceux que vous aviez.
# Assurez-vous que le chemin 'planner/data' correspond bien à l'emplacement de vos fichiers JSON.
# GRAPHS_CACHE_DIR_DJANGO est le plus pertinent pour les API et OSMnx.
GRAPHS_CACHE_DIR_DJANGO = BASE_DIR / "city_graphs_cache_django"


SECRET_KEY = 'django-insecure-!!REMPLACEZ-MOI-AVEC-UNE-VRAIE-CLE-SECRETE!!'
DEBUG = True
ALLOWED_HOSTS = ['127.0.0.1', 'localhost', '192.168.1.14','192.168.153.79']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'planner',
     # Third-party applications
    'crispy_forms',
    'crispy_bootstrap5',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'trip_planner_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'], # Pour templates/registration/
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'trip_planner_project.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]

LANGUAGE_CODE = 'fr-fr'
TIME_ZONE = 'UTC' # Laissez UTC pour la base de données, ajustez l'affichage en frontend
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
# STATICFILES_DIRS = [BASE_DIR / "static_global"] # Si vous avez un dossier static global au projet

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'


MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Configuration du Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module}:{lineno} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'root': { # Logger racine, capture tout par défaut
        'handlers': ['console'],
        'level': 'INFO', # Mettre à DEBUG pour plus de détails pendant le développement
    },
    'loggers': {
        'django': { # Logger pour les messages de Django lui-même
            'handlers': ['console'],
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO'),
            'propagate': False,
        },
        'planner': { # Logger spécifique pour votre application planner
            'handlers': ['console'],
            'level': 'DEBUG', # NOUVEAU : Niveau de log DEBUG pour votre app
            'propagate': True, # Laisse les messages remonter au root logger aussi
        },
    },
}

CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"

# --- CONFIGURATIONS POUR LES CLÉS API ET VÉRIFICATION DES RÉPERTOIRES ---

# Clés API : Il est crucial de les gérer via des variables d'environnement en production.
# En développement, vous pouvez les définir ici directement pour les tests, mais NE PAS faire cela en production.
# Exemple pour le développement (à remplacer par vos vraies clés) :
# TAVILY_API_KEY = "VOTRE_CLE_TAVILY_ICI"
# SERPAPI_API_KEY = "VOTRE_CLE_SERPAPI_ICI"

# Utilisation des variables d'environnement (méthode préférée) :
# Si les variables d'environnement ne sont pas définies, elles recevront None ou une chaîne vide.
# Il est ensuite de la responsabilité de votre code (e.g., dans utils.py) de vérifier leur présence.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Assurez-vous que les répertoires nécessaires existent au démarrage
# Ils sont définis au début du fichier et ici on s'assure de leur existence.
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Création du répertoire DATA_DIR: {DATA_DIR}")

if not GRAPHS_CACHE_DIR_DJANGO.exists():
    GRAPHS_CACHE_DIR_DJANGO.mkdir(parents=True, exist_ok=True)
    print(f"Création du répertoire GRAPHS_CACHE_DIR_DJANGO: {GRAPHS_CACHE_DIR_DJANGO}")

if hasattr(os, 'W_OK') and not os.access(GRAPHS_CACHE_DIR_DJANGO, os.W_OK):
    print(f"AVERTISSEMENT: Le répertoire de cache des graphes {GRAPHS_CACHE_DIR_DJANGO} n'est pas accessible en écriture pour le processus Django.")
