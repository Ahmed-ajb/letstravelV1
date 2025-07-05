Voici le fichier prêt à être copié dans un fichier `README.md` :

---

````markdown
# 🌍 MarocPlanner – Application de Planification de Voyage

Bienvenue sur **MarocPlanner** !  
Cette application web, développée avec **Django**, permet de générer des itinéraires de voyage personnalisés au **Maroc**, et intègre un **chatbot intelligent** pour assister les utilisateurs.

---

## 🚀 Fonctionnalités principales

- Génération d'itinéraires personnalisés au Maroc
- Intégration d'un chatbot IA via **Ollama**
- Interface d’administration Django
- Configuration simple et rapide

---

## 🛠️ Prérequis

Avant de commencer, assurez-vous d'avoir installé les logiciels suivants :

- 🐍 [Python 3.10+](https://www.python.org/downloads/)
- 🧰 [Git](https://git-scm.com/downloads)
- 🤖 [Ollama](https://ollama.com/) (pour le chatbot)

---

## ⚙️ Installation

### 📥 Étape A : Cloner le projet

```bash
git clone https://github.com/Ahmed-ajb/letstravelV1.git
cd letstravelV1
````

### 🧪 Étape B : Créer un environnement virtuel

```bash
# Créer un environnement virtuel nommé "env"
python -m venv env

# Activer l'environnement virtuel
# Sur Windows :
.\env\Scripts\activate

# Sur macOS/Linux :
source env/bin/activate
```

### 📦 Étape C : Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

### 🔑 Étape A : Variables d’environnement

Créez un fichier `.env` à la racine du projet et ajoutez votre clé API :

```env
TAVILY_API_KEY="votre_cle_api_ici"
```

### 🗃️ Étape B : Préparer la base de données

```bash
python manage.py makemigrations
python manage.py migrate
```

### 👤 Étape C : Créer un super-utilisateur

```bash
python manage.py createsuperuser
```

Suivez les instructions pour choisir un nom d'utilisateur, une adresse e-mail et un mot de passe.

---

## 💡 Lancer l'application

### 🤖 Étape A : Démarrer Ollama

Assurez-vous que **Ollama** est installé et fonctionne, puis téléchargez les modèles nécessaires :

```bash
ollama pull phi3
ollama pull mistral
```

### 🌐 Étape B : Démarrer le serveur Django

```bash
python manage.py runserver
```

Ensuite, ouvrez votre navigateur et allez à l'adresse :

👉 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## ✅ Félicitations !

Votre application **MarocPlanner** est maintenant **prête à l’emploi** ! Bon voyage ! ✈️🇲🇦

---

