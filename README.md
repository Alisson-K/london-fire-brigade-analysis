# 🔥 Prédiction du Temps d'Intervention de la London Fire Brigade

Ce projet de data science analyse les données de 2024 de la brigade des pompiers de Londres pour construire un modèle de **régression** capable de prédire le **temps d'intervention** (`AttendanceTimeSeconds`) du premier véhicule sur le lieu d'un incident.

L'objectif est d'identifier les facteurs influençant ce temps de réponse et de fournir un outil prédictif qui pourrait, à terme, aider à l'optimisation des ressources.

Ce repository contient le script d'entraînement complet du modèle, le code de l'application web de démonstration, ainsi que la documentation du projet.

## 📊 Aperçu de l'Application

Voici un aperçu de l'application de prédiction interactive construite avec Streamlit. L'utilisateur peut renseigner les paramètres d'un incident pour obtenir une estimation en temps réel du temps de première intervention.

![Aperçu de l'application Streamlit de prédiction](assets/app_screenshot.jpg)

## 🛠️ Stack Technique

-   **Analyse & Modélisation :** Python, Pandas, NumPy, Scikit-learn
-   **Modèle de Machine Learning :** **LightGBM** (`LGBMRegressor`)
-   **Optimisation :** **RandomizedSearchCV** pour la recherche d'hyperparamètres
-   **Application Web :** Streamlit (`app.py`)
-   **Environnement :** Jupyter Notebook, VS Code

---

## 📂 Structure du Projet

```
├── app/              # Code source de l'application Streamlit de démonstration
├── assets/           # Images et logos pour le README
├── data/             # Données (les fichiers complets doivent être téléchargés séparément)
├── notebooks/        # Notebooks Jupyter pour l'analyse exploratoire et la modélisation
├── saved_models/     # Modèles entraînés (ignorés par Git)
├── .gitignore        # Fichiers et dossiers à ignorer par Git
├── README.md         # Ce fichier
└── requirements.txt  # Dépendances Python du projet
```

---

## 🚀 Installation et Utilisation

#### **Prérequis**

* Python 3.9+
* Un environnement virtuel (recommandé)

#### **Instructions**

1.  **Clonez le repository :**
    ```bash
    git clone [https://github.com/Alisson-K/london-fire-brigade-analysis.git](https://github.com/Alisson-K/london-fire-brigade-analysis.git)
    cd london-fire-brigade-analysis
    ```

2.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Source des Données :**
    Les données complètes (`LFB Incident data...` et `LFB Mobilisation data...`) ne sont pas incluses dans ce repository. Veuillez les télécharger depuis le **[London Datastore](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)** et les placer à la racine du projet.

4.  **Entraîner le modèle :**
    Le script fourni (`[analyse_2.ipynb]`) exécute le pipeline complet : chargement, nettoyage, feature engineering, optimisation et sauvegarde des artefacts du modèle.
    ```bash
    python analyse_2.ipynb
    ```

5.  **Lancer l'application de démonstration :**
    L'application Streamlit utilise les fichiers `.joblib` sauvegardés pour faire des prédictions en temps réel.
    ```bash
    streamlit run app/app.py
    ```

---

## 📈 Démarche du Projet

Le projet suit un pipeline de machine learning structuré :

1.  **Nettoyage et Fusion :** Les jeux de données sur les incidents et les mobilisations de 2024 sont chargés, filtrés et fusionnés. Les valeurs manquantes et les doublons sont traités.

2.  **Feature Engineering :**
    * Extraction de features temporelles à partir des dates et heures (`Month`, `DayOfWeek`, `Hour`).
    * Création de variables pertinentes comme `IsWeekend` et `TimeOfDay` (Nuit, Matin, etc.).

3.  **Encodage et Préparation :**
    * Les variables géographiques (`BoroughCode`, `WardCode`) sont encodées avec `LabelEncoder`.
    * Les variables catégorielles (`IncidentGroup`, `PropertyCategory`) sont transformées avec `OneHotEncoder` pour éviter toute relation d'ordre implicite.

4.  **Optimisation du Modèle :**
    * Un modèle **LightGBM**, réputé pour sa rapidité et ses performances, a été choisi.
    * Les meilleurs hyperparamètres ont été recherchés à l'aide de `RandomizedSearchCV` sur 50 itérations avec une validation croisée en 5 folds, en optimisant le score R².

5.  **Évaluation :** Le modèle final est évalué sur un jeu de test mis de côté (20% des données) pour mesurer sa performance réelle sur des données inconnues.
