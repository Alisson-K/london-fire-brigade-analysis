import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
from PIL import Image
from pathlib import Path 

BASE_DIR = Path(__file__).parent.parent # Remonte au dossier racine 'london-fire-brigade-analysis'
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
ASSETS_DIR = BASE_DIR / "assets"

@st.cache_resource
def load_model(path=SAVED_MODELS_DIR / 'model_poc.joblib'):
    try: return joblib.load(path)
    except FileNotFoundError: st.error(f"ERREUR : Fichier modèle '{path}' non trouvé."); st.stop()
    except Exception as e: st.error(f"Erreur chargement modèle: {e}"); st.stop()

@st.cache_resource
def load_scaler(path=SAVED_MODELS_DIR / 'scaler_poc.joblib'):
    try: return joblib.load(path)
    except FileNotFoundError: st.error(f"ERREUR : Fichier scaler '{path}' non trouvé."); st.stop()
    except Exception as e: st.error(f"Erreur chargement scaler: {e}"); st.stop()

@st.cache_resource
def load_label_encoders(path=SAVED_MODELS_DIR / 'label_encoders_poc.joblib'):
    try: return joblib.load(path)
    except FileNotFoundError: st.error(f"ERREUR : Fichier encodeurs '{path}' non trouvé."); st.stop()
    except Exception as e: st.error(f"Erreur chargement encodeurs: {e}"); st.stop()

@st.cache_data
def load_metadata(path=SAVED_MODELS_DIR / 'categories_poc.joblib'):
    try:
        saved_data = joblib.load(path)
        metadata = {
            'ward_name_to_code': saved_data.get('ward_name_to_code', {}),
            'borough_name_to_code': saved_data.get('borough_name_to_code', {}),
            'incident_groups': saved_data.get('IncidentGroup', []),
            'property_categories': saved_data.get('PropertyCategory', []),
            'stations_codes': saved_data.get('IncidentStationGround', []),
            'deployed_locations': saved_data.get('DeployedFromLocation', []),
            'stop_codes': saved_data.get('StopCodeDescription', []),
            'time_of_day_cats': saved_data.get('TimeOfDay', ['Night', 'Morning', 'Afternoon', 'Evening']),
            'model_columns': saved_data.get('model_columns', None)
        }
        metadata['wards_names'] = sorted(list(metadata['ward_name_to_code'].keys()))
        metadata['borough_names'] = sorted(list(metadata['borough_name_to_code'].keys()))

        if metadata['model_columns'] is None:
             st.error("ERREUR: Liste 'model_columns' manquante."); st.stop()
        
        return metadata
    except FileNotFoundError:
        st.error(f"ERREUR: Fichier métadonnées '{path}' manquant."); st.info("Exécutez le script d'entraînement."); st.stop()
    except Exception as e:
        st.error(f"Erreur chargement métadonnées : {e}"); st.stop()

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Prédiction Temps d'Intervention LFB",
    page_icon="🚒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chargement Effectif des Artefacts ---
model = load_model()
scaler = load_scaler()
label_encoders = load_label_encoders()
metadata = load_metadata()

# --- Titre Principal ---
st.title("Prédiction du Temps d'Intervention de la London Fire Brigade")
st.caption(f"Modèle prédictif basé sur les données de 2024")
st.divider()

# --- Interface Utilisateur ---
st.header("🚒 Paramètres de l'Incident")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🕒 Temps")
    date_appel = st.date_input("Date", datetime.date.today(), help="Date de l'appel initial.")
    heure_appel_saisie = st.time_input("Heure", datetime.time(12, 0), help="Heure de l'appel initial (HH:MM).")
    heure_appel_datetime = datetime.datetime.combine(datetime.date.today(), heure_appel_saisie)

with col2:
    st.markdown("##### 📍 Lieu")
    selected_borough_name = st.selectbox("Arrondissement", metadata['borough_names'], help="Sélectionnez l'arrondissement (Borough).")
    selected_ward_name = st.selectbox("Quartier", metadata['wards_names'], help="Sélectionnez le quartier spécifique (Ward).")
    selected_station_code = st.selectbox("Station d'Origine", metadata['stations_codes'], help="Code de la station initialement assignée.")

with col3:
    st.markdown("##### 🔥 Incident & Mobilisation")
    selected_incident_group = st.selectbox("Type d'Incident", metadata['incident_groups'], help="Classification générale.")
    selected_property_cat = st.selectbox("Type de Propriété", metadata['property_categories'], help="Type général du lieu.")
    selected_deployed_loc = st.selectbox("Lieu de Départ", metadata['deployed_locations'], help="D'où la première pompe est-elle partie ?")

with st.expander("➕ Détails Supplémentaires"):
     selected_stop_code = st.selectbox("Description de l'Arrêt (Stop Code)", metadata['stop_codes'], index=None, placeholder="Sélectionnez si pertinent...")

st.divider()

# --- Bouton et Zone de Prédiction ---
if st.button("🔮 Prédire le Temps", type="primary", use_container_width=True):
    with st.spinner("🧠 Analyse en cours..."):
        try:
            # --- Préparation des données d'entrée ---
            selected_ward_code = metadata['ward_name_to_code'].get(selected_ward_name)
            selected_borough_code = metadata['borough_name_to_code'].get(selected_borough_name)
            if selected_ward_code is None or selected_borough_code is None: raise ValueError("Nom Ward/Borough invalide.")
            final_stop_code = selected_stop_code if selected_stop_code is not None else metadata['stop_codes'][0]
            
            # --- Feature Engineering ---
            year, month, day_of_week, hour = date_appel.year, date_appel.month, date_appel.weekday(), heure_appel_datetime.hour
            is_weekend = 1 if day_of_week >= 5 else 0
            time_bins, time_labels = [0, 6, 12, 18, 24], ['Night', 'Morning', 'Afternoon', 'Evening']
            time_of_day = pd.cut([hour], bins=time_bins, labels=time_labels, right=False, include_lowest=True)[0]

            input_data = {
                'Year': year, 'Month': month, 'DayOfWeek': day_of_week, 'Hour': hour,
                'IsWeekend': is_weekend, 'TimeOfDay': time_of_day,
                'IncidentGroup': selected_incident_group, 'PropertyCategory': selected_property_cat,
                'IncGeo_WardCode': selected_ward_code, 'IncGeo_BoroughCode': selected_borough_code,
                'IncidentStationGround': selected_station_code, 'DeployedFromLocation': selected_deployed_loc,
                'StopCodeDescription': final_stop_code, 'PumpOrder': 1, 'DelayCodeId': 1
            }
            input_df = pd.DataFrame([input_data])

            # --- Encodage ---
            for col, encoder in label_encoders.items():
                if col in input_df.columns: input_df[col] = encoder.transform(input_df[col].astype(str))
            
            # --- One-Hot Encoding et Alignement des colonnes ---
            input_df_encoded = pd.get_dummies(input_df).reindex(columns=metadata['model_columns'], fill_value=0)

            # --- Scaling & Prédiction ---
            input_scaled = scaler.transform(input_df_encoded)
            prediction = model.predict(input_scaled)
            predicted_time = max(0, round(prediction[0]))

            # --- Affichage Résultat ---
            minutes, secondes = divmod(predicted_time, 60)
            st.metric(
                label="⏱️ Temps d'Intervention Estimé",
                value=f"{minutes} min {secondes} s",
                delta=f"({predicted_time} secondes au total)",
                delta_color="off"
            )

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
else:
    st.info("Renseignez les paramètres de l'incident et cliquez sur 'Prédire le Temps'.")

# --- Section Importance des Variables ---
st.divider()
with st.expander("🔍 Importance des Variables pour le Modèle"):
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'Feature': metadata['model_columns'],
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False).head(20)
            
            st.markdown("Top 20 des variables les plus importantes pour le modèle :")
            st.bar_chart(feature_importance_df.set_index('Feature'))
    except Exception as e: st.error(f"Erreur lors de l'affichage de l'importance: {e}")

# --- Contenu de la Sidebar ---
st.sidebar.markdown("---")

try:
    logo_path = ASSETS_DIR / 'logoLFB.jpg'
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning(f"⚠️ Logo non trouvé au chemin: {logo_path}")
except Exception as e:
    st.sidebar.error(f"Erreur chargement logo: {e}")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Application de Démonstration**\n\n"
    "**Modèle :** LightGBM Regressor\n"
    "**Données :** London Fire Brigade 2024"
)
st.sidebar.caption(f"© {datetime.date.today().year} - Allisson Kyriakidis")