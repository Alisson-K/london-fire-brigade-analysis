import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
from PIL import Image # <--- IMPORT AJOUTÉ ICI

# --- Fonctions de Chargement Mises en Cache ---
@st.cache_resource
def load_model(path='model_poc.joblib'):
    try: return joblib.load(path)
    except FileNotFoundError: st.error(f"ERREUR : Fichier modèle '{path}' non trouvé."); st.stop()
    except Exception as e: st.error(f"Erreur chargement modèle: {e}"); st.stop()

@st.cache_resource
def load_scaler(path='scaler_poc.joblib'):
    try: return joblib.load(path)
    except FileNotFoundError: st.error(f"ERREUR : Fichier scaler '{path}' non trouvé."); st.stop()
    except Exception as e: st.error(f"Erreur chargement scaler: {e}"); st.stop()

@st.cache_resource
def load_label_encoders(path='label_encoders_poc.joblib'):
    try: return joblib.load(path)
    except FileNotFoundError: st.error(f"ERREUR : Fichier encodeurs '{path}' non trouvé."); st.stop()
    except Exception as e: st.error(f"Erreur chargement encodeurs: {e}"); st.stop()

@st.cache_data
def load_metadata(path='categories_poc.joblib'):
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

        # Vérifications (gardées du code précédent)
        if not metadata['ward_name_to_code'] or not metadata['borough_name_to_code']:
             st.error("ERREUR: Mappings Nom->Code manquants."); st.stop()
        if metadata['model_columns'] is None:
             st.error("ERREUR: Liste 'model_columns' manquante."); st.stop()
        if not metadata['wards_names'] or not metadata['borough_names']:
             st.warning("Listes Ward/Borough vides.")
        required_lists = ['incident_groups', 'property_categories', 'stations_codes', 'deployed_locations', 'stop_codes']
        for key in required_lists:
            if not metadata[key]: st.warning(f"Liste pour '{key}' vide.")

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

# --- CSS Personnalisé (Optionnel, gardé du code précédent) ---
custom_css = """
<style>
    /* ... (votre CSS personnalisé ici si vous en avez défini un) ... */
    /* Exemple simple pour que la section soit présente */
    h1 { color: #D70000; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- Chargement Effectif ---
model = load_model()
scaler = load_scaler()
label_encoders = load_label_encoders()
metadata = load_metadata()


# --- Titre Principal ---
st.title("Prédiction du Temps d'Intervention LFB")
st.caption(f"Modèle basé sur les données 2024 - {datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))).strftime('%d %B %Y, %H:%M:%S %Z')}")
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
    selected_station_code = st.selectbox("Station d'Origine", metadata['stations_codes'], help="Code de la station initialement assignée (si connu).")

with col3:
    st.markdown("##### 🔥 Incident & Mobilisation")
    selected_incident_group = st.selectbox("Type Incident", metadata['incident_groups'], help="Classification générale.")
    selected_property_cat = st.selectbox("Type Propriété", metadata['property_categories'], help="Type général du lieu.")
    selected_deployed_loc = st.selectbox("Lieu Départ Pompe", metadata['deployed_locations'], help="D'où la première pompe est-elle partie ?")

with st.expander("➕ Détails Supplémentaires (Optionnel)"):
     selected_stop_code = st.selectbox("Description Arrêt (Stop Code)", metadata['stop_codes'], index=None, placeholder="Sélectionnez si pertinent...")

st.divider()

# --- Bouton et Zone de Prédiction ---
col_btn, col_pred_zone = st.columns([1, 2])

with col_btn:
    predict_button = st.button("🔮 Prédire le Temps", type="primary", use_container_width=True)

with col_pred_zone:
    if predict_button:
        with st.spinner("🧠 Analyse en cours..."):
            try:
                # --- Récupérer les CODES ---
                selected_ward_code = metadata['ward_name_to_code'].get(selected_ward_name)
                selected_borough_code = metadata['borough_name_to_code'].get(selected_borough_name)
                if selected_ward_code is None or selected_borough_code is None: raise ValueError("Nom Ward/Borough sélectionné invalide.")
                final_stop_code = selected_stop_code if selected_stop_code is not None else metadata['stop_codes'][0]
                if selected_stop_code is None: st.caption("Stop Code non fourni, valeur par défaut utilisée.")

                # --- Préparer features ---
                year, month, day_of_week, hour = date_appel.year, date_appel.month, date_appel.weekday(), heure_appel_datetime.hour
                is_weekend = 1 if day_of_week >= 5 else 0
                time_bins, time_labels = [0, 6, 12, 18, 24], metadata['time_of_day_cats']
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
                cols_to_onehot = ['TimeOfDay', 'IncidentGroup', 'StopCodeDescription', 'PropertyCategory', 'DeployedFromLocation']
                input_df_encoded = pd.get_dummies(input_df, columns=cols_to_onehot, drop_first=False)

                # --- Alignement & Scaling & Prédiction ---
                final_input_df = pd.DataFrame(columns=metadata['model_columns']).fillna(0)
                final_input_df = input_df_encoded.combine_first(final_input_df)
                final_input_df = final_input_df.reindex(columns=metadata['model_columns'], fill_value=0)
                input_scaled = scaler.transform(final_input_df)
                prediction = model.predict(input_scaled)
                predicted_time = max(0, round(prediction[0]))

                # --- Affichage Résultat ---
                minutes, secondes = divmod(predicted_time, 60)
                rmse_model = 84.91 # A CHANGER si besoin
                st.markdown(f"""
                <div class="result-container">
                    <div class="stMetric">
                        <div class="st-emotion-cache-1 S{{metric_label}}">⏱️ Temps d'Intervention Estimé</div>
                        <div class="st-emotion-cache-1 S{{metric_value}}">{minutes} min {secondes} s</div>
                        <div class="st-emotion-cache-1 S{{metric_delta}}">({predicted_time} s)</div>
                    </div>
                    <p style="font-size: 0.9em; color: #666;"><i>Note: Erreur typique ±{rmse_model:.0f}s (RMSE). Estimation.</i></p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {e}")
    else:
        st.info("Cliquez sur 'Prédire le Temps' après avoir renseigné les paramètres.")


# --- Section Importance des Variables ---
st.divider()
with st.expander("🔍 Importance des Variables pour le Modèle"):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = metadata['model_columns']
            if len(importances) == len(feature_names):
                feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0]
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(20)
                chart_df = feature_importance_df.set_index('Feature')
                st.markdown("Top 20 variables (après encodage) importantes :")
                st.bar_chart(chart_df)
                st.caption("Note: Inclut features encodées (ex: 'TimeOfDay_Morning').")
            else: st.warning("Incohérence nombre importances/features.")
        else: st.info("Modèle sans 'feature_importances_'.")
    except Exception as e: st.error(f"Erreur affichage importance: {e}")


# --- Sidebar Content (Info + Logo) ---
st.sidebar.markdown("---")

# --- AJOUT DU LOGO ICI ---
try:
    logo = Image.open('logoLFB.jpg') # Charger VOTRE image
    st.sidebar.image(logo, use_container_width=True) # Afficher dans la sidebar
except FileNotFoundError:
    st.sidebar.warning("⚠️ Logo 'logoLFB.jpg' non trouvé.")
    st.sidebar.caption("Placer le fichier dans le dossier de app.py.")
except Exception as e:
    st.sidebar.error(f"Erreur chargement logo: {e}")
# --- FIN AJOUT LOGO ---

st.sidebar.markdown("---")
st.sidebar.info(
    "**Application PoC - WCS**\n\n"
    "**Modèle :** LightGBM\n"
    "**Données :** LFB 2024"
)
st.sidebar.caption(f"© {datetime.date.today().year}")
# --- Fin du Script ---