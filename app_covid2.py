import streamlit as st
import pandas as pd
import joblib
import requests
import io

# Cargar el modelo desde GitHub
@st.cache_resource
def load_model():
    url = "https://github.com/JoseSC2025/predictor_covid_mortalidad_2/blob/main/best_model.pkl"  # Reemplaza con tu URL real
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

model = load_model()

st.title("🦠 Predicción de Mortalidad por COVID-19")

st.markdown("Ingrese los datos del paciente para predecir el **riesgo de mortalidad**.")

# Lista de variables
columnas = [
    'edad', 'sexo', 'fiebre_Si', 'malestar_gen_Si', 'tos_Si', 'dolor_garganta_Si',
    'congestion_nasal_Si', 'dificultad_resp_Si', 'diarrea_Si', 'vomitos_Si',
    'cefalea_Si', 'confusion_Si', 'dolor_musc_Si', 'dolor_abdom_Si', 'dolor_torax_Si',
    'dolor_art_Si', 'disosmia_disgeusia_Si', 'dolor_oido_Si', 'embarazo_Si', 'aborto_Si',
    'enf_cardiovascular_Si', 'hta_Si', 'dislipidemia_Si', 'diabetes_Si', 'tiroidopatia_Si',
    'hepatopatia_Si', 'enf_neurologica_Si', 'inmunodeficiencia_Si', 'enf_renal_Si',
    'enf_pulmonar_Si', 'asma_Si', 'cancer_Si', 'obesidad_Si', 'tbc_Si'
]

# Diccionario para guardar inputs
datos = {}

# Entrada para edad y sexo
datos['edad'] = st.slider("Edad", 0, 120, 30)
datos['sexo'] = st.radio("Sexo", ["Hombre", "Mujer"])
datos['sexo'] = 1 if datos['sexo'] == "Hombre" else 0

# Entradas para síntomas y comorbilidades
for col in columnas[2:]:
    datos[col] = st.radio(f"{col.replace('_Si','').replace('_',' ').capitalize()}", ["No", "Sí"]) == "Sí"
    datos[col] = int(datos[col])

# Botón de predicción
if st.button("🔍 Predecir"):
    input_df = pd.DataFrame([datos])
    pred = model.predict(input_df)[0]
    if pred == 0:
        st.success("✅ Riesgo mortalidad ausente (muerte_Si = 0)")
    else:
        st.error("⚠️ Riesgo mortalidad presente (muerte_Si = 1)")
