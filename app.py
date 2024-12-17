import streamlit as st
import torch
from PIL import Image
import yaml
import os

# Titre de l'application
st.title("Application de Scanning avec best.pt")

# Charger le modèle
@st.cache_resource  # Pour éviter de recharger à chaque interaction
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

# Charger les noms de classes depuis le fichier YAML
def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('names', [])

# Chemins des fichiers
MODEL_PATH = "best.pt"
YAML_PATH = "dama.yaml"

# Charger le modèle et les catégories
model = load_model(MODEL_PATH)
categories = load_yaml(YAML_PATH)

# Section pour uploader une image
uploaded_file = st.file_uploader("Chargez une image à analyser", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Analyser l'image avec le modèle
    results = model(image)

    # Afficher les résultats
    st.subheader("Résultats de l'analyse :")
    results.print()  # Affiche les résultats dans le terminal (utile pour debug)

    # Visualisation des résultats sur l'image
    st.image(results.render()[0], caption="Résultats détectés", use_column_width=True)

    # Liste des objets détectés
    detections = results.pandas().xyxy[0]  # Obtenir les résultats au format pandas
    if not detections.empty:
        for index, row in detections.iterrows():
            st.write(f"Objet détecté : {categories[int(row['class'])]} (Confiance : {row['confidence']:.2f})")
    else:
        st.write("Aucun objet détecté.")

