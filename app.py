import subprocess
import sys
import os
import torch
import streamlit as st
from PIL import Image
from ultralytics import YOLO  # Assure-toi que la bibliothèque ultralytics est installée

# Essayer d'importer torch, sinon l'installer manuellement
try:
    import torch
except ModuleNotFoundError:
    print("Torch non trouvé, installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"])
    import torch
    print("Torch installé avec succès.")

# Charger le modèle YOLOv8
def load_model():
    model_path = "best.pt"  # Modifie ce chemin selon l'emplacement de ton modèle
    if os.path.exists(model_path):
        model = YOLO(model_path)
        return model
    else:
        st.error(f"Le modèle {model_path} est introuvable.")
        return None

# Interface utilisateur Streamlit
st.title("Application de détection d'objets avec YOLOv8")

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image à analyser", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Charger le modèle
    model = load_model()
    if model:
        # Effectuer la détection d'objets
        results = model(image)  # Prédiction sur l'image téléchargée

        # Afficher les résultats
        st.write("Prédictions du modèle :")
        st.write(results.pandas().xywh)  # Affiche les résultats sous forme de dataframe

        # Afficher l'image avec les boîtes de détection
        annotated_image = results.plot()
        st.image(annotated_image, caption="Image avec détection d'objets", use_column_width=True)
    else:
        st.write("Erreur lors du chargement du modèle.")


