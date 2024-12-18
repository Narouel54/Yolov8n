import os
import subprocess
import sys
import torch
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import yaml
import pandas as pd

# Essayer d'importer torch, sinon l'installer manuellement
try:
    import torch
except ModuleNotFoundError:
    print("Torch non trouvé, installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"])
    import torch
    print("Torch installé avec succès.")

# Charger le modèle YOLOv8 et le fichier data.yaml
def load_model_and_data():
    model_path = "best.pt"  # Modifie ce chemin selon l'emplacement de ton modèle
    data_path = "data.yaml"  # Chemin vers le fichier data.yaml

    # Vérification de l'existence du modèle et du fichier YAML
    if os.path.exists(model_path) and os.path.exists(data_path):
        # Charger les informations de classes à partir de data.yaml
        with open(data_path, 'r') as file:
            data = yaml.safe_load(file)
            class_names = data['names']  # Liste des classes
        model = YOLO(model_path)  # Charger le modèle YOLO
        return model, class_names
    else:
        st.error(f"Le modèle {model_path} ou le fichier {data_path} est introuvable.")
        return None, None

# Sauvegarder l'image téléchargée dans un dossier local
def save_uploaded_image(uploaded_file):
    # Créer un dossier 'uploads' s'il n'existe pas
    upload_folder = "uploads"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    # Obtenir le nom de l'image et la sauvegarder dans le dossier
    image_path = os.path.join(upload_folder, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return image_path

# Interface utilisateur Streamlit
st.title("Application de détection d'objets avec YOLOv8")

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image à analyser", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Sauvegarder l'image dans le dossier local
    image_path = save_uploaded_image(uploaded_file)
    st.success(f"Image sauvegardée sous : {image_path}")

    # Ouvrir et afficher l'image
    image = Image.open(image_path)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Charger le modèle et les classes
    model, class_names = load_model_and_data()
    if model and class_names:
        # Ajout d'un slider pour sélectionner la confiance minimum
        min_confidence = st.slider('Sélectionner la confiance minimale (%)', 0, 100, 50) / 100.0  # Valeur entre 0 et 1

        # Effectuer la détection d'objets
        results = model(image)  # Prédiction sur l'image téléchargée

        # Vérification si les résultats sont disponibles
        if results:
            # Extraire les résultats sous forme de DataFrame
            boxes = results[0].boxes.xywh  # Boîtes de détection au format xywh
            confidences = results[0].boxes.conf  # Confiance (score) des détections
            class_ids = results[0].boxes.cls  # IDs des classes

            # Filtrer les résultats en fonction du seuil de confiance
            valid_results = []
            for i, confidence in enumerate(confidences):
                if confidence >= min_confidence:
                    valid_results.append({
                        'class_id': class_ids[i],
                        'class_name': class_names[int(class_ids[i])],
                        'confidence': confidence,
                        'xmin': boxes[i, 0],
                        'ymin': boxes[i, 1],
                        'xmax': boxes[i, 2],
                        'ymax': boxes[i, 3]
                    })

            # Organiser les résultats dans un DataFrame
            predictions_df = pd.DataFrame(valid_results)

            # Afficher les résultats
            st.write("### Prédictions du modèle :")
            st.write(predictions_df)

            # Afficher les catégories d'objets détectées avec leurs classes et leur confiance
            st.write("### Objets détectés :")
            for _, row in predictions_df.iterrows():
                class_name = row['class_name']
                confidence = row['confidence']
                st.write(f"Objet: {class_name} (Confiance: {confidence:.2f})")

            # Afficher l'image avec les boîtes de détection
            annotated_image = results[0].plot()  # Utilisation de la méthode .plot() pour l'image annotée

            # Créer deux colonnes pour afficher les images côte à côte
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Image téléchargée", use_container_width=True)

            with col2:
                st.image(annotated_image, caption="Image avec détection d'objets", use_container_width=True)
        else:
            st.write("Aucun objet détecté.")
    else:
        st.write("Erreur lors du chargement du modèle ou des classes.")


