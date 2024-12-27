from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image, ImageOps
import tempfile
import base64
import matplotlib.pyplot as plt

# Initialisation de l'application FastAPI
app = FastAPI()

# =========================================
# Fonctions personnalisées pour le modèle
# =========================================
@register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    intersection = np.sum(y_true * y_pred, axis=[1, 2, 3])
    union = np.sum(y_true, axis=[1, 2, 3]) + np.sum(y_pred, axis=[1, 2, 3])
    return (2. * intersection + smooth) / (union + smooth)

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

@register_keras_serializable()
def iou_metric(y_true, y_pred):
    y_true = np.round(y_true).astype(bool)
    y_pred = np.round(y_pred).astype(bool)
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    return (intersection + 1e-10) / (union + 1e-10)

# Charger le modèle
MODEL_PATH = "./models/U-Net Miniaug.keras"
try:
    model = load_model(MODEL_PATH, custom_objects={
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "iou_metric": iou_metric
    })
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

# =========================================
# Routes
# =========================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Page d'accueil avec un bouton pour passer à la page d'analyse.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>S.O.P.H.I.A - Accueil</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                text-align: center;
                background: #ffffff;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                font-size: 16px;
                color: white;
                background-color: #2980b9;
                text-decoration: none;
                border-radius: 5px;
                margin-top: 20px;
            }
            .btn:hover {
                background-color: #1c598a;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bienvenue sur S.O.P.H.I.A</h1>
            <h2>Segmentation Optimisée Pour l'Harmonie Intelligente des Automobiles</h2>
            <p>Aucune image n'est stockée sur l'application.</p>
            <a href="/analyze" class="btn">Passer à l'analyse</a>
        </div>
    </body>
    </html>
    """

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page():
    """
    Page permettant d'uploader une image pour analyse.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>S.O.P.H.I.A - Analyse</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                text-align: center;
                background: #ffffff;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            form {
                margin-top: 20px;
            }
            button {
                padding: 10px 20px;
                font-size: 16px;
                color: white;
                background-color: #3498db;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #2980b9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Analyse</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Analyser l'image</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    """
    Traite l'image uploadée, retourne l'image et le masque prédit.
    """
    try:
        img = Image.open(file.file).convert("RGB")
        img_resized = ImageOps.fit(img, (256, 256), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        predicted_mask = np.argmax(prediction, axis=-1)
        predicted_mask_colored = apply_palette(predicted_mask, PALETTE)

        temp_dir = tempfile.gettempdir()
        img_path = os.path.join(temp_dir, "uploaded_image.png")
        mask_path = os.path.join(temp_dir, "predicted_mask.png")

        img.save(img_path)
        plt.imsave(mask_path, predicted_mask_colored)

        img_base64 = encode_image_to_base64(img_path)
        mask_base64 = encode_image_to_base64(mask_path)

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>S.O.P.H.I.A - Résultats</title>
            <style>
                img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <h1>Résultats de l'analyse</h1>
            <h2>Image originale</h2>
            <img src="data:image/png;base64,{img_base64}" alt="Image originale">
            <h2>Masque prédit</h2>
            <img src="data:image/png;base64,{mask_base64}" alt="Masque prédit">
        </body>
        </html>
        """

    except Exception as e:
        return f"<h1>Erreur : {str(e)}</h1>"

# =========================================
# Utilitaires
# =========================================

def apply_palette(mask, palette):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_mask[mask == class_id] = color
    return color_mask

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

PALETTE = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
]

CLASS_LABELS = [
    "Flat",
    "Human",
    "Vehicle",
    "Construction",
    "Object",
    "Nature",
    "Sky",
    "Void",
]
