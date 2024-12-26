from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import uvicorn

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
def bce_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

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
        "bce_loss": bce_loss,
        "iou_metric": iou_metric
    })
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

# Configuration
INPUT_SIZE = (256, 256)

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

def apply_palette(mask, palette):
    """
    Applique une palette de couleurs à un masque d'indices de classes.
    """
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_mask[mask == class_id] = color
    return color_mask

# =========================================
# ROUTES
# =========================================

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
    <head>
        <title>API de Prédiction</title>
    </head>
    <body>
        <h1>Uploader une Image pour la Prédiction</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <button type="submit">Envoyer</button>
        </form>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    # Charger l'image
    try:
        img = Image.open(file.file).convert('RGB')
        img_resized = ImageOps.fit(img, (256, 256), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prédire le masque
        prediction = model.predict(img_array)[0]
        predicted_mask = np.argmax(prediction, axis=-1)

        # Appliquer la palette
        predicted_mask_colored = apply_palette(predicted_mask, PALETTE)

        # Sauvegarder temporairement l'image et le masque
        temp_dir = tempfile.gettempdir()
        original_image_path = os.path.join(temp_dir, "uploaded_image.png")
        predicted_mask_path = os.path.join(temp_dir, "predicted_mask.png")

        img.save(original_image_path)
        plt.imsave(predicted_mask_path, predicted_mask_colored)

        # Retourner l'interface avec l'image et le masque
        return f"""
        <html>
        <head>
            <title>Résultat de Prédiction</title>
        </head>
        <body>
            <h1>Résultat de Prédiction</h1>
            <h2>Image originale :</h2>
            <img src="data:image/png;base64,{read_image_as_base64(original_image_path)}" alt="Image originale">
            <h2>Masque prédit :</h2>
            <img src="data:image/png;base64,{read_image_as_base64(predicted_mask_path)}" alt="Masque prédit">
            <form action="/">
                <button type="submit">OK</button>
            </form>
        </body>
        </html>
        """
    except Exception as e:
        return f"<h1>Erreur : {str(e)}</h1>"

# Fonction utilitaire pour encoder une image en Base64
def read_image_as_base64(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

