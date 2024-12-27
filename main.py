from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tempfile
import base64

# Initialisation de l'application FastAPI
app = FastAPI()

# Dossier des templates
templates = Jinja2Templates(directory="templates")

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
async def home(request: Request):
    """
    Page d'accueil avec un bouton pour passer à la page d'analyse.
    """
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    """
    Page permettant d'uploader une image pour analyse.
    """
    return templates.TemplateResponse("analyze.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Traite l'image uploadée, retourne l'image et le masque prédit.
    """
    try:
        # Charger l'image
        img = Image.open(file.file).convert("RGB")
        img_resized = ImageOps.fit(img, (256, 256), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prédiction
        prediction = model.predict(img_array)[0]
        predicted_mask = np.argmax(prediction, axis=-1)

        # Palette appliquée
        predicted_mask_colored = apply_palette(predicted_mask, PALETTE)

        # Sauvegarde temporaire
        temp_dir = tempfile.gettempdir()
        original_image_path = os.path.join(temp_dir, "uploaded_image.png")
        predicted_mask_path = os.path.join(temp_dir, "predicted_mask.png")

        img.save(original_image_path)
        plt.imsave(predicted_mask_path, predicted_mask_colored)

        # Encodage Base64 pour affichage
        original_image_base64 = encode_image_to_base64(original_image_path)
        predicted_mask_base64 = encode_image_to_base64(predicted_mask_path)

        # Retourne le template avec les images et la légende
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "original_image": original_image_base64,
                "predicted_mask": predicted_mask_base64,
                "legend": [{"label": label, "color": f"rgb({color[0]},{color[1]},{color[2]})"} for label, color in zip(CLASS_LABELS, PALETTE)],
            },
        )

    except Exception as e:
        return f"<h1>Erreur : {str(e)}</h1>"

# =========================================
# Utilitaires
# =========================================

def apply_palette(mask, palette):
    """
    Applique une palette de couleurs à un masque d'indices de classes.
    """
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_mask[mask == class_id] = color
    return color_mask

def encode_image_to_base64(image_path):
    """
    Encode une image en Base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Palette et labels
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
