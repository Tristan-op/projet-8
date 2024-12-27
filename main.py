from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tempfile
import base64

# Désactiver CUDA si non nécessaire
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
# Palette et labels
# =========================================

PALETTE = [
    (0, 0, 0),       # Black for "Flat"
    (128, 0, 0),     # Red for "Human"
    (0, 128, 0),     # Green for "Vehicle"
    (128, 128, 0),   # Yellow for "Construction"
    (0, 0, 128),     # Blue for "Object"
    (128, 0, 128),   # Magenta for "Nature"
    (0, 128, 128),   # Cyan for "Sky"
    (128, 128, 128), # Gray for "Void"
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

# =========================================
# Routes
# =========================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Page d'accueil HTML intégrée.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>S.O.P.H.I.A - Accueil</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; text-align: center; padding: 20px; }
            h1 { color: #2c3e50; }
            a { font-size: 1.2rem; color: #ffffff; text-decoration: none; background-color: #3498db; padding: 10px 20px; border-radius: 5px; }
            a:hover { background-color: #2980b9; }
        </style>
    </head>
    <body>
        <h1>Bienvenue sur S.O.P.H.I.A</h1>
        <p>Bienvenue dans notre application de segmentation d'images.</p>
        <a href="/analyze">Analyser une image</a>
    </body>
    </html>
    """

@app.get("/analyze", response_class=HTMLResponse)
async def analyze():
    """
    Page HTML pour analyser une image.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>S.O.P.H.I.A - Analyse</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; text-align: center; padding: 20px; }
            h1 { color: #2c3e50; }
            form { margin-top: 20px; }
            input[type="file"] { margin: 20px 0; }
            button { font-size: 1.2rem; color: #ffffff; background-color: #3498db; padding: 10px 20px; border: none; border-radius: 5px; }
            button:hover { background-color: #2980b9; }
        </style>
    </head>
    <body>
        <h1>Analyse d'image</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Analyser</button>
        </form>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    """
    Traite l'image et retourne le résultat HTML avec une légende des classes.
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

        # Sauvegarder l'image originale et le masque
        temp_dir = tempfile.gettempdir()
        original_image_path = os.path.join(temp_dir, "original_image.png")
        predicted_mask_path = os.path.join(temp_dir, "predicted_mask.png")

        img.save(original_image_path)
        mask_with_colors = apply_palette(predicted_mask, PALETTE)
        Image.fromarray(mask_with_colors).save(predicted_mask_path)

        # Encodage des images en Base64
        original_image_base64 = encode_image_to_base64(original_image_path)
        predicted_mask_base64 = encode_image_to_base64(predicted_mask_path)

        # Générer la légende HTML
        legend_html = "".join(
            f"<div style='display:flex; align-items:center; margin-bottom:10px;'>"
            f"<div style='width:20px; height:20px; background-color:rgb({color[0]},{color[1]},{color[2]}); margin-right:10px;'></div>"
            f"<span>{label}</span></div>"
            for label, color in zip(CLASS_LABELS, PALETTE)
        )

        # Retourner le HTML avec les résultats
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>S.O.P.H.I.A - Résultats</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; text-align: center; padding: 20px; }}
                h1 {{ color: #2c3e50; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                a {{ font-size: 1rem; color: #ffffff; text-decoration: none; background-color: #3498db; padding: 10px 20px; border-radius: 5px; }}
                a:hover {{ background-color: #2980b9; }}
                .legend {{ margin-top: 20px; text-align: left; display: inline-block; }}
            </style>
        </head>
        <body>
            <h1>Résultats de l'analyse</h1>
            <h2>Image originale :</h2>
            <img src="data:image/png;base64,{original_image_base64}" alt="Image originale">
            <h2>Masque prédit :</h2>
            <img src="data:image/png;base64,{predicted_mask_base64}" alt="Masque prédit">
            <div class="legend">
                <h3>Légende :</h3>
                {legend_html}
            </div>
            <a href="/">Retour à l'accueil</a>
        </body>
        </html>
        """
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


