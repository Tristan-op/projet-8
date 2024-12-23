from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tempfile

# Initialisation de l'application Flask
app = Flask(__name__)

# =========================================
# Fonctions personnalisées pour le modèle
# =========================================
@register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return (2. * intersection + smooth) / (union + smooth)

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

@register_keras_serializable()
def bce_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss
    """
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

@register_keras_serializable()
def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(tf.round(y_pred), tf.bool)
    intersection = tf.reduce_sum(tf.cast(y_true & y_pred, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true | y_pred, tf.float32))
    return (intersection + 1e-10) / (union + 1e-10)

# Charger le modèle
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "models", "U-Net Miniaug.keras.keras")

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

# Répertoires principaux
IMAGE_BASE_FOLDER = os.path.join(BASE_DIR, "api_image")
MASK_BASE_FOLDER = os.path.join(BASE_DIR, "api_mask")
ALÉATOIRE_FOLDER = os.path.join(BASE_DIR, "aléatoire")
INPUT_SIZE = (256, 256)

# Palette de couleurs pour les classes
PALETTE = [
    (0, 0, 0),        # Flat : Noir
    (128, 0, 0),      # Human : Rouge foncé
    (0, 128, 0),      # Vehicle : Vert foncé
    (128, 128, 0),    # Construction : Jaune foncé
    (0, 0, 128),      # Object : Bleu foncé
    (128, 0, 128),    # Nature : Violet foncé
    (0, 128, 128),    # Sky : Cyan foncé
    (128, 128, 128)   # Void : Gris
]

CLASS_LABELS = [
    "Flat",
    "Human",
    "Vehicle",
    "Construction",
    "Object",
    "Nature",
    "Sky",
    "Void"
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
# ROUTES PRINCIPALES
# =========================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/accueil')
def accueil():
    return render_template('accueil.html')

@app.route('/list_cities')
def list_cities():
    if not os.path.exists(IMAGE_BASE_FOLDER):
        return "Le dossier api_image est introuvable.", 500
    cities = [d for d in os.listdir(IMAGE_BASE_FOLDER) if os.path.isdir(os.path.join(IMAGE_BASE_FOLDER, d))]
    return render_template('list_cities.html', cities=cities)

@app.route('/city/<city_name>')
def city_images(city_name):
    city_path = os.path.join(IMAGE_BASE_FOLDER, city_name)
    if not os.path.exists(city_path):
        return f"Ville '{city_name}' introuvable.", 404
    images = [f for f in os.listdir(city_path) if f.endswith('_leftImg8bit.png')]
    return render_template('city.html', city=city_name, images=images)

@app.route('/random_image')
def random_image():
    all_images = [f for f in os.listdir(ALÉATOIRE_FOLDER) if f.endswith('.png')]

    if not all_images:
        return "Aucune image disponible dans le répertoire 'aléatoire'.", 404

    image_name = random.choice(all_images)
    return redirect(url_for('random_image_details', image_name=image_name))

@app.route('/random_image_details/<image_name>')
def random_image_details(image_name):
    image_path = os.path.join(ALÉATOIRE_FOLDER, image_name)
    if not os.path.exists(image_path):
        return f"Image '{image_name}' introuvable dans le répertoire 'aléatoire'.", 404

    try:
        # Prétraitement de l'image
        img = Image.open(image_path).convert('RGB')
        img_resized = ImageOps.fit(img, INPUT_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prédiction du modèle
        prediction = model.predict(img_array)[0]
        predicted_mask = np.argmax(prediction, axis=-1)

        # Appliquer la palette de couleurs au masque prédit
        predicted_mask_colored = apply_palette(predicted_mask, PALETTE)

        # Sauvegarde de l'image temporaire
        temp_dir = tempfile.gettempdir()
        prediction_filename = f"{image_name.replace('.png', '')}_random_visualization.png"
        prediction_path = os.path.join(temp_dir, prediction_filename)
        plt.imsave(prediction_path, predicted_mask_colored)

        # Redirection avec le chemin temporaire
        return redirect(url_for('serve_tmp_image', filename=prediction_filename))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tmp/<filename>')
def serve_tmp_image(filename):
    """
    Sert les fichiers temporairement sauvegardés.
    """
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    if os.path.exists(file_path):
        return send_from_directory(temp_dir, filename)
    else:
        return "Fichier temporaire introuvable.", 404

# =========================================
# Lancer l'application Flask
# =========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
