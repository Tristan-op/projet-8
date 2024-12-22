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
MODEL_PATH = "./models/VGG16_aug.keras"

try:
    model = load_model(MODEL_PATH, custom_objects={
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "bce_loss": bce_loss,  # Ajout explicite de bce_loss
        "iou_metric": iou_metric
    })
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# Répertoires principaux
IMAGE_BASE_FOLDER = "./api_image/"
MASK_BASE_FOLDER = "./api_mask/"
ALÉATOIRE_FOLDER = "./aléatoire/"
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
    all_images = []
    for image in os.listdir(ALÉATOIRE_FOLDER):
        if image.endswith('.png'):
            all_images.append(image)

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

        # Visualisation
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_resized)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask_colored)
        plt.title("Masque prédit")
        plt.axis("off")

        # Ajouter une légende
        handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(color)/255) for color in PALETTE]
        plt.legend(handles, CLASS_LABELS, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # Sauvegarde de l'image temporaire
        temp_dir = tempfile.gettempdir()
        prediction_filename = f"{image_name.replace('.png', '')}_random_visualization.png"
        prediction_path = os.path.join(temp_dir, prediction_filename)
        plt.savefig(prediction_path, bbox_inches='tight')
        plt.close()

        # Redirection avec le chemin temporaire
        return redirect(url_for('serve_tmp_image', filename=prediction_filename))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/image/<city_name>/<image_name>')
def image_details(city_name, image_name):
    image_path = os.path.join(IMAGE_BASE_FOLDER, city_name, image_name)
    mask_path = os.path.join(MASK_BASE_FOLDER, city_name, image_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png'))

    if not os.path.exists(image_path):
        return f"Image '{image_name}' introuvable dans la ville '{city_name}'.", 404
    if not os.path.exists(mask_path):
        return f"Le masque réel pour l'image '{image_name}' est introuvable.", 404

    try:
        # Prétraitement de l'image
        img = Image.open(image_path).convert('RGB')
        img_resized = ImageOps.fit(img, INPUT_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prédiction du modèle
        prediction = model.predict(img_array)[0]
        predicted_mask = np.argmax(prediction, axis=-1)

        # Charger le masque réel
        true_mask = Image.open(mask_path).resize(INPUT_SIZE, Image.NEAREST)
        true_mask = np.array(true_mask)

        # Appliquer la palette de couleurs au masque prédit
        predicted_mask_colored = apply_palette(predicted_mask, PALETTE)

        # Visualisation
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img_resized)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap="gray")  # Masque réel en niveaux de gris
        plt.title("Masque réel")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask_colored)
        plt.title("Masque prédit")
        plt.axis("off")

        # Ajouter une légende
        handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(color)/255) for color in PALETTE]
        plt.legend(handles, CLASS_LABELS, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # Sauvegarde de l'image temporaire
        temp_dir = tempfile.gettempdir()
        visualization_filename = f"{image_name.replace('_leftImg8bit.png', '')}_visualization.png"
        visualization_path = os.path.join(temp_dir, visualization_filename)
        plt.savefig(visualization_path, bbox_inches='tight')
        plt.close()

        # Redirection avec le chemin temporaire
        return redirect(url_for('serve_tmp_image', filename=visualization_filename))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================
# ROUTE : Servir les fichiers temporaires
# =========================================
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
