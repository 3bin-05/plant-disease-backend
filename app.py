from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import GlorotUniform

from utils.preprocess import preprocess_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Load model (FIXED)
model = load_model(
    "model/model.h5",
    custom_objects={"GlorotUniform": GlorotUniform},
    compile=False
)

# ✅ Load disease info
with open("data/disease_info.json") as f:
    disease_info = json.load(f)

# ⚠️ IMPORTANT: Update these labels based on your model
class_labels = [
    "Healthy Leaf",
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus"
]

@app.route("/")
def home():
    return "Plant Disease Detection API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Preprocess
        img = preprocess_image(filepath)

        # Predict
        prediction = model.predict(img)
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # 🔥 DEBUG (VERY IMPORTANT)
        print("Prediction shape:", prediction.shape)
        print("Predicted index:", predicted_index)

        # ✅ SAFE LABEL HANDLING
        if predicted_index < len(class_labels):
            label = class_labels[predicted_index]
        else:
            label = f"Class {predicted_index}"

        # Get disease info
        info = disease_info.get(label, {
            "description": "No info available",
            "solution": "Consult expert"
        })

        return jsonify({
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "description": info["description"],
            "solution": info["solution"]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)