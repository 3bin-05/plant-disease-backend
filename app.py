from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os

from config import Config
from utils.predict import predict_disease
from utils.ai_helper import get_disease_info

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def home():
    """
    Health check route
    """
    return {"message": "🌿 Plant Disease Detection API Running"}


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint
    """

    # Check if file is present
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save file temporarily
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        # Open image
        image = Image.open(filepath).convert("RGB")

        # Predict disease
        label, confidence = predict_disease(image)

        # =========================
        # 🧠 CONFIDENCE CHECK (NEW 🔥)
        # =========================
        if confidence < 0.7:
            return jsonify({
                "disease": "Unknown",
                "confidence": f"{round(confidence * 100, 2)}%",
                "description": "The model is not confident enough to determine the disease.",
                "causes": [],
                "remedies": []
            })

        # =========================
        # 🧠 CLEAN LABEL PROPERLY
        # =========================
        parts = label.split("___")

        if len(parts) == 2:
            plant, disease = parts
            disease_name = f"{plant} {disease}".replace("_", " ").strip()
        else:
            disease_name = label.replace("_", " ").strip()

        # =========================
        # 🚫 INVALID LABEL FILTER
        # =========================
        if "plantvillage" in label.lower():
            return jsonify({
                "disease": "Unknown",
                "confidence": f"{round(confidence * 100, 2)}%",
                "description": "no description available.",
                "causes": [],
                "remedies": []
            })

        # =========================
        # 🌿 HEALTHY CHECK
        # =========================
        if "healthy" in label.lower():
            response = {
                "disease": disease_name,
                "confidence": f"{round(confidence * 100, 2)}%",
                "description": "The plant appears healthy with no visible disease symptoms.",
                "causes": [],
                "remedies": []
            }

        else:
            # 🤖 Call LLM ONLY if diseased
            disease_info = get_disease_info(disease_name)

            response = {
                "disease": disease_name,
                "confidence": f"{round(confidence * 100, 2)}%",
                "description": disease_info.get("description", "No description available."),
                "causes": disease_info.get("causes", []),
                "remedies": disease_info.get("remedies", [])
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # 🧹 Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True)