import numpy as np
import json
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image

# Load model once (important for performance)
model = load_model("model/plant_model.h5")

# Load labels
with open("model/labels.json") as f:
    labels = json.load(f)

def predict_disease(image):
    """
    Predict plant disease from image
    """
    processed = preprocess_image(image)

    predictions = model.predict(processed)[0]

    index = np.argmax(predictions)
    confidence = float(predictions[index])

    label = labels[index]

    return label, confidence