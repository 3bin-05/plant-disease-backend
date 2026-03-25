import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Resize and normalize the image for model prediction
    """
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image