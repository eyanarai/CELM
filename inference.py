import tensorflow as tf
import numpy as np
from PIL import Image

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    image = load_image(img_path)
    prediction = model.predict(image)[0][0]
    label = "Cardiomegaly" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"🧠 Prediction: {label} ({confidence * 100:.2f}% confidence)")
    return prediction

# Example usage:
# predict('exports/vgg16/VGG16.h5', 'test_images/sample.png')
