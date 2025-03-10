from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('ASL_Detection_model.h5')

# Class labels for ASL signs
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['space', 'delete', 'nothing']

# Prediction Function
def predict_asl(image):
    # Preprocess image
    image = image.resize((64, 64))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)

    # Model Prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence_score = np.max(predictions) * 100  # Convert to percentage
    predicted_label = class_labels[predicted_class]
    
    return predicted_label, confidence_score

# Define the API route for prediction
@app.route('/', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read image file
        image = Image.open(io.BytesIO(file.read()))
        predicted_label, confidence_score = predict_asl(image)

        # Return the prediction and confidence score
        return jsonify({
            "predicted_label": predicted_label,
            "confidence_score": confidence_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
