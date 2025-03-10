import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Load the trained model
MODEL_PATH = "ASL_Detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load class labels from the training data
DATASET_PATH = "asl_datasets/train"
class_labels = sorted(os.listdir(DATASET_PATH))

# Define test image directory
TEST_DIR = "asl_datasets/test/all_letters"

def predict_image(image_path):
    """ Predict the ASL sign from an input image """
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    # Display the image and prediction
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

# Loop through all images in the test folder
for img_file in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img_file)
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        print(f"Predicting for: {img_file}")
        predict_image(img_path)
