import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame
import time

# Load the trained model
model = load_model('ASL_Detection_model.h5')

# Class labels for ASL signs
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['space', 'delete', 'nothing']

# Initialize Pygame for display
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("ASL Real-Time Sign Language Detection")
font = pygame.font.Font(None, 60)
sentence = ""

# Open the webcam
cap = cv2.VideoCapture(0)

# Prediction delay to avoid rapid multiple detections
prediction_delay = 2  # 2 seconds
last_prediction_time = time.time()

# Initialize predicted_label to avoid NameError
predicted_label = ""
confidence_score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Mirror the image
    
    # Adjusted Region of Interest (ROI) for hand detection
    roi_x_start = 350
    roi_y_start = 100
    roi_width = 300
    roi_height = 300
    roi_x_end = roi_x_start + roi_width
    roi_y_end = roi_y_start + roi_height
    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # Improved Image Preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blur)
    rgb_img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    
    # Resize and normalize image for model input
    img = cv2.resize(rgb_img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make a prediction with a delay to avoid rapid multiple predictions
    current_time = time.time()
    if current_time - last_prediction_time > prediction_delay:
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        confidence_score = np.max(predictions) * 100  # Convert to percentage
        predicted_label = class_labels[predicted_class]
        
        # Handle the predicted sign
        if predicted_label == 'space':
            sentence += ' '
        elif predicted_label == 'delete':
            sentence = sentence[:-1]  # Remove last character
        elif predicted_label != 'nothing':
            sentence += predicted_label
        
        # Limit sentence length
        if len(sentence) > 30:
            sentence = ""
        
        last_prediction_time = current_time
    
    # Display the ROI and Enhanced Input Box
    cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
    cv2.putText(frame, "Input:", (roi_x_start, roi_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, predicted_label, (roi_x_start, roi_y_end + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence_score:.2f}%", (roi_x_start, roi_y_end + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # Display the sentence on the camera feed
    cv2.putText(frame, "Sentence: " + sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("ASL Detection", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()
