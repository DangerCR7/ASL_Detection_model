import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Load the trained ASL detection model
model = load_model('ASL_Detection_model.h5')

# Class labels for ASL signs
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['space', 'delete', 'nothing']

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Sentence building
sentence = ""
prediction_delay = 2  # 2 seconds delay to avoid multiple rapid predictions
last_prediction_time = time.time()

# Initialize predicted_label to avoid NameError
predicted_label = ""
confidence_score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Mirror the image for natural interaction
    
    # Define Region of Interest (ROI) for hand detection
    roi_x_start = 350
    roi_y_start = 100
    roi_width = 300
    roi_height = 300
    roi_x_end = roi_x_start + roi_width
    roi_y_end = roi_y_start + roi_height
    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # Image Preprocessing (same as training)
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make a prediction with a delay to avoid multiple rapid predictions
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
    
    # Display the ROI and Input Box
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
