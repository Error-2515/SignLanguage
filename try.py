import streamlit as st
import cv2
import numpy as np
from function import *  # Import your custom functions
from keras.models import model_from_json

# Load the model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

colors = [(245, 117, 16) for _ in range(20)]  # Assuming 20 colors for classes

# ... (define your other functions and constants here)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set mediapipe model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Make detections
        cropframe = frame[40:400, 0:300]
        cropframe = cv2.flip(cropframe, 1)

        # Perform your logic here...

        # Display the output
        st.image(frame, channels="BGR")

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close the Streamlit app
cap.release()
cv2.destroyAllWindows()
