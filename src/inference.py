import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('model.h5')

# Function to preprocess the captured image
def preprocess_image(frame):
    # Resize the image to the size expected by the model
    img = cv2.resize(frame, (64, 64))  # Adjust to the correct size expected by the model

    # Convert the image from BGR (OpenCV format) to RGB (format expected by Keras)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Add an extra dimension to simulate the batch
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image
    img_array = img_array / 255.0

    return img_array

# Initialize video capture (default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error opening the camera.")
    exit()

while True:
    # Capture an image from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error capturing the image.")
        break

    # Preprocess the image
    processed_image = preprocess_image(frame)

    # Make a prediction with the model
    predictions = model.predict(processed_image)

    # Find the predicted class (adjust according to your model's output)
    predicted_class = np.argmax(predictions, axis=1)

    # Display the predicted class on the image
    cv2.putText(frame, f"Class: {predicted_class[0]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the image with the prediction
    cv2.imshow('Camera - Gesture Prediction', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
