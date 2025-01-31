import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, f1_score  # Metrics for evaluation
import os
from tkinter import filedialog  # For file selection dialog
from tkinter import Tk  # To use Tkinter without showing the window

# Load the trained model
model_path = r'C:\Users\ashwi\GUVI_Projects\Emotion\best_model.keras'  # Set your model path
model_best = load_model(model_path)

# Define emotion classes (ensure these match the training dataset labels)
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Ask user to select an image file for testing
print("📂 Please select test images for emotion classification:")

# Initialize Tkinter and hide main window (for file dialog)
root = Tk()
root.withdraw()  # Don't show the Tkinter root window

# Allow user to select multiple image files
file_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

# Initialize lists to store predictions
y_pred = []

# Process each selected image
for img_path in file_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    if img is None:
        print(f"❌ Error: Could not load image '{img_path}'. Skipping.")
        continue

    # Resize image to match model input size
    face_image = cv2.resize(img, (224, 224))

    # Convert grayscale to 3-channel (if model expects RGB input)
    face_image = np.stack((face_image,)*3, axis=-1)  # Convert to (224, 224, 3)

    # Preprocess for model
    face_image = image.img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
    face_image /= 255.0  # Normalize pixel values

    # Predict emotion
    predictions = model_best.predict(face_image)
    predicted_emotion = class_names[np.argmax(predictions)]

    # Store predicted labels
    y_pred.append(predicted_emotion)

    print(f"🖼️ Image: {os.path.basename(img_path)} | 🤖 Predicted: {predicted_emotion}")

# Ensure there are valid predictions before calculating accuracy and F1 score
if y_pred:
    print("\n📊 **Model Predictions on Selected Images:**")
    print(f"✅ Predictions: {y_pred}")
else:
    print("\n⚠️ No valid predictions were made. Please select valid images.")
