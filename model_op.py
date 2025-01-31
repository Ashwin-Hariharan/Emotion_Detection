import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load a pre-trained emotion classification model
model = tf.keras.models.load_model('emotion_detection_model_improved.keras')

# Emotion labels (ensure these match the model's output classes)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize image
    return img_array

# Function to predict emotion
def classify_emotion(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_class]
    return predicted_emotion

# Open file dialog to select an image
Tk().withdraw()  # Hide the root window
img_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

if img_path:
    # Classify the image
    emotion = classify_emotion(img_path)
    print(f"The emotion in the image is: {emotion}")
    
    # Display the image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted Emotion: {emotion}")
    plt.show()
else:
    print("No image selected.")
