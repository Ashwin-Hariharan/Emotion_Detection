import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
MODEL_PATH = r'C:\Users\ashwi\GUVI_Projects\Emotion\best_emotion_model.h5'
model = load_model(MODEL_PATH)

# Emotion Classes
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
image_size = (48, 48)  # Model expects 48x48 grayscale images

# Streamlit App UI
st.title("Emotion Recognition App")
st.write("Upload an image, and the model will predict the emotion!")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to grayscale (1 channel)
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    
    # Resize the image to match model input size
    image = image.resize(image_size)

    # Convert to NumPy array and preprocess
    face_image = img_to_array(image)  # Convert to array
    face_image = np.expand_dims(face_image, axis=-1)  # Ensure shape (48, 48, 1)
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension (1, 48, 48, 1)
    face_image = face_image / 255.0  # Normalize pixel values
    
    # Model Prediction
    predictions = model.predict(face_image)
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions))  # Softmax-like scaling
    predicted_class_index = np.argmax(probabilities)
    predicted_emotion = class_names[predicted_class_index]
    confidence = np.max(probabilities) * 100  # Convert to percentage

    # Display Image & Prediction
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write(f"### Predicted Emotion: **{predicted_emotion.capitalize()}**")
    
