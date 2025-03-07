Emotion Detection Using Transfer Learning

** Project Overview**
This project implements emotion detection using transfer learning on grayscale images. It allows users to upload an image via a Streamlit web app, where the model classifies the dominant emotion using VGG16, MobileNetV2, and ResNet50.

The best-performing model (ResNet50) achieved 64% accuracy and was saved for real-time inference.

** Features**
Pretrained CNN models (VGG16, MobileNetV2, ResNet50)
Transfer learning with grayscale input support
Data augmentation for improved generalization
Real-time image upload & prediction in a Streamlit web app
Optimized model with EarlyStopping & Learning Rate Scheduling

**Key Technologies Used**
TensorFlow/Keras (VGG16, MobileNetV2, ResNet50)
Streamlit (Real-time Web App)
OpenCV & PIL (Image Processing)
Scikit-learn (Model Evaluation)
