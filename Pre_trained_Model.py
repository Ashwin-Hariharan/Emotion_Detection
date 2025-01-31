import os
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Define directories for training and validation (test)
train_dir = r'C:\Users\ashwi\GUVI_Projects\Emotion\DS\Dataset\train'
val_dir = r'C:\Users\ashwi\GUVI_Projects\Emotion\DS\Dataset\test'

# Image size for the model (assuming input size of (48, 48, 1) for grayscale)
image_size = (48, 48)

# Function to load images from the directory
def load_images_from_directory(directory, image_size=(48, 48)):
    images = []
    labels = []
    class_names = os.listdir(directory)  # Get the subfolder names (emotion labels)

    for label in class_names:
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                if img is not None:
                    img = cv2.resize(img, image_size)  # Resize to the desired image size
                    img = np.expand_dims(img, axis=-1)  # Add the channel dimension (48, 48, 1)
                    images.append(img)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

# Load train and validation data
x_train, y_train = load_images_from_directory(train_dir, image_size)
x_val, y_val = load_images_from_directory(val_dir, image_size)

# Normalize image data
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_train = to_categorical(y_train, num_classes=7)
y_val = label_encoder.transform(y_val)
y_val = to_categorical(y_val, num_classes=7)

# Reshape data to add batch dimension (for model input)
x_train = x_train.reshape(-1, 48, 48, 1)
x_val = x_val.reshape(-1, 48, 48, 1)

print("Train data shape:", x_train.shape)
print("Validation data shape:", x_val.shape)

# DEFINING A DEEP CONVOLUTIONARY NEURAL NETWORK MODEL

model = Sequential()

# 1. LAYER
model.add(Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal", input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 2. LAYER
model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.6)) # 60% forgetting process (neuron deletion-dropout)

# 3. LAYER
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 4. LAYER
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 5. LAYER
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.6)) # 60% forgetting process (neuron deletion-dropout)

# FULL CONNECTION LAYER
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6)) # 60% forgetting process (neuron deletion-dropout)

# OUTPUT LAYER
model.add(Dense(7))  # 7 emotions to classify
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summarize model architecture
model.summary()

# Training the model with ImageDataGenerator and ModelCheckpoint

batch_size = 32
epochs = 10

# Data augmentation generator
datagen = ImageDataGenerator()

train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

# Model checkpoint to save the best model
checkpointer = ModelCheckpoint(filepath='best_model.h5', verbose=1, save_best_only=True)

# Train the model
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpointer], verbose=2)

# Save the trained model to 'bestmodel.keras'
model.save('best_model.keras')  # Save the model in the .keras format

# Evaluate the model on the validation dataset
score = model.evaluate(x_val, y_val, verbose=1)
print(f"Validation Loss: {score[0]}")
print(f"Validation Accuracy: {score[1] * 100:.2f}%")

# Plot training history
plt.figure(figsize=(14,3))

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.title('Loss')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()

plt.show()
