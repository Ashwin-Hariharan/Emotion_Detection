import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Dataset paths
train_dir = r'C:\Users\ashwi\GUVI_Projects\Emotion\DS\Dataset\train1\train'
class_names = ["angry", "fear", "disgust", "happy", "neutral", "sad", "surprise"]

# Data Preprocessing with Augmentation
data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.1  # 90% train, 10% validation
)

train_generator = data_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # Updated to match emotion dataset
    batch_size=32,
    class_mode='categorical',
    color_mode="rgb",  # Convert grayscale images to RGB
    subset='training'
)

val_generator = data_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    color_mode="rgb",
    subset='validation'
)

# Define a function to modify pretrained models to accept 48x48 input
def use_pretrained_model(model_class, model_name):
    # Load the base model with ImageNet weights and original input shape
    base_model = model_class(weights="imagenet", include_top=False, input_shape=(48, 48, 3))

    # Attach custom classifier
    global_avg_pool = GlobalAveragePooling2D()(base_model.output)
    dense = Dense(128, activation='relu')(global_avg_pool)
    dropout = Dropout(0.5)(dense)
    output = Dense(len(class_names), activation='softmax')(dropout)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=25,
        callbacks=[early_stopping]
    )

    model.save(f'{model_name}.h5')

    return history

# Use pre-trained models and store histories
models = {
    #'VGG16': VGG16,  # Replaced EfficientNetB0 with VGG16
    #'MobileNetV2': MobileNetV2,
    'ResNet50': ResNet50
}

histories = {}
for model_name, model_class in models.items():
    print(f'Using {model_name} with custom classifier on 48×48 images...')
    histories[model_name] = use_pretrained_model(model_class, model_name)

# Plot accuracy and loss
plt.figure(figsize=(12, 6))
for model_name, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{model_name} Train Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Val Accuracy')
plt.title('Model Accuracy (48x48)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for model_name, history in histories.items():
    plt.plot(history.history['loss'], label=f'{model_name} Train Loss')
    plt.plot(history.history['val_loss'], label=f'{model_name} Val Loss')
plt.title('Model Loss (48x48)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
