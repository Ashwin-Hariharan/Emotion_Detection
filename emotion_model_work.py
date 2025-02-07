import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import pandas as pd

# ✅ Define Dataset Paths
train_dir = r'C:\Users\ashwi\GUVI_Projects\Emotion\DS\Dataset\train'
val_dir = r'C:\Users\ashwi\GUVI_Projects\Emotion\DS\Dataset\test'

# ✅ Define Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = (96, 96)  # Image size for models
AUTOTUNE = tf.data.AUTOTUNE  

# ✅ Data Augmentation (Grayscale)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# ✅ Load Training & Validation Data (Grayscale Input)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale", class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale", class_mode='categorical')

# ✅ Class Names
class_names = list(train_generator.class_indices.keys())

# ✅ Define Transfer Learning Models (Adjust for Grayscale Input)
def create_transfer_model(base_model, input_shape=(96, 96, 1), num_classes=7):
    base_model.trainable = False  # Freeze base model for feature extraction
    
    # Add a grayscale input layer (1 channel)
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(3, (3, 3), padding="same", activation="relu")(inputs)  # Convert 1-channel → 3-channel
    x = base_model(x, training=False)

    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# ✅ Define Pretrained Models
models_to_compare = {
    "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(96, 96, 3)),
    "MobileNetV2": MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3)),
    "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3)),
}

# ✅ Train & Evaluate Models
performance_metrics = {}
trained_models = {}

for model_name, base_model in models_to_compare.items():
    print(f"\n🚀 **Training {model_name} Model with Grayscale Input...**")
    
    # Create Transfer Learning Model with Grayscale Compatibility
    model = create_transfer_model(base_model, input_shape=(96, 96, 1), num_classes=len(class_names))

    # Compile Model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        ModelCheckpoint(f'{model_name}_best.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # Train Model
    history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=callbacks)

    # Save Model
    model.save(f"{model_name}_finetuned.keras")
    trained_models[model_name] = model

# ✅ Evaluate Models Using Metrics
for model_name, model in trained_models.items():
    print(f"\n📊 **Evaluating {model_name}...**")

    # Get predictions
    val_predictions = model.predict(val_generator)
    y_pred = np.argmax(val_predictions, axis=1)
    y_true = val_generator.classes

    # Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    performance_metrics[model_name] = (accuracy, precision, recall, f1)

# ✅ Compare Model Performance
df_performance = pd.DataFrame(performance_metrics, index=["Accuracy", "Precision", "Recall", "F1-score"]).T
print("\n🏆 **Best Performing Model:**", df_performance["Accuracy"].idxmax())

# ✅ Save the Best Model
best_model_name = df_performance["Accuracy"].idxmax()
best_model = trained_models[best_model_name]
best_model.save("best_emotion_model_grayscale.keras")
print(f"✅ Best model saved as `best_emotion_model_grayscale.keras`")

# ✅ Display Performance Summary
print("\n📊 **Model Performance Summary:**")
print(df_performance)
