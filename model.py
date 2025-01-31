import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


# Plot training history
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(model_history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(model_history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].legend()
    axs[1].plot(model_history.history['loss'], label='Train Loss')
    axs[1].plot(model_history.history['val_loss'], label='Validation Loss')
    axs[1].set_title('Model Loss')
    axs[1].legend()
    plt.show()

# Define directories for training and validation (test)
train_dir = r'C:\Users\ashwi\GUVI_Projects\Emotion\DS\Dataset\train'
val_dir = r'C:\Users\ashwi\GUVI_Projects\Emotion\DS\Dataset\test'

batch_size = 64
epochs = 20

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for validation
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=batch_size, color_mode="grayscale", class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=batch_size, color_mode="grayscale", class_mode='categorical')


# Class weights to handle imbalance
class_weights = {0: 1.0, 1: 5.0, 2: 1.2, 3: 1.0, 4: 1.0, 5: 1.5, 6: 1.3}



def depthwise_separable_conv_block(x, filters, strides):
    """Creates a depthwise separable convolution block."""
    # Depthwise Convolution
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    # Pointwise Convolution
    x = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    return x

# Reuse your existing model
def create_model(input_shape=(224, 224, 1), num_classes=7):
    """Constructs the MobileNet architecture manually with a classification head."""
    # Input layer
    inputs = Input(shape=input_shape)

    # Initial Convolutional Layer
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    # Depthwise Separable Convolution Blocks
    x = depthwise_separable_conv_block(x, filters=64, strides=(1, 1))
    x = depthwise_separable_conv_block(x, filters=128, strides=(2, 2))
    x = depthwise_separable_conv_block(x, filters=128, strides=(1, 1))
    x = depthwise_separable_conv_block(x, filters=256, strides=(2, 2))
    x = depthwise_separable_conv_block(x, filters=256, strides=(1, 1))
    x = depthwise_separable_conv_block(x, filters=512, strides=(2, 2))

    # 5 additional blocks with 512 filters and stride 1
    for _ in range(5):
        x = depthwise_separable_conv_block(x, filters=512, strides=(1, 1))

    x = depthwise_separable_conv_block(x, filters=1024, strides=(2, 2))
    x = depthwise_separable_conv_block(x, filters=1024, strides=(1, 1))

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer for classification
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs, outputs)
    return model



model = create_model(input_shape=(224, 224, 1), num_classes=7)


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# Plot training history
plot_model_history(history)

# Save the trained model
model.save('emotion_detection_model_improved.keras')

model.summary()

# Evaluate the model
val_labels = val_generator.classes
val_predictions = model.predict(val_generator)
val_pred_classes = np.argmax(val_predictions, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(val_labels, val_pred_classes, target_names=list(val_generator.class_indices.keys())))

# Confusion matrix
cm = confusion_matrix(val_labels, val_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_generator.class_indices.keys()))
disp.plot(cmap='viridis')
plt.show()
