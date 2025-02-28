import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import KMeans
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt

# Load pretrained MobileNetV2 model (without classifier)
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

dataset_path_filter = r'C:\Users\ashwi\GUVI_Projects\Emotion\DS\Dataset\train1\train'
class_names = ["angry", "fear", "disgust", "happy", "neutral", "sad", "surprise"]

# Step 1: Filter Out-of-Context Images
for class_name in class_names:
    #Handling Outliners
    """
    class_folder = os.path.join(dataset_path_filter, class_name)
    if not os.path.isdir(class_folder):
        continue
    
    image_paths = [os.path.join(class_folder, img) for img in os.listdir(class_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    print(f"Total images before removal in '{class_name}': {len(image_paths)}")
    
    if not image_paths:
        continue
    
    def extract_features(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    
    feature_list = np.array([extract_features(img_path) for img_path in image_paths])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(feature_list)
    outlier_cluster = 0 if np.sum(labels) > len(labels) // 2 else 1
    outliers = [image_paths[i] for i in range(len(labels)) if labels[i] == outlier_cluster]
    
    print(f"\nRemoved Out-of-Context Images in '{class_name}':")
    
    for outlier in outliers:
        try:
            os.remove(outlier)
        except Exception as e:
            print(f"Could not delete {outlier}: {e}")
    
    image_paths = [img for img in image_paths if img not in outliers]
    print(f"Total images after removal in '{class_name}': {len(image_paths)}")
    """
#Balancing the DS
class_counts = {}
print("\nInitial image counts:")
for class_name in class_names:
    class_folder = os.path.join(dataset_path_filter, class_name)
    if os.path.isdir(class_folder):
        class_counts[class_name] = len([img for img in os.listdir(class_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))])
        print(f"{class_name}: {class_counts[class_name]}")

# Target image count (Between 1600 and 1700)
target_count = int(max(class_counts.values()) * 1.2)

for class_name in class_names:
    class_folder = os.path.join(dataset_path_filter, class_name)
    if not os.path.isdir(class_folder):
        continue

    image_paths = [os.path.join(class_folder, img) for img in os.listdir(class_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    # **Undersampling (If Too Many Images)**
    if len(image_paths) > target_count:
        num_to_remove = len(image_paths) - target_count
        images_to_delete = random.sample(image_paths, num_to_remove)
        
        for img_path in images_to_delete:
            os.remove(img_path)

    # **Oversampling (If Too Few Images)**
    elif len(image_paths) < target_count:
        num_to_add = target_count - len(image_paths)
        while num_to_add > 0:
            img_path = random.choice(image_paths)  # Pick a random image
            base, ext = os.path.splitext(img_path)
            new_img_path = f"{base}_copy{num_to_add}{ext}"

            img = Image.open(img_path)
            img.save(new_img_path)
            image_paths.append(new_img_path)  # Add to list
            num_to_add -= 1

# Step 3: Get Final Image Counts
print("\nFinal image counts:")
for class_name in class_names:
    class_folder = os.path.join(dataset_path_filter, class_name)
    if os.path.isdir(class_folder):
        final_count = len([img for img in os.listdir(class_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))])
        print(f"{class_name}: {final_count}")
