# Step 1: Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Step 2: Load pre-trained MobileNetV2 (on ImageNet)
model = MobileNetV2(weights='imagenet')
IMG_SIZE = 224

# Step 3: Load and classify images from local path
while True:
    print("\n📤 Type the image path (or press Enter to exit):")
    path = input("🖼 ")

    if not path.strip():
        print("🛑 Exiting...")
        break

    if not os.path.exists(path):
        print("❌ File not found. Please try again.")
        continue

    try:
                # Load and preprocess the image
        img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        predictions = model.predict(img_array)
        decoded_preds = decode_predictions(predictions, top=3)[0]

        # Show image with top prediction
        plt.imshow(img)
        title = "\n".join([f"{label}: {score:.2%}" for (_, label, score) in decoded_preds])
        plt.title(f"🔍 Prediction:\n{title}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("❌ Error processing the image:",e)