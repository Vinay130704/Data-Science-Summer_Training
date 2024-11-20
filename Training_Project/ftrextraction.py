import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tqdm import tqdm

# Define paths
images_dir = 'Images'  # Directory where your images are stored (replace this with the correct path)
features_path = 'features.npy'  # Output file path for features
weights_path = 'interceptionv3.h5'  # Path to the manually downloaded weights file

# Load InceptionV3 model with pre-trained weights (without the top classification layer)
model = InceptionV3(weights=None, include_top=False, pooling='avg')  # We don't load the weights yet
model.load_weights(weights_path)  # Load the manually downloaded weights

# Function to extract features from an image
def extract_features(image_path):
    # Load image, resize to target size, and preprocess for InceptionV3
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)  # Prepare the image for the model
    
    # Extract features using InceptionV3
    feature = model.predict(image)
    return feature

# Initialize an empty dictionary to store features
features = {}

# Loop over each image in the directory and extract features
for image_name in tqdm(os.listdir(images_dir)):
    # Decode the file name correctly
    image_name = os.fsdecode(image_name)
    
    try:
        image_path = os.path.join(images_dir, image_name)
        feature = extract_features(image_path)
        features[image_name] = feature  # Store features with the image name as the key
    except Exception as e:
        print(f"Error processing image {image_name}: {e}")
        continue

# Save features to a numpy file
np.save(features_path, features)
print(f"Features have been successfully saved to {features_path}")
