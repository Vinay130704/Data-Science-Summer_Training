from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np
from PIL import Image
import os

# Extract features from images
def extract_features(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = {}
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        image = Image.open(img_path).resize((299, 299))
        image = np.array(image)
        if image.shape[2] == 4:  # Handle RGBA images
            image = image[:, :, :3]
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[img_name.split('.')[0]] = feature
    return features
