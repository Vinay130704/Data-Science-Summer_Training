import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, add
from tensorflow.keras.callbacks import ModelCheckpoint
from data_utils import create_tokenizer, max_caption_length, data_generator

# Load descriptions
def load_descriptions(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    
    descriptions = {}
    # Iterate through each line of the file
    for line in text.strip().split('\n'):
        # Split by comma
        if ',' in line:  # Ensure that a comma exists before attempting to split
            image_id, description = line.split(',', 1)  # Only split on the first comma
            # Store the descriptions in a dictionary
            if image_id not in descriptions:
                descriptions[image_id] = []
            descriptions[image_id].append(description)
        else:
            print(f"Skipping malformed line: {line}")
    
    return descriptions


# Paths
descriptions_path = 'descriptions.txt'
photo_features_path = 'features.npy'
model_save_path = 'model.h5'

# Load resources
descriptions = load_descriptions(descriptions_path)
photos = np.load(photo_features_path, allow_pickle=True).item()

# Create tokenizer
tokenizer = create_tokenizer(descriptions)
max_length = max_caption_length(descriptions)

# Define the model
vocab_size = len(tokenizer.word_index) + 1

# Image feature extractor model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# Decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# Final model
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
batch_size = 32
steps = len(descriptions) // batch_size

generator = data_generator(descriptions, photos, tokenizer, max_length, batch_size)
checkpoint = ModelCheckpoint(model_save_path, monitor='loss', save_best_only=True)

model.fit(generator, epochs=20, steps_per_epoch=steps, callbacks=[checkpoint])
