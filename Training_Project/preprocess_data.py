from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import os
import string
from numpy import array

# Load and clean captions
def load_captions(filepath):
    with open(filepath, 'r') as file:
        text = file.read()
    captions = {}
    for line in text.split('\n'):
        if len(line) < 2:
            continue
        image_id, caption = line.split('\t')
        image_id = image_id.split('.')[0]
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append('startseq ' + caption.lower() + ' endseq')
    return captions

# Save cleaned captions to a file
def save_clean_captions(captions, filepath):
    with open(filepath, 'w') as file:
        for image_id, caption_list in captions.items():
            for caption in caption_list:
                file.write(f"{image_id}\t{caption}\n")
