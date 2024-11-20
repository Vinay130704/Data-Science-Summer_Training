from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import numpy as np

# Create tokenizer
def create_tokenizer(descriptions):
    tokenizer = Tokenizer()
    all_desc = []
    for key in descriptions.keys():
        all_desc.extend(descriptions[key])
    tokenizer.fit_on_texts(all_desc)
    return tokenizer

# Get maximum caption length
def max_caption_length(descriptions):
    return max(len(caption.split()) for desc_list in descriptions.values() for caption in desc_list)

# Data generator for training
def data_generator(descriptions, photos, tokenizer, max_length, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for image_id, captions in descriptions.items():
            photo = photos[image_id]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = np.zeros(tokenizer.num_words + 1)
                    out_seq[out_seq] = 1.0
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
                n += 1
                if n == batch_size:
                    yield [[np.array(X1), np.array(X2)], np.array(y)]
                    X1, X2, y = [], [], []
                    n = 0
