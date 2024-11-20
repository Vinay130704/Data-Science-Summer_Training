from keras.models import load_model
from data_utils import pad_sequences
import numpy as np

# Generate captions
def generate_caption(model, tokenizer, feature, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = next((word for word, idx in tokenizer.word_index.items() if idx == yhat), None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Test image
if __name__ == '__main__':
    model = load_model('model.h5')
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    max_length = 34
    test_feature = extract_features('example.jpg')
    caption = generate_caption(model, tokenizer, test_feature, max_length)
    print(f"Generated Caption: {caption}")
