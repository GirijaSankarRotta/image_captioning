import os
import gdown
import pickle
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import streamlit as st

# ====== Custom Keras Layer ======
class NotEqualMask(Layer):
    def __init__(self, **kwargs):
        super(NotEqualMask, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        return input_shape

# ====== Download Keras Model from Google Drive ======
model_path = 'Model/vit_model.keras'
file_id = '1g083q5QMQ8ldcAYHRVA8ztbxsbGmZVbf'

if not os.path.exists(model_path):
    os.makedirs('Model', exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)

# ====== Load Model and Tokenizer ======
model = load_model(model_path, custom_objects={'NotEqualMask': NotEqualMask})

with open('Tokenizer/vit30k_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# ====== Load ViT Feature Extractor ======
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def extract_features(image):
    image = Image.open(image).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
    return features.numpy().squeeze()

def generate_beam_caption(model, tokenizer, photo, max_length=80, beam_width=3):
    start_seq = [tokenizer.word_index.get('startseq', 0)]
    start_word = [[start_seq, 0.0]]
    photo = np.array(photo).reshape(1, 768)

    while len(start_word[0][0]) < max_length:
        temp = []
        sequences = [seq for seq, _ in start_word]
        scores = [score for _, score in start_word]

        sequences_padded = pad_sequences(sequences, maxlen=max_length, padding='post')
        photos_batch = np.repeat(photo, len(sequences_padded), axis=0)
        preds = model.predict([photos_batch, sequences_padded], verbose=0)

        for i in range(len(preds)):
            top_k_indices = np.argsort(preds[i])[-beam_width:]
            for index in top_k_indices:
                word_seq = sequences[i][:] + [index]
                prob = scores[i] + np.log(preds[i][index] + 1e-10)
                temp.append([word_seq, prob])

        start_word = sorted(temp, key=lambda l: l[1], reverse=True)[:beam_width]

    final_seq = start_word[0][0]
    final_caption = ' '.join(tokenizer.index_word.get(i, '') for i in final_seq if i != 0)
    return final_caption.replace('startseq ', '').replace(' endseq', '')

def generate_greedy_caption(model, tokenizer, photo, max_length=80):
    in_text = 'startseq'
    photo = np.array(photo).reshape(1, 768)

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, '')
        if not word:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    return in_text.replace('startseq ', '').replace(' endseq', '')

# ====== Streamlit UI ======
st.title("🖼️ Image Captioning using ViT + LSTM")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating captions..."):
            image_features = extract_features(uploaded_file)
            beam_caption = generate_beam_caption(model, tokenizer, image_features)
            greedy_caption = generate_greedy_caption(model, tokenizer, image_features)
            st.success(f"**Beam Search Caption:** {beam_caption}")
            st.success(f"**Greedy Search Caption:** {greedy_caption}")
