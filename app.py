import gdown
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences
from transformers import ViTFeatureExtractor, ViTModel
import torch
import pickle
import numpy as np
from PIL import Image
import streamlit as st

# Re-defining the NotEqualMask class
class NotEqualMask(Layer):
    def __init__(self, **kwargs):
        super(NotEqualMask, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        return input_shape

# Cache Model Download
@st.cache_resource
def download_model():
    model_path = 'Model/vit_model.keras'
    if not os.path.exists(model_path):
        os.makedirs('Model', exist_ok=True)
        url = 'https://drive.google.com/uc?id=1g083q5QMQ8ldcAYHRVA8ztbxsbGmZVbf'
        gdown.download(url, model_path, quiet=False)
    return model_path

# Cache ViT Feature Extractor and Model
@st.cache_resource
def load_vit_models():
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    return feature_extractor, vit_model

# Load models
model_file_path = download_model()
model = load_model(model_file_path, custom_objects={'NotEqualMask': NotEqualMask})

# Load tokenizer
with open('Tokenizer/vit30k_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load feature extractor and ViT model
feature_extractor, vit_model = load_vit_models()

# Feature extraction function
def extract_features(image):
    image = Image.open(image).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = vit_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token representation
    
    image_features = features.numpy().squeeze()
    return image_features

# Generate beam caption
def generate_beam_caption(model, tokenizer, photo, max_length=80, beam_width=3):
    start_seq = [tokenizer.word_index.get('startseq', 0)]
    start_word = [[start_seq, 0.0]]
    photo = np.array(photo).reshape(1, 768)
    
    while len(start_word[0][0]) < max_length:
        temp = []
        sequences = []
        scores = []
        
        for seq, score in start_word:
            sequences.append(seq)
            scores.append(score)
        
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

# Generate greedy caption
def generate_greedy_caption(model, tokenizer, photo, max_length=80):
    in_text = 'startseq'
    photo = np.array(photo).reshape(1, 768)
    for i in range(max_length):
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

# Streamlit UI
st.title("Image captioning with ViT + LSTM")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)
    if st.button("Generate Caption"):
        with st.spinner("Generating..."):
            image_features = extract_features(uploaded_file)
            beam_caption = generate_beam_caption(model, tokenizer, image_features)
            greedy_caption = generate_greedy_caption(model, tokenizer, image_features)
            st.success(f"Beam caption: {beam_caption}")
            st.success(f"Greedy caption: {greedy_caption}")
