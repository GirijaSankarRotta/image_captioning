from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences
from transformers import ViTFeatureExtractor, ViTModel
import tensorflow as tf
import torch
import pickle
import numpy as np
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st

# Re-defining the NotEqualMask class
class NotEqualMask(Layer):
    def __init__(self, **kwargs):
        super(NotEqualMask, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        return input_shape

# Now load the model with the custom layer
model = load_model('Model/vit_model.keras', custom_objects={'NotEqualMask': NotEqualMask})

# # Use the model
# model.summary()

# Load the tokenizer from the tokenizer.pkl file
with open('Tokenizer/vit30k_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# # Print the tokenizer word index to confirm it has been loaded correctly
# print('vocabulary : ', len(tokenizer.word_index)+1)
# Load pretrained feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def extract_features(image):
    # Open and process the image
    image = Image.open(image).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Extract features
    with torch.no_grad():
        outputs = vit_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token representation
    
    # Convert to numpy array
    image_features = features.numpy().squeeze()
    return image_features

#generate Beam caption
def generate_beam_caption(model, tokenizer, photo, max_length=80, beam_width=3):
    start_seq = [tokenizer.word_index.get('startseq', 0)]
    start_word = [[start_seq, 0.0]]
    photo = np.array(photo).reshape(1, 768)
    
    while len(start_word[0][0]) < max_length:
        temp = []
        sequences = []
        scores = []
        
        # Prepare all sequences at once
        for seq, score in start_word:
            sequences.append(seq)
            scores.append(score)
        
        sequences_padded = pad_sequences(sequences, maxlen=max_length, padding='post')
        photos_batch = np.repeat(photo, len(sequences_padded), axis=0)  # Repeat the photo for each sequence
        
        # Predict in a single batch
        preds = model.predict([photos_batch, sequences_padded], verbose=0)
        
        # Expand each prediction
        for i in range(len(preds)):
            top_k_indices = np.argsort(preds[i])[-beam_width:]
            for index in top_k_indices:
                word_seq = sequences[i][:] + [index]
                prob = scores[i] + np.log(preds[i][index] + 1e-10)  # Add small value to avoid log(0)
                temp.append([word_seq, prob])
        
        # Keep only top beam_width sequences
        start_word = sorted(temp, key=lambda l: l[1], reverse=True)[:beam_width]
    
    # Get the sequence with the best score
    final_seq = start_word[0][0]
    final_caption = ' '.join(tokenizer.index_word.get(i, '') for i in final_seq if i != 0)
    
    return final_caption.replace('startseq ', '').replace(' endseq', '')

#generate beam caption
def generate_greedy_caption(model, tokenizer, photo, max_length = 80):
    in_text = 'startseq'
    photo = np.array(photo)
    photo = photo.reshape(1,768)
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

def generate_caption(image):
    image_features = extract_features(image)
    beam_caption = generate_beam_caption(model, tokenizer, image_features)
    greedy_caption = generate_greedy_caption(model, tokenizer, image_features)
    show_image(image)
    print(beam_caption)
    print(greedy_caption)
# generate_caption('man.jpg')

#======streamlit UI======
st.title("Image captioning with ViT + LSTM")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="uploaded Image", use_container_width=True)
    if st.button("Generate Caption"):
        with st.spinner("Generating..."):
            image_features = extract_features(uploaded_file)
            beam_caption = generate_beam_caption(model, tokenizer, image_features)
            greedy_caption = generate_greedy_caption(model, tokenizer, image_features)
            st.success(f"Beam caption : {beam_caption}")
            st.success(f"greedy caption : {greedy_caption}")