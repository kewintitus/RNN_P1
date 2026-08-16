import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from pathlib import Path


word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "simplernn_imdb_model.h5"

model = load_model(model_path)

# Helper functions


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess the input text


def preprocess_input_text(input_text, maxlen=500):
    # Tokenize the input text
    tokens = input_text.lower().split()

    # Convert tokens to their corresponding indices in the IMDB dataset
    # 2 is for unknown words
    encoded_review = [word_index.get(token, 2) + 3 for token in tokens]

    # Pad the sequence to ensure it has the same length as the training data
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)

    return padded_review


# Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_input_text(review)

    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# streamlit application
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below, and the model will predict whether the sentiment is positive or negative.")

user_input = st.text_area("Movie Review", "")

if st.button("Predict Sentiment"):
    if user_input:
        sentiment, probability = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: {sentiment}")
        st.write(f"Confidence: {probability:.2f}")
    else:
        st.write("Please enter a movie review to analyze.")
