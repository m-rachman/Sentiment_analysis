import streamlit as st
import pandas as pd
import re
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.keras')

# Load stopwords
stpwds_id = list(set(stopwords.words('english')))

# Text preprocessing function
def text_preprocessing(text):
    # Case folding
    text = text.lower()

    # Mention removal
    text = re.sub("@[A-Za-z0-9_]+", " ", text)

    # Hashtags removal
    text = re.sub("#[A-Za-z0-9_]+", " ", text)

    # Newline removal (\n)
    text = re.sub(r"\\n", " ",text)

    # Whitespace removal
    text = text.strip()

    # URL removal
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www.\S+", " ", text)

    # Non-letter removal (such as emoticons, symbols, etc.)
    text = re.sub("[^A-Za-z\s']", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords removal
    tokens = [word for word in tokens if word not in stpwds_id]

    # Combining Tokens
    text = ' '.join(tokens)

    return text

# Define the Streamlit interface
st.title('Sentiment Analysis App')

# Get user input
user_input = st.text_area("Enter the text for sentiment analysis:")

if st.button('Analyze'):
    if user_input:
        # Preprocess the input text
        processed_text = text_preprocessing(user_input)
        prediction = model.predict([[processed_text]])
        sentiment = "Positive" if prediction[0] > 0.5 else "Negative"

        # Display the result
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter some text.")

