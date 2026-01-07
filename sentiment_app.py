import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources (run once)
nltk.download("stopwords")
nltk.download("wordnet")

# Load model and vectorizer
model = joblib.load("results/sentiment_model.pkl")
vectorizer = joblib.load("results/vectorizer.pkl")

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Streamlit app
st.title("Sentiment Analysis App")
st.write("Type a review and get sentiment prediction!")

# User input
user_input = st.text_area("Enter your review:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        clean_input = clean_text(user_input)
        vector = vectorizer.transform([clean_input])
        prediction = model.predict(vector)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter a review.")
