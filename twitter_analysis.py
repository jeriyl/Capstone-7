import streamlit as st
import pickle
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

# Load the pre-trained model
loaded_model = pickle.load(open(r'C:\Users\91822\OneDrive\Documents\Final Project\Twitter-Sentiment-Analysis\trained_model_logisticRegression.pkl', "rb"))

# Load the TfidfVectorizer
vectorizer = pickle.load(open(r'C:\Users\91822\OneDrive\Documents\Final Project\Twitter-Sentiment-Analysis\trained_model_tfidfvectorizer.pkl', "rb"))

# Define the stemming function
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

# Set up the Streamlit app
st.title("Twitter Sentiment Analysis")

# Get the input tweet from the user
tweet = st.text_area("Enter a tweet:", height=200)

# Add a "Analyze" button
if st.button("Analyze"):
    # Preprocess the input tweet
    preprocessed_tweet = stemming(tweet)

    # Convert the preprocessed tweet to a numerical format
    X_input = vectorizer.transform([preprocessed_tweet])

    # Make the prediction
    prediction = loaded_model.predict(X_input)

    # Display the sentiment
    if prediction[0] == 0:
        st.write("Negative Tweet")
    else:
        st.write("Positive Tweet")