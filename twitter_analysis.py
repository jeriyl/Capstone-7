import streamlit as st
import pickle
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from streamlit_lottie import st_lottie
import requests
import nltk
nltk.download('stopwords')

st.set_page_config(page_title="Twitter Sentiment Analysis",page_icon=":blue_heart:")
def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background:url("https://c4.wallpaperflare.com/wallpaper/154/47/615/twitter-social-networks-wallpaper-preview.jpg");
                        background-size: cover}}
                     </style>""", unsafe_allow_html=True)
setting_bg()
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#lottie_coding=load_lottieurl("https://lottie.host/304662f4-751f-4ec6-bf94-5633f32de03e/aOWpvskzxm.json")

# Load the pre-trained model
loaded_model = pickle.load(open(r'trained_model_logisticRegression.pkl', "rb"))
loaded_model1 = pickle.load(open(r'trained_model_RandomForestClassifier.pkl',"rb"))
# Load the TfidfVectorizer
vectorizer = pickle.load(open(r'trained_model_tfidfvectorizer.pkl', "rb"))

# Define the stemming function
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

title_column, lottie_column = st.columns([1, 0.3])

# Title in the left column
with title_column:
    st.markdown("<p style='font-size:38px; color:White; text-align:right; font-weight:bold;'>Twitter Sentiment Analysis</p>", unsafe_allow_html=True)

tweet = st.text_area("Enter a tweet:", height=170)

if st.button("Predict the Status"):
    # Preprocess the input tweet
    preprocessed_tweet = stemming(tweet)

    # Convert the preprocessed tweet to a numerical format
    X_input = vectorizer.transform([preprocessed_tweet])

    # Make the prediction
    prediction = loaded_model.predict(X_input)

    # Display the sentiment
    if prediction[0] == 0:
        st.markdown('<p style="color:red; font-size: 24px; font-family: Arial, sans-serif; font-style: italic; font-weight: bold; border: 1px solid black; padding: 10px; text-align: center; background-color: pink;">Negative Tweet</p>', unsafe_allow_html=True)


    else:
        st.markdown('<p style="color:green; font-size: 24px; font-family: Arial, sans-serif; font-style: italic; font-weight: bold; border: 1px solid black; padding: 10px; text-align: center; background-color: lightgreen;">Positive Tweet</p>', unsafe_allow_html=True)

st.markdown("<p style='font-size:28px; color:White; font-weight:bold;'>Sample Tweets:</p>", unsafe_allow_html=True)

# Display the sample tweets
sample_tweets = [
    "Congratulations Team India for a spectacular win! 🇮🇳🏆 Dominating performance that makes the nation proud. #TeamIndia #Cricket #Victory",
    "Though India lost the match, the spirit of the game lives on. Chin up, Team India! 🇮🇳 #Cricket #TeamIndia #KeepFighting",
    "Today has been a rough day. Nothing seems to be going right. 😔 #BadDay"

]
for idx, sample_tweet in enumerate(sample_tweets, start=1):
    st.write(f"<span style='color:white;font-size:18px'>{idx}. {sample_tweet}</span>", unsafe_allow_html=True)

