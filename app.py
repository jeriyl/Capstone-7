import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

st.set_page_config(page_title="Twitter Sentiment Analysis",page_icon="🏨",layout="wide")
def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background:url("https://cdn.vox-cdn.com/thumbor/ja-AARxC-3kEHgMewcykITPI_3c=/0x0:2040x1360/1200x628/filters:focal(1020x680:1021x681)/cdn2.vox-cdn.com/uploads/chorus_asset/file/8966105/acastro_170726_1777_0010.jpg");
                        background-size: cover}}
                     </style>""", unsafe_allow_html=True)
setting_bg()

df=pd.read_csv(r'C:\Users\91822\OneDrive\Documents\Final Project\Twitter-Sentiment-Analysis\stemmed_dataframe.csv')


# Load your sentiment analysis model
model = pickle.load(open('',"rb"))

# Define the Streamlit app
def main():
    st.title("Twitter Sentiment Analysis")

    # User input for Twitter handle or search query
    tweet=st.text_input("Write a Tweet😊 What is happening in your World?🌍")

    if st.button("Analyze"):
        # Perform sentiment analysis
        model=LogisticRegression(max_iter=1000)
        prediction=model.predict(model, tweet)

        # Display results
        st.write("Sentiment:", prediction)

if __name__ == "__main__":
    main()


