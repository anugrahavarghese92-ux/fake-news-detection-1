import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load('svm_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Clean news text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# App design
st.title("Fake News Detection")
st.write("Enter a news article to check whether it is Fake or Real.")

news = st.text_area("Enter News Text")

if st.button("Predict"):
    if news.strip():
        cleaned_news = clean_text(news)
        vectorized_news = tfidf.transform([cleaned_news])
        prediction = model.predict(vectorized_news)[0]

        if prediction == 0:
            st.error("FAKE NEWS")
        else:
            st.success("REAL NEWS")
    else:
        st.warning("Please enter news text.")