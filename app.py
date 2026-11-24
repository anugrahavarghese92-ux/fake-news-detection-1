
import streamlit as st
import joblib
import re

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# Load trained model and vectorizer
model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    text-align: center;
    color: #172554;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="title">📰 Fake News Detection</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">An NLP and Machine Learning Based Application</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

st.subheader("🔍 Analyse a News Article")
st.write("Enter an English news article below to predict whether it is Fake or Real.")

news = st.text_area(
    "News Article",
    placeholder="Paste your news article here...",
    height=250
)

if st.button("Predict News", use_container_width=True):
    if news.strip():
        cleaned_news = clean_text(news)
        vectorized_news = tfidf.transform([cleaned_news])
        prediction = model.predict(vectorized_news)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 0:
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

    else:
        st.warning("Please enter a news article before predicting.")

st.divider()

st.caption(
    "Note: This application predicts news based on patterns learned "
    "from the training dataset. It is not a substitute for independent fact-checking."
)
