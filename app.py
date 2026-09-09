import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords", quiet=True)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english"))


# Text cleaning function
def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Split words
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


# Streamlit page
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake Job Detection System")

st.write(
    "Enter a job posting below to check whether it is likely to be real or fake."
)

# Text input
job_text = st.text_area(
    "📄 Enter Job Description",
    height=250,
    placeholder="Paste the complete job description here..."
)


# Prediction
if st.button("🔎 Check Job"):

    if job_text.strip() == "":
        st.warning("Please enter a job description.")

    else:

        # Clean text
        cleaned_text = clean_text(job_text)

        # Convert to TF-IDF
        vector = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vector)[0]

        # Probability
        probability = model.predict_proba(vector)[0]

        if prediction == 1:

            st.error("⚠️ This job posting is likely FAKE.")

            st.write(
                f"Fake Probability: {probability[1] * 100:.2f}%"
            )

        else:

            st.success("✅ This job posting is likely REAL.")

            st.write(
                f"Real Probability: {probability[0] * 100:.2f}%"
            )
