from flask import Flask, render_template, request
import pickle
import re
import nltk

app = Flask(__name__)

print("🚀 App is starting...")

# 🔹 Load model safely
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

# 🔹 Text cleaning (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# 🔹 Home route
@app.route('/')
def home():
    return render_template('index.html')

# 🔹 Prediction route
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    job_text = request.form['job_text']
    text = job_text.lower()

    # 🔴 Rule-based fraud detection (strong signals)
    fraud_keywords = [
        "registration fee", "payment required", "pay fee",
        "earn money fast", "no experience required",
        "whatsapp", "telegram", "click here",
        "limited slots", "work from home earn",
        "guaranteed income"
    ]

    for word in fraud_keywords:
        if word in text:
            return render_template(
                'index.html',
                prediction_text="FAKE JOB ❌ (rule-based detection)",
                color="red"
            )

    # 🔹 ML prediction (if no rule triggered)
    cleaned = clean_text(job_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "FAKE JOB ❌"
        color = "red"
    else:
        result = "REAL JOB ✅"
        color = "green"

    return render_template('index.html', prediction_text=result, color=color)

# 🔹 Run app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
