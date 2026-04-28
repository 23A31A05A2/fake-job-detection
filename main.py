# Fake Job Detection System using Logistic Regression

# Import libraries
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("dataset.csv")

# Display first rows
print(data.head())

# Check dataset information
print(data.info())

# Fill missing values
data = data.fillna('')

# Combine important text columns
data['text'] = data['title'] + " " + data['company_profile'] + " " + data['description'] + " " + data['requirements'] + " " + data['benefits']

# Target column (0 = Real Job, 1 = Fake Job)
y = data['fraudulent']

# Text preprocessing function
stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Apply cleaning
data['text'] = data['text'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['text'])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# Function to predict new job description
def predict_job(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    if prediction == 0:
        print("This job posting is REAL")
    else:
        print("This job posting is FAKE")


# Test Example
sample_job = """
We are hiring data entry operators. Work from home.
Earn 5000 dollars weekly with no experience required.
"""

predict_job(sample_job)