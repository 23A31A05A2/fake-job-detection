# ============================================
# FAKE JOB DETECTION SYSTEM
# Using NLP + TF-IDF + Logistic Regression
# ============================================

# Import libraries
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# 1. DOWNLOAD NLTK STOPWORDS
# ============================================

nltk.download('stopwords')

# ============================================
# 2. LOAD DATASET
# ============================================

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
data.info()

# ============================================
# 3. HANDLE MISSING VALUES
# ============================================

data = data.fillna('')

# ============================================
# 4. COMBINE IMPORTANT TEXT COLUMNS
# ============================================

data['text'] = (
    data['title'] + " " +
    data['company_profile'] + " " +
    data['description'] + " " +
    data['requirements'] + " " +
    data['benefits']
)

# ============================================
# 5. TARGET VARIABLE
# ============================================

# 0 = Real Job
# 1 = Fake Job

y = data['fraudulent']

# ============================================
# 6. TEXT PREPROCESSING
# ============================================

stop_words = set(stopwords.words('english'))

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


print("\nCleaning text...")

data['text'] = data['text'].apply(clean_text)

# ============================================
# 7. TRAIN-TEST SPLIT
# ============================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    data['text'],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))

# ============================================
# 8. TF-IDF FEATURE EXTRACTION
# ============================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.7
)

print("\nCreating TF-IDF features...")

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("TF-IDF training shape:", X_train.shape)
print("TF-IDF testing shape:", X_test.shape)

# ============================================
# 9. TRAIN LOGISTIC REGRESSION MODEL
# ============================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("Model training completed!")

# ============================================
# 10. PREDICTION
# ============================================

y_pred = model.predict(X_test)

# ============================================
# 11. MODEL EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n============================================")
print("MODEL EVALUATION")
print("============================================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================
# 12. PREDICT NEW JOB POSTING
# ============================================

def predict_job(text):

    # Clean input
    cleaned = clean_text(text)

    # Convert text into TF-IDF
    vector = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(vector)[0]

    # Probability
    probability = model.predict_proba(vector)[0]

    if prediction == 0:

        print("\n================================")
        print("RESULT: REAL JOB")
        print("================================")
        print(
            f"Real Job Probability: {probability[0] * 100:.2f}%"
        )

    else:

        print("\n================================")
        print("RESULT: FAKE JOB")
        print("================================")
        print(
            f"Fake Job Probability: {probability[1] * 100:.2f}%"
        )


# ============================================
# 13. TEST EXAMPLE
# ============================================

sample_job = """
We are hiring data entry operators.
Work from home.
Earn 5000 dollars weekly with no experience required.
"""

predict_job(sample_job)
