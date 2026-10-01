"""
Experiment 15: End-to-End Text Classification Pipeline
using NLTK, Scikit-learn and Flask

Builds a small sentiment classifier (NLTK preprocessing + TF-IDF +
Logistic Regression), trains and tests it, then exposes it behind a
Flask REST API at POST /predict.

Steps 1-7 (preprocessing, dataset, pipeline, training, testing) run
unconditionally every time this module is executed, so the pipeline
output below is always produced. Step 8 only defines the Flask app;
the dev server itself is started by STEP 9, and only when this file
is run directly (`python experiment15.py`), not when it is imported.
"""

import re

import nltk
import numpy as np
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from flask import Flask, request, jsonify

# ---------------------------------------------------------------
# Step 1: Download NLTK stopwords
# ---------------------------------------------------------------
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# ---------------------------------------------------------------
# Step 2: Text preprocessing
# ---------------------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


# ---------------------------------------------------------------
# Step 3: Dataset
# ---------------------------------------------------------------
texts = [
    "excellent product good quality",
    "amazing service fast delivery",
    "great useful product",
    "happy purchase",
    "fantastic quality excellent service",

    "terrible product bad quality",
    "poor service slow delivery",
    "unhappy purchase",
    "disappointing useless product",
    "awful quality horrible service",
]

labels = np.array([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
])


# ---------------------------------------------------------------
# Step 4: Preprocess dataset
# ---------------------------------------------------------------
cleaned_texts = [preprocess_text(text) for text in texts]


# ---------------------------------------------------------------
# Step 5: Build ML pipeline
# ---------------------------------------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(random_state=42)),
])


# ---------------------------------------------------------------
# Step 6: Train model
# ---------------------------------------------------------------
model.fit(cleaned_texts, labels)


# ---------------------------------------------------------------
# Step 7: Test model
# ---------------------------------------------------------------
test_texts = [
    "excellent product with great quality",
    "terrible service and poor product",
]

for text in test_texts:
    cleaned = preprocess_text(text)
    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned])[0, 1]

    print("Original:", text)
    print("Cleaned :", cleaned)
    print("Prediction:", "Positive" if prediction == 1 else "Negative")
    print("Positive probability:", round(probability, 3))
    print()


# ---------------------------------------------------------------
# Step 8: Flask application
# ---------------------------------------------------------------
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    cleaned = preprocess_text(text)
    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned])[0, 1]

    return jsonify({
        "text": text,
        "prediction": "Positive" if prediction == 1 else "Negative",
        "positive_probability": round(float(probability), 4),
    })


# ---------------------------------------------------------------
# Step 9: Run Flask (only when executed directly, e.g. `python
# experiment15.py`; importing this module for the pipeline output
# above never starts the server)
# ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
