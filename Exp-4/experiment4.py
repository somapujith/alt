"""
Experiment 4: Text classification (sentiment analysis) with Logistic Regression
using Scikit-learn.

Pipeline: TF-IDF vectoriser -> Logistic Regression.
Classes : Positive / Negative / Neutral
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 1. Load the labelled dataset
df = pd.read_csv("customer_reviews_with_sentiment.csv", encoding="utf-8")
print("Dataset shape:", df.shape)
print("Class distribution:\n", df["Sentiment"].value_counts(), "\n")

X = df["Review"]
y = df["Sentiment"]

# 2. Stratified train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}  Test size: {len(X_test)}\n")

# 3. Pipeline: TF-IDF features + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("logistic_regression", LogisticRegression(max_iter=1000)),
])

# 4. Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Test-set predictions
results = pd.DataFrame({
    "Review": X_test.values,
    "Actual": y_test.values,
    "Predicted": y_pred,
})
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)
print("Predictions on the test set:")
print(results.to_string(index=False), "\n")

# 6. Evaluation
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4), "\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

labels = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Confusion Matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=labels, columns=labels), "\n")

# 7. Predict on new, unseen reviews
new_reviews = [
    "Absolutely fantastic phone, the camera and battery are superb!",
    "Terrible quality, it broke in two days. Very disappointed.",
    "The box contains a charger and the warranty card.",
]
new_pred = model.predict(new_reviews)
proba = model.predict_proba(new_reviews)
print("Predictions for new unseen reviews:")
for text, label, p in zip(new_reviews, new_pred, proba):
    print(f'  "{text}" -> {label} (confidence {p.max():.2f})')
