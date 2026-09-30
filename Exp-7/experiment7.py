# Experiment 7: Text classification of bank customer reviews using Random Forest
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load the dataset (path is relative to this script, so it runs from any cwd)
DATA_PATH = Path(__file__).resolve().parent / "bank_reviews_dataset.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Preview")
print(df.head())

# Features and labels
X = df["text"]
y = df["label"]

# 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# TF-IDF + Random Forest pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Actual vs predicted
results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print("\nActual vs Predicted:")
print(results)

# Predictions on new reviews
new_reviews = [
    "The customer support was excellent.",
    "Hidden charges were very disappointing.",
    "Internet banking is easy to use.",
]
new_predictions = model.predict(new_reviews)

print("\nPredictions on New Reviews:")
for review, label in zip(new_reviews, new_predictions):
    print(f"{review} --> {label}")
