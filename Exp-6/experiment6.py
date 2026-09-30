# Experiment 6: Text Classification using Support Vector Machine (SVM)
# TF-IDF + linear SVC on a news dataset (Politics / Sports / Technology)

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load dataset (path is relative to this script so it runs from any directory)
DATA_PATH = Path(__file__).resolve().parent / "news_dataset.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df["category"].value_counts())
print()

# Features and labels
X = df["text"]
y = df["category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Pipeline: TF-IDF vectorization + SVM classifier
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", SVC(kernel="linear")),
])

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Actual vs Predicted
results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print("\nActual vs Predicted:\n", results)

# Predict new text
new_text = [
    "Artificial Intelligence is changing industries",
    "The football team won the championship",
    "The government passed a budget",
]
new_pred = model.predict(new_text)

print("\nNew Text Predictions:")
for sentence, category in zip(new_text, new_pred):
    print(sentence, "->", category)
