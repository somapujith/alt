"""Experiment 13: Text Classification Evaluation Metrics."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

# Step 1: Text dataset (8 positive reviews, 8 negative reviews)
reviews = [
    "excellent product great quality",
    "amazing service fast delivery",
    "happy purchase great quality",
    "product works perfectly excellent",
    "good value useful product",
    "fantastic quality wonderful service",
    "very good product and service",
    "excellent purchase fast delivery",
    "terrible product bad quality",
    "poor service slow delivery",
    "unhappy purchase poor quality",
    "product failed disappointing",
    "bad value useless product",
    "awful quality horrible service",
    "terrible service bad product",
    "poor quality disappointing purchase",
]

# 1 = Positive, 0 = Negative
y = np.array([1] * 8 + [0] * 8)

# Step 2: Train-test split (stratified so both classes appear in the test set)
X_train, X_test, y_train, y_test = train_test_split(
    reviews, y, test_size=0.30, random_state=42, stratify=y
)

# Step 3: Build a TF-IDF + Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(random_state=42)),
])

# Step 4: Train
model.fit(X_train, y_train)

# Step 5: Predictions and prediction probabilities
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Step 6: Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"], zero_division=0
    )
)

# ---------------------------------------------------------------------------
# EXTRA (not part of the lab PDF): show the confusion matrix as a labeled
# predicted-vs-actual table for easier reading.
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("\n" + "=" * 60)
print("EXTRA: CONFUSION MATRIX AS A LABELED TABLE (not in lab PDF)")
print("=" * 60)
print(f"{'':15}{'Predicted Negative':>20}{'Predicted Positive':>20}")
print(f"{'Actual Negative':15}{cm[0][0]:>20}{cm[0][1]:>20}")
print(f"{'Actual Positive':15}{cm[1][0]:>20}{cm[1][1]:>20}")

print("\n" + "=" * 60)
print("LAB COMPLETED SUCCESSFULLY")
print("=" * 60)
