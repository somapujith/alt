# ============================================================
# LAB EXPERIMENT-5
# Title : Multinomial Naive Bayes Text Classification (Spam / Ham)
# Course: Applied Machine Learning for Text Analysis
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------
# Step 1: Read the dataset
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "spam_ham_text_dataset.csv"))

# ------------------------------------------------------------
# Step 2: Display the dataset
# ------------------------------------------------------------
print("========== DATASET ==========")
print(df.to_string())
print("\nClass distribution:")
print(df["category"].value_counts())

# ------------------------------------------------------------
# Step 3: Separate features (X) and labels (y)
# ------------------------------------------------------------
X = df["text"]
y = df["category"]

# ------------------------------------------------------------
# Step 4: Split into training (70%) and testing (30%) sets
#         stratify=y keeps the Ham/Spam ratio the same in both
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ------------------------------------------------------------
# Step 5: Build the pipeline
#         CountVectorizer -> word-count features
#         MultinomialNB   -> classifier suited to count data
# ------------------------------------------------------------
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# ------------------------------------------------------------
# Step 6: Train the model
# ------------------------------------------------------------
model.fit(X_train, y_train)

# ------------------------------------------------------------
# Step 7: Predict on the test set
# ------------------------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------------------------
# Step 8: Display the prediction results
# ------------------------------------------------------------
results = pd.DataFrame({
    "Text": X_test.values,
    "Actual": y_test.values,
    "Predicted": y_pred
})
print("\n========== PREDICTION RESULTS ==========")
print(results.to_string())

# ------------------------------------------------------------
# Step 9: Accuracy and classification report
# ------------------------------------------------------------
print("\n========== ACCURACY ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# Step 10: Confusion matrix (rows = actual, columns = predicted)
# ------------------------------------------------------------
print("========== CONFUSION MATRIX ==========")
cm = confusion_matrix(y_test, y_pred, labels=["Ham", "Spam"])
print(pd.DataFrame(cm,
                   index=["Actual Ham", "Actual Spam"],
                   columns=["Predicted Ham", "Predicted Spam"]))

# ------------------------------------------------------------
# Extra A: Top 5 most indicative words per class
#          Score = log P(word | class) - log P(word | other class)
# ------------------------------------------------------------
vectorizer = model.named_steps["vectorizer"]
classifier = model.named_steps["classifier"]
words = np.array(vectorizer.get_feature_names_out())
log_prob = classifier.feature_log_prob_          # shape (n_classes, n_words)
classes = list(classifier.classes_)

print("\n========== TOP 5 INDICATIVE WORDS PER CLASS ==========")
for i, cls in enumerate(classes):
    other = 1 - i                                # two classes only
    score = log_prob[i] - log_prob[other]
    top = np.argsort(score)[::-1][:5]
    print(f"{cls}:")
    for idx in top:
        print(f"  {words[idx]:<12} log P(w|{cls}) = {log_prob[i][idx]:.3f}  "
              f"(log-ratio vs other class = {score[idx]:.3f})")

# ------------------------------------------------------------
# Extra B: Predict 3 new, unseen messages
# ------------------------------------------------------------
new_messages = [
    "Win a free prize now click to claim your reward",
    "Please review the attached report before the meeting",
    "Can you send me the lecture notes tomorrow",
]
new_pred = model.predict(new_messages)
new_proba = model.predict_proba(new_messages)

print("\n========== PREDICTIONS ON NEW MESSAGES ==========")
for msg, pred, proba in zip(new_messages, new_pred, new_proba):
    prob_str = ", ".join(f"P({c})={p:.3f}" for c, p in zip(classes, proba))
    print(f"Message   : {msg}")
    print(f"Predicted : {pred}   [{prob_str}]\n")
