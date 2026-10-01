"""
Experiment 14: Explainable AI using SHAP

Trains a small TF-IDF + Logistic Regression sentiment classifier and uses
SHAP (SHapley Additive exPlanations) to explain WHY the model made a given
prediction, by attributing the prediction to individual words.
"""

import numpy as np
import shap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------
# Step 1: Sample labelled texts (1 = positive, 0 = negative)
# ---------------------------------------------------------------
texts = [
    "excellent amazing wonderful great product",
    "good useful quality fast service",
    "fantastic excellent product performance",
    "happy with purchase",

    "bad terrible awful product",
    "poor quality disappointing service",
    "horrible useless product experience",
    "unhappy purchase",

    "excellent quality good service",
    "terrible quality bad service"
]

labels = np.array([
    1, 1, 1, 1,
    0, 0, 0, 0,
    1, 0
])

# ---------------------------------------------------------------
# Step 2: TF-IDF vectorization
# ---------------------------------------------------------------
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    texts
)

print("TF-IDF matrix shape:", X.shape)

# ---------------------------------------------------------------
# Step 3: Train a Logistic Regression classifier
# ---------------------------------------------------------------
model = LogisticRegression(
    random_state=42
)

model.fit(
    X,
    labels
)

print("Model trained. Training accuracy:", round(model.score(X, labels), 4))

# ---------------------------------------------------------------
# Step 4: Build a SHAP explainer for the linear model
# ---------------------------------------------------------------
explainer = shap.LinearExplainer(
    model,
    X
)

shap_values = explainer(X)

# ---------------------------------------------------------------
# Step 5: Explain the first document
# ---------------------------------------------------------------
document_index = 0

feature_names = (
    vectorizer
    .get_feature_names_out()
)

values = shap_values.values[
    document_index
]

contributions = sorted(
    zip(feature_names, values),
    key=lambda x: abs(x[1]),
    reverse=True
)

print("\n" + "=" * 60)
print("SHAP EXPLANATION FOR DOCUMENT", document_index)
print("=" * 60)

print("\nText explained:")
print(texts[document_index])

print("\nTop SHAP feature contributions:")

for word, value in contributions[:8]:

    print(
        f"{word:15s} "
        f"{value:+.4f}"
    )

# ---------------------------------------------------------------
# Step 6: Show the model's prediction alongside the explanation
# ---------------------------------------------------------------
prediction = model.predict(
    X[document_index]
)[0]

probability = model.predict_proba(
    X[document_index]
)[0, 1]

print(
    "\nPrediction:",
    "Positive" if prediction == 1
    else "Negative"
)

print(
    "Positive probability:",
    round(probability, 4)
)

print("\n" + "=" * 60)
print("LAB COMPLETED SUCCESSFULLY")
print("=" * 60)
