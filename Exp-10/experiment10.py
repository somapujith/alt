"""Experiment 10: Principal Component Analysis (PCA) on TF-IDF text vectors."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: save the figure instead of showing it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent

# Step 1: Sample documents
documents = [
    "machine learning algorithms analyze data and make predictions",
    "deep learning uses neural networks for complex data analysis",
    "natural language processing analyzes human language and text",
    "text classification uses machine learning for document prediction",
    "neural networks are useful for deep learning applications",
    "natural language processing is used for sentiment analysis",
    "machine learning improves prediction accuracy from data",
    "text mining extracts useful information from documents",
    "deep learning and machine learning are important in AI",
    "sentiment analysis classifies text into positive and negative categories",
    "document classification is a common natural language processing task",
    "data analysis and visualization help understand machine learning results",
]

# Step 2: TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_sparse = vectorizer.fit_transform(documents)

print("===== BEFORE PCA =====")
print("Number of documents :", X_sparse.shape[0])
print("Number of features  :", X_sparse.shape[1])
print("TF-IDF matrix shape :", X_sparse.shape)

# Step 3: Convert sparse matrix to dense array
X = X_sparse.toarray()

# Step 4: Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("\n===== AFTER PCA =====")
print("Reduced matrix shape       :", X_pca.shape)
print("Explained variance ratio   :", np.round(pca.explained_variance_ratio_, 4))
print("Total variance preserved   :", round(pca.explained_variance_ratio_.sum(), 4))

# Step 5: Display transformed features
result = pd.DataFrame(
    X_pca,
    columns=["Principal_Component_1", "Principal_Component_2"],
    index=[f"Document_{i}" for i in range(1, len(documents) + 1)],
)
print("\n===== PCA TRANSFORMED FEATURES =====")
print(result.round(4).to_string())

# Step 6: Visualization
plt.figure(figsize=(9, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=80)
for i, (x, y) in enumerate(X_pca):
    plt.annotate(f"D{i + 1}", (x, y), textcoords="offset points", xytext=(6, 6))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of TF-IDF Text Features")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_DIR / "pca_visualization.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# EXTRA (not part of the PDF program): cumulative explained variance using all
# components, printed only.
# ---------------------------------------------------------------------------
full_pca = PCA().fit(X)
cumulative = np.cumsum(full_pca.explained_variance_ratio_)
print("\n===== EXTRA: CUMULATIVE EXPLAINED VARIANCE (ALL COMPONENTS) =====")
for k, c in enumerate(cumulative, start=1):
    print(f"PC1..PC{k:<2d}: {c:.4f}")
