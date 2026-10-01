"""Experiment 11: K-Means Text Clustering using TF-IDF."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Step 1: Sample documents
documents = [
    "smartphone camera processor battery mobile phone",
    "mobile phone smartphone camera internet connectivity",
    "smartphone processor battery camera technology",

    "football team player match goal stadium",
    "football player scored goals team championship",
    "football training player team fitness goals",

    "online shopping customer product price discount",
    "customers compare product prices shopping online",
    "ecommerce shopping products discount delivery customer"
]

# Step 2: Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)

# Step 3: Apply K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X)

# Step 4: Display results
print("Number of documents:", X.shape[0])
print("Number of TF-IDF features:", X.shape[1])

for i, label in enumerate(labels, 1):
    print(f"Document {i}: Cluster {label}")

# Step 5: Display important terms
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("\nTop terms in each cluster:")

for cluster in range(3):
    top_terms = [
        terms[index]
        for index in order_centroids[cluster, :5]
    ]

    print(
        f"Cluster {cluster}: "
        + ", ".join(top_terms)
    )

print("\nCluster sizes:")
print(pd.Series(labels).value_counts().sort_index())
