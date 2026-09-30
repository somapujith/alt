"""Experiment 3: TF-IDF vectorization using Scikit-learn."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("cleaned_reviews.csv")
df["Cleaned_Review"] = df["Cleaned_Review"].fillna("")

print("Cleaned Reviews:")
print(df["Cleaned_Review"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Cleaned_Review"])

vocabulary = vectorizer.get_feature_names_out()
print("\nVocabulary:")
print(vocabulary)

tfidf_matrix = X.toarray()
print("\nTF-IDF Matrix:")
print(tfidf_matrix)

tfidf_df = pd.DataFrame(tfidf_matrix, columns=vocabulary)
tfidf_df.insert(0, "ReviewID", df["ReviewID"])
tfidf_df.to_csv("tfidf_features.csv", index=False)
print("\nTF-IDF features saved to tfidf_features.csv")

# Extra: top 3 highest-weighted terms per review
print("\nTop 3 TF-IDF terms per review:")
for i, row in enumerate(tfidf_matrix):
    top = sorted(range(len(row)), key=lambda j: (-row[j], vocabulary[j]))[:3]  # ties broken alphabetically
    terms = ", ".join(f"{vocabulary[j]} ({row[j]:.4f})" for j in top if row[j] > 0)
    print(f"Review {df['ReviewID'].iloc[i]}: {terms if terms else '(no terms)'}")
