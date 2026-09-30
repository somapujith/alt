# Experiment 2: Bag-of-Words (BoW) model using Scikit-learn
# Reads cleaned_reviews.csv produced by Experiment 1 and converts the
# cleaned reviews into numerical feature vectors with CountVectorizer.

# Import pandas for reading the CSV file
import pandas as pd

# Import CountVectorizer for Bag-of-Words
from sklearn.feature_extraction.text import CountVectorizer

# Read the cleaned reviews file created by Experiment 1
df = pd.read_csv("cleaned_reviews.csv")

# A review can become empty after cleaning, so replace missing values
df["Cleaned_Review"] = df["Cleaned_Review"].fillna("")

# Display the cleaned reviews
print("Cleaned Reviews:")
print(df["Cleaned_Review"])

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Convert cleaned reviews into Bag-of-Words numerical vectors
X = vectorizer.fit_transform(df["Cleaned_Review"])

# Get the vocabulary (unique words)
vocabulary = vectorizer.get_feature_names_out()

# Display the vocabulary
print("\nVocabulary:")
print(vocabulary)

# Convert the BoW sparse matrix into a normal array
bow_matrix = X.toarray()

# Display the Bag-of-Words matrix
print("\nBag-of-Words Matrix:")
print(bow_matrix)

# Create a DataFrame for the BoW matrix
bow_df = pd.DataFrame(
    bow_matrix,
    columns=vocabulary
)

# Save the BoW feature vectors to a CSV file
bow_df.to_csv("bow_features.csv", index=False)

# Display confirmation
print("\nBag-of-Words features saved as bow_features.csv")
