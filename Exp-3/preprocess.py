"""Program 1 (text preprocessing) - produces cleaned_reviews.csv for Experiment 3.

Steps: lowercase -> word_tokenize -> remove stopwords / punctuation tokens
       -> PorterStemmer -> WordNetLemmatizer -> join with spaces.
"""
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    if pd.isna(text):
        return ""
    tokens = word_tokenize(str(text).lower())
    tokens = [
        t for t in tokens
        if t not in stop_words and not all(ch in string.punctuation for ch in t)
    ]
    tokens = [lemmatizer.lemmatize(stemmer.stem(t)) for t in tokens]
    return " ".join(tokens)


df = pd.read_csv("customer_reviews.csv")
df["Cleaned_Review"] = df["Review"].apply(clean_text)
df[["ReviewID", "Review", "Cleaned_Review"]].to_csv("cleaned_reviews.csv", index=False)
print("Preprocessing complete. Saved cleaned_reviews.csv")
print(df[["ReviewID", "Cleaned_Review"]])
