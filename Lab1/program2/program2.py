# ============================================================

# LAB EXPERIMENT-2

# Title : Tokenization, Lemmatization and Stemming

# Course: Applied Machine Learning for Text Analysis

# ============================================================

 

import pandas as pd

import nltk

 

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

 

# Download required resources (only first time)

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('omw-1.4')

nltk.download('punkt_tab')

 

 

# ------------------------------------------------------------

# Load Dataset

# ------------------------------------------------------------

 

df = pd.read_csv("cleaned_reviews.csv")

 

print("="*80)

print("INPUT DATASET")

print("="*80)

print(df)

 

# ------------------------------------------------------------

# Create Objects

# ------------------------------------------------------------

 

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

 

# ------------------------------------------------------------

# Functions

# ------------------------------------------------------------

 

def sentence_tokenizer(text):

   return sent_tokenize(str(text))

 

def word_tokenizer(text):

   return word_tokenize(str(text))

 

def stemming(text):

   words = word_tokenize(str(text))

   return " ".join([stemmer.stem(word) for word in words])

 

def lemmatization(text):

   words = word_tokenize(str(text))

   return " ".join([lemmatizer.lemmatize(word) for word in words])

 

# ------------------------------------------------------------

# Apply NLP Operations

# ------------------------------------------------------------

 

df["Sentence_Tokens"] = df["Cleaned_Review"].apply(sentence_tokenizer)

 

df["Word_Tokens"] = df["Cleaned_Review"].apply(word_tokenizer)

 

df["Stemmed_Text"] = df["Cleaned_Review"].apply(stemming)

 

df["Lemmatized_Text"] = df["Cleaned_Review"].apply(lemmatization)

 

# ------------------------------------------------------------

# Display Results

# ------------------------------------------------------------

 

print("\n")

print("="*100)

print("TOKENIZATION, STEMMING & LEMMATIZATION RESULTS")

print("="*100)

 

for i in range(len(df)):

   print("\nReview:", df.loc[i, "Cleaned_Review"])

   print("Sentence Tokens :", df.loc[i, "Sentence_Tokens"])

   print("Word Tokens     :", df.loc[i, "Word_Tokens"])

   print("Stemmed Text    :", df.loc[i, "Stemmed_Text"])

   print("Lemmatized Text :", df.loc[i, "Lemmatized_Text"])

   print("-"*100)

 

# ------------------------------------------------------------

# Save Output

# ------------------------------------------------------------

 

df.to_csv("tokenized_reviews.csv", index=False)

 

print("\n")

print("="*80)

print("Processing Completed Successfully.")

print("Output saved as tokenized_reviews.csv")

print("="*80)

 