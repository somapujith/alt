# Experiment 1: Text pre-processing using NLTK
# Tokenization, lowercasing, stop-word removal, punctuation removal,
# stemming and lemmatization.

# Import pandas for reading and saving CSV files
import pandas as pd

# Import NLTK libraries
import nltk
import string

# Import tokenization, stop words, stemming and lemmatization
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Read the customer reviews dataset
df = pd.read_csv("customer_reviews.csv")

# Create objects for stop words, stemming and lemmatization
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# Define a function for text preprocessing
def preprocess_text(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Remove stop words and punctuation
    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    # Apply stemming
    stemmed_words = [
        stemmer.stem(word)
        for word in tokens
    ]

    # Apply lemmatization
    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in stemmed_words
    ]

    # Join the words back into a single sentence
    return " ".join(lemmatized_words)


# Apply preprocessing to the Review column
df["Cleaned_Review"] = df["Review"].apply(preprocess_text)

# Save the cleaned reviews to a new CSV file
df.to_csv("cleaned_reviews.csv", index=False)

# Display the original and cleaned reviews
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
print(df[["Review", "Cleaned_Review"]])

# Display the output file name
print("\nCleaned reviews saved as cleaned_reviews.csv")
