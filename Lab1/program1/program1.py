import pandas as pd 
import re 
import string

df = pd.read_csv("customer_reviews.csv")

print("="*80)
print("Original Dataset")
print("="*80)
print(df)


# define text cleaning function 


def clean_text(text):

    text = str(text)

    text = re.sub(r'<.*?>', '', text)  # remove HTML tags

    # Remove URLs

    text = re.sub(r'http\S+|www\S+', '', text)

 

   # Remove Email IDs

    text = re.sub(r'\S+@\S+', '', text)

 

   # Remove Numbers

    text = re.sub(r'\d+', '', text)

 

   # Remove Punctuation

    text = text.translate(str.maketrans('', '', string.punctuation))

 

   # Remove Special Characters and Emojis

    text = re.sub(r'[^a-zA-Z\s]', '', text)

 

   # Remove Extra Spaces

    text = re.sub(r'\s+', ' ', text).strip()

 

    return text

# ------------------------------------------------------------

# Step 3: Apply Cleaning Function

# ------------------------------------------------------------

 

df["Cleaned_Review"] = df["Review"].apply(clean_text)

 

# ------------------------------------------------------------

# Step 4: Display Results

# ------------------------------------------------------------

 

print("\n")

print("="*80)

print("CLEANED DATASET")

print("="*80)

 

print(df[["Review","Cleaned_Review"]])

 

# ------------------------------------------------------------

# Step 5: Save Cleaned Dataset

# ------------------------------------------------------------

 

df.to_csv("cleaned_reviews.csv", index=False)

 

print("\n")

print("="*80)

print("Text Cleaning Completed Successfully.")

print("Cleaned dataset saved as 'cleaned_reviews.csv'")

print("="*80)