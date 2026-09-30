# Experiment 1 - Text Pre-processing with NLTK

**Aim:** Apply tokenization, lowercasing, stop-word removal, punctuation removal, stemming and lemmatization to prepare clean text for NLP tasks.

## Run
```
cd Exp-1
python experiment1.py
```
Input: `customer_reviews.csv` (ReviewID, Review). Output: `cleaned_reviews.csv` (adds `Cleaned_Review`); console output in `output.txt`.

## Steps
1. Lowercase the review
2. Tokenize with `word_tokenize`
3. Remove English stop words and punctuation tokens
4. Stem with `PorterStemmer`
5. Lemmatize with `WordNetLemmatizer`
6. Join tokens back into a sentence

## Sample result (real run)
| Review | Cleaned_Review |
|---|---|
| Worst product ever!!! Waste of money!!!! | worst product ever wast money |
| Battery backup is good, but delivery was late. | batteri backup good deliveri late |
| Delivery was on time, excellent service!! | deliveri time excel servic |

The manual's steps do not strip URLs, e-mails, HTML tags or emojis, so fragments such as `http`, `shop.com` or `😍` stay in some rows. See `output.txt` for all 15 rows.
