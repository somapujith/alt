# Experiment 3: TF-IDF Vectorization using Scikit-learn

## Aim
Apply TF-IDF vectorization using Scikit-learn in Python to calculate the importance of words within a collection of documents (customer reviews).

## Files
| File | Purpose |
|---|---|
| `customer_reviews.csv` | Raw input (ReviewID, Review) |
| `preprocess.py` | Program 1: text preprocessing, writes `cleaned_reviews.csv` |
| `cleaned_reviews.csv` | ReviewID, Review, Cleaned_Review |
| `experiment3.py` | TF-IDF vectorization, writes `tfidf_features.csv` |
| `tfidf_features.csv` | ReviewID + one TF-IDF column per vocabulary term |
| `output.txt` | Console output of `experiment3.py` |

## How to run (in this order, from inside `Exp-3/`)
```
python preprocess.py     # customer_reviews.csv -> cleaned_reviews.csv
python experiment3.py    # cleaned_reviews.csv  -> tfidf_features.csv
```
Requires pandas, nltk (with punkt, stopwords, wordnet data) and scikit-learn.

## Workflow
`customer_reviews.csv` -> Program 1 preprocessing (lowercase, tokenize, stop-word/punctuation removal, Porter stemming, WordNet lemmatization) -> `cleaned_reviews.csv` -> Bag of Words -> TF-IDF -> `tfidf_features.csv`

## TF-IDF explained
TF-IDF scores how important a word is to a document relative to the whole collection.

- Term Frequency: `TF(t, d) = (count of t in d) / (total terms in d)`
- Inverse Document Frequency: `IDF(t) = log(N / df(t))`, where N is the number of documents and df(t) the number of documents containing t.
- `TF-IDF(t, d) = TF(t, d) x IDF(t)`

Words frequent in one review but rare across all reviews get high weights; words common to many reviews get low weights. Scikit-learn's `TfidfVectorizer` uses the smoothed variant `IDF(t) = ln((1 + N) / (1 + df(t))) + 1` and then L2-normalises each row, so every review vector has unit length (this is why a review with two terms shows 0.7071 for each).

## Output (real, from `output.txt`; the full 15 x 51 TF-IDF matrix is omitted here, see `output.txt` and `tfidf_features.csv`)
```
Cleaned Reviews:
0     absolut love mobil 😍 work perfectli visit http...
1                         worst product ever wast money
2                     batteri backup good deliveri late
3                       email u support shop.com assist
4                            b excel /b qualiti product
5                                         price ₹25,999
6                                       phone awesom 😊😊
7                          visit www.flipkart.com offer
8                                  camera qualiti 10/10
9                                 n't recommend product
10                             laptop amaz fast perform
11                                      call 9876543210
12                visit websit http //example.com today
13                             bad packag product damag
14                           deliveri time excel servic
Name: Cleaned_Review, dtype: str

Vocabulary:
['10' '25' '9876543210' '999' 'absolut' 'amaz' 'amazon' 'assist' 'awesom'
 'backup' 'bad' 'batteri' 'call' 'camera' 'com' 'damag' 'deliveri' 'email'
 'ever' 'example' 'excel' 'fast' 'flipkart' 'good' 'http' 'in' 'laptop'
 'late' 'love' 'mobil' 'money' 'offer' 'packag' 'perfectli' 'perform'
 'phone' 'price' 'product' 'qualiti' 'recommend' 'servic' 'shop' 'support'
 'time' 'today' 'visit' 'wast' 'websit' 'work' 'worst' 'www']


[... TF-IDF Matrix: 15 rows x 51 columns, see output.txt ...]

TF-IDF features saved to tfidf_features.csv

Top 3 TF-IDF terms per review:
Review 1: absolut (0.3460), amazon (0.3460), in (0.3460)
Review 2: ever (0.4717), money (0.4717), wast (0.4717)
Review 3: backup (0.4586), batteri (0.4586), good (0.4586)
Review 4: assist (0.4662), email (0.4662), shop (0.4662)
Review 5: excel (0.6138), qualiti (0.6138), product (0.4965)
Review 6: 25 (0.5774), 999 (0.5774), price (0.5774)
Review 7: awesom (0.7071), phone (0.7071)
Review 8: flipkart (0.4879), offer (0.4879), www (0.4879)
Review 9: 10 (0.8338), camera (0.4169), qualiti (0.3620)
Review 10: recommend (0.8183), product (0.5748)
Review 11: amaz (0.5000), fast (0.5000), laptop (0.5000)
Review 12: 9876543210 (0.7071), call (0.7071)
Review 13: example (0.4492), today (0.4492), websit (0.4492)
Review 14: bad (0.5350), damag (0.5350), packag (0.5350)
Review 15: servic (0.5339), time (0.5339), deliveri (0.4636)
```
