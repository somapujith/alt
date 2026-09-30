# Experiment 2 - Bag-of-Words (BoW) with Scikit-learn

**Aim:** Convert text documents into numerical feature vectors with `CountVectorizer`, for text classification and information retrieval.

## Run
```
cd Exp-2
python experiment2.py
```
Input: `cleaned_reviews.csv` (produced by Experiment 1 and copied here). Output: `bow_features.csv` (one row per review, one column per word, cells are word counts); console output in `output.txt`.

## Workflow
`customer_reviews.csv` -> Experiment 1 preprocessing -> `cleaned_reviews.csv` -> `CountVectorizer` -> Bag-of-Words matrix -> `bow_features.csv`

## Real result
15 reviews x 51 vocabulary words. Each cell is how many times the word occurs in the review (e.g. `10` appears twice in the "10/10" review). Word order is ignored, which is the defining property of BoW. Full vocabulary and matrix are in `output.txt`.
