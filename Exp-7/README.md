# Experiment 7 - Random Forest Text Classification on Bank Customer Reviews

## Aim
Classify bank customer reviews as Positive or Negative using TF-IDF features and a Random Forest classifier, and evaluate it with accuracy, a classification report and a confusion matrix.

## How to run
```
python experiment7.py
```
The script reads `bank_reviews_dataset.csv` from its own folder, so it works from any working directory. It needs pandas and scikit-learn. The captured output is in `output.txt`.

## Files
- `experiment7.py` - the experiment program (settings exactly as in the lab manual).
- `bank_reviews_dataset.csv` - dataset with columns `text` and `label`.
- `output.txt` - real output of the run.

## Dataset
Hand-written (synthetic) bank customer reviews: **149 rows** (74 Positive, 75 Negative), instead of the 15-row dataset in the lab PDF (which gave only 20% accuracy). Topics include fees and charges, mobile/net banking, customer support, loans, cards, branches, wait times, ATMs and complaint handling. No duplicate texts, and the three new-review sentences used for prediction are not in the dataset. The dataset was written once, before running, and was not edited after seeing any score.

## Method
1. Load the CSV, print the preview (`df.head()`).
2. `X = text`, `y = label`; `train_test_split(test_size=0.30, random_state=42)` gives 104 training and 45 test reviews.
3. Pipeline: `TfidfVectorizer()` (defaults) followed by `RandomForestClassifier(n_estimators=100, random_state=42)`.
4. Fit on the training set, predict on the test set, and report the metrics.
5. Predict labels for three new reviews.

## Results (from `output.txt`)
- **Accuracy: 0.5778 (57.8%)** on 45 test reviews (26 correct).
- Confusion matrix (rows = actual, columns = predicted; order Negative, Positive):

| | Pred Negative | Pred Positive |
|---|---|---|
| **Actual Negative** | 15 | 7 |
| **Actual Positive** | 12 | 11 |

- Negative: precision 0.56, recall 0.68, F1 0.61 (support 22). Positive: precision 0.61, recall 0.48, F1 0.54 (support 23). Macro avg F1 0.57.
- Predictions on new reviews:
  - The customer support was excellent. --> Negative
  - Hidden charges were very disappointing. --> Negative
  - Internet banking is easy to use. --> Negative

## Notes and interpretation
- Accuracy improved from 33% (50 rows, 15 test) and 20% (the PDF's 15 rows) to about 58% with 149 rows, which is above the 50% chance level for a balanced problem but still weak.
- Caveat: the dataset is still small and synthetic/hand-written, so the scores are indicative only. With 45 test reviews, one review changes accuracy by about 2.2 points, so a single split is noisy.
- The model leans towards Negative: it labelled all three new reviews Negative, so only "Hidden charges were very disappointing." is right (the other two are clearly positive). The reviews use varied wording, so most informative words appear only once or twice in training, and Random Forest on sparse TF-IDF features has little to learn from.
- No tuning was done: the model settings are exactly those of the manual, and the dataset was not adjusted in response to scores.
- Possible improvements: much more data, stop-word removal and n-grams, cross-validation, or a linear model (Logistic Regression / Naive Bayes), which usually handles sparse text better on small data.
