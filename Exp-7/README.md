# Experiment 7 - Random Forest Text Classification on Bank Customer Reviews

## Aim
Classify bank customer reviews as Positive or Negative using TF-IDF features and a Random Forest classifier, and evaluate it with accuracy, a classification report and a confusion matrix.

## How to run
```
python experiment7.py
```
The script reads `bank_reviews_dataset.csv` from its own folder, so it works from any working directory. It needs pandas and scikit-learn. The captured output is in `output.txt`.

## Files
- `experiment7.py` - the experiment program.
- `bank_reviews_dataset.csv` - dataset with columns `text` and `label`.
- `output.txt` - real output of the run.

## Dataset
Hand-written (synthetic) bank customer reviews: **50 rows** (25 Positive, 25 Negative), instead of the 15-row dataset in the lab PDF. Topics include branch service, mobile/net banking, loans, cards, fees, ATMs and complaint handling. The three new-review sentences used for prediction are not in the dataset.

## Method
1. Load the CSV, print the preview (`df.head()`).
2. `X = text`, `y = label`; `train_test_split(test_size=0.30, random_state=42)` gives 35 training and 15 test reviews.
3. Pipeline: `TfidfVectorizer()` followed by `RandomForestClassifier(n_estimators=100, random_state=42)`.
4. Fit on the training set, predict on the test set, and report the metrics.
5. Predict labels for three new reviews.

## Results (from `output.txt`)
- **Accuracy: 0.3333 (33.3%)** on 15 test reviews (5 correct).
- Confusion matrix (rows = actual, columns = predicted; order Negative, Positive):

| | Pred Negative | Pred Positive |
|---|---|---|
| **Actual Negative** | 3 | 5 |
| **Actual Positive** | 5 | 2 |

- Negative: precision 0.38, recall 0.38, F1 0.38 (support 8). Positive: precision 0.29, recall 0.29, F1 0.29 (support 7).
- Predictions on new reviews:
  - The customer support was excellent. --> Positive
  - Hidden charges were very disappointing. --> Positive
  - Internet banking is easy to use. --> Negative

## Notes and interpretation
- Accuracy is at or below chance level for a balanced two-class problem. The lab PDF's own run on 15 rows reached only 20%, so a poor score is expected for this setup.
- The main cause is the tiny data size: with 35 training reviews, TF-IDF vocabulary is very sparse, and the reviews are varied, so the test reviews share few informative words with the training ones. The model mostly sees words such as "the", "and", "my", which carry no sentiment.
- On 15 test samples, one review changes accuracy by 6.7 points, so the number is very noisy and should not be over-interpreted.
- Two of the three new reviews are misclassified (as one would expect from this weak model); "Hidden charges were very disappointing." is predicted Positive even though it is clearly negative.
- Possible improvements: a much larger dataset, stop-word removal and n-grams in TF-IDF, cross-validation instead of a single split, or a linear model (Logistic Regression / Naive Bayes), which usually handles sparse text better than Random Forest on small data.
