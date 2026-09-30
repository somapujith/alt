# Experiment 5 - Multinomial Naive Bayes Text Classification

## Aim
Apply the Multinomial Naive Bayes classifier using Scikit-learn in Python to classify text documents (here: spam vs. ham messages; the same pipeline works unchanged for news categories).

## Files
| File | Purpose |
|------|---------|
| `spam_ham_text_dataset.csv` | Dataset (columns `text`, `category` = Ham / Spam) |
| `experiment5.py` | The experiment, written in the manual's 10-step style plus two extras |
| `output.txt` | Real console output of one run |

## How to run
```
cd Exp-5
python experiment5.py
```
Requires `pandas`, `numpy`, `scikit-learn`. On Windows set `PYTHONIOENCODING=utf-8` if redirecting output to a file. Results are deterministic (`random_state=42`).

## Dataset note
The manual's `spam_ham_text_dataset.csv` was not available, so it was recreated. The manual describes 20 messages (10 spam / 10 ham); this dataset is **extended to 40 rows (20 Ham / 20 Spam)** so the 30% test split holds 12 messages instead of 6 and is more meaningful. The sample messages quoted in the manual are included verbatim.

## Workflow
1. Read the CSV with pandas.
2. Display the dataset and class balance.
3. `X = df["text"]`, `y = df["category"]`.
4. `train_test_split(test_size=0.30, random_state=42, stratify=y)` -> 28 train / 12 test, class ratio preserved.
5. `Pipeline([("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())])`.
6. `fit` on the training data (the vectorizer learns the vocabulary and the classifier learns class priors and per-class word probabilities).
7. `predict` on the test data.
8. Show a Text / Actual / Predicted table.
9. `accuracy_score` and `classification_report`.
10. `confusion_matrix(labels=["Ham", "Spam"])`.
- Extra A: top 5 most indicative words per class from `feature_log_prob_`, scored as `log P(w|class) - log P(w|other class)`.
- Extra B: predictions (with probabilities) for 3 new unseen messages.

## Real output (from `output.txt`)
Split: 28 training / 12 testing samples.

```
Accuracy: 1.0

              precision    recall  f1-score   support

         Ham       1.00      1.00      1.00         6
        Spam       1.00      1.00      1.00         6

    accuracy                           1.00        12
   macro avg       1.00      1.00      1.00        12
weighted avg       1.00      1.00      1.00        12

             Predicted Ham  Predicted Spam
Actual Ham               6               0
Actual Spam              0               6
```

Top indicative words:
```
Ham : the, meeting, me, when, of
Spam: now, claim, offer, win, free
```

Predictions on new unseen messages:
| Message | Predicted | Probabilities |
|---------|-----------|---------------|
| Win a free prize now click to claim your reward | Spam | P(Spam)=1.000 |
| Please review the attached report before the meeting | Ham | P(Ham)=1.000 |
| Can you send me the lecture notes tomorrow | Ham | P(Ham)=1.000 |

## Why Multinomial NB suits count features
`CountVectorizer` turns each document into a vector of word counts. Multinomial NB models a document as counts drawn from a multinomial distribution over the vocabulary, one per class: it estimates `P(word | class)` from the total word counts in that class (with Laplace smoothing, `alpha=1`, so unseen words do not give zero probability) and classifies with `argmax [ log P(class) + sum(count_w * log P(w | class)) ]`. This matches count data directly, is very fast, works well with high-dimensional sparse vectors, and its learned log-probabilities are easy to inspect (as in Extra A). Gaussian NB would assume continuous, bell-shaped features, which word counts are not.

## Observations and caveats
- 100% accuracy is **not** a reliable estimate of real-world performance. The dataset is tiny (40 messages, 12 test), the spam messages share very obvious vocabulary (free, now, claim, win, offer), and one wrongly classified message would change accuracy by about 8 points. Real spam filters need thousands of messages and cross-validation.
- Top "Ham" words include stop words such as "the", "me", "of", because no stop-word removal was applied and the tiny sample makes function words look discriminative. Using `CountVectorizer(stop_words="english")` or more data would give more meaningful ham words.
- The 3 new-message predictions are very confident (probability ~1.0). Naive Bayes probabilities are typically over-confident because of the independence assumption, so treat them as scores rather than calibrated probabilities.
