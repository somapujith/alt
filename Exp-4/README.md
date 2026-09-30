# Experiment 4: Text Classification with Logistic Regression (Scikit-learn)

## Aim
Apply the Logistic Regression classifier using Scikit-learn in Python to classify textual data into predefined categories, demonstrated on sentiment analysis (Positive / Negative / Neutral) of customer reviews. The same pipeline applies to document classification.

## Files
| File | Description |
|------|-------------|
| `customer_reviews_with_sentiment.csv` | Hand-labelled dataset (60 reviews, columns `ReviewID, Review, Sentiment`) |
| `experiment4.py` | Training, evaluation and prediction script |
| `output.txt` | Real console output of one run |

## How to run
```
cd Exp-4
python experiment4.py
```
Requires Python 3, `pandas` and `scikit-learn`. On Windows set `PYTHONIOENCODING=utf-8` if the console fails to print emoji or the rupee sign.

## Dataset note
The lab manual's dataset was not available, so it was recreated. It has 60 reviews, balanced at 20 Positive, 20 Negative and 20 Neutral.
- Reviews 1-15 are the 15 reviews from Experiment 1 (`Exp-1/customer_reviews.csv`), now labelled.
- Reviews 16-60 are new, realistic product reviews written and labelled by hand.
- Neutral means informational or no-opinion text (specs, prices, contact details, URLs, shipping and return policy). Reviews 3 ("Battery backup is good, but delivery was late.") and 59 are mixed-sentiment and were labelled Neutral.
- **Label correction:** the manual labels "I don't recommend this product..." as Positive, which is wrong. It is labelled **Negative** here.

## Workflow
1. Load the CSV; `X = Review`, `y = Sentiment`.
2. Split with `train_test_split(test_size=0.2, random_state=42, stratify=y)`, giving 48 training and 12 test reviews (4 per class).
3. Build a `Pipeline`: `TfidfVectorizer(lowercase=True, stop_words="english")` then `LogisticRegression(max_iter=1000)`.
4. Fit on the training set and predict the test set.
5. Print a table of Review / Actual / Predicted, the accuracy, the classification report (`zero_division=0`) and the confusion matrix (labels Negative, Neutral, Positive).
6. Predict the sentiment of 3 new, unseen reviews.

## Results (from `output.txt`)
- Accuracy: **0.3333** (4 of 12 correct)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.20 | 0.25 | 0.22 | 4 |
| Neutral | 0.40 | 0.50 | 0.44 | 4 |
| Positive | 0.50 | 0.25 | 0.33 | 4 |
| macro avg | 0.37 | 0.33 | 0.33 | 12 |

Confusion matrix (rows = actual, columns = predicted):

|  | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1 | 2 | 1 |
| **Neutral** | 2 | 2 | 0 |
| **Positive** | 2 | 1 | 1 |

Predictions on new unseen reviews:
- "Absolutely fantastic phone, the camera and battery are superb!" -> Positive (confidence 0.43)
- "Terrible quality, it broke in two days. Very disappointed." -> Negative (confidence 0.48)
- "The box contains a charger and the warranty card." -> Neutral (confidence 0.48)

## Caveat
The dataset is very small: the model trains on only 48 short reviews and is scored on 12, so each test review is worth about 8 percentage points and the metrics are noisy. With 3 classes, chance level is about 33%, so 0.33 accuracy is no better than guessing. TF-IDF can only use words seen in training, and most test words (e.g. "stitching", "sound", "value") appear in reviews of both polarities or never in training, so the model has little signal to learn from. The three unseen-review predictions are correct, but their confidence is low (0.43 to 0.48), showing the probabilities are close to uniform. These numbers should not be read as a measure of how well the method works. A larger dataset (hundreds to thousands of labelled reviews), n-gram features and cross-validation would give a far more reliable and higher-scoring result.
