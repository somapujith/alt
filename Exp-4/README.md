# Experiment 4: Text Classification with Logistic Regression (Scikit-learn)

## Aim
Apply the Logistic Regression classifier using Scikit-learn in Python to classify textual data into predefined categories, demonstrated on sentiment analysis (Positive / Negative / Neutral) of customer reviews. The same pipeline applies to document classification.

## Files
| File | Description |
|------|-------------|
| `customer_reviews_with_sentiment.csv` | Hand-labelled dataset (180 reviews, columns `ReviewID, Review, Sentiment`) |
| `experiment4.py` | Training, evaluation and prediction script |
| `output.txt` | Real console output of one run |

## How to run
```
cd Exp-4
python experiment4.py
```
Requires Python 3, `pandas` and `scikit-learn`. On Windows set `PYTHONIOENCODING=utf-8` if the console fails to print emoji or the rupee sign.

## Dataset note
The lab manual's dataset was not available, so it was recreated by hand. It has **180 reviews**, balanced at 60 Positive, 60 Negative and 60 Neutral (no duplicate reviews).
- Reviews 1-15 are the 15 reviews from Experiment 1 (`Exp-1/customer_reviews.csv`), now labelled.
- Reviews 16-180 are new, hand-written product reviews (phones, laptops, audio, clothing, appliances, apps, books, delivery and service).
- Positive and Negative rows use clear sentiment vocabulary (great, excellent, love versus bad, worst, poor, waste) with varied wording.
- Neutral means informational or no-opinion text (specs, prices, contact details, URLs, shipping, payment and return policy). Reviews 3 ("Battery backup is good, but delivery was late.") and 59 are mixed-sentiment and were labelled Neutral.
- **Label correction:** the manual labels "I don't recommend this product..." as Positive, which is wrong. It is labelled **Negative** here.
- The model settings and the split (`test_size=0.2, random_state=42, stratify=y`) are exactly the manual's and were not tuned. An earlier 60-review version of the dataset gave only 0.33 accuracy, which is why the dataset was enlarged.

## Workflow
1. Load the CSV; `X = Review`, `y = Sentiment`.
2. Split with `train_test_split(test_size=0.2, random_state=42, stratify=y)`, giving 144 training and 36 test reviews (12 per class).
3. Build a `Pipeline`: `TfidfVectorizer(lowercase=True, stop_words="english")` then `LogisticRegression(max_iter=1000)`.
4. Fit on the training set and predict the test set.
5. Print a table of Review / Actual / Predicted, the accuracy, the classification report (`zero_division=0`) and the confusion matrix (labels Negative, Neutral, Positive).
6. Predict the sentiment of 3 new, unseen reviews.

## Results (from `output.txt`)
- Accuracy: **0.7222** (26 of 36 correct)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.69 | 0.92 | 0.79 | 12 |
| Neutral | 0.70 | 0.58 | 0.64 | 12 |
| Positive | 0.80 | 0.67 | 0.73 | 12 |
| macro avg | 0.73 | 0.72 | 0.72 | 36 |

Confusion matrix (rows = actual, columns = predicted):

|  | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 11 | 1 | 0 |
| **Neutral** | 3 | 7 | 2 |
| **Positive** | 2 | 2 | 8 |

Predictions on new unseen reviews:
- "Absolutely fantastic phone, the camera and battery are superb!" -> Positive (confidence 0.48)
- "Terrible quality, it broke in two days. Very disappointed." -> Negative (confidence 0.67)
- "The box contains a charger and the warranty card." -> Neutral (confidence 0.55)

## Caveat
The dataset is small and hand-written by the author, so it is cleaner and less varied than real customer reviews. With only 36 test reviews, each one is worth about 2.8 percentage points, so the metrics are noisy and would shift with a different split or different labelling choices. Mixed-sentiment and no-opinion texts are the hardest to separate (Neutral recall is lowest), and TF-IDF only knows words seen in training, so unseen words such as "book" or "headphones" can push a neutral review towards Positive or Negative. Confidence scores are moderate (0.48 to 0.67) because a regularised Logistic Regression on 144 short texts produces soft probabilities. These numbers show that the pipeline works, not how it would perform on real-world data. A larger real dataset, n-gram features and cross-validation would give a more reliable result.
