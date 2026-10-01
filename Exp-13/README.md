# Experiment 13 - Model Evaluation Metrics

## Aim
Train a simple text classifier (TF-IDF + Logistic Regression) on a small labeled review dataset and
evaluate it with the standard classification metrics: accuracy, precision, recall, F1-score, ROC-AUC,
the confusion matrix, and the full classification report.

## Files
- `experiment13.py` - the program
- `output.txt` - captured console output of a real run

## How to run
```
pip install scikit-learn numpy
python experiment13.py
```

## Steps
1. Define 16 short reviews, 8 labeled positive (`1`) and 8 labeled negative (`0`).
2. Split into train/test with `train_test_split(test_size=0.30, random_state=42, stratify=y)`, so the
   5-row test set keeps the same class balance as the full set.
3. Build a `Pipeline` of `TfidfVectorizer` followed by `LogisticRegression(random_state=42)`.
4. Train the pipeline on `X_train, y_train`.
5. Get class predictions with `predict()` and positive-class probabilities with `predict_proba()[:, 1]`.
6. Compute accuracy, precision, recall, F1-score, and ROC-AUC, then print the confusion matrix and the
   full `classification_report`.
7. Extra (not in the lab PDF): print the confusion matrix again as a labeled predicted-vs-actual table.

## Results (from `output.txt`)
| Metric | Value |
|---|---|
| Accuracy | 0.6000 |
| Precision | 0.5000 |
| Recall | 1.0000 |
| F1-Score | 0.6667 |
| ROC-AUC | 1.0000 |

### Confusion matrix (predicted vs actual)
|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 1 | 2 |
| **Actual Positive** | 0 | 2 |

### Classification report
```
              precision    recall  f1-score   support

    Negative       1.00      0.33      0.50         3
    Positive       0.50      1.00      0.67         2

    accuracy                           0.60         5
   macro avg       0.75      0.67      0.58         5
weighted avg       0.80      0.60      0.57         5
```

## Interpretation
The 5-row test set has 3 actual negatives and 2 actual positives. The model predicts "Positive" too
freely: it catches both positives (recall = 1.00 for Positive) but misclassifies 2 of the 3 negatives
as positive, so negative recall is only 0.33 and overall precision is just 0.50. ROC-AUC is a perfect
1.00 because the model's predicted *probabilities* rank every positive review above every negative one,
even though its default 0.5 decision threshold does not separate them perfectly. Accuracy alone (0.60)
hides all of this: it does not say which class is being missed, and a model that always predicted
"Positive" would also score 0.60 here. This small demonstration dataset produces different metric
values, illustrating why accuracy alone should not be used to evaluate a classification model.

## Note
Exact numbers depend on library versions (this run: scikit-learn 1.9.1, numpy 2.5.1). With only 16
training examples and a 5-row test set, the metrics are sensitive to the exact train/test split;
`random_state=42` and `stratify=y` were used so the result is reproducible on any machine with the
same library versions. The numbers obtained here match the lab PDF's reference output exactly.
