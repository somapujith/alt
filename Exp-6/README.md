# Experiment 6: Text Classification using SVM (TF-IDF + linear SVC)

## Aim
Classify short news sentences into **Politics**, **Sports** or **Technology** using TF-IDF features and a Support Vector Machine (`SVC(kernel="linear")`) inside a scikit-learn `Pipeline`.

## How to run
```
python experiment6.py
```
The script reads `news_dataset.csv` from its own folder, so it works from any working directory. Requires `pandas` and `scikit-learn`. Captured output is in `output.txt`.

## Dataset
`news_dataset.csv` has columns `text,category`. It is **synthetic**: 51 hand-written news-style headlines (17 per category), not taken from a real corpus. The lab PDF used only 15 rows (and got 40% accuracy); the larger set is used here so the results are more meaningful. With `test_size=0.30`, `random_state=42`: 35 training rows, 16 test rows.

## Method
1. Load CSV, `X = text`, `y = category`.
2. `train_test_split(test_size=0.30, random_state=42)`.
3. `Pipeline([("tfidf", TfidfVectorizer()), ("svm", SVC(kernel="linear"))])`, fit on train.
4. Predict on test set; report accuracy, classification report, confusion matrix, Actual vs Predicted table.
5. Predict three new sentences.

## Results (real run)
**Accuracy: 0.6875** (11 of 16 correct)

```
              precision    recall  f1-score   support

    Politics       1.00      0.33      0.50         6
      Sports       0.71      1.00      0.83         5
  Technology       0.57      0.80      0.67         5

    accuracy                           0.69        16
   macro avg       0.76      0.71      0.67        16
weighted avg       0.78      0.69      0.66        16
```

Confusion matrix (rows = actual, columns = predicted; order Politics, Sports, Technology):
```
[[2 1 3]
 [0 5 0]
 [0 1 4]]
```

New text predictions:
```
Artificial Intelligence is changing industries -> Technology
The football team won the championship -> Sports
The government passed a budget -> Politics
```

## Notes
- Accuracy (68.75%) is well above the PDF's 40% on 15 rows, but still modest because the test set is only 16 sentences, so a single mistake moves accuracy by about 6 points.
- Politics has perfect precision but low recall (0.33): 4 of 6 Politics test sentences were assigned to Sports or Technology. Politics sentences share few distinctive words with the small training set, so TF-IDF has little to match on.
- Sports is easiest (recall 1.00) thanks to distinctive words such as match, win, team.
- All three new sentences are classified correctly, which is encouraging, though three examples is not a robust test.
- More data, stop-word removal or n-grams would likely help; results vary with the split seed on a dataset this small.
