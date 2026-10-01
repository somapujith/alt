# Experiment 14: Explainable AI using SHAP

## Aim
Train a small TF-IDF + Logistic Regression sentiment classifier and use SHAP (SHapley Additive
exPlanations) to explain *why* the model made a given prediction, by attributing the prediction to
individual words.

## How to run
```bash
pip install shap scikit-learn numpy
python3 experiment14.py > output.txt 2>&1
```
This run produced no stderr output (no SHAP loader/masker warnings on this `shap`/`scikit-learn`
version), so stdout and stderr were safely combined into `output.txt`. If a different SHAP version
prints a background-data/masker warning to stderr, it is harmless and can be ignored or separated
the same way Exp-9 does (redirect stdout only).

## Steps
1. Define 10 short labelled texts (1 = positive, 0 = negative).
2. Vectorize the texts with `TfidfVectorizer`.
3. Train a `LogisticRegression` classifier on the TF-IDF features.
4. Build a `shap.LinearExplainer` around the trained linear model, using the TF-IDF matrix as the
   background/reference distribution.
5. Compute SHAP values for every document, then pull out the explanation for document 0
   (`"excellent amazing wonderful great product"`).
6. Sort that document's per-word SHAP contributions by absolute value and print the top 8.
7. Print the model's prediction and positive-class probability for the same document, alongside the
   explanation.

## Results (from `output.txt`)
```
TF-IDF matrix shape: (10, 24)
Model trained. Training accuracy: 1.0

Text explained:
excellent amazing wonderful great product

Top SHAP feature contributions:
excellent       +0.1224
amazing         +0.0874
great           +0.0874
wonderful       +0.0874
bad             +0.0421
terrible        +0.0421
good            -0.0418
unhappy         +0.0247

Prediction: Positive
Positive probability: 0.6124
```

| Word | SHAP value | Direction |
|---|---|---|
| excellent | +0.1224 | pushes toward Positive |
| amazing | +0.0874 | pushes toward Positive |
| great | +0.0874 | pushes toward Positive |
| wonderful | +0.0874 | pushes toward Positive |
| bad | +0.0421 | pushes toward Positive (absent from this doc, but shifts the baseline) |
| terrible | +0.0421 | pushes toward Positive (absent from this doc, but shifts the baseline) |
| good | -0.0418 | pushes toward Negative |
| unhappy | +0.0247 | pushes toward Positive |

The model correctly predicts **Positive** for `"excellent amazing wonderful great product"`, with a
positive-class probability of **0.6124**.

## Interpretation
SHAP answers the question "why did the model make this prediction?" by splitting the prediction into
additive per-feature contributions (Shapley values from cooperative game theory). For this document,
the strongly positive words `excellent`, `amazing`, `great`, and `wonderful` dominate and push the
prediction toward Positive, which matches intuition. Words that do not even appear in this document
(`bad`, `terrible`, `good`, `unhappy`) still get a nonzero SHAP value because `LinearExplainer`
measures each feature's contribution relative to its average (background) value across the whole
TF-IDF matrix — a word being *absent* when it is often present in negative documents can itself shift
the prediction, which is why `bad`/`terrible` show up with small positive contributions here.

This kind of explanation is useful anywhere a model's decision needs to be justified to a human:
sentiment analysis, spam detection, fake-news classification, medical text classification,
customer-review analysis, and explainable NLP systems generally.

## Note
SHAP values explain this specific trained model's behavior on this specific dataset — they are not
causal effects and should not be read as "this word causes positive sentiment" in general. Exact
values can differ slightly across `shap`/`scikit-learn` versions; this run used `shap` 0.52.0 and
reproduced the lab PDF's reference numbers exactly. `shap.LinearExplainer` may print a background/mean
warning to stderr on some versions — that is harmless and does not affect the results.
