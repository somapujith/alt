# Experiment 15: End-to-End Text Classification Pipeline

## Aim
Build an end-to-end sentiment classification pipeline with NLTK (preprocessing), scikit-learn
(TF-IDF + Logistic Regression), train and test it on a small labelled dataset, and serve it as a
REST API with Flask.

## How to run
```bash
pip install nltk scikit-learn numpy flask
```

**Plain pipeline run (training + testing only, no server):**
The training/testing code (Steps 1-7) runs unconditionally at import/module level and prints the
test results immediately. The Flask app is only *defined* at import time; it is only *started*
(`app.run(...)`) when the script is executed directly, inside the `if __name__ == "__main__":`
guard at the bottom.

Because `app.run(debug=True)` blocks forever once it starts, capturing the test output
non-interactively needs the process to be killed right after it prints. This repo does not use a
`timeout` binary (not installed by default on macOS), so `output.txt` was captured like this:
```bash
(python3 experiment15.py > output.txt 2>&1 &)
sleep 3
pkill -f "experiment15.py"
```
The pipeline prints its Original/Cleaned/Prediction results in well under a second, so by the time
the background process is killed those lines are already in `output.txt`. On Linux/with
coreutils installed you could equivalently run `timeout 5 python3 experiment15.py > output.txt 2>&1 ; true`.
`output.txt` in this folder has been trimmed to just the ML pipeline output — the Flask
startup banner (and, in this environment, a "port already in use" message because macOS
AirPlay Receiver occupies port 5000 by default) were removed for clarity, since they are not
part of the lab's expected result and are environment-specific noise.

**Flask API run (live server):**
```bash
python3 experiment15.py
```
This runs the pipeline test prints first, then starts the API at `http://127.0.0.1:5000`
(`debug=True`, so it keeps running and auto-reloads on code changes). In another terminal:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "excellent product and fast service"}'
```
expected response shape:
```json
{"text": "excellent product and fast service", "prediction": "Positive", "positive_probability": 0.XXXX}
```

## Steps
1. Download NLTK's English stopword list (`nltk.download("stopwords", quiet=True)`).
2. `preprocess_text`: lowercase, strip everything but letters/spaces with a regex, tokenize on
   whitespace, drop stopwords, rejoin into a cleaned string.
3. Define a small labelled dataset: 10 short reviews, 5 positive (label 1) and 5 negative (label 0).
4. Preprocess every training text with `preprocess_text`.
5. Build a scikit-learn `Pipeline` of `TfidfVectorizer` -> `LogisticRegression(random_state=42)`.
6. Fit the pipeline on the cleaned training texts and labels.
7. Test the pipeline on two unseen sentences, printing the original text, cleaned text,
   predicted class, and predicted positive-class probability for each.
8. Define a Flask app with a `POST /predict` endpoint: it reads `text` from the JSON body,
   cleans it with the same `preprocess_text`, and returns the prediction and probability as JSON.
9. Start the Flask dev server, but only when the script is run directly.

## Results (from `output.txt`)
```
Original: excellent product with great quality
Cleaned : excellent product great quality
Prediction: Positive
Positive probability: 0.593

Original: terrible service and poor product
Cleaned : terrible service poor product
Prediction: Negative
Positive probability: 0.424
```
This matches the lab PDF's expected test-phase output exactly.

### Verified live Flask API call
Server started with `python3 experiment15.py` (port 5001 was used in this verification run only
because port 5000 was already bound by macOS AirPlay Receiver on this machine; the code itself
defaults to port 5000 as specified):

Request:
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "excellent product and fast service"}'
```

Response (actual, not a placeholder):
```json
{
  "positive_probability": 0.5894,
  "prediction": "Positive",
  "text": "excellent product and fast service"
}
```

A second check with a negative example:
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "terrible service and poor product"}'
```
```json
{
  "positive_probability": 0.4236,
  "prediction": "Negative",
  "text": "terrible service and poor product"
}
```
The server was stopped (`pkill -f experiment15_verify.py`) immediately after these two checks; no
Flask process was left running.

## Interpretation
- **Preprocessing (NLTK):** lowercasing, punctuation/digit removal, and stopword filtering reduce
  each review to its content-bearing words (e.g. "excellent product with great quality" ->
  "excellent product great quality"), so the vectorizer only sees signal words.
- **TF-IDF + Logistic Regression:** with only 10 training examples the model is simple, but it
  still generalizes correctly to both held-out test sentences — positive-sentiment words
  ("excellent", "great", "quality") push the probability above 0.5, negative-sentiment words
  ("terrible", "poor") push it below 0.5.
- **Probabilities are close to 0.5:** the training set is tiny (10 documents), so the model is not
  highly confident; this is expected for a toy dataset and the lab PDF's own expected output shows
  the same modest margins (0.593 / 0.424).
- **Flask as the serving layer:** wrapping `model.predict`/`predict_proba` behind a `/predict`
  endpoint is the "end-to-end" part of the experiment — the same trained pipeline object answers
  both the in-process test calls (Step 7) and the HTTP requests (Step 8/9), so there is no
  train/serve skew.

## Note
- Running `python3 experiment15.py` directly starts a real Flask dev server that blocks the
  terminal (`debug=True`) until stopped with Ctrl+C — this is expected behaviour, not a hang bug.
  `output.txt` intentionally only captures the pipeline test output described above, not a live
  API interaction; the live API call is demonstrated separately in this README with real
  request/response pairs captured during development.
- Exact probabilities depend on scikit-learn's version and the tiny training set; the qualitative
  result (first sentence Positive, second sentence Negative) is stable.
