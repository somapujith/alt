# Experiment 8: Word2Vec Embeddings with Gensim

## Aim

Train a skip-gram Word2Vec model with Gensim on a small toy corpus, inspect the learned word vectors, query the most similar words, measure the cosine similarity between word pairs, and save the trained model.

## Files

| File | Description |
|------|-------------|
| `experiment8.py` | The program (15-sentence inline corpus, so no dataset file is needed) |
| `output.txt` | Real captured output of one run (UTF-8) |
| `word2vec_gensim_model.model` | Saved Gensim model produced by the run (26 KB, no `.npy` sidecar files were needed for this small model) |

## How to run

```bash
cd Exp-8
python experiment8.py
```

Requires `gensim` and `numpy`. The model is saved next to the script (via `pathlib`), so the script works from any current directory. To reload it:

```python
from gensim.models import Word2Vec
model = Word2Vec.load("word2vec_gensim_model.model")
```

## Method

1. Build a 15-sentence corpus (royalty, people, capitals, fruit, animals, vehicles).
2. Tokenize with `sentence.lower().split()`.
3. Train `Word2Vec(vector_size=50, window=3, min_count=1, workers=1, sg=1, epochs=100, seed=42)`.
4. Print vocabulary size, embedding dimension and the first 10 values of the vector for `king`.
5. Print the top 3 `most_similar` words for `king`, `queen`, `apple`, `dog`.
6. Print cosine similarity for four word pairs.
7. Save the model.

### Key concepts

- **Skip-gram (`sg=1`)**: the model takes a centre word and learns to predict the words around it. It tends to work better than CBOW for small datasets and rare words.
- **`window=3`**: context size. Up to 3 words on each side of the centre word count as its context.
- **`vector_size=50`**: length of each word embedding (number of dimensions).
- **`min_count=1`**: keep every word, even those seen once (needed for such a tiny corpus).
- **`epochs=100`**: number of passes over the corpus. Many are used because there is very little data.
- **Reproducibility**: `workers=1` (single thread) and `seed=42` make training deterministic on the same machine and library version. Multi-threaded training or a different Gensim/NumPy version can give different vectors.

## Real results (from `output.txt`)

```
===== WORD2VEC MODEL INFORMATION =====
Vocabulary size: 42
Embedding dimension: 50

Embedding of 'king' (first 10 values):
[-0.0044  0.0101 -0.02    0.0194  0.0044 -0.0161 -0.0115  0.0025 -0.0107
 -0.0134]
```

Most similar words (top 3):

| Query | 1st | 2nd | 3rd |
|-------|-----|-----|-----|
| king  | son (0.5700) | palace (0.4508) | the (0.4473) |
| queen | royal (0.4381) | the (0.4339) | king (0.4274) |
| apple | the (0.3646) | son (0.3505) | girl (0.3414) |
| dog   | fruit (0.3648) | kingdom (0.3539) | is (0.3129) |

Semantic similarity:

| Pair | Cosine similarity |
|------|-------------------|
| (king, queen) | 0.427 |
| (man, woman) | -0.039 |
| (king, apple) | 0.280 |
| (dog, cat) | 0.046 |

## Notes and observations

- **The results are noisy and not reliably semantic.** With only 15 short sentences (42 unique words) there is far too little data for Word2Vec to learn real meaning. Vectors are mostly shaped by the random initialisation and by a handful of co-occurrences.
- Some results look sensible: `queen` is close to `royal` and `king`, and `king` is close to `palace`. Others are clearly not: `apple` is closest to `the`, `son` and `girl`, and `dog` is closest to `fruit` and `kingdom`.
- `man` and `woman` have a slightly negative similarity (-0.039) and `dog`/`cat` are almost unrelated (0.046), even though a human would expect high similarity. Meanwhile `king`/`apple` (0.280) is not far below `king`/`queen` (0.427), so the ranking of pairs is not trustworthy.
- Frequent function words such as `the`, `is` and `and` appear as neighbours because they co-occur with many words.
- Meaningful embeddings need a much larger corpus (millions of tokens) or pre-trained vectors (e.g. Google News Word2Vec, GloVe).
- The numbers above are from the run stored in `output.txt`. Rerunning on the same machine and library versions should give identical values; other environments may differ.
