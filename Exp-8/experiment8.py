"""
Experiment 8: Word2Vec Embeddings with Gensim (Skip-gram)

Trains a Word2Vec model on a small 15-sentence toy corpus, inspects the
learned embeddings, finds similar words, computes pairwise similarity and
saves the model next to this script.
"""

from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

# ---------------------------------------------------------------------------
# 1. Toy corpus
# ---------------------------------------------------------------------------
corpus = [
    "king queen prince princess royal palace",
    "king rules the kingdom and queen rules the palace",
    "prince is the son of king and princess is the daughter of queen",
    "man woman boy girl",
    "man and woman are people",
    "boy and girl are children",
    "paris france london england",
    "paris is the capital of france",
    "london is the capital of england",
    "apple orange banana fruit",
    "apple and orange are fruit",
    "dog cat animal pet",
    "dog and cat are common pets",
    "car bus train vehicle transport",
    "car bus and train are vehicles",
]

# ---------------------------------------------------------------------------
# 2. Tokenization
# ---------------------------------------------------------------------------
tokenized_sentences = [sentence.lower().split() for sentence in corpus]

# ---------------------------------------------------------------------------
# 3. Train Word2Vec (skip-gram, single worker + fixed seed => reproducible)
# ---------------------------------------------------------------------------
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    sg=1,
    epochs=100,
    seed=42,
)

# ---------------------------------------------------------------------------
# 4. Model information
# ---------------------------------------------------------------------------
print("===== WORD2VEC MODEL INFORMATION =====")
print("Vocabulary size:", len(model.wv.index_to_key))
print("Embedding dimension:", model.vector_size)
print("\nEmbedding of 'king' (first 10 values):")
print(np.round(model.wv["king"][:10], 4))

# ---------------------------------------------------------------------------
# 5. Most similar words
# ---------------------------------------------------------------------------
print("\n===== MOST SIMILAR WORDS =====")
for word in ["king", "queen", "apple", "dog"]:
    print(f"\nTop 3 words similar to '{word}':")
    for similar_word, score in model.wv.most_similar(word, topn=3):
        print(f"  {similar_word}: {score:.4f}")

# ---------------------------------------------------------------------------
# 6. Semantic similarity between word pairs
# ---------------------------------------------------------------------------
print("\n===== SEMANTIC SIMILARITY =====")
word_pairs = [
    ("king", "queen"),
    ("man", "woman"),
    ("king", "apple"),
    ("dog", "cat"),
]
for w1, w2 in word_pairs:
    s = model.wv.similarity(w1, w2)
    print(f"Similarity({w1}, {w2}) = {s:.3f}")

# ---------------------------------------------------------------------------
# 7. Save model (next to this script, independent of the current directory)
# ---------------------------------------------------------------------------
model_path = Path(__file__).resolve().parent / "word2vec_gensim_model.model"
model.save(str(model_path))
print("\nWord2Vec model saved successfully.")
