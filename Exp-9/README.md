# Experiment 9: Contextual Word Embeddings using BERT

## Aim
To generate contextual word embeddings with a pre-trained BERT model (`bert-base-uncased`) and show that the same word ("bank") gets different vectors in different sentences.

## How to run
```bash
pip install transformers torch numpy
python experiment9.py
```
The first run downloads `bert-base-uncased` (about 440 MB) into the default Hugging Face cache (outside this repo). Later runs use the cache. Model weights are not committed. The sample output was captured with `PYTHONIOENCODING=utf-8 python experiment9.py > output.txt`. Loader warnings (for the unused `cls.*` pre-training head weights) go to stderr and are harmless.

## Steps
1. Load the tokenizer and model, and call `model.eval()`. Hidden size is 768.
2. Tokenize two sentences in which "bank" has different meanings.
3. Run the model under `torch.no_grad()` and take `last_hidden_state`.
4. Pull out the vector at the "bank" token position in each sentence.
5. Compare the two "bank" vectors with cosine similarity.
6. Build sentence vectors by mean pooling over the token vectors.
7. Extra (not in the lab PDF): compare against a third sentence, "I deposited money in the bank."

## Results (from `output.txt`)
| Sentence | Tokens | Embedding shape |
|---|---|---|
| I went to the bank to deposit money. | `[CLS] i went to the bank to deposit money . [SEP]` (11) | `[1, 11, 768]` |
| The fisherman sat on the bank of the river. | `[CLS] the fisherman sat on the bank of the river . [SEP]` (12) | `[1, 12, 768]` |

- "bank" first 10 values, sentence 1 (money): `[0.7091 -0.259 -0.0186 -0.0936 1.2637 0.0223 -0.3096 0.9714 -0.1028 0.2012]`
- "bank" first 10 values, sentence 2 (river): `[-0.2521 -0.4272 -0.2305 0.1381 -0.4004 0.05 0.0232 1.675 0.0897 -0.5353]`
- **Cosine similarity between the two "bank" vectors: 0.4744**. It is well below 1, so the model represents the two senses differently.
- Sentence vectors (mean pooling) are 768-dimensional.
- Extra comparison: "bank" in "I deposited money in the bank." has similarity **0.9115** with the financial "bank" of sentence 1, but only **0.4677** with the river "bank" of sentence 2. So financial-vs-financial is much higher than financial-vs-river.

## What it shows: contextual vs static embeddings
- **Word2Vec / GloVe (static):** each word has one fixed vector, so "bank" would have the same vector in every sentence (cosine similarity 1.0) and would blend all its senses together.
- **BERT (contextual):** the vector for a token is computed by the whole Transformer from the entire sentence, so it changes with the surrounding words. "bank" near "deposit money" is close to other financial uses and far from "bank of the river".

## Note
Exact numbers depend on the library versions (this run: transformers 5.17.0, torch 2.14.0 CPU) and on the model checkpoint, so they may differ slightly on other setups. The qualitative result (different vectors for different senses) stays the same.
