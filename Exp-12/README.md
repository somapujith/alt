# Experiment 12: LDA Topic Modeling using Gensim

## Aim
To discover hidden thematic structure in a small collection of text documents using Latent
Dirichlet Allocation (LDA) with the `gensim` library, and to identify the dominant topic for
each document.

## How to run
```bash
pip install gensim
python experiment12.py
```
The sample output was captured with `python3 experiment12.py > output.txt 2>&1`.

## Steps
1. Define 9 sample documents: 3 about football, 3 about machine learning / AI, and 3 about
   online shopping.
2. Tokenize each document by lowercasing and splitting on whitespace.
3. Build a `gensim.corpora.Dictionary` (word <-> id mapping) from the tokenized texts.
4. Convert each document into a Bag-of-Words vector with `dictionary.doc2bow`.
5. Train an `LdaModel` with `num_topics=3`, `random_state=42`, `passes=20`, `iterations=100`.
6. Print the top 6 words for each of the 3 discovered topics.
7. For each document, get its topic distribution with `get_document_topics(bow, minimum_probability=0)` and print the dominant (highest-probability) topic.

## Results (from `output.txt`)

| Stage | Value |
|---|---|
| Number of documents | 9 |
| Vocabulary size | 33 |

### Discovered topics

| Topic | Top words | Theme |
|---|---|---|
| Topic 0 | football, team, goal, player, stadium, training | Football and Sports |
| Topic 1 | learning, model, algorithm, intelligence, machine, data | Machine Learning and AI |
| Topic 2 | shopping, online, delivery, product, discount, customer | Online Shopping and E-commerce |

### Dominant topic per document

| Document | Dominant topic | Probability |
|---|---|---|
| Document 1 | Topic 0 (Football and Sports) | 0.916 |
| Document 2 | Topic 0 (Football and Sports) | 0.916 |
| Document 3 | Topic 0 (Football and Sports) | 0.905 |
| Document 4 | Topic 1 (Machine Learning and AI) | 0.916 |
| Document 5 | Topic 1 (Machine Learning and AI) | 0.916 |
| Document 6 | Topic 1 (Machine Learning and AI) | 0.916 |
| Document 7 | Topic 2 (Online Shopping and E-commerce) | 0.916 |
| Document 8 | Topic 2 (Online Shopping and E-commerce) | 0.888 |
| Document 9 | Topic 2 (Online Shopping and E-commerce) | 0.904 |

Every document is assigned, with high confidence (probability 0.888-0.916), to the topic
matching the theme it was written for: documents 1-3 (football) to Topic 0, documents 4-6
(machine learning) to Topic 1, and documents 7-9 (online shopping) to Topic 2.

## Interpretation
- **Topic 0 - Football and Sports:** dominated by `football, team, goal, player, stadium,
  training` - clearly the vocabulary of sports/match reporting.
- **Topic 1 - Machine Learning and AI:** dominated by `learning, model, algorithm,
  intelligence, machine, data` - clearly the vocabulary of ML/AI.
- **Topic 2 - Online Shopping and E-commerce:** dominated by `shopping, online, delivery,
  product, discount, customer` - clearly the vocabulary of e-commerce.

**LDA successfully identified the hidden thematic structures in the document collection.**

## Note
LDA is a probabilistic, iterative algorithm, so the numeric topic-word probabilities (and in
general which integer topic id gets which theme) can vary slightly across gensim versions or
machines even with a fixed `random_state`. This run (gensim 4.4.0) happened to assign Topic 0
to football, Topic 1 to machine learning, and Topic 2 to shopping, matching the lab manual's
reference output exactly; in general, always confirm the topic-to-theme mapping from the
actual top words printed, not from a fixed topic number.
