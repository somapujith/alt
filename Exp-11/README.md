# Experiment 11: K-Means Text Clustering

## Aim
To group a small collection of text documents into clusters using TF-IDF vectorization and the K-Means clustering algorithm, without using any document labels.

## How to run
```bash
pip install scikit-learn pandas
python experiment11.py
```
The sample output was captured with `PYTHONIOENCODING=utf-8 python3 experiment11.py > output.txt 2>&1`. There is no model download and no stderr noise for this script — scikit-learn and pandas run fully offline.

## Steps
1. Define 9 sample documents covering three topics: smartphones (3), football (3), and online shopping (3).
2. `TfidfVectorizer(stop_words="english")` converts the documents into TF-IDF vectors.
3. `KMeans(n_clusters=3, random_state=42, n_init=10)` fits the model and assigns each document to one of 3 clusters.
4. Print the cluster assignment for every document.
5. Sort each cluster's centroid weights to find and print its top 5 terms.
6. Print the number of documents in each cluster with `pandas.Series(labels).value_counts()`.

## Results (from `output.txt`)
| Document | Text (topic) | Cluster |
|---|---|---|
| 1 | smartphone camera processor battery mobile phone | 2 |
| 2 | mobile phone smartphone camera internet connectivity | 2 |
| 3 | smartphone processor battery camera technology | 2 |
| 4 | football team player match goal stadium | 0 |
| 5 | football player scored goals team championship | 0 |
| 6 | football training player team fitness goals | 0 |
| 7 | online shopping customer product price discount | 1 |
| 8 | customers compare product prices shopping online | 1 |
| 9 | ecommerce shopping products discount delivery customer | 1 |

- Number of documents: 9
- Number of TF-IDF features: 32
- Top terms per cluster:
  - **Cluster 0**: player, team, football, goals, training
  - **Cluster 1**: shopping, customer, discount, product, online
  - **Cluster 2**: smartphone, camera, battery, processor, phone
- Cluster sizes: Cluster 0 = 3, Cluster 1 = 3, Cluster 2 = 3

## Interpretation
| Cluster | Topic | Documents |
|---|---|---|
| 0 | Football / Sports | 3 |
| 1 | Online Shopping / E-commerce | 3 |
| 2 | Smartphones / Technology | 3 |

K-Means successfully grouped the documents into three meaningful clusters based on textual similarity.

## Note
This run used scikit-learn 1.9.1 and pandas 3.0.5 on Python 3.12.10. The exact cluster **numbers** (0/1/2) depend on `KMeans`'s internal centroid initialization order, so they can differ between library versions even with `random_state=42` fixed — the lab PDF's reference run happened to land on the same numbering used here (0=Football, 1=Shopping, 2=Tech). The **grouping itself** (which documents end up together) is identical to the PDF's expected output. Two of the 15 printed top-terms also differ slightly from the PDF reference: Cluster 0's 5th term is "training" here vs "championship" in the PDF, and Cluster 1's term order is "shopping, customer, discount, product, online" here vs "shopping, product, online, discount, customer" in the PDF — both are TF-IDF weighting ties/near-ties resolved slightly differently across versions, and do not change the clustering result or its interpretation. With newer pandas, `value_counts()` also prints a trailing `Name: count, dtype: int64` line, which is pandas 3.x's standard `Series` repr footer.
