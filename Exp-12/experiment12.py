"""
Experiment 12: LDA Topic Modeling using Gensim

Discovers hidden topics in a small document collection with Latent Dirichlet
Allocation (LDA) and shows the dominant topic for each document.
"""

from gensim import corpora
from gensim.models import LdaModel

# ---------------------------------------------------------------
# Step 1: Sample documents (3 topics x 3 documents each)
# ---------------------------------------------------------------
documents = [
    "football team player match goal training stadium",
    "football player scored goal team championship match",
    "football training player team goal stadium",

    "machine learning artificial intelligence model data algorithm",
    "deep learning neural network artificial intelligence model",
    "machine learning data science model prediction algorithm",

    "online shopping customer product price discount delivery",
    "ecommerce shopping product online payment",
    "online shopping customers products discounts delivery",
]

# ---------------------------------------------------------------
# Step 2: Tokenization
# ---------------------------------------------------------------
texts = [
    document.lower().split()
    for document in documents
]

# ---------------------------------------------------------------
# Step 3: Create dictionary (word <-> id mapping)
# ---------------------------------------------------------------
dictionary = corpora.Dictionary(texts)

# ---------------------------------------------------------------
# Step 4: Convert documents into Bag-of-Words vectors
# ---------------------------------------------------------------
corpus = [
    dictionary.doc2bow(text)
    for text in texts
]

# ---------------------------------------------------------------
# Step 5: Create and train the LDA model
# ---------------------------------------------------------------
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    random_state=42,
    passes=20,
    iterations=100,
)

print("Number of documents:", len(texts))
print("Vocabulary size:", len(dictionary))

# ---------------------------------------------------------------
# Step 6: Display the discovered topics (top words per topic)
# ---------------------------------------------------------------
print("\nDiscovered Topics:")

for topic_id, topic_words in lda_model.show_topics(
        num_topics=3,
        num_words=6,
        formatted=False):

    words = [
        word
        for word, probability in topic_words
    ]

    print(
        f"Topic {topic_id}: "
        + ", ".join(words)
    )

# ---------------------------------------------------------------
# Step 7: Display the dominant topic for every document
# ---------------------------------------------------------------
print("\nDominant topic for each document:")

for i, bow in enumerate(corpus, 1):

    distribution = lda_model.get_document_topics(
        bow,
        minimum_probability=0,
    )

    dominant_topic = max(
        distribution,
        key=lambda x: x[1],
    )

    print(
        f"Document {i}: "
        f"Topic {dominant_topic[0]} "
        f"(probability={dominant_topic[1]:.3f})"
    )
