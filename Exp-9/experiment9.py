"""
Experiment 9: Contextual Word Embeddings using BERT

Shows that the SAME word ("bank") receives DIFFERENT vectors depending on
the sentence it appears in (unlike static embeddings such as Word2Vec).
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------
# Step 1: Load the pre-trained BERT model and tokenizer
# ---------------------------------------------------------------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

print("BERT model loaded successfully!")
print("Model:", model_name)
print("Hidden size:", model.config.hidden_size)

# ---------------------------------------------------------------
# Step 2: Define sentences where "bank" has different meanings
# ---------------------------------------------------------------
sentences = [
    "I went to the bank to deposit money.",
    "The fisherman sat on the bank of the river.",
]

# ---------------------------------------------------------------
# Step 3: Tokenize and generate contextual embeddings
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TOKENIZATION AND EMBEDDING GENERATION")
print("=" * 60)

results = []
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print("\nSentence:", sentence)
    print("Tokens:", tokens)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state
    print("Embeddings shape:", embeddings.shape)

    results.append((tokens, embeddings[0]))

# ---------------------------------------------------------------
# Step 4: Extract the contextual embedding of the word "bank"
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("CONTEXTUAL EMBEDDING OF 'BANK'")
print("=" * 60)

bank_vectors = []
for i, (tokens, embeddings) in enumerate(results):
    idx = tokens.index("bank")
    vec = embeddings[idx]
    bank_vectors.append(vec)
    print(f"\nSentence {i + 1}: {sentences[i]}")
    print(f"Position of 'bank': {idx}")
    print("First 10 values:", np.round(vec[:10].numpy(), 4))

# ---------------------------------------------------------------
# Step 5: Compare the two "bank" vectors (contextual similarity)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("CONTEXTUAL SIMILARITY")
print("=" * 60)

similarity = torch.nn.functional.cosine_similarity(
    bank_vectors[0].unsqueeze(0), bank_vectors[1].unsqueeze(0)
).item()
print("Cosine similarity between 'bank' (money) and 'bank' (river):",
      round(similarity, 4))

# ---------------------------------------------------------------
# Step 6: Sentence representations (mean pooling over tokens)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SENTENCE REPRESENTATIONS")
print("=" * 60)

sentence_vectors = []
for i, (tokens, embeddings) in enumerate(results):
    sent_vec = embeddings.mean(dim=0)
    sentence_vectors.append(sent_vec)
    print(f"\nSentence {i + 1}: {sentences[i]}")
    print("Sentence vector dimension:", sent_vec.shape[0])
    print("First 10 values:", np.round(sent_vec[:10].numpy(), 4))

# ---------------------------------------------------------------
# EXTRA (not part of the lab PDF): a third sentence with the financial
# sense of "bank" to show financial-vs-financial > financial-vs-river.
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("EXTRA: THIRD SENTENCE COMPARISON (not in lab PDF)")
print("=" * 60)

extra_sentence = "I deposited money in the bank."
inputs = tokenizer(extra_sentence, return_tensors="pt")
extra_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
with torch.no_grad():
    extra_emb = model(**inputs).last_hidden_state[0]
extra_bank = extra_emb[extra_tokens.index("bank")]

print("\nSentence 3:", extra_sentence)
print("Tokens:", extra_tokens)

cos = torch.nn.functional.cosine_similarity
sim_fin_fin = cos(extra_bank.unsqueeze(0), bank_vectors[0].unsqueeze(0)).item()
sim_fin_river = cos(extra_bank.unsqueeze(0), bank_vectors[1].unsqueeze(0)).item()
print("Similarity 'bank' (sentence 3) vs 'bank' (sentence 1, financial):",
      round(sim_fin_fin, 4))
print("Similarity 'bank' (sentence 3) vs 'bank' (sentence 2, river):",
      round(sim_fin_river, 4))
print("Financial-vs-financial > financial-vs-river:", sim_fin_fin > sim_fin_river)

print("\n" + "=" * 60)
print("LAB COMPLETED SUCCESSFULLY")
print("=" * 60)
