# Bi-encoders

A bi-encoder is a neural retrieval architecture that encodes queries and documents
independently using two separate (or shared weight) encoder networks, producing one
dense vector per query and one dense vector per document. Relevance is then computed
as the similarity between these two vectors. Because documents are encoded offline
and independently of queries, bi-encoders enable efficient dense retrieval over large
corpora at query time.

## Intuition

The core constraint in first-stage retrieval is speed. At query time you have
milliseconds to retrieve from millions of documents. A cross-encoder that jointly
reads query and document together is far too slow — you would need one full BERT
forward pass per document per query.

The bi-encoder solves this with a simple insight: encode documents once at index
time and store their vectors. At query time, encode only the query (one forward
pass), then find the nearest document vectors using ANN search. The expensive
encoding work is front-loaded to index time, making query time fast regardless
of corpus size.

The tradeoff is accuracy — because query and document are encoded independently,
there is no direct interaction between query tokens and document tokens during
encoding. The vectors must capture everything needed for relevance without seeing
each other.

## Architecture

```
Query encoder:
    q_vec = Encoder_Q([CLS] query tokens [SEP])   → 768-dim vector

Document encoder:
    d_vec = Encoder_D([CLS] doc tokens [SEP])     → 768-dim vector

Relevance score:
    score(q, d) = sim(q_vec, d_vec)
```

sim is typically dot product or cosine similarity. If vectors are L2-normalized,
dot product and cosine similarity are equivalent.

### Shared vs separate encoders

- **Shared weights** — same encoder for both query and document. Simpler, fewer
  parameters, often performs similarly to separate encoders.
- **Separate weights** — different encoders for query and document. Allows the model
  to learn query-specific and document-specific representations independently. DPR
  uses separate encoders.

In practice, sentence-transformers models typically use a single shared encoder
with mean pooling, which works well across most retrieval tasks.

## Training Bi-encoders

### Objective

Pull together vectors of relevant (query, document) pairs and push apart vectors
of irrelevant pairs. This is metric learning / contrastive learning.

### InfoNCE loss (in-batch negatives)

Given a batch of B (query, positive_doc) pairs:

```
For each query qᵢ:
    positive: dᵢ+
    negatives: all other dⱼ+ where j ≠ i

L = (1/B) × Σᵢ -log( exp(sim(qᵢ, dᵢ+) / τ) /
                      Σⱼ exp(sim(qᵢ, dⱼ+) / τ) )
```

Larger batch sizes = more negatives per query = stronger training signal. This
is why bi-encoder training typically uses large batch sizes (512-4096).

### Triplet loss

Older approach, still used in some settings:

```
L = max(0, sim(q, d-) - sim(q, d+) + margin)
```

Ensures the positive document scores higher than the negative by at least margin.
Simpler but less effective than InfoNCE with large batches.

### Hard negative mining

Random negatives are easy. The model quickly learns to separate clearly irrelevant
documents. Hard negatives — documents that are retrieved by BM25 or an earlier
retriever but are not relevant — force the model to make finer distinctions.

Standard training recipe:

1. Train initial bi-encoder with in-batch random negatives
2. Use trained model to retrieve top-k candidates for all training queries
3. Label candidates — relevant ones are positives, others are hard negatives
4. Retrain with hard negatives
5. Repeat (iterative hard negative mining)

This iterative approach significantly improves retrieval quality.

### Knowledge distillation from cross-encoder

A cross-encoder is more accurate than a bi-encoder but too slow for retrieval.
Use the cross-encoder as a teacher — train the bi-encoder to mimic cross-encoder
relevance scores:

```
Teacher (cross-encoder):  score_CE(q, d)    → soft labels
Student (bi-encoder):     score_BE(q, d)    → trained to match teacher

L = KL_divergence(softmax(score_CE), softmax(score_BE))
```

This distillation approach closes much of the accuracy gap between bi-encoder and
cross-encoder while preserving the bi-encoder's speed advantage.

## Pooling Strategies

How the sequence of token vectors is collapsed into one fixed-size vector matters:

### CLS token

```
vec = encoder_output[0]    # position 0 is always [CLS]
```

Simple, fast. Works well when the model is specifically fine-tuned with CLS pooling.

### Mean pooling

```
vec = mean(encoder_output[non_padding_tokens])
```

Generally outperforms CLS pooling for semantic similarity. Requires applying the
attention mask to exclude padding tokens from the mean.

### Max pooling

```
vec = max(encoder_output, dim=token_dim)
```

Takes the maximum activation across all token positions per dimension. Less common
than mean pooling for retrieval.

### Weighted mean pooling

Weight each token by its attention score or TF-IDF weight before averaging.
Gives more weight to semantically important tokens.

sentence-transformers uses mean pooling with attention mask by default — this is
the most widely adopted approach.

## Prominent Bi-encoder Models

| Model                      | Base model    | Training data    | Notes                 |
| -------------------------- | ------------- | ---------------- | --------------------- |
| DPR                        | BERT-base     | NQ, TriviaQA     | Separate encoders, QA |
| TAS-B                      | DistilBERT    | MS MARCO         | Topic-aware sampling  |
| ANCE                       | RoBERTa-base  | MS MARCO         | Async hard negatives  |
| sentence-transformers      | Various       | Multi-task       | General purpose       |
| msmarco-distilbert-base-v4 | DistilBERT    | MS MARCO         | Efficient, strong     |
| all-MiniLM-L6-v2           | MiniLM        | Multi-task       | Fast, lightweight     |
| E5                         | bert-base     | Large multi-task | Strong zero-shot      |
| GTE                        | BERT variants | Large scale      | Strong across BEIR    |

For most retrieval tasks, start with `all-MiniLM-L6-v2` (fast) or `E5-base`
(strong) from sentence-transformers.

## Bi-encoder vs Cross-encoder

This comparison is central to understanding the neural IR architecture:

| Property              | Bi-encoder               | Cross-encoder             |
| --------------------- | ------------------------ | ------------------------- | ---------- | -------- |
| Query-doc interaction | None during encoding     | Full joint attention      |
| Document encoding     | Offline, precomputed     | Online, per query         |
| Query time complexity | O(1) encode + ANN search | O(                        | candidates | ) × BERT |
| Accuracy              | Lower                    | Higher                    |
| Suitable for          | First-stage retrieval    | Second-stage reranking    |
| Corpus size supported | Millions to billions     | Hundreds to thousands     |
| Latency               | ~10ms                    | ~500ms for 100 candidates |

The standard production pipeline uses both:

```
Bi-encoder → top-1000 candidates → cross-encoder → top-10 final results
```

## The Independence Assumption — Why It Hurts

The bi-encoder's fundamental limitation is that query and document vectors are
computed without seeing each other. Consider:

Query: "What Python libraries are used for data visualization?"
Doc A: "Matplotlib and Seaborn are Python libraries for creating charts and plots."
Doc B: "Python is a versatile language used in many domains including visualization."

A well-trained bi-encoder should rank Doc A higher. But if it encodes the query
into a vector capturing "Python + libraries + visualization" and Doc B has strong
individual term representations, the bi-encoder may incorrectly rank Doc B higher.

A cross-encoder reading both together immediately sees that Doc A matches more
specifically — it can attend to "matplotlib", "seaborn", "libraries" and "charts"
jointly with the query terms.

This is why reranking with a cross-encoder after bi-encoder retrieval consistently
improves results.

## Code

```python
# pip install sentence-transformers faiss-cpu

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util

# Load a pretrained bi-encoder fine-tuned for retrieval
model = SentenceTransformer("msmarco-distilbert-base-v4")

# Corpus
documents = {
    "D1": "Information retrieval finds relevant documents for a query.",
    "D2": "Python matplotlib and seaborn are used for data visualization.",
    "D3": "BERT is a transformer model for natural language understanding.",
    "D4": "Dense retrieval encodes queries and documents as neural vectors.",
    "D5": "Heart disease prevention includes diet exercise and medication.",
    "D6": "Cardiovascular mortality rates have declined over recent decades.",
    "D7": "Search engines build inverted indexes for efficient keyword lookup.",
    "D8": "Bi-encoders enable fast retrieval by precomputing document vectors.",
}

doc_ids   = list(documents.keys())
doc_texts = list(documents.values())

# Offline — encode all documents once
print("Encoding documents offline...")
doc_embeddings = model.encode(
    doc_texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True
)
doc_embeddings = np.array(doc_embeddings, dtype=np.float32)
print(f"Index shape: {doc_embeddings.shape}")    # → (8, 768)

# Build FAISS index
d = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(d)   # cosine sim (vectors are normalized)
index.add(doc_embeddings)

def biencoder_retrieve(query: str, k: int = 5) -> list[tuple[str, float, str]]:
    """Retrieve top-k documents using bi-encoder + FAISS."""
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )
    query_embedding = np.array(query_embedding, dtype=np.float32)

    scores, indices = index.search(query_embedding, k)

    return [
        (doc_ids[idx], float(score), doc_texts[idx])
        for score, idx in zip(scores[0], indices[0])
    ]


# Query time — only one forward pass needed
queries = [
    "cardiovascular disease prevention",
    "fast neural document retrieval",
    "python charting libraries",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = biencoder_retrieve(query, k=3)
    for doc_id, score, text in results:
        print(f"  {doc_id} ({score:.4f}): {text[:65]}...")


# Semantic similarity without FAISS (small scale)
query = "neural retrieval system"
query_embedding = model.encode(query, normalize_embeddings=True)

print(f"\nSemantic similarity scores for '{query}':")
for doc_id, text in documents.items():
    doc_emb = model.encode(text, normalize_embeddings=True)
    score = util.cos_sim(query_embedding, doc_emb).item()
    print(f"  {doc_id} ({score:.4f}): {text[:60]}...")


# Training a bi-encoder from scratch (minimal example)
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

# Training pairs: (query, positive_doc)
train_examples = [
    InputExample(texts=[
        "information retrieval systems",
        "Information retrieval is the task of finding relevant documents."
    ]),
    InputExample(texts=[
        "neural dense retrieval",
        "Dense retrieval encodes queries and documents as neural vectors."
    ]),
    InputExample(texts=[
        "heart disease prevention",
        "Heart disease prevention includes diet exercise and medication."
    ]),
]

train_dataloader = DataLoader(train_examples, batch_size=2, shuffle=True)

# MultipleNegativesRankingLoss = InfoNCE with in-batch negatives
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=10,
    show_progress_bar=True
)

print("\nFine-tuning complete.")
```

## Efficient Indexing at Scale

For corpora of millions of documents:

```python
# IVF index for faster search over large corpora
nlist = 100    # number of clusters
nprobe = 10    # clusters to search at query time

quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist,
                                faiss.METRIC_INNER_PRODUCT)
index_ivf.train(doc_embeddings)
index_ivf.add(doc_embeddings)
index_ivf.nprobe = nprobe

# HNSW for fastest query time
index_hnsw = faiss.IndexHNSWFlat(d, 32)
index_hnsw.add(doc_embeddings)
```

For billion-scale corpora, use product quantization (PQ) to compress vectors
and reduce memory:

```python
m = 8          # number of subquantizers
bits = 8       # bits per subquantizer
index_pq = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
index_pq.train(doc_embeddings)
index_pq.add(doc_embeddings)
```

## Where This Fits in the Progression

```
Word Embeddings     → static dense vectors
BERT for IR         → contextual dense vectors
Dense Retrieval     → applying vectors to retrieval at scale
Bi-encoders         → the architecture enabling dense retrieval  ← you are here
Cross-encoders      → accurate reranking of bi-encoder candidates
SPLADE              → learned sparse retrieval
```

The bi-encoder is the workhorse of modern first-stage neural retrieval. It is the
component you interact with most when building IR systems — sentence-transformers
makes it accessible in a few lines of code. Its limitation (no query-document
interaction) is exactly what the cross-encoder addresses.

## My Summary

A bi-encoder encodes queries and documents independently into dense vectors and
computes relevance as vector similarity, enabling precomputation of document vectors
at index time for fast retrieval. Training uses contrastive loss with in-batch
negatives — pulling relevant pairs together and pushing irrelevant ones apart —
with hard negative mining and cross-encoder distillation improving quality
significantly. The bi-encoder trades accuracy for speed relative to cross-encoders:
no direct query-document interaction during encoding means relevance judgments are
less precise, but document vectors are precomputed so query time is just one encoder
forward pass plus ANN search. sentence-transformers provides pretrained bi-encoders
ready for use. The bi-encoder is always the first stage; the cross-encoder is always
the second.
