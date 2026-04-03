# Dense Retrieval

Dense retrieval is a first-stage retrieval paradigm that represents both queries and
documents as dense low-dimensional vectors and retrieves documents by finding those
whose vectors are closest to the query vector in embedding space. It replaces the
sparse term-matching of BM25 with learned neural representations that capture
semantic meaning, enabling retrieval of relevant documents even when they share no
terms with the query.

## Intuition

BM25 retrieves documents that contain the query terms. If a user searches for
"cardiovascular disease risk factors" and a relevant document discusses "heart attack
prevention" without using the exact query terms, BM25 misses it entirely. This is the
vocabulary mismatch problem.

Dense retrieval fixes this by operating in semantic space rather than term space.
Both the query and document are encoded into vectors that capture meaning. A document
about "heart attack prevention" ends up near a query about "cardiovascular disease risk
factors" in the embedding space — not because they share terms, but because they share
meaning.

The geometry is the same as the Vector Space Model from Phase 4 — cosine similarity
between vectors. The difference is what the vectors represent: hand-crafted TF-IDF
weights in VSM versus deep contextual representations learned by a neural encoder
in dense retrieval.

## The Dense Retrieval Pipeline

```bash
Offline (index time):
    for each document d in corpus:
        doc_vector = encoder(d)         → 768-dim dense vector
        store doc_vector in vector index

Online (query time):
    query_vector = encoder(query)       → 768-dim dense vector
    top_k_docs   = ANN_search(query_vector, vector_index, k)
    return top_k_docs
```

Two key components: a **neural encoder** that produces the vectors, and an
**approximate nearest neighbour (ANN) index** that retrieves the closest vectors
efficiently.

## The Encoder

The encoder is typically a pretrained transformer (BERT or similar) fine-tuned on
query-document relevance pairs. It maps variable-length text to a fixed-size dense
vector.

Two common pooling strategies:

### CLS token pooling

Use the [CLS] token output as the sequence representation:

```bash
vector = BERT([CLS] text [SEP])[CLS_position]   → 768-dim
```

### Mean pooling

Average all non-padding token representations:

```bash
vector = mean(BERT([CLS] text [SEP])[all_tokens])   → 768-dim
```

Mean pooling generally outperforms CLS pooling for semantic similarity tasks.
sentence-transformers uses mean pooling by default.

## Training Dense Retrievers

A dense retriever must be fine-tuned to produce vectors where relevant query-document
pairs are close and irrelevant pairs are far apart. This requires labeled data and a
suitable training objective.

### Contrastive learning with in-batch negatives

The standard training approach:

Given a batch of (query, positive_document) pairs, treat all other documents in the
batch as negatives for each query. Train the model to maximize similarity to the
positive and minimize similarity to all negatives.

```bash
Batch: [(q1, d1+), (q2, d2+), (q3, d3+)]

For q1:
  positive: d1+ → maximize cosine_similarity(enc(q1), enc(d1+))
  negatives: d2+, d3+ → minimize cosine_similarity(enc(q1), enc(d2+))
                         minimize cosine_similarity(enc(q1), enc(d3+))
```

Loss function — InfoNCE (contrastive loss):

```bash
L = -log( exp(sim(q, d+) / τ) / Σ exp(sim(q, dᵢ) / τ) )
```

Where τ (temperature) controls the sharpness of the distribution.

### Hard negatives

Random in-batch negatives are easy — they are unlikely to be relevant to the query.
Hard negatives are documents that look relevant but are not — they force the model
to learn finer-grained distinctions.

Sources of hard negatives:

- BM25 top results that are not the positive (retrieved but not relevant)
- Another dense retriever's top results
- Adversarially mined negatives

Training with hard negatives significantly improves retrieval quality.

## DPR — Dense Passage Retrieval

Introduced by Karpukhin et al. (Facebook AI, 2020). The first major dense retrieval
system to outperform BM25 on open-domain question answering benchmarks.

Architecture:

```bash
Two separate BERT-base encoders:
    query_encoder(q)    → 768-dim query vector
    passage_encoder(p)  → 768-dim passage vector

score(q, p) = dot_product(query_encoder(q), passage_encoder(p))
```

DPR uses two separate encoders (one for queries, one for passages) rather than
sharing weights. Training uses in-batch negatives + BM25 hard negatives.

Results on Natural Questions:

```bash
BM25:           Top-20 recall = 59.1%
DPR:            Top-20 recall = 79.4%
```

This was a major result — dense retrieval decisively outperforming BM25 on a
standard benchmark for the first time.

## Approximate Nearest Neighbour Search

At query time, the encoded query vector must be compared against millions of
precomputed document vectors. Exact search is O(n × d) — too slow for large corpora.
ANN search trades a small amount of accuracy for orders-of-magnitude speedup.

### FAISS (Facebook AI Similarity Search)

The standard library for ANN search in IR. Supports multiple index types:

**Flat index (exact search)**

```python
import faiss
import numpy as np

d = 768           # vector dimension
index = faiss.IndexFlatIP(d)   # inner product (equivalent to cosine if normalized)
index.add(doc_vectors)         # add all document vectors
D, I = index.search(query_vector, k=10)  # returns distances D and indices I
```

Exact but O(n) — use only for small corpora or as a baseline.

**IVF (Inverted File Index)**
Clusters vectors into nlist cells. At query time, search only the nearest nprobe
cells:

```python
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist=100)
index.train(doc_vectors)
index.add(doc_vectors)
index.nprobe = 10              # search 10 nearest cells
D, I = index.search(query_vector, k=10)
```

**HNSW (Hierarchical Navigable Small World)**
Graph-based ANN. Fastest query time among major methods:

```python
index = faiss.IndexHNSWFlat(d, 32)   # 32 = graph connectivity
index.add(doc_vectors)
D, I = index.search(query_vector, k=10)
```

### ANN accuracy-speed tradeoff

```bash
Index type   Build time   Query time   Recall@10   Memory
──────────────────────────────────────────────────────────
Flat         Fast         Slow         100%        High
IVF          Medium       Fast         ~95%        Medium
HNSW         Slow         Fastest      ~98%        High
```

For production retrieval systems, HNSW is the most common choice. FAISS also
supports product quantization (PQ) for memory compression at the cost of some
accuracy.

## Dense vs Sparse Retrieval

| Property              | BM25 (sparse)     | Dense retrieval       |
| --------------------- | ----------------- | --------------------- |
| Vocabulary mismatch   | Fails             | Handles well          |
| Exact keyword match   | Strong            | Weaker                |
| Rare terms / entities | Strong (high IDF) | Weaker                |
| Out-of-domain         | Robust            | Can degrade           |
| Training data needed  | None              | Large labeled dataset |
| Index size            | Small (sparse)    | Large (dense vectors) |
| Query latency         | Very fast         | Fast with ANN         |
| Interpretability      | High              | Low                   |

Neither dominates in all cases — which is why hybrid search (covered in
07-advanced/02-hybrid-search.md) combines both.

## When Dense Retrieval Fails

### Rare entities and exact match

BM25 handles rare named entities and exact string matches better. If a user queries
"BERT4Rec" (a specific model name), BM25 finds it via exact term match. Dense
retrieval may miss it if the encoder has not learned a strong representation for
that rare term.

### Out-of-domain generalization

Dense retrievers trained on MS MARCO (web questions) can degrade significantly on
other domains (biomedical, legal, code). BM25 is domain-agnostic. BEIR benchmark
specifically tests this generalization gap.

### No training data

Dense retrieval requires labeled query-document pairs for fine-tuning. BM25 requires
none. For low-resource domains and languages, BM25 remains the practical choice.

## The Two-Stage Retrieval Pipeline

Dense retrieval is almost always used as the first stage in a two-stage pipeline:

```bash
Query
  ↓
Stage 1 — Dense Retrieval (or BM25 or hybrid)
  → fast, approximate
  → retrieves top-1000 candidates from full corpus
  ↓
Stage 2 — Cross-encoder Reranking
  → slow, accurate
  → reranks top-100 candidates
  ↓
Final top-10 results
```

Dense retrieval is optimized for speed and recall at this first stage — it must
not miss relevant documents. The cross-encoder at stage 2 is optimized for
precision — it accurately scores the shortlist.

## Where This Fits in the Progression

```bash
Word Embeddings     → static dense vectors
BERT for IR         → contextual dense vectors, pretraining
Dense Retrieval     → applying BERT vectors to first-stage retrieval  ← you are here
Bi-encoders         → the specific architecture enabling dense retrieval
Cross-encoders      → accurate second-stage reranking
SPLADE              → learned sparse retrieval
```

Dense retrieval is the application layer built on BERT — it answers the question
of how to use contextual embeddings for retrieval at scale. The bi-encoder is the
specific architectural pattern that makes this efficient, and is covered next.

## My Summary

Dense retrieval represents queries and documents as dense neural vectors and retrieves
documents by approximate nearest neighbour search in embedding space. It solves the
vocabulary mismatch problem of BM25 by operating in semantic space where "heart attack
prevention" and "cardiovascular disease risk factors" are geometrically close. DPR
was the first major system to outperform BM25 on open-domain QA. Dense retrieval
requires labeled training data, is weaker on exact keyword matching and rare entities,
and can degrade out-of-domain — which is why hybrid search combining dense and sparse
retrieval consistently outperforms either alone. FAISS provides the ANN infrastructure
that makes dense retrieval practical at scale.
