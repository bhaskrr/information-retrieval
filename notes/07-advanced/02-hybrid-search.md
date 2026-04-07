# Hybrid Search

Hybrid search is a retrieval strategy that combines sparse retrieval (BM25 or SPLADE)
with dense retrieval (bi-encoders) to produce a single ranked result list that is
stronger than either approach alone. It exploits the complementary strengths of both
paradigms — sparse retrieval's exactness and dense retrieval's semantic understanding
— and merges their signals into a unified ranking.

## Intuition

No single retrieval model dominates across all query types:

- BM25 excels at exact keyword matching, rare terms, and named entities. A query
  for "BERT4Rec" or "ISO 27001 compliance" is handled well by BM25 because it
  matches exact strings with high IDF weight.

- Dense retrieval excels at semantic queries where the vocabulary of the query
  differs from the vocabulary of the relevant document. A query for "heart attack
  prevention" retrieves documents about "cardiovascular disease management" that
  BM25 misses entirely.

Neither approach handles all cases well. Hybrid search combines both, consistently
outperforming either alone across diverse query sets and domains.

The intuition is simple: run both retrievers, get two ranked lists, merge them.
The challenge is how to merge them faithfully when the scores from different systems
are not on the same scale.

## Why Hybrid Consistently Outperforms

Failure modes of each approach:

```
BM25 fails when:
  - Query and document use different vocabulary (vocabulary mismatch)
  - Query is semantically complex ("explain why transformers work")
  - Paraphrases and synonyms are involved

Dense retrieval fails when:
  - Query contains rare named entities ("Karpukhin et al. 2020")
  - Exact string matching is critical ("error code 404")
  - Out-of-domain queries (trained on MS MARCO, evaluated on biomedical text)
  - Short queries with high ambiguity
```

These failure modes rarely overlap. A query that confuses the dense retriever
usually does not confuse BM25, and vice versa. Combining both covers each other's
blind spots.

Empirically, hybrid search typically improves NDCG@10 by 2-5 points over the
best single retriever on most benchmarks.

## Score Fusion Methods

The core challenge: BM25 scores and cosine similarity scores are on completely
different scales. BM25 scores are unbounded positive numbers; cosine similarities
are between 0 and 1. You cannot directly add them without normalization.

### Method 1 — Reciprocal Rank Fusion (RRF)

The simplest and most robust approach. Ignores raw scores entirely and combines
based on rank positions:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```

Where the sum is over all retrieval systems r, rank_r(d) is the rank of document d
in system r, and k is a smoothing constant (typically 60).

Documents not retrieved by a system are assigned rank = ∞, contributing 0 to RRF.

```
Example with k=60:

           BM25 rank   Dense rank   RRF score
D1:        1           3            1/(61) + 1/(63) = 0.0164 + 0.0159 = 0.0323
D2:        2           1            1/(62) + 1/(61) = 0.0161 + 0.0164 = 0.0325
D3:        3           not retrieved 1/(63) + 0      = 0.0159
D4:        not retrieved 2           0 + 1/(62)      = 0.0161

RRF ranking: D2 (0.0325) > D1 (0.0323) > D4 (0.0161) > D3 (0.0159)
```

D2 wins because it ranks highly in both systems. D3 drops because it was not
retrieved by the dense retriever at all.

Advantages of RRF:

- No score calibration needed
- Robust to outlier scores
- Works with any number of systems
- Simple to implement
- Performs surprisingly well in practice

Disadvantage: discards score magnitude information — a document with score 0.99
from the dense retriever is treated the same as one with score 0.51.

### Method 2 — Linear Score Combination

Normalize scores from each system then combine with a weighted sum:

```
hybrid_score(d) = α × dense_score(d) + (1 - α) × sparse_score(d)
```

Where α is a mixing parameter tuned on a validation set (typically 0.3-0.7).

Score normalization is essential before combining. Common normalization:

**Min-max normalization:**

```
normalized_score(d) = (score(d) - min_score) / (max_score - min_score)
```

**Z-score normalization:**

```
normalized_score(d) = (score(d) - mean_score) / std_score
```

**Softmax normalization:**

```
normalized_score(d) = exp(score(d)) / Σ exp(score(dᵢ))
```

Advantage: preserves score magnitude information, α is interpretable.
Disadvantage: normalization across different query types is unstable — min/max
depends on the current query's result set and varies unpredictably.

### Method 3 — Learned Score Fusion

Train a lightweight model to predict the optimal combination:

```
hybrid_score(d) = f(dense_score(d), sparse_score(d), query_features)
```

Where f is a linear model or small neural network trained on validation data.
Features might include: dense score, sparse score, score rank, query length,
query type (navigational vs informational).

More flexible than fixed α but requires labeled data and adds complexity.

### Method 4 — Convex Combination with Score Normalization (CCSN)

A more principled version of linear combination used in production systems:

```
hybrid_score(d) = α × normalize(dense_score(d)) + (1-α) × normalize(bm25_score(d))
```

Where normalization maps each score distribution to the same range. The key
insight is that normalization must be done per-query (relative to other candidates
in this query's result set), not globally.

## Practical Implementation Patterns

### Pattern 1 — Parallel retrieval + RRF

```
Query
  ↓
┌─────────────────┐    ┌─────────────────┐
│  BM25 retrieval │    │ Dense retrieval │
│  → top-1000     │    │ → top-1000      │
└────────┬────────┘    └────────┬────────┘
         └──────────┬───────────┘
                    ↓
            RRF score fusion
                    ↓
              top-1000 merged
                    ↓
         Cross-encoder reranking
                    ↓
              top-10 results
```

### Pattern 2 — Sequential retrieval (dense → sparse expansion)

Use dense retrieval to find a broad candidate set, then use BM25 to expand:

```
Dense retrieval → top-500
BM25 retrieval  → top-500
Union           → up to 1000 unique candidates
Cross-encoder   → top-10
```

### Pattern 3 — SPLADE + Dense

Replace BM25 with SPLADE in the hybrid:

```
SPLADE retrieval (sparse, learned) → top-1000
Dense retrieval (bi-encoder)       → top-1000
RRF fusion                         → top-1000
Cross-encoder reranking            → top-10
```

SPLADE + Dense is currently one of the strongest combinations on BEIR benchmarks
because SPLADE already handles vocabulary expansion — adding dense retrieval on top
covers any remaining semantic gaps.

## Hybrid Search in Production Systems

### Elasticsearch

Elasticsearch supports hybrid search natively via the `knn` query combined with
standard BM25 queries:

```json
{
  "query": {
    "bool": {
      "should": [
        {"match": {"content": "cardiovascular disease"}},
        {"knn": {"field": "embedding", "query_vector": [...], "num_candidates": 100}}
      ]
    }
  }
}
```

### OpenSearch

Similar to Elasticsearch with built-in hybrid search support via the `hybrid`
query type and normalization processors.

### Weaviate, Qdrant, Pinecone

Modern vector databases with built-in hybrid search — BM25 and dense retrieval
are run together and fused internally.

## Tuning α — When to Weight Each Signal

```
Query type                          Recommended α (dense weight)
──────────────────────────────────────────────────────────────────
Semantic / conceptual queries       α = 0.7  (weight dense more)
Keyword / exact match queries       α = 0.3  (weight sparse more)
Named entity queries                α = 0.2  (weight sparse more)
Mixed / general queries             α = 0.5  (equal weight)
```

In practice, α = 0.5 with min-max normalization or RRF with k=60 are strong
defaults that work well across query types without per-query tuning.

## Benchmark Results — Hybrid vs Single System

Typical results on BEIR (zero-shot evaluation), NDCG@10:

```
System                          Avg NDCG@10 across BEIR
────────────────────────────────────────────────────────
BM25                            0.428
Dense (msmarco-distilbert)      0.396
SPLADE                          0.453
Dense + BM25 (RRF)              0.461
SPLADE + Dense (RRF)            0.482
+ Cross-encoder reranking       0.510+
```

Hybrid consistently outperforms single systems. SPLADE + Dense is currently one
of the strongest combinations without reranking.

## Where This Fits in the Progression

```
Dense Retrieval     → neural first-stage retrieval
Bi-encoders         → efficient neural first-stage
Cross-encoders      → accurate second-stage reranking
SPLADE              → learned sparse first-stage
Reranking           → full two-stage pipeline
Hybrid Search       → combining multiple first-stage signals  ← you are here
ColBERT             → alternative: late interaction model
RAG                 → retrieval feeding generation
```

Hybrid search represents the current practical best-practice for first-stage
retrieval in production systems. The full pipeline — hybrid first stage + cross-
encoder reranking — is the standard architecture for state-of-the-art IR systems.

## My Summary

Hybrid search combines sparse retrieval (BM25 or SPLADE) and dense retrieval
(bi-encoder) to cover each other's failure modes — sparse retrieval handles exact
keyword matching and rare terms, dense retrieval handles vocabulary mismatch and
semantic queries. Score fusion is the core challenge: RRF combines rank positions
without requiring score calibration and is the most robust practical approach;
linear combination with min-max normalization preserves score magnitude but is less
stable across query types. SPLADE + Dense + RRF is currently one of the strongest
first-stage retrieval combinations. In production, hybrid retrieval is almost always
followed by cross-encoder reranking to produce the final ranked list.
