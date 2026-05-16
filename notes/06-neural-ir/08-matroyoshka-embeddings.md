# Matryoshka Embeddings

Matryoshka embeddings are dense vector representations trained with a nested
learning objective that produces a single embedding where every prefix - the
first 32 dimensions, the first 64 dimensions, the first 128 dimensions, and so
on up to the full embedding - is itself a useful representation of the input.
Named after the Russian nested doll (матрёшка), the full embedding contains
smaller, complete embeddings within it. This means you can truncate a 1024-dim
Matryoshka embedding to any smaller power-of-two dimension and still get a
competitive representation - without retraining any model. Introduced by
Kusupati et al. at the University of Washington in 2022 and now the default
training objective for E5, GTE, and most major embedding models, Matryoshka
Representation Learning (MRL) is the primary technique for making embedding
deployments adaptive to latency and storage constraints.

## Intuition

Standard embedding models produce fixed-size vectors. A model trained to produce
768-dim embeddings always produces 768-dim embeddings. If you want a smaller
embedding - because you need faster ANN search, lower memory usage, or reduced
storage - your only option is to train an entirely new model at the smaller
dimension. This is expensive and you lose the quality of the large model.

Matryoshka embeddings solve this with a different training objective. Instead of
optimizing a single loss on the full embedding, MRL optimizes a weighted sum of
losses at multiple prefix lengths simultaneously:

```
Standard training:
  L = loss(embed[:768], label)     ← one loss

MRL training:
  L = loss(embed[:32],  label) × w₁
    + loss(embed[:64],  label) × w₂
    + loss(embed[:128], label) × w₃
    + loss(embed[:256], label) × w₄
    + loss(embed[:512], label) × w₅
    + loss(embed[:768], label) × w₆   ← six losses, one embedding
```

The model must pack as much information as possible into the first few
dimensions - because those dimensions are evaluated independently at every
scale. The first 32 dimensions must encode the most critical semantic signal.
Dimensions 33-64 add to it. Dimensions 65-128 refine further. And so on.

The result: a 768-dim Matryoshka embedding truncated to 128 dimensions gives
competitive retrieval quality to a model trained at 128 dimensions natively -
without any additional training. You get a whole spectrum of models for the
price of one.

## Why This Matters for IR

In IR, embedding dimensionality affects three distinct costs:

### 1. ANN search latency

ANN algorithms (HNSW, IVF, IVFPQ) scale with embedding dimension. Higher
dimension = more computation per similarity comparison:

```
Exact search (Flat):    O(n × d)
HNSW search:            approximately O(log n × d)
```

Halving the embedding dimension roughly halves the similarity computation cost
at each node of the HNSW graph, which translates to 30-50% latency reduction
in practice.

### 2. Index storage

Dense vector indexes store one vector per document:

```
Storage = n_documents × d × bytes_per_value

1M documents, 768-dim, float32:
  1,000,000 × 768 × 4 = 3.07 GB

1M documents, 128-dim, float32:
  1,000,000 × 128 × 4 = 0.51 GB

6x storage reduction by truncating to 128 dimensions
```

For large corpora (100M+ documents), this difference is the gap between a
system that fits in memory and one that requires expensive disk-based ANN.

### 3. Quantization interaction

Matryoshka embeddings combine multiplicatively with quantization. A 768-dim
float32 embedding truncated to 128-dim and quantized to int8:

```
Standard 768-dim float32:  768 × 4 = 3,072 bytes per vector
Matryoshka 128-dim int8:   128 × 1 =   128 bytes per vector

24x storage reduction, approximately 3-5x latency reduction
with modest quality loss
```

This combination is increasingly common in production deployments where
storage and latency are primary constraints.

## MRL Training Objective

The formal MRL loss for a model with embedding dimension d and nesting
dimensions {m₁, m₂, ..., mₗ} where m₁ < m₂ < ... < mₗ = d:

```
L_MRL = Σᵢ cᵢ × L(W^(mᵢ) × z^(mᵢ), y)
```

Where:

- z^(mᵢ) = first mᵢ dimensions of the full embedding z
- W^(mᵢ) = linear classifier/projector at scale mᵢ
- L = task loss (contrastive loss for retrieval, cross-entropy for classification)
- cᵢ = weight for scale mᵢ (typically uniform or geometrically decreasing)
- y = label (relevant/irrelevant for retrieval)

For retrieval, L is typically the InfoNCE contrastive loss:

```
L_InfoNCE(z^(m), y) = -log(
    exp(sim(q^(m), d₊^(m)) / τ) /
    Σⱼ exp(sim(q^(m), dⱼ^(m)) / τ)
)
```

Where sim is cosine similarity computed on only the first m dimensions.

### Weight schedules

The weights cᵢ for each scale affect which prefix dimensions are prioritized:

```
Uniform weights:          c₁ = c₂ = ... = cₗ = 1/l
  → Equal pressure at every scale
  → Most common in practice

Geometrically increasing: cᵢ ∝ mᵢ
  → Larger scales get more weight
  → Full-dimensional quality is prioritized

Geometrically decreasing: cᵢ ∝ 1/mᵢ
  → Smaller scales get more weight
  → Very small truncations work better, full-dimension quality slightly lower
```

Uniform weights are the standard choice and what E5, GTE, and most production
models use.

## MRL vs Alternative Approaches

### MRL vs training separate models per dimension

| Approach              | Cost               | Quality                               |
| --------------------- | ------------------ | ------------------------------------- |
| Train separate models | O(k) training runs | Each model optimized for its dim      |
| MRL                   | 1 training run     | Single model, competitive at all dims |

MRL quality at any given dimension is typically 1-3% below a model trained
specifically at that dimension - a small cost for enormous training efficiency.

### MRL vs PCA/linear projection

A natural alternative is to train a full-size model and then apply PCA or a
learned linear projection to reduce dimension:

```
PCA reduction:
  z_full → PCA → z_reduced (lower dim)
  Quality: significantly worse than MRL at the same dimension
  Reason: PCA optimizes variance preservation, not task performance
```

MRL outperforms PCA because the embedding is directly trained to be useful
at every prefix dimension - the model learns to pack task-relevant information
into the early dimensions, not variance-preserving information.

### MRL vs matryoshka adapter fine-tuning

For models not trained with MRL (e.g., standard BERT-base), you can add
MRL-style fine-tuning:

```
Take existing non-MRL model
Add MRL loss at multiple prefix dimensions
Fine-tune for 1-5 epochs on task data
```

Quality is better than PCA but worse than full MRL training from scratch.
Useful when you have an existing trained model you want to make adaptive.

## Models Using MRL

### E5 family (Microsoft)

All E5 models (e5-small, e5-base, e5-large, e5-mistral) are trained with MRL.
The Matryoshka dimensions for E5 are:

```
e5-small (384-dim):   [32, 64, 128, 256, 384]
e5-base  (768-dim):   [32, 64, 128, 256, 512, 768]
e5-large (1024-dim):  [32, 64, 128, 256, 512, 1024]
```

### GTE family (Alibaba)

GTE models (gte-base, gte-large, gte-Qwen) are MRL-trained. Same dimension
nesting as E5.

### OpenAI text-embedding-3

The text-embedding-3 API models are MRL-trained. The API accepts a `dimensions`
parameter to request truncated embeddings:

```python
response = openai_client.embeddings.create(
    input=["your text"],
    model="text-embedding-3-large",
    dimensions=256   # truncate to 256 from 3072
)
```

This is MRL truncation at the API level - the model produces a 3072-dim
embedding internally and returns the first 256 dimensions.

### Cohere Embed v3

MRL-trained. Supports adaptive truncation via API.

### Nomic Embed

MRL-trained. Available locally via sentence-transformers.

## Choosing the Right Truncation Dimension

The quality-cost tradeoff curve for a typical MRL model:

| Dimension  | Relative quality | Storage (vs full) | Latency (vs full) |
| ---------- | ---------------- | ----------------- | ----------------- |
| Full (768) | 1.000 (baseline) | 1.0×              | 1.0×              |
| 512        | 0.992            | 0.67×             | 0.68×             |
| 256        | 0.975            | 0.33×             | 0.35×             |
| 128        | 0.952            | 0.17×             | 0.18×             |
| 64         | 0.910            | 0.08×             | 0.09×             |
| 32         | 0.845            | 0.04×             | 0.05×             |

(Approximate values based on MTEB results for e5-base. Exact tradeoffs vary
by model, dataset, and task type.)

### Decision rules

| Scenario                             | Recommended dimension                    |
| ------------------------------------ | ---------------------------------------- |
| Research / maximum quality           | Full dimension                           |
| Production, latency < 50ms           | 256 (good quality, meaningful speedup)   |
| Production, large corpus (>10M docs) | 128 (significant storage reduction)      |
| Production, very large (>100M docs)  | 64-128 + int8 quantization               |
| Mobile / edge deployment             | 32-64 (very small memory footprint)      |
| First-stage recall, reranker follows | 128 (recall@100 is very similar to full) |

The key insight: for retrieval where a reranker follows, the first-stage model
only needs to maintain recall@100 (are the right documents in the top-100?),
not precision at rank 1. Lower dimensions maintain recall@100 much better than
they maintain NDCG@10 - making smaller truncations safe when a reranker handles
final ranking.

## Matryoshka + Quantization

The strongest production optimization combines Matryoshka truncation with
int8 or binary quantization:

```
Full embedding:             768-dim float32 = 3,072 bytes, NDCG@10 = 0.480
Matryoshka 256-dim float32: 256-dim float32 = 1,024 bytes, NDCG@10 = 0.468
Matryoshka 256-dim int8:    256-dim int8    =   256 bytes, NDCG@10 = 0.461
Matryoshka 128-dim int8:    128-dim int8    =   128 bytes, NDCG@10 = 0.445
Matryoshka 128-dim binary:  128-dim binary  =    16 bytes, NDCG@10 = 0.418
```

For many production scenarios the 128-dim int8 configuration - 24x smaller
and 3-5x faster than the full float32 embedding - is the practical sweet spot.

## Adaptive Retrieval with MRL

The most sophisticated MRL deployment pattern uses multiple dimensions in a
two-stage retrieval cascade:

```
Stage 1 - Coarse retrieval:
  Use small truncation (64-dim or 128-dim) for fast approximate search
  Retrieve top-1000 candidates from the full corpus
  Cost: fast and cheap (small dimension, large corpus)

Stage 2 - Re-retrieval or reranking:
  Re-score top-1000 with full-dimension embedding OR cross-encoder
  Return top-10 results
  Cost: slower but applied only to 1000 candidates (much smaller set)
```

This cascade uses MRL to get cheap first-stage recall, then applies more
expensive full-dimension scoring only where it matters. The total cost is much
lower than running full-dimension ANN search over the full corpus.

## MRL in Production - Deployment Decision Guide

| Constraint                | Recommended config                           |
| ------------------------- | -------------------------------------------- |
| Latency < 50ms, 1M docs   | 128-dim float32 (good quality, fast)         |
| Latency < 50ms, 10M docs  | 128-dim int8 (small index, stays in memory)  |
| Latency < 100ms, 10M docs | 256-dim float32 (strong quality)             |
| Latency unconstrained     | 512-768-dim float32 (maximum quality)        |
| First-stage + reranker    | 64-128-dim (recall@100 robust to truncation) |
| Mobile / edge             | 32-64-dim int8 (tiny footprint)              |
| RAG with reranker follows | 128-dim + cross-encoder reranker             |
| Maximum quality RAG       | 512-dim + cross-encoder reranker             |

## Common Mistakes with MRL

| Mistake                                              | Impact                           | Fix                                       |
| ---------------------------------------------------- | -------------------------------- | ----------------------------------------- |
| Truncating without renormalizing                     | Dot product != cosine similarity | Always L2-normalize after truncation      |
| Using non-MRL model and truncating                   | Very poor quality                | Check if model supports MRL               |
| Mixing dims at index and query time                  | Dimension mismatch error         | Always use same dim for query and docs    |
| Truncating to arbitrary dim (not in matryoshka_dims) | Unsupported dim may degrade more | Only use dims in for query and docs       |
| Not benchmarking per-use-case                        | Wrong dim selection              | Always measure quality at your target dim |

## My Summary

Matryoshka Representation Learning trains a single embedding model such that every
prefix of the full embedding - the first 32 dimensions, first 64, first 128, and so
on - is itself a useful dense representation. This is achieved by computing
contrastive loss at multiple prefix dimensions simultaneously during training and
optimizing a weighted sum. The result is one model that can be truncated to any
supported dimension at inference time without retraining, giving a spectrum of
quality-cost tradeoffs from a single training run. In IR, smaller dimensions reduce
ANN search latency (roughly linear in d), reduce index storage (directly linear
in d), and interact multiplicatively with int8 quantization for up to 24x total
storage reduction. The quality degradation is modest - a 128-dim MRL embedding
typically retains 95% of full-dimension NDCG@10 - and is acceptable for first-stage
retrieval where recall@100 rather than precision@1 is the relevant metric. All major
modern embedding models (E5, GTE, OpenAI text-embedding-3, Cohere Embed v3, Nomic
Embed) now use MRL as their default training objective, making it the standard
rather than an advanced technique.
