# Two-Stage Retrieval Architecture

Two-stage retrieval architecture is the standard production design pattern for
high-quality retrieval systems, where a fast first-stage retriever generates a
candidate set of hundreds of documents and a slower but more accurate second-stage
reranker reorders them to produce the final ranked list. The first stage - BM25,
dense retrieval, or hybrid - is optimized for recall: retrieve every potentially
relevant document without missing anything. The second stage - typically a
cross-encoder or learned reranker - is optimized for precision: among the retrieved
candidates, correctly identify and rank the most relevant ones first. The architecture
decouples the coverage problem from the ordering problem, allowing each stage to
use the best tool for its specific task without being constrained by the other
stage's requirements.

## Intuition

Retrieval and ranking are fundamentally different problems that are poorly served
by a single model:

**The retrieval problem** is about coverage. Given a query and a corpus of millions
of documents, which 100-1000 are worth considering? Speed and recall are the primary
constraints. The model must process every document in the corpus for every query,
so it must be extremely fast. Missing a relevant document at this stage is
unrecoverable - if a document is not retrieved, it cannot appear in the final result.

**The ranking problem** is about discrimination. Given 100-1000 candidates that
all look potentially relevant, which 10 should appear first? Accuracy is the primary
constraint. The model only processes the small candidate set, so speed is less
critical. Getting the order wrong is bad but recoverable - the user can scroll
down or reformulate.

No single model optimally solves both problems simultaneously:

- A bi-encoder (used for dense retrieval) can process millions of documents per
  second because it encodes query and document independently. But independent
  encoding means the model cannot directly compare query and document - it encodes
  each in isolation and hopes the resulting vectors are nearby for relevant pairs.
  This limitation produces strong recall but imperfect precision.

- A cross-encoder (used for reranking) jointly encodes query and document, allowing
  full attention across both. This produces far more accurate relevance assessments.
  But joint encoding requires a forward pass for every (query, document) pair -
  running it over millions of documents is orders of magnitude too slow for a
  production system.

Two-stage architecture resolves this tension by using each model where it excels:
bi-encoder for fast recall-oriented retrieval, cross-encoder for accurate
precision-oriented reranking.

## Why First-Stage Recall Matters More Than Precision

A critical asymmetry governs the two-stage design: recall in the first stage is
more important than precision.

If the first stage has Recall@100 = 0.90 (finds 90% of relevant documents in
its top 100), the best the second stage can ever achieve is NDCG@10 corresponding
to 90% recall - the missing 10% of relevant documents are permanently lost.

If the first stage has Recall@100 = 0.70, even a perfect reranker cannot compensate.

```
First-stage Recall@100    Maximum possible final NDCG@10
──────────────────────────────────────────────────────────
0.95                       ~0.89 (near perfect reranker)
0.80                       ~0.72 (near perfect reranker)
0.60                       ~0.52 (near perfect reranker)
0.40                       ~0.33 (near perfect reranker)
```

This asymmetry has three practical implications:

**1. First-stage k should be large**
Retrieving top-1000 candidates before reranking is better than top-100, even
if you eventually return only top-10. More candidates = higher recall = higher
ceiling for the reranker.

**2. First-stage should optimize for recall, not NDCG**
A first-stage model tuned for NDCG@10 might sacrifice recall@100. A model tuned
for recall@100 directly is better as a first-stage retriever.

**3. Hybrid first stage is usually better than single method**
BM25 and dense retrieval have complementary failure modes. BM25 misses semantic
matches; dense misses exact keyword matches. Their union covers both, giving higher
recall@100 than either alone even when their NDCG@10 is similar.

## The Architecture in Detail

### Stage 1 - Candidate retrieval

```
Query
  ↓
[First-stage retriever]
  - BM25 (fast, exact match)
  - Dense bi-encoder (semantic match)
  - Hybrid BM25 + dense (best recall)
  - SPLADE (sparse + semantic)
  ↓
Top-k candidates (k = 100-1000)
  - Speed requirement: < 50ms for most applications
  - Quality requirement: Recall@k ≥ 0.90 on domain benchmark
```

### Stage 2 - Reranking

```
Query + Top-k candidates
  ↓
[Second-stage reranker]
  - Cross-encoder (standard, most accurate)
  - ColBERT (late interaction, faster than cross-encoder)
  - MonoT5 (generative reranker)
  - LLM-based reranker (highest quality)
  ↓
Top-n final results (n = 10-50)
  - Speed requirement: < 200ms additional latency
  - Quality requirement: NDCG@10 maximized
```

### Full pipeline timing breakdown

For a typical production deployment (1M document corpus, single GPU):

```
Stage            Time          Notes
──────────────────────────────────────────────────────────────────
Query encoding   2-10ms        bi-encoder forward pass
ANN search       5-30ms        FAISS/Qdrant lookup, top-1000
BM25 search      2-10ms        parallel if hybrid
RRF fusion       1ms           merge ranked lists
Reranker input   1ms           batch (query, doc) pairs
Cross-encoder    50-200ms      depends on k and model size
Final sort       <1ms          sort reranked scores
Total            ~60-250ms     well within typical SLAs
```

## Recall at Different k Values

Understanding how Recall@k grows with k guides the choice of first-stage k:

```
First-stage method    R@10    R@50    R@100    R@500    R@1000
──────────────────────────────────────────────────────────────────
BM25                 0.44    0.61    0.72     0.83     0.87
Dense (E5-base)      0.55    0.73    0.82     0.91     0.93
Hybrid (RRF)         0.60    0.79    0.88     0.94     0.96
SPLADE               0.52    0.71    0.80     0.90     0.92
```

(Approximate values based on MS MARCO dev set averages)

Observations:

- Recall increases significantly from k=10 to k=100 (every document matters)
- Returns diminish from k=100 to k=1000 (marginal documents are marginal)
- Hybrid consistently outperforms single-method for recall at all k values
- The gap between methods narrows at higher k (all methods cover most relevant docs
  eventually)

For most applications, k=100 is sufficient and k=1000 is the practical maximum.

## Reranker Types and Their Tradeoffs

### Cross-encoder (standard)

BERT-based cross-encoder takes concatenated (query, document) pair as input:

```
Input:  [CLS] query [SEP] document [SEP]
Output: relevance score ∈ ℝ

Pros:
  + Highest quality among non-LLM rerankers
  + Full cross-attention between query and document
  + Small models available (MiniLM) with fast inference

Cons:
  - One forward pass per (query, document) pair
  - Latency scales linearly with k (number of candidates)
  - Cannot precompute document representations
```

Standard models: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast),
`cross-encoder/ms-marco-electra-base` (accurate)

### ColBERT late interaction

Precomputes document token embeddings offline, performs MaxSim at query time:

```
Offline: encode all document tokens → store as matrices
Query:   encode query tokens → compute MaxSim vs stored document matrices

Pros:
  + Faster than cross-encoder at query time (precomputed doc representations)
  + Per-token matching more precise than single-vector bi-encoder
  + Can rerank large candidate sets faster

Cons:
  - Much larger index (one vector per token per document)
  - Still requires document token lookup at query time
  - Higher memory requirements
```

### MonoT5 (generative reranker)

T5-based reranker that generates "true" or "false" for each (query, document) pair:

```
Input:  "Query: {query} Document: {document} Relevant:"
Output: probability of "true" token as relevance score

Pros:
  + Can leverage T5's generative capabilities for complex reasoning
  + Handles long documents better than BERT-based models
  + Strong performance on diverse tasks

Cons:
  - Slower than cross-encoder (generative decoding overhead)
  - Larger model sizes
  - Less common in production
```

### LLM-as-reranker

Uses a large language model to score or rank documents:

```
Approaches:
  A. Pointwise: LLM assigns relevance score per document
  B. Listwise:  LLM reorders all candidates at once
  C. Pairwise:  LLM compares pairs of documents

Pros:
  + Highest quality for complex queries
  + Can reason about nuanced relevance
  + No additional training needed (prompt-based)

Cons:
  - Very slow (LLM inference per candidate or per batch)
  - Expensive (API costs or GPU requirements)
  - Latency typically 500ms-5s per query
  - Best used as offline evaluation, not real-time serving
```

## Optimizing the First Stage for the Second Stage

A common mistake is optimizing the first-stage model independently of the second
stage. The correct objective is to maximize Recall@k of the first stage, measured
on the queries and relevant documents specific to your application.

### First-stage model selection for two-stage systems

```
Priority order for first-stage selection:
1. Maximize Recall@k (not NDCG@10)
2. Latency budget (must complete before second stage starts)
3. Index size / infrastructure constraints
```

Specifically, when a reranker follows:

- A dense model with Recall@100 = 0.88 is better than one with NDCG@10 = 0.55
  if the latter has Recall@100 = 0.82
- The NDCG@10 of the first stage is almost irrelevant - the reranker will reorder
  the candidates anyway

### First stage k selection

```
Reranker latency budget    Recommended k
──────────────────────────────────────────────
< 50ms (tight latency)     50-100
50-100ms                   100-200
100-200ms                  200-500
200-500ms                  500-1000
> 500ms or offline         1000+
```

Note: latency scales with k for cross-encoders. Doubling k roughly doubles
reranker latency. The optimal k is where Recall@k stops growing significantly.

## Stage Calibration and Score Combination

After reranking, scores from different stages can be combined to improve the final
ranking:

### Score interpolation

```
final_score = α × first_stage_score + (1 - α) × reranker_score
```

Typically α = 0.1-0.3 - mostly trust the reranker but let first-stage scores
break ties for documents the reranker considered similarly relevant.

### Cascade filtering

Use the first-stage score as a hard filter before expensive reranking:

```
first_stage_results = retrieve(query, k=1000)
hard_cutoff         = 0.5 × max_first_stage_score

# Only rerank documents above threshold
candidates_for_reranking = [
    doc for doc in first_stage_results
    if doc.first_stage_score >= hard_cutoff
]
reranked = reranker.score(query, candidates_for_reranking[:100])
```

Cascade filtering reduces reranker input size, cutting latency without
significantly hurting recall.

## Failure Modes of Two-Stage Architecture

```
Failure mode                  Cause                   Fix
──────────────────────────────────────────────────────────────────────────
Reranker can't help           Low first-stage recall  Increase k or improve
(relevant doc not in pool)    (<0.80)                 first stage retrieval

Reranker wrong order          Domain mismatch or      Fine-tune reranker on
despite correct pool          model generalization    domain data
                              failure

Latency exceeds budget        k too large or          Reduce k, use smaller
                              reranker too slow       reranker (MiniLM vs
                                                      larger), async pipeline

Memory overflow               Too many candidates     Reduce k or switch to
                              in GPU reranking        batch reranking with
                                                      cascade filter

First stage distributional    Index not updated after Use incremental indexing
drift                         corpus updates          or periodic reindexing
```

## My Summary

Two-stage retrieval architecture separates the coverage problem (retrieve every
potentially relevant document - first stage) from the ordering problem (correctly
rank the retrieved candidates - second stage), allowing each stage to use the
optimal tool for its task. The first stage uses a fast bi-encoder or hybrid
BM25 - dense retrieval to produce top-k candidates, optimizing for Recall@k because
documents not retrieved are permanently lost. The second stage uses a cross-encoder
or other reranker that jointly encodes query and document, optimizing for NDCG@10.
First-stage recall is more important than precision - the ceiling for the final
system is bounded by first-stage recall@k. Hybrid first stages (BM25 + dense + RRF)
consistently achieve higher recall@100 than either method alone, making them the
recommended default. The choice of k involves a latency-recall tradeoff - more
candidates give higher recall but increase reranker latency linearly for cross-
encoders. Typical production configurations use k=100-500 with a MiniLM cross-
encoder, achieving 50-200ms additional reranking latency and 5-15% NDCG@10
improvement over first-stage-only retrieval.
