# Reranking

## What is it?

Reranking is the process of taking a candidate set of documents retrieved by a
fast first-stage retriever and re-ordering them using a slower, more accurate
scoring model. It is the second stage in the standard two-stage retrieval pipeline.
The first stage optimizes for recall — retrieve everything that might be relevant.
The second stage optimizes for precision — put the most relevant documents first.

## Intuition

First-stage retrieval (BM25 or bi-encoder) is fast but imperfect. It retrieves a
broad set of candidates that contains the relevant documents but in a suboptimal
order. The relevant documents are in there somewhere but may not be at the top.

Reranking fixes the ordering. It applies a more powerful model — typically a cross-
encoder — that can afford to be slow because it only operates on a small candidate
set rather than the full corpus. The result is a ranked list where the most relevant
documents appear first.

The intuition is the same as a hiring pipeline: a fast resume screen narrows 10,000
applications to 100 candidates, then an in-depth interview process selects the best 10. The screen optimizes for not missing good candidates (recall). The interviews
optimize for identifying the best ones (precision).

## The Two-Stage Pipeline

```
Full corpus (millions of documents)
      ↓
Stage 1 — First-stage retrieval
  Model:   BM25, bi-encoder, or SPLADE
  Speed:   ~10ms
  Output:  top-1000 candidates (high recall, imperfect order)
      ↓
Stage 2 — Reranking
  Model:   cross-encoder or listwise reranker
  Speed:   ~200-500ms (for top-100)
  Output:  top-10 final results (high precision, correct order)
      ↓
Results returned to user
```

The key design decision: how many candidates to pass from stage 1 to stage 2.
More candidates = better recall fed to reranker but higher reranking latency.
Typical production values: top-100 to top-1000 from stage 1, rerank top-50 to
top-100.

## Types of Reranking Models

### Pointwise reranking

Score each candidate independently. The cross-encoder is the standard pointwise
reranker — it scores each (query, document) pair in isolation and ranks by score.

```
score(q, d₁) = 0.92
score(q, d₂) = 0.71
score(q, d₃) = 0.85
→ ranking: d₁ > d₃ > d₂
```

Advantages: simple, parallelizable, each document gets an absolute relevance score.
Disadvantages: does not directly optimize ranking — documents are scored without
reference to each other.

### Pairwise reranking

Compare documents in pairs and determine which is more relevant:

```
is d₁ more relevant than d₂ for query q? → yes
is d₁ more relevant than d₃ for query q? → yes
is d₃ more relevant than d₂ for query q? → yes
→ ranking: d₁ > d₃ > d₂
```

More directly aligned with the ranking objective. Requires O(n²) comparisons for
n candidates — expensive but feasible for small candidate sets.

### Listwise reranking

Consider all candidates together and produce a ranking jointly. Models like
RankGPT and LLM-based rerankers take the full list as input and output a permutation.

```
Input:  query + [d₁, d₂, d₃, d₄, d₅] as one prompt
Output: ranked permutation [d₁, d₃, d₅, d₂, d₄]
```

Most powerful but most expensive — requires fitting all candidates in one context.

## Cross-encoder Reranking — The Standard

The cross-encoder (covered in 05-cross-encoders.md) is the dominant reranking model:

```
Input:  [CLS] query [SEP] document [SEP]
Output: relevance score (scalar)
```

The cross-encoder reads query and document jointly — every query token attends to
every document token. This joint attention is what makes it more accurate than a
bi-encoder that encodes them separately.

### MonoBERT

BERT fine-tuned on MS MARCO for pointwise relevance classification. The original
strong cross-encoder baseline:

```
BM25 alone:                    MRR@10 = 0.184
BM25 + MonoBERT reranking:     MRR@10 = 0.365
```

### MonoT5

T5 fine-tuned to generate "true" or "false" given (query, document). Relevance
score = P("true"). Stronger than MonoBERT:

```
BM25 + MonoT5-3B reranking:    MRR@10 ≈ 0.398
```

### Duos — pairwise T5 reranker

An extension of MonoT5 that compares document pairs:

```
Input:  "Query: {q} Document0: {d₁} Document1: {d₂} Relevant:"
Output: "0" (d₁ more relevant) or "1" (d₂ more relevant)
```

More accurate than pointwise but O(n²) in candidates — practical only for
reranking top-20 or fewer.

## LLM-based Reranking

Recent trend: use large language models as rerankers. Two main approaches:

### Pointwise LLM scoring (RankGPT-style)

Prompt the LLM to assess relevance of each document independently:

```
Prompt: "Given the query '{query}', is the following document relevant?
Document: {document}
Answer yes or no."

Score = P("yes") from LLM output probabilities
```

### Listwise LLM reranking

Prompt the LLM to directly produce a ranking permutation:

```
Prompt: "Rank the following documents for the query '{query}' from most
to least relevant. Output the document IDs in order.
[1] {doc1}
[2] {doc2}
[3] {doc3}
..."

Output: "[2] > [1] > [3]"
```

LLM-based rerankers are currently state-of-the-art on many benchmarks but are
expensive — GPT-4 level models are ~100x slower than cross-encoders. Smaller
open models (Mistral, LLaMA) fine-tuned for reranking offer better tradeoffs.

## Cascade Reranking

For very large candidate sets or strict latency budgets, use multiple reranking
stages:

```
Stage 1:  BM25 → top-1000             (10ms)
Stage 2:  lightweight bi-encoder reranker → top-200    (20ms)
Stage 3:  cross-encoder → top-20      (100ms)
Stage 4:  LLM reranker → top-5        (500ms)
```

Each stage is slower and more accurate. The cascade minimizes the number of
expensive model calls by eliminating most candidates early.

## Score Normalization and Fusion

When combining first-stage and reranker scores:

### Replace

Simply use the reranker score — discard the first-stage score entirely. Standard
approach when the reranker is much more accurate.

### Linear combination

```
final_score = α × reranker_score + (1-α) × first_stage_score
```

Useful when the first-stage retriever captures signals the reranker misses (e.g.
BM25 is strong on exact keyword matching).

### Reciprocal Rank Fusion (RRF)

Combine rankings from multiple systems without needing calibrated scores:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```

Where the sum is over all retrieval systems r, rank_r(d) is document d's rank in
system r, and k is a smoothing constant (typically 60).

RRF is robust, parameter-free beyond k, and works well in practice for combining
multiple retrieval signals.

## Reranking for Different Tasks

```
Task                   First stage        Reranker              Metric
──────────────────────────────────────────────────────────────────────────
Web search             BM25 + dense       CrossEncoder          NDCG@10
Open-domain QA         DPR                MonoBERT              MRR@10
Code search            BM25               CodeBERT cross-enc    MRR@10
Legal retrieval        BM25               Domain-fine-tuned CE  NDCG@10
Conversational IR      Dense              LLM reranker          NDCG@3
```

Domain-specific fine-tuning of the reranker is often necessary for specialized
domains. A cross-encoder fine-tuned on MS MARCO may degrade significantly on
biomedical or legal text.

## Practical Reranking with sentence-transformers

```python
# pip install sentence-transformers rank-bm25

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

# Cross-encoder reranker
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)

# Corpus
documents = {
    "D1": "Information retrieval finds relevant documents for user queries.",
    "D2": "Python is a popular language for machine learning and data science.",
    "D3": "Search engines index billions of documents using inverted indexes.",
    "D4": "Neural reranking models improve ranking quality using transformers.",
    "D5": "Heart disease prevention requires dietary changes and regular exercise.",
    "D6": "Cardiovascular mortality has declined due to improved medical care.",
    "D7": "BERT-based cross-encoders significantly improve passage reranking.",
    "D8": "Dense retrieval uses approximate nearest neighbour search at scale.",
    "D9": "Reranking re-orders retrieved candidates using a more powerful model.",
    "D10": "BM25 is the standard sparse retrieval baseline in information retrieval.",
}

doc_ids   = list(documents.keys())
doc_texts = list(documents.values())

# First stage — BM25
tokenized = [text.lower().split() for text in doc_texts]
bm25 = BM25Okapi(tokenized)


def rerank_pipeline(query: str,
                    first_stage_k: int = 8,
                    final_k: int = 5) -> dict:
    """
    Full two-stage pipeline with BM25 first stage and cross-encoder reranking.
    Returns both stage results for comparison.
    """
    # Stage 1 — BM25
    bm25_scores = bm25.get_scores(query.lower().split())
    first_stage_indices = np.argsort(bm25_scores)[::-1][:first_stage_k]
    first_stage_results = [
        (doc_ids[i], bm25_scores[i], doc_texts[i])
        for i in first_stage_indices
    ]

    # Stage 2 — cross-encoder reranking
    candidates = [(doc_ids[i], doc_texts[i]) for i in first_stage_indices]
    pairs = [(query, text) for _, text in candidates]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip([doc_id for doc_id, _ in candidates],
            rerank_scores,
            [text for _, text in candidates]),
        key=lambda x: x[1],
        reverse=True
    )[:final_k]

    return {
        "first_stage": first_stage_results,
        "reranked":    reranked
    }


def reciprocal_rank_fusion(rankings: list[list[str]],
                            k: int = 60) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: list of ranked doc ID lists from different systems
        k:        smoothing constant (default 60)

    Returns:
        list of (doc_id, rrf_score) sorted by score descending
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Run pipeline
queries = [
    "cardiovascular disease prevention",
    "fast neural retrieval systems",
    "how does reranking improve search",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = rerank_pipeline(query, first_stage_k=8, final_k=5)

    print("  BM25 first stage (top-5):")
    for doc_id, score, text in results["first_stage"][:5]:
        print(f"    {doc_id} ({score:.3f}): {text[:55]}...")

    print("  After reranking (top-5):")
    for doc_id, score, text in results["reranked"]:
        print(f"    {doc_id} ({score:.3f}): {text[:55]}...")


# RRF fusion example
bm25_ranking  = ["D1", "D3", "D7", "D10", "D4"]
dense_ranking = ["D9", "D4", "D7", "D1",  "D8"]
splade_ranking = ["D7", "D1", "D9", "D3", "D4"]

fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking, splade_ranking])
print("\nRRF fusion of three systems:")
for doc_id, score in fused[:5]:
    print(f"  {doc_id} ({score:.4f}): {documents[doc_id][:55]}...")
```

## Reranking Quality vs Latency Tradeoffs

```
Model                          MRR@10   Latency (100 docs)   Params
───────────────────────────────────────────────────────────────────
BM25 (no reranking)            0.184    ~1ms                 0
MiniLM-L6 cross-encoder        0.334    ~50ms                22M
MiniLM-L12 cross-encoder       0.345    ~80ms                33M
BERT-base cross-encoder        0.358    ~200ms               110M
MonoT5-base                    0.378    ~400ms               250M
MonoT5-3B                      0.398    ~2000ms              3B
```

For most production use cases, MiniLM-L6 offers the best tradeoff — strong
improvement over BM25 with manageable latency.

## Where This Fits in the Progression

```
Dense Retrieval     → first-stage neural retrieval
Bi-encoders         → efficient first-stage retrieval
Cross-encoders      → accurate second-stage reranking
SPLADE              → learned sparse first-stage retrieval
Reranking           → the full two-stage pipeline  ← you are here
Hybrid Search       → combining multiple first-stage signals
ColBERT             → alternative to two-stage: late interaction
RAG                 → retrieval feeding into generation
```

Reranking is where all the neural IR components come together into a complete
system. The subsequent notes cover how to extend this pipeline — hybrid search
adds multiple first-stage signals, ColBERT proposes an alternative to the strict
bi-encoder/cross-encoder split, and RAG uses the full pipeline to power generation.

## My Summary

Reranking is the second stage of the standard two-stage retrieval pipeline — it
takes a candidate set from a fast first-stage retriever and re-orders it using a
more accurate but slower model. Cross-encoders are the dominant reranking model,
providing full query-document token interaction at the cost of not being able to
precompute document representations. The key engineering tradeoff is how many
candidates to rerank — more improves quality but increases latency. Cascade
reranking addresses this by using progressively more expensive models on
progressively smaller candidate sets. RRF provides a simple and effective way
to combine rankings from multiple systems without requiring calibrated scores.
Reranking consistently provides the largest quality jump in any retrieval pipeline
and is a standard component in every production search system.
