# Listwise Reranking

## What is it?

Listwise reranking is a family of reranking approaches that treat the full set of
retrieved candidates as a single input and produce a global ranking over all of them
simultaneously, rather than scoring each candidate independently. Pointwise rerankers
(cross-encoders) score each (query, document) pair in isolation and rank by those
scores. Pairwise rerankers compare two documents at a time. Listwise rerankers see
the entire candidate list at once - the score assigned to any document can depend on
what other documents are in the list. This global perspective allows listwise models
to optimize the ranking objective directly, avoid redundant results, and produce
rankings that are coherent as a set rather than as independent scores that happen
to be sorted.

## Intuition

Imagine you are asked to rate ten job candidates on a scale of 1 to 10, one at a
time, without seeing the others. You rate candidate A as 8. Then you see candidate
B, who is much stronger - but you already committed to 8 for A. Now your rating
scale for A is wrong relative to B. This is the pointwise problem: scores are not
globally calibrated.

Now imagine instead you see all ten candidates at once and are asked to rank them.
You can directly compare: B is clearly better than A. C has similar strengths to A
but in different areas - should they rank together or apart? D is good but very
similar to B - putting them at ranks 1 and 2 would be redundant if users only read
the top result. Seeing all candidates simultaneously allows you to make these
globally coherent judgments.

Listwise reranking gives the model this global view. Instead of asking "how relevant
is this document to the query?" it asks "given all these documents, which is the
best ordering?" This framing directly matches the actual evaluation metric (NDCG,
which is defined over a complete ranked list) and allows the model to make
calibrated relative judgments rather than independent absolute scores.

## Why Pointwise Reranking Falls Short

### The calibration problem

Cross-encoders produce uncalibrated scores. Two queries may have very different
score distributions:

```
Query A: 10 highly relevant candidates
  Scores: [0.95, 0.92, 0.91, 0.89, 0.88, 0.87, 0.85, 0.84, 0.82, 0.81]
  Ranking: easy, all scores well-separated

Query B: 2 relevant candidates and 8 irrelevant
  Scores: [0.71, 0.68, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.38, 0.37]
  Ranking: the top-2 are correct but absolute scores are low
```

If you threshold at 0.70 to return "confident" results, Query B returns only one
result despite having two relevant documents. The cross-encoder's absolute scores
do not carry consistent meaning across queries.

Listwise models avoid this: they produce relative orderings, not absolute scores.
The output is "document 3 before document 7 before document 1" - no miscalibration
possible.

### The redundancy problem

Cross-encoders score documents independently. If five documents are nearly identical
(different web pages reporting the same news event), the cross-encoder will rank all
five highly because each is individually relevant. The resulting top-5 list is
redundant and useless.

A listwise model that sees all five simultaneously can recognize their similarity
and intersperse more diverse results - similar to MMR from the diversity module,
but learned rather than heuristic.

### The global ranking objective

NDCG, MAP, and MRR are defined over complete ranked lists, not individual document
scores. Optimizing them requires optimizing the full list jointly. Pointwise models
optimize a proxy (individual relevance score) that is correlated but not identical
to the true objective. Listwise models can directly optimize a surrogate of the
list-level metric.

## Listwise Approaches

### Approach 1 - Sorting-based listwise models

Process all candidates jointly with a model that directly outputs a permutation:

```
Input:  [CLS] query [SEP] doc₁ [SEP] doc₂ [SEP] ... [SEP] docₖ [SEP]
Output: permutation π over {1, ..., k}
```

**Limitations:** Transformer context window limits k. BERT-base handles ~512 tokens
total - with a query of 20 tokens and documents of 100 tokens each, you can fit
at most ~4 documents per forward pass. Even with long-context models, encoding 100
documents of 500 tokens each requires 50K context tokens.

In practice, sliding window approaches process overlapping subsets of candidates
and aggregate.

### Approach 2 - Sequence-to-sequence (RankT5, LiT5)

Frame reranking as a sequence generation problem. The model generates a ranking
as a sequence of document identifiers:

```
Input:  "Query: {query}\nDoc[1]: {doc₁}\nDoc[2]: {doc₂}\n...\nDoc[k]: {docₖ}"

Output: "Doc[3] Doc[1] Doc[5] Doc[2] Doc[4] ..."
         ↑ generate the ranking as a token sequence
```

T5-based models (RankT5, LiT5) are trained to generate the permutation of
document identifiers that corresponds to the correct relevance ranking. The decoder
generates identifiers one by one, effectively performing a learned insertion sort.

**Advantages:** T5's decoder autoregressively places documents, allowing each
placement decision to depend on previous ones - truly listwise.

**Disadvantages:** Generation is slow (one token per generation step), and the
output depends on the order documents are presented in the input (position bias).

### Approach 3 - LLM sliding window (RankGPT, SetRank)

Use a large language model to rank a window of documents at a time, sliding the
window over the full candidate list:

```
Window size w = 20 documents

Iteration 1: rank documents [1..20] → get ordering O₁
Iteration 2: slide window, rank documents [11..30] → get ordering O₂
Iteration 3: slide window, rank documents [21..40] → get ordering O₃
...

Aggregate orderings with bubble sort pass:
  Documents that consistently appear at top of windows → high rank
  Documents that consistently appear at bottom → low rank
```

This approach, used in RankGPT and similar LLM-based systems, can rank 50-100
documents in a few forward passes while maintaining the listwise property within
each window.

**Advantages:** Works with any LLM through prompting, no training required for
the ranking component, extremely high quality.

**Disadvantages:** High latency (multiple LLM forward passes), expensive,
ordering within windows may not be globally consistent.

## RankT5

RankT5 (Zhuang et al., 2022) is the most influential learned listwise reranker.
It fine-tunes T5 to directly optimize ranking metrics:

### Training

**Direct NDCG optimization:**
RankT5 uses a novel training approach that directly optimizes an approximation
of NDCG rather than a surrogate loss:

```
Approximate NDCG loss (differentiable):
  L_NDCG = 1 - NDCG_approximation(predicted_scores, labels)

Where NDCG_approximation uses softmax to make the ranking operation differentiable:
  permutation(scores) ≈ softmax(scores/τ)   (Gumbel-softmax or similar)
```

This is a fundamental advance over pairwise or pointwise losses - the model
is directly told "make your ranking have high NDCG" rather than "make relevant
documents score higher than irrelevant ones."

**Training data:** MS MARCO with graded relevance labels from TREC-DL.

**Architecture variants:**

```
RankT5-base:    T5-base (250M params), fast
RankT5-large:   T5-large (770M params), strong
RankT5-3b:      T5-3b (3B params), strongest
```

### Inference

RankT5 uses a pointwise-style inference (despite listwise training) - it scores
each document independently against the query - but the training objective makes
the scores better calibrated globally:

```
For each document:
  prompt = f"Query: {query}\nDocument: {doc}\nRelevant:"
  score  = T5_logit(token="true") - T5_logit(token="false")
```

The T5 model has learned to produce well-calibrated relevance probabilities
because the listwise training objective forced global calibration during training.
At inference, the per-document scoring is fast (no interaction between documents
required) while benefiting from the listwise training signal.

## LiT5

LiT5 (Ma et al., 2023) takes the sequence generation approach further with
a key insight: instead of generating a full permutation, generate a sorted list
progressively and use the generation probability as the ranking score.

### Listify: the key innovation

LiT5 trains T5 to generate a "listified" form of the ranking problem:

```
Input: Query: {query}
       Passage[1]: {doc₁}
       Passage[2]: {doc₂}
       ...
       Passage[k]: {docₖ}

Output: Passage[3] Passage[1] Passage[5] ...
        (identifiers in ranked order, most relevant first)
```

The generation probability of each identifier (e.g., "Passage[3]") given the
preceding context provides a listwise score for that document. Documents that the
model generates early and with high probability are ranked highest.

### Distill: efficient inference

LiT5 Distill reduces the cost by training a smaller model (T5-base) to match
the outputs of a larger teacher (T5-3b). This produces a fast listwise reranker
that approaches the quality of much larger models.

## SetRank

SetRank frames reranking as a set-level problem using a different architecture:

```
Input: each document encoded separately (bi-encoder style)
Interaction: self-attention over the full set of document embeddings
Output: relevance scores for all documents simultaneously
```

Unlike cross-encoders that see each document alone, SetRank's self-attention
between document embeddings allows it to compare documents to each other:

```
Document 3's score is high
Document 7's score is high but similar to Document 3 → deprioritize
Document 2's score is moderate but covers aspects Documents 3 and 7 miss → promote
```

This diversity-aware ranking emerges naturally from the set-level attention
without requiring explicit diversity constraints.

## Sliding Window Reranking

For production use with large candidate sets (k₁ > 20), sliding window
reranking is the practical approach:

```
Candidate list: [D₁, D₂, ..., D₁₀₀]
Window size: w = 20
Step size: s = 10 (50% overlap)

Pass 1 (front-to-back):
  Window [D₁..D₂₀]:  rank internally → partial order
  Window [D₁₁..D₃₀]: rank internally → partial order
  ...
  Window [D₈₁..D₁₀₀]: rank internally → partial order

Bubble sort aggregation:
  Top documents in each window bubble toward the front
  After one pass: approximate global ranking

Multiple passes for refinement:
  Pass 2 (back-to-front): refines the ordering
  Convergence: typically 2-3 passes
```

This is how RankGPT operates with LLMs - multiple sliding window passes over
the candidate list, each refining the ordering. It achieves near-listwise quality
on 100 candidates with O(k/s) forward passes rather than O(k²) pairwise comparisons.

## Listwise vs Pointwise vs Pairwise

```
Property                  Pointwise     Pairwise       Listwise
────────────────────────────────────────────────────────────────────
Input per inference       1 doc         2 docs         k docs
Score calibration         Poor          Moderate       Good
Diversity awareness       None          None           Implicit
Optimizes NDCG            Indirectly    Indirectly     Directly (RankT5)
Inference complexity      O(k)          O(k²)          O(1) or O(k/s)
Context window needed     Small         Small          Large
Position bias risk        Low           Low            High (input order)
Training complexity       Simple        Simple         Complex
Best NDCG@10 quality      Good          Better         Best
Production maturity       High          Medium         Growing
```

## Position Bias in Listwise Reranking

A critical challenge: listwise models that see multiple documents in context are
sensitive to the order those documents are presented in the input. Documents
presented earlier in the context tend to receive higher relevance scores - not
because they are more relevant but because of attention patterns that favor earlier
positions.

### Position bias manifestations

```
Input order:  [D₁, D₂, D₃, D₄, D₅]
Output order: [D₁, D₃, D₂, D₅, D₄]   ← D₁ boosted by being first

Shuffled input: [D₃, D₅, D₁, D₄, D₂]
Output order:   [D₃, D₅, D₁, D₄, D₂]  ← D₃ now boosted by being first
```

The model is partially learning "rank what is first" rather than "rank what is
most relevant."

### Mitigations

**Random shuffling during training:**
Train with randomly shuffled input orders so the model cannot rely on position.

**Permutation-invariant architectures:**
Use architectures that are structurally invariant to input permutation - such
as SetRank's set-level attention which treats input as a set, not a sequence.

**Multiple shuffle inference:**
At inference, run the model multiple times with different shuffles and aggregate:

```
score(doc) = mean(scores from 5 random input shuffles)
```

Expensive but eliminates position bias at inference time.

**Calibration with special tokens:**
Add position identifiers in the input that the model learns to ignore during
training, helping it rely on content rather than position.

## Practical Deployment Considerations

### When to use listwise over pointwise

```
Use listwise when:
  → Diversity in results is critical (avoid redundant rankings)
  → Candidate set is small (≤ 20 documents fit in context comfortably)
  → Latency is not critical (batch or offline ranking)
  → State-of-the-art quality is needed and LLM budget is available
  → NDCG optimization is explicitly needed (RankT5 with direct metric training)

Use pointwise when:
  → Large candidate sets (> 50 documents) require fast per-document scoring
  → Latency budget is tight (< 100ms)
  → Simple, well-understood scoring is preferred
  → A reranked list with possible redundancy is acceptable
```

### Hybrid: pointwise + listwise

The strongest production pattern combines both:

```
Stage 1: BM25/dense → top-1000 candidates
Stage 2: Cross-encoder (pointwise) → top-50 candidates    ← fast, filters
Stage 3: Listwise reranker → top-10 final results          ← precise, diverse
```

The pointwise reranker filters efficiently at scale. The listwise reranker applies
global reasoning only to the 50 most promising candidates - small enough to fit
in context with quality window coverage.

### Computational cost comparison

For reranking 100 candidates with a 512-token max per document:

```
Method                    Forward passes    Approx latency (GPU)
────────────────────────────────────────────────────────────────────
Cross-encoder (pointwise)  100               70ms (MiniLM)
RankT5-base (listwise)     ~10 windows       120ms
LiT5-Distill (listwise)    ~5 windows        80ms
RankGPT-4 (LLM sliding)    ~20 windows       2-10s
ColBERT (late interaction)  1 (precomputed)  20ms
```

RankT5 and LiT5 are typically 1.5-2x slower than a cross-encoder for the same
candidate set but produce meaningfully better NDCG.

Listwise reranking sits at the quality ceiling of the reranking module. It
addresses the fundamental limitations of pointwise approaches - score calibration,
redundancy, and indirect metric optimization - by treating the ranking problem as
it actually is: a global ordering task, not a collection of independent scoring
tasks.

## My Summary

Listwise reranking optimizes a global ordering over all retrieved candidates
simultaneously rather than scoring each document independently. The key advantages
over pointwise cross-encoders are score calibration (relative orderings rather than
uncalibrated absolute scores), diversity awareness (the model sees all documents and
can avoid redundant rankings), and direct NDCG optimization (RankT5 directly
optimizes an approximation of NDCG during training rather than a proxy). The main
approaches are T5-based sequence generation (RankT5, LiT5) that frames ranking as
generating a sequence of document identifiers, sliding window LLM reranking
(RankGPT) that applies a language model to overlapping windows of candidates, and
set-level attention models (SetRank) that compute self-attention across all candidate
embeddings simultaneously. Position bias - the tendency to favor documents presented
early in the input - is the central challenge and is addressed through random
shuffling during training, permutation-invariant architectures, and multi-shuffle
inference. The practical production pattern is a three-stage cascade: BM25/dense
retrieval → pointwise cross-encoder filtering to top-50 → listwise reranker for
final top-10.
