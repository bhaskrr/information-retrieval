# Recall-Oriented Evaluation

Recall-oriented evaluation is the practice of measuring retrieval quality with
metrics and protocols specifically designed to assess first-stage retrieval -
the component responsible for finding all relevant documents in a large corpus
before any reranking occurs. Standard evaluation metrics like NDCG@10 and MRR@10
measure precision at the top ranks, which is the right objective for the final
ranked list a user sees. But the first stage of a two-stage retrieval pipeline is
not trying to rank perfectly - it is trying to ensure that relevant documents are
not missed entirely, so the reranker has the opportunity to surface them. Evaluating
a first-stage retrieval with NDCG@10 misses its actual objective and leads to poor
design decisions. Recall-oriented evaluation uses metrics like Recall@100, Recall@1000,
and hole rate that directly measure whether a retrieval component successfully
gathered the relevant candidates it needs to pass to the next stage.

## Intuition

Consider a library assistant whose job is to gather all books that might be
relevant to a patron's research question, before a librarian specialist reviews the
selection and curates the final reading list. The assistant should not be evaluated
on whether the first book in their stack is perfect - they should be evaluated on
whether they missed any important books. If they bring back 100 books and the
librarian can always find what the patron needs within that stack, the assistant
has done their job well. If the librarian frequently says "you forgot this entire
relevant category of books," the assistant has failed - regardless of whether the
one book they placed on top was excellent.

This is exactly the two-stage retrieval situation. The first stage is the library
assistant: its job is to bring back a comprehensive candidate set, not to perfectly
rank it. The second stage (reranker) is the librarian specialist: it takes the
candidate set and produces the final curated ranking. Evaluating the library
assistant with ranking metrics misses the point entirely.

The practical consequence is significant. Two first-stage retrieval systems with
identical NDCG@10 = 0.45 may have Recall@100 of 0.73 and 0.87 respectively. The
system with higher Recall@100 will produce significantly better final results after
reranking because it gives the reranker access to more relevant documents. The
NDCG@10 measurement of the first stage completely obscured this difference.

## The Two Objectives of a Two-Stage Pipeline

```
First stage objective:    Maximize Recall@k₁
                          "Have all relevant documents in the candidate set"

Second stage objective:   Maximize NDCG@k₂ given the candidate set
                          "Rank the candidate set correctly"

End-to-end objective:     Maximize NDCG@k₂ of the full pipeline
                          "Return the best possible final ranking"

The relationship:
  NDCG@k₂(full_pipeline) ≤ NDCG_ceiling(first_stage_recall)
  → First-stage recall determines the maximum possible final quality
  → Second-stage reranking can only improve within the recall ceiling
```

This hierarchy means first-stage recall is a prerequisite for final quality,
not a secondary concern. Improving Recall@100 from 0.73 to 0.87 raises the
maximum achievable NDCG@10 and allows a better reranker to actually reach that
ceiling. No matter how good the reranker is, it cannot surface documents the
first stage never retrieved.

## Recall@K

The primary metric for first-stage evaluation:

```
Recall@K = |{relevant documents in top-K retrieved}| / |{all relevant documents}|
```

For a query with 5 relevant documents where retrieval returns 100 candidates:

```
If 4 of the 5 relevant documents appear in the top-100: Recall@100 = 0.80
If all 5 relevant documents appear in the top-100: Recall@100 = 1.00
If only 2 relevant documents appear in the top-100: Recall@100 = 0.40
```

### Choosing K

K should match the candidate set size passed to the reranker:

```
Common configurations:
  Recall@20:   for tight latency budgets where reranker sees only 20 candidates
  Recall@100:  standard first-stage evaluation, most common in literature
  Recall@1000: for research systems where the reranker is a BM25 re-retrieval
```

Using a different K than the actual pipeline configuration is a common mistake.
If your reranker receives 50 candidates but you evaluate Recall@100, your
evaluation is optimistic - the first stage appears better than it actually is
in context.

### Recall@K vs NDCG@K

```
Recall@100 = 0.85:
  85% of all relevant documents appear somewhere in the top-100
  Does not care what order they appear in
  Does not penalize putting an irrelevant document at rank 1

NDCG@10 = 0.45:
  Rewards having relevant documents at the very top ranks
  Penalizes relevant documents at rank 6-10 compared to rank 1-5
  Completely ignores documents at ranks 11+

For first-stage retrieval, Recall@100 is the right metric.
For final ranked output, NDCG@10 is the right metric.
Using NDCG@10 to evaluate a first stage that feeds into a reranker
gives misleading results.
```

## Hole Rate

A complementary metric to Recall@K that measures the frequency of complete failure

- queries where zero relevant documents appear in the candidate set:

```
Hole rate = fraction of queries where Recall@k₁ = 0
          = count(queries with no relevant document in top-k₁) / total_queries
```

A system with Recall@100 = 0.85 might still have a hole rate of 15% if all
its failures are complete misses (zero relevant documents found) rather than
partial misses (some but not all relevant documents found). Hole rate directly
measures the fraction of queries the pipeline has no chance of answering correctly.

### Why hole rate matters independently from Recall@K

```
System A: Recall@100 = 0.83, Hole rate = 5%
  Most queries: retrieves some but not all relevant docs
  Few queries: complete failures

System B: Recall@100 = 0.83, Hole rate = 20%
  Many queries: perfect recall (all relevant docs retrieved)
  Many queries: complete failures (zero relevant docs retrieved)

Both have identical Recall@100 but System A is more consistent.
System B has bimodal behavior - great for easy queries, terrible for
hard queries - which may be unacceptable for a production system.
```

Users experience hole-rate failures as "search returned nothing useful" -
the most frustrating retrieval failure mode. Tracking hole rate separately from
average recall reveals whether a system fails gracefully (partial misses) or
catastrophically (complete misses).

## Recall Ceiling Analysis

Recall ceiling analysis measures how much first-stage recall limits the maximum
achievable final quality, revealing the headroom that first-stage improvement
would unlock:

### Computing the ceiling

```
For a given test set:
  first_stage_recall = compute_recall_at_k1(retrieval_system, k1=100)

Ceiling NDCG@10 = NDCG@10 that would be achieved if the reranker were perfect
                 (i.e., it perfectly ranks the first-stage candidates)

Ceiling NDCG@10 ≈ NDCG@10 computed assuming the first-stage candidates
                  are the only available documents and the oracle ranking
                  (relevance-sorted) is applied
```

The gap between actual final NDCG@10 and ceiling NDCG@10 has two components:

```
Gap = (ceiling NDCG@10 - actual first stage NDCG@10) → reranker headroom
    + (maximum possible NDCG@10 - ceiling NDCG@10)   → recall headroom

Reranker headroom:  how much better reranking could be with the current candidates
Recall headroom:    how much better the pipeline could be with more recall
```

This analysis directly guides investment decisions:

```
Scenario A: Large reranker headroom, small recall headroom
  → Invest in better reranker
  → First stage is sufficient, reranker is the bottleneck

Scenario B: Small reranker headroom, large recall headroom
  → Invest in better first stage (higher Recall@100)
  → Reranker is already near optimal, recall is the bottleneck

Scenario C: Both large
  → Invest in first stage first (unlocks ceiling)
  → Then invest in reranker (fills up to the new ceiling)
```

## Recall Depth Profile

Rather than a single Recall@K number, the recall depth profile shows how recall
grows as K increases - providing a complete picture of first-stage behavior:

```
K          BM25      Dense     Hybrid
──────────────────────────────────────
10         0.431     0.572     0.611
25         0.612     0.721     0.762
50         0.723     0.812     0.849
100        0.821     0.871     0.912
200        0.878     0.908     0.946
500        0.921     0.942     0.968
1000       0.951     0.966     0.982
```

The profile reveals:

- How steeply recall grows with K (steep = efficiently ordered candidates)
- Where diminishing returns set in (flat plateau = marginal documents beyond that K)
- Whether a system finds relevant documents early or only with large K

A retrieval system that achieves Recall@100 = 0.85 but Recall@10 = 0.35 is
distributing relevant documents sparsely through the candidate list - they are
there but buried. The reranker must sift through many irrelevant candidates to
find them. A system with Recall@10 = 0.65 and Recall@100 = 0.87 is more efficiently
ordered - relevant documents appear earlier, making the reranker's job easier.

## Multi-Stage Recall Cascade

For three-stage pipelines (first stage → intermediate filter → final reranker),
recall must be tracked at each handoff:

```
Full corpus (10M docs)
  ↓ BM25/dense first stage
Top-1000 candidates (Recall@1000 = 0.95)
  ↓ lightweight bi-encoder filter
Top-100 candidates (Recall@100 among top-1000 = 0.88)
  ↓ full cross-encoder reranker
Top-10 final results

Overall pipeline Recall@10 (from full corpus) ≈ ?
  Maximum possible ≈ min(0.95, 0.88) × per-stage efficiency
  ≈ 0.95 × 0.88 × (reranker precision@10)
  ≈ 0.836 × (reranker precision)

If reranker achieves 90% precision@10 from its candidates:
  ≈ 0.75 end-to-end recall at top-10
```

Each stage introduces recall loss. The cascade structure means first-stage recall
loss is amplified by subsequent stages - a 10% recall loss at the first stage
becomes a ~10% degradation ceiling for all downstream stages.

### Recall drop between stages

Tracking recall drop at each handoff is essential for diagnosing pipeline failures:

```
Stage 1→2 recall drop:
  Recall@1000 = 0.95
  Recall retained in top-100 = 0.88
  Drop = 0.07 → 7% of relevant documents lost at intermediate filter

Stage 2→3 recall drop:
  Relevant docs in top-100 = 88% of all relevant
  Relevant docs in top-10 = 65% of all relevant
  Drop = 0.23 → 23% lost at final reranker (ranking precision issue)
```

Diagnosing which stage causes the most recall loss determines where to
invest optimization effort.

## Recall vs Precision Tradeoffs for Different Applications

The right Recall@K target depends on the application's tolerance for missed
relevant documents:

### High-recall applications

Legal discovery, patent search, and systematic literature review require
finding every relevant document - missing relevant documents has legal or
scientific consequences:

```
Legal e-discovery:
  Recall@K target: 0.95+ at large K
  Acceptable K: 1,000-10,000 documents (human reviewers will screen them)
  Metric: Recall@5000 or even Recall@10000
  NDCG@K: secondary - precision matters less than coverage

Systematic literature review:
  Recall@K target: 0.98+ (near-complete coverage needed)
  K: may be set to entire corpus (screen everything above a threshold)
  NDCG@K: irrelevant (all retrieved documents are reviewed)
```

### Precision-recall balanced applications

E-commerce search, enterprise knowledge bases, and general web search require
balancing recall and precision because users examine only a few results:

```
E-commerce:
  Recall@100 target: 0.85+ (reranker sees 100 candidates)
  Recall@10 target: 0.65+ (some relevant products in final list)
  NDCG@10: primary metric for final quality

Enterprise search:
  Recall@50 target: 0.80+ (employees want to find the right document)
  NDCG@10: important (employees read first few results)
  Hole rate: < 5% (complete failures are very visible and costly)
```

### Low-recall-tolerant applications

Recommendation systems, trending content, and casual browsing tolerate missing
some relevant content because the abundance of alternatives reduces the cost:

```
Content recommendation:
  Recall@10 target: 0.5+ (one good recommendation is sufficient)
  NDCG@5: primary metric
  High recall less critical: many alternatives exist, finding all is not needed

News feed:
  Recall@20: 0.6+ (capture key stories in user's interest)
  Diversity: important (variety matters more than exhaustive recall)
```

## Calibrating First-Stage K₁

Recall depth profiles enable principled selection of K₁ (how many candidates
the first stage retrieves and passes to the reranker):

### The K₁ selection problem

```
Small K₁ (e.g., k₁ = 20):
  Advantages: reranker is fast, low latency
  Risk: low Recall@20, ceiling quality limited

Large K₁ (e.g., k₁ = 500):
  Advantages: high Recall@500, high ceiling quality
  Risk: reranker is slow, high latency

Optimal K₁: the point where the recall depth profile plateaus
            Additional candidates yield diminishing recall gains
            while reranker cost continues to increase linearly
```

### Finding the optimal K₁

From the recall depth profile, find the "knee" where the marginal recall gain
per additional candidate decreases sharply:

```
K    Recall   Marginal gain
──────────────────────────
10   0.43     -
25   0.61     0.18/15 = 0.012 per doc
50   0.72     0.11/25 = 0.0044 per doc
100  0.85     0.13/50 = 0.0026 per doc   ← diminishing returns begin here
200  0.89     0.04/100 = 0.0004 per doc  ← large drop in marginal gain
500  0.93     0.04/300 = 0.0001 per doc

Knee ≈ K = 100: marginal gain per document drops sharply after 100
→ Optimal K₁ ≈ 100 for this system
```

This recall depth profile analysis is the principled method for selecting K₁
rather than using the common rule-of-thumb of K₁ = 100.

## Recall Evaluation Across Query Subgroups

Aggregate recall metrics can mask systematic failures on specific query types.
Segment-level recall analysis reveals which query categories the first stage
handles poorly:

### Common segmentation dimensions

**Query length:**

```
Short queries (1-2 words):   Recall@100 = 0.78  ← often worse
Medium queries (3-5 words):  Recall@100 = 0.86
Long queries (6+ words):     Recall@100 = 0.91  ← often better
```

Short queries are ambiguous and tend to have lower recall because the first
stage cannot determine which interpretation to retrieve for.

**Query type:**

```
Navigational queries:  Recall@100 = 0.94 (find specific page → high recall)
Informational queries: Recall@100 = 0.83 (broad topic → moderate recall)
Transactional queries: Recall@100 = 0.79 (find product/service → variable)
```

**Domain specificity:**

```
General queries:         Recall@100 = 0.87
Domain-specific queries: Recall@100 = 0.71  ← recall drops in specialized domains
Technical queries:       Recall@100 = 0.75
```

**Vocabulary mismatch level:**

```
Low mismatch (query terms appear in documents):  Recall@100 = 0.91
Medium mismatch (some synonyms needed):          Recall@100 = 0.83
High mismatch (different vocabulary required):   Recall@100 = 0.67
```

Segment analysis reveals which query types need targeted improvement (vocabulary
expansion, domain adaptation) versus which are already well served.

## Recall Evaluation for Dense vs Sparse Retrieval

A systematic empirical finding from BEIR and related work: sparse and dense
retrieval have complementary recall profiles that explain why hybrid search
consistently improves both recall and quality:

```
Query type                BM25 Recall@100   Dense Recall@100
────────────────────────────────────────────────────────────────
Keyword queries           0.91              0.82  ← BM25 wins
Natural language queries  0.78              0.89  ← Dense wins
Technical acronym queries 0.88              0.74  ← BM25 wins
Synonym-heavy queries     0.65              0.88  ← Dense wins
Named entity queries      0.93              0.85  ← BM25 wins

Hybrid:                   0.93              0.94  (consistently better)
```

The complementarity is the empirical justification for hybrid search. It is not
just about average performance - it is about covering recall gaps that each method
has independently.

### Recall failure mode analysis for each method

**BM25 recall failures:**

- Vocabulary mismatch: query uses synonyms not in relevant documents
- Paraphrase queries: documents express the relevant content in different words
- Cross-lingual: query and document in different languages

**Dense recall failures:**

- Rare terms: specific identifiers, version numbers, proper nouns not in training
- Long queries: semantic compression loses specific constraint information
- Domain shift: relevant documents in domain poorly covered by pretraining

Understanding these failure modes guides targeted recall improvement:
add SPLADE to BM25 for vocabulary expansion, add hybrid for complementary coverage.

## Relating Recall Metrics to User Experience

Recall metrics are internal technical metrics. Translating them to user
experience helps justify engineering investment:

```
Recall@100 = 0.75:
  25% of relevant documents never seen by the reranker
  If a query has 5 relevant documents: expected 1.25 relevant docs missing from pool
  User experience: occasionally (perhaps 20-30% of queries) the best answer is missing
  User behavior: more reformulations, longer sessions, lower satisfaction

Recall@100 = 0.90:
  10% of relevant documents never seen
  If a query has 5 relevant documents: expected 0.5 relevant docs missing
  User experience: best answer missing on ~10% of queries
  User behavior: some reformulations, moderate satisfaction

Recall@100 = 0.95:
  5% of relevant documents never seen
  User experience: best answer missing very occasionally
  User behavior: low reformulation rate, high satisfaction
```

This translation is useful for communicating the value of first-stage improvements
to non-technical stakeholders - "improving Recall@100 from 75% to 90% means
users get the best available answer on 15% more queries."

Recall-oriented evaluation connects the evaluation module to the practical
systems concerns covered in 11-practical-systems/ and 16-reranking-deep-dive/.
Understanding which metrics to optimize at each pipeline stage is what prevents
the common mistake of using NDCG@10 to make decisions about first-stage retrieval
systems - a metric mismatch that leads to optimizing the wrong objective and
building retrieval pipelines that underperform despite good component-level scores.

## My Summary

Recall-oriented evaluation measures whether a first-stage retrieval component
successfully gathers all relevant documents into the candidate set passed to the
reranker, using Recall@K and hole rate rather than NDCG@K which measures precision
at top ranks. The core insight is that first-stage retrieval has a different
objective than the full pipeline: its job is to maximize recall (not miss any
relevant documents) while the reranker's job is to maximize precision (rank correctly).
Evaluating the first stage with NDCG@10 is a metric mismatch that hides important
differences - two systems with identical NDCG@10 can have Recall@100 values of 0.73
and 0.87, producing dramatically different reranking ceilings. Recall ceiling analysis
reveals the maximum achievable final NDCG given the first-stage candidates and
identifies whether to invest in better first-stage recall (raise the ceiling) or
better reranking (fill up to the existing ceiling). The recall depth profile shows
how recall grows with K, identifying the optimal K₁ where marginal gains from
additional candidates diminish. Sparse and dense retrieval have complementary recall
failure modes - BM25 fails on vocabulary mismatch, dense fails on rare exact-match
terms - which is the fundamental empirical justification for hybrid search's
consistently superior recall across diverse query types.
