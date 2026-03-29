# Mean Reciprocal Rank (MRR)

## What is it?

Mean Reciprocal Rank (MRR) is an evaluation metric that measures where the first
relevant document appears in a ranked retrieval result. For each query, the score
is the reciprocal of the rank of the first relevant document. MRR is the mean of
these reciprocal rank scores across all queries in the test collection.

## Intuition

Some retrieval tasks have exactly one correct answer. If you ask "who invented the
telephone?" there is one right answer. If you search for the homepage of a company,
there is one correct page. In these cases, you do not care about the full ranked list
— you only care about how quickly the system surfaces the first correct result.

MRR captures this precisely. If the correct answer appears at rank 1, you score 1.0.
At rank 2, you score 0.5. At rank 3, you score 0.333. The further down the first
relevant document is, the lower the score. If no relevant document is retrieved, the
score is 0.

## Formula

For a single query q:

```
RR(q) = 1 / rank_first_relevant
```

Where rank_first_relevant is the rank position of the first relevant document in
the retrieved list. If no relevant document is retrieved, RR = 0.

For a set of queries Q:

```
MRR = (1 / |Q|) × Σ RR(qᵢ)
```

## When to Use MRR

MRR is designed for tasks where:

- There is exactly one correct answer or one most-relevant document
- The user only needs to find the first relevant result
- Position of that result is the primary quality signal

```
Task                            MRR appropriate?
──────────────────────────────────────────────────────
Navigational web search         Yes — one correct homepage
Question answering              Yes — one correct answer passage
Entity lookup                   Yes — one correct entity page
Fact retrieval                  Yes — one correct fact
Ad-hoc document retrieval       No — many relevant docs exist
Legal discovery                 No — recall over full list matters
Recommendation                  No — quality of full list matters
```

## MRR vs MAP vs NDCG

| Property          | MRR            | MAP            | NDCG           |
| ----------------- | -------------- | -------------- | -------------- |
| Cares about       | First hit only | All hits       | All hits       |
| Graded relevance  | No             | No             | Yes            |
| Recall sensitive  | No             | Yes            | Partial        |
| Best for          | Single-answer  | Multi-doc IR   | Ranked list IR |
|                   | tasks          |                |                |
| Interpretability  | High           | Medium         | Medium         |
| Common benchmarks | MS MARCO, QA   | TREC, academic | TREC DL, BEIR  |

## MRR@K

In practice MRR is always computed with a rank cutoff K. If the first relevant
document does not appear in the top K, RR = 0 for that query.

```
MRR@10: only consider the top 10 results
MRR@100: only consider the top 100 results
```

MS MARCO uses MRR@10 as its primary evaluation metric for the passage retrieval task.
This directly reflects user behavior — if the correct passage is not in the first 10
results, the system has failed for that query.

## Limitations of MRR

### Only cares about the first relevant document

Everything after the first relevant document is completely ignored. A system that
returns one relevant document at rank 1 and nothing else scores MRR=1.0 — identical
to a system that returns relevant documents at every rank. For tasks with multiple
relevant documents this is a serious blind spot.

```
System A: [rel, rel, rel, rel, rel] → RR = 1.0
System B: [rel, not, not, not, not] → RR = 1.0
Both score the same despite System A being far more useful.
```

### Binary relevance

Like MAP, MRR assumes binary relevance. A highly relevant document at rank 2
scores the same as a marginally relevant one at rank 2.

### Sensitive to the one-answer assumption

MRR is only well-calibrated when the task genuinely has one correct answer. For
open-ended retrieval tasks it gives a misleading picture of system quality.

### High variance on small query sets

With few queries, a single query where the relevant document is not retrieved
(RR=0) can significantly drag down MRR. MAP is more stable across varied query
difficulty.

## Where This Fits in the Progression

```bash
Precision & Recall  → unranked set metrics
F-Measure           → combines precision and recall
Precision@K         → precision at a fixed rank cutoff
Average Precision   → precision integrated over all relevant positions
MAP                 → mean AP across all queries
MRR                 → rank of the first relevant result  ← you are here
NDCG                → graded relevance with position discounting
```

MRR is the most focused metric in this progression — it reduces the entire ranked
list to a single number based on one event: where does the first relevant document
appear? This makes it ideal for single-answer tasks but inappropriate for anything
requiring a complete ranked list. NDCG is the final step — it handles graded
relevance, considers the full ranked list, and applies a logarithmic position
discount that rewards relevant documents appearing early.

## My Summary

MRR measures how quickly a retrieval system surfaces its first relevant result,
scoring each query as 1 divided by the rank of the first relevant document and
averaging across all queries. It is simple, interpretable, and ideal for tasks
with a single correct answer such as question answering and navigational search.
Its core limitation is that it completely ignores everything after the first relevant
document — making it unsuitable for tasks where multiple relevant documents matter.
MRR@10 is the primary metric for MS MARCO passage retrieval. NDCG is the natural
next step for tasks that require evaluating the quality of the full ranked list with
graded relevance.
