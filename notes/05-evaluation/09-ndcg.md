# Normalized Discounted Cumulative Gain (NDCG)

NDCG is a ranking evaluation metric that measures the quality of a ranked retrieval
result while accounting for both the position of relevant documents and how relevant
they are. It handles graded relevance — documents can be highly relevant, partially
relevant, or not relevant at all — and applies a logarithmic discount that reduces
the contribution of relevant documents appearing lower in the ranked list.

NDCG is the dominant evaluation metric in modern IR research and most production
ranking systems.

## Intuition

MAP and MRR assume binary relevance — a document is either relevant or not. Real
relevance is a spectrum. A document that perfectly answers a query is more valuable
than one that tangentially mentions the query terms. NDCG captures this distinction.

Two additional intuitions drive NDCG:

1. **Position matters** — a highly relevant document at rank 1 is more valuable than
   the same document at rank 10. Users are less likely to read lower results. The
   metric should reflect this.

2. **Ideal comparison** — a raw score is meaningless without context. NDCG normalizes
   by the best possible score achievable given the available relevant documents, making
   scores comparable across queries with different numbers of relevant documents.

## Building Up to NDCG — Three Steps

### Step 1 — Cumulative Gain (CG)

Simply sum the relevance scores of retrieved documents:

```bash
CG@K = Σ rel(i)   for i = 1 to K
```

Where rel(i) is the relevance score of the document at rank i.

```bash
Ranked list:  [rel=3, rel=0, rel=2, rel=1, rel=0]
CG@5 = 3 + 0 + 2 + 1 + 0 = 6
```

Problem: CG ignores position entirely. The same documents in any order give the
same CG. A highly relevant document at rank 5 counts the same as one at rank 1.

### Step 2 — Discounted Cumulative Gain (DCG)

Apply a logarithmic discount that reduces the contribution of documents at lower ranks:

```bash
DCG@K = Σ rel(i) / log₂(i + 1)   for i = 1 to K
```

The discount at each rank:

```bash
Rank 1:  log₂(2) = 1.000  → no discount
Rank 2:  log₂(3) = 1.585  → divide by 1.585
Rank 3:  log₂(4) = 2.000  → divide by 2
Rank 4:  log₂(5) = 2.322  → divide by 2.322
Rank 5:  log₂(6) = 2.585  → divide by 2.585
Rank 10: log₂(11)= 3.459  → divide by 3.459
```

The log₂ discount grows slowly — rank 2 is penalized less than rank 3, which is
penalized less than rank 4. This reflects the diminishing marginal cost of lower
positions, consistent with how users scan results.

Alternative DCG formula (used in some benchmarks, emphasizes highly relevant docs):

```bash
DCG@K = Σ (2^rel(i) - 1) / log₂(i + 1)   for i = 1 to K
```

This variant gives exponentially more weight to highly relevant documents (e.g.
rel=3 contributes 7, rel=2 contributes 3, rel=1 contributes 1). The standard
formula is more common in academic IR; the exponential variant is used in some
industry benchmarks.

### Step 3 — Normalized DCG (NDCG)

DCG scores are not comparable across queries — a query with 10 highly relevant
documents will naturally have a higher DCG than one with 2 marginally relevant ones.
Normalize by the Ideal DCG (IDCG) — the DCG achieved by a perfect ranking:

```bash
NDCG@K = DCG@K / IDCG@K
```

IDCG@K is computed by sorting all known relevant documents by their relevance score
in descending order and computing DCG on that ideal ranking.

Range: 0 (worst) to 1 (perfect ranking)

## Worked Example

Relevant documents and their grades for query q:

```bash
D1: relevance = 3  (highly relevant)
D3: relevance = 2  (relevant)
D5: relevance = 1  (marginally relevant)
D7: relevance = 0  (not relevant)
D9: relevance = 0  (not relevant)
```

System ranked list:

```bash
Rank 1:  D3  → rel = 2
Rank 2:  D1  → rel = 3
Rank 3:  D9  → rel = 0
Rank 4:  D5  → rel = 1
Rank 5:  D7  → rel = 0
```

### DCG@5 (standard formula)

```bash
DCG@5 = 2/log₂(2) + 3/log₂(3) + 0/log₂(4) + 1/log₂(5) + 0/log₂(6)
      = 2/1.000 + 3/1.585 + 0/2.000 + 1/2.322 + 0/2.585
      = 2.000 + 1.893 + 0.000 + 0.431 + 0.000
      = 4.324
```

### IDCG@5 — ideal ranking sorts by relevance descending

```bash
Ideal rank 1: D1 → rel = 3
Ideal rank 2: D3 → rel = 2
Ideal rank 3: D5 → rel = 1
Ideal rank 4: D7 → rel = 0
Ideal rank 5: D9 → rel = 0

IDCG@5 = 3/log₂(2) + 2/log₂(3) + 1/log₂(4) + 0/log₂(5) + 0/log₂(6)
       = 3/1.000 + 2/1.585 + 1/2.000 + 0 + 0
       = 3.000 + 1.262 + 0.500 + 0 + 0
       = 4.762
```

### NDCG@5

```bash
NDCG@5 = DCG@5 / IDCG@5
       = 4.324 / 4.762
       = 0.908
```

The system scores 0.908 — close to ideal but penalized for placing D1 (rel=3) at
rank 2 instead of rank 1, and for placing D5 (rel=1) at rank 4 instead of rank 3.

### What if D1 were ranked first?

```bash
Rank 1: D1 → rel = 3
Rank 2: D3 → rel = 2
Rank 3: D9 → rel = 0
Rank 4: D5 → rel = 1
Rank 5: D7 → rel = 0

DCG@5 = 3/1.000 + 2/1.585 + 0/2.000 + 1/2.322 + 0/2.585
      = 3.000 + 1.262 + 0.000 + 0.431 + 0.000
      = 4.693

NDCG@5 = 4.693 / 4.762 = 0.986
```

Ranking the most relevant document first improves NDCG from 0.908 to 0.986.

## NDCG@K

NDCG is almost always reported with a rank cutoff K:

```bash
Context                         Typical K
──────────────────────────────────────────
Web search                      NDCG@10
TREC Deep Learning track        NDCG@10, NDCG@100
BEIR benchmark                  NDCG@10
Recommendation systems          NDCG@10, NDCG@20
Neural IR papers                NDCG@10
```

NDCG@10 is the standard metric in virtually every modern IR benchmark. It reflects
the first page of web search results and is the number reported in most neural IR
papers.

## NDCG vs Other Metrics

| Property                  | MAP           | MRR       | NDCG              |
| ------------------------- | ------------- | --------- | ----------------- |
| Graded relevance          | No            | No        | Yes               |
| Position discounting      | Yes (partial) | First hit | Yes (logarithmic) |
| Normalized across queries | No            | No        | Yes (0 to 1)      |
| Handles multiple rel docs | Yes           | No        | Yes               |
| Standard in modern IR     | Less common   | QA tasks  | Most common       |
| Requires graded qrels     | No            | No        | Yes               |

## Limitations of NDCG

### Requires graded relevance judgments

Binary qrels (0/1) can be used with NDCG but the metric then behaves similarly to
MAP — the graded relevance benefit is lost. Getting high-quality graded judgments
is expensive.

### IDCG assumes perfect knowledge

IDCG is computed from known relevant documents. Due to pooling, some relevant
documents may be unjudged and assumed non-relevant — making IDCG an underestimate
and NDCG potentially inflated for systems that retrieve unjudged relevant documents.

### Choice of discount function

The log₂ discount is standard but arbitrary. Different discount functions would
produce different scores. The standard formula and exponential formula give
different rankings of systems in some cases.

### Not directly interpretable

NDCG@10 = 0.72 is hard to interpret in absolute terms. It is most useful as a
relative metric for comparing systems, not as an absolute quality indicator.

## The Full Evaluation Metrics Picture

Now that all metrics have been covered, here is how they relate:

```bash
Metric          Relevance    Position    Multi-doc    Best for
──────────────────────────────────────────────────────────────────────────
Precision       Binary       No          Yes          Simple set retrieval
Recall          Binary       No          Yes          Coverage tasks
F-Measure       Binary       No          Yes          NLP classification
P@K             Binary       Partial     Yes          Quick quality check
AP / MAP        Binary       Yes         Yes          Academic IR, TREC
MRR             Binary       First hit   No           Single-answer tasks
NDCG            Graded       Yes         Yes          Modern IR, rankings
```

## Where This Fits in the Progression

```bash
Precision & Recall  → unranked set metrics
F-Measure           → single number combining both
Precision@K         → precision at a fixed cutoff
Average Precision   → precision over all relevant positions
MAP                 → mean AP across queries
MRR                 → rank of first relevant result
NDCG                → graded relevance + position discount  ← you are here
```

NDCG is the culmination of the evaluation module. Every limitation of the previous
metrics — no ranking, binary relevance, first-hit only — is addressed here. The
progression from precision to NDCG is the progression from asking "did you find
anything useful?" to asking "did you find the most useful things and put them
where users will actually see them?"

## My Summary

NDCG measures ranked retrieval quality by summing relevance scores discounted by
their logarithmic rank position, then normalizing by the ideal score achievable
given the available relevant documents. It handles graded relevance — distinguishing
highly relevant from marginally relevant — and rewards surfacing the most relevant
documents early. NDCG@10 is the standard metric in virtually every modern IR
benchmark. Its main requirements are graded relevance judgments and a defined rank
cutoff K. The full evaluation progression — from precision and recall through MAP,
MRR, and finally NDCG — reflects the evolution of understanding what "good retrieval"
actually means: not just finding relevant documents, but finding the right ones and
putting them first.
