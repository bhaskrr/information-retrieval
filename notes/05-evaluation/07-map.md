# Mean Average Precision (MAP)

Mean Average Precision (MAP) is the mean of Average Precision (AP) scores across a set of queries. It is a single number that summarizes ranked retrieval quality over an entire test collection. MAP was the dominant evaluation metric in IR research for over a decade and remains widely used in academic benchmarks today.

## Intuition

P@K measures precision at a single rank cutoff. Average Precision extends this by
measuring precision at every position where a relevant document appears, then
averaging those values - rewarding systems that rank relevant documents as early
as possible.

MAP then averages those AP scores across all queries in the test collection -
giving a single number that reflects system performance across a diverse set of
information needs.

A system that consistently ranks relevant documents early, across many different
queries, achieves a high MAP.

## Building Up to MAP

### Step 1 - Precision at each relevant document

For a ranked list, compute precision at every rank where a relevant document appears.
Ignore ranks where the retrieved document is not relevant.

### Step 2 - Average Precision (AP) for one query

Average the precision values computed in Step 1.

$AP = \frac{1}{|Relevant|}\times\sum P@k \times rel(k)$

Where:

- |Relevant| = total number of relevant documents for this query
- P@k = precision at rank k
- rel(k) = 1 if document at rank k is relevant, 0 otherwise

The sum runs over all retrieved documents but only accumulates when a relevant
document is encountered.

### Step 3 - MAP across all queries

$MAP = \frac{1}{|Q|} \times \sum AP(q_i)$

Where |Q| is the number of queries in the test collection.

## Worked Example

Relevant documents for query q: {D1, D3, D5, D7, D9}: 5 relevant docs total

Ranked retrieval result:

```bash
Rank 1:  D1   relevant   P@1  = 1/1  = 1.000
Rank 2:  D2   not relevant
Rank 3:  D3   relevant   P@3  = 2/3  = 0.667
Rank 4:  D4   not relevant
Rank 5:  D5   relevant   P@5  = 3/5  = 0.600
Rank 6:  D8   not relevant
Rank 7:  D7   relevant   P@7  = 4/7  = 0.571
Rank 8:  D6   not relevant
Rank 9:  D9   relevant   P@9  = 5/9  = 0.556
Rank 10: D10  not relevant
```

Precision values at relevant document positions: 1.000, 0.667, 0.600, 0.571, 0.556

```bash
AP = (1.000 + 0.667 + 0.600 + 0.571 + 0.556) / 5
   = 3.394 / 5
   = 0.679
```

Now consider a worse system that ranks relevant documents lower:

```bash
Rank 1:  D2   not relevant
Rank 2:  D4   not relevant
Rank 3:  D1   relevant   P@3  = 1/3  = 0.333
Rank 4:  D6   not relevant
Rank 5:  D3   relevant   P@5  = 2/5  = 0.400
Rank 6:  D8   not relevant
Rank 7:  D5   relevant   P@7  = 3/7  = 0.429
Rank 8:  D10  not relevant
Rank 9:  D7   relevant   P@9  = 4/9  = 0.444
Rank 10: D9   relevant   P@10 = 5/10 = 0.500
```

```bash
AP = (0.333 + 0.400 + 0.429 + 0.444 + 0.500) / 5
   = 2.106 / 5
   = 0.421
```

The first system scores AP=0.679 vs AP=0.421 for the second - MAP correctly
penalizes the system that ranks relevant documents lower, even though both
eventually retrieve all 5 relevant documents.

### MAP across multiple queries

```bash
Query q1: AP = 0.679
Query q2: AP = 0.512
Query q3: AP = 0.834
Query q4: AP = 0.290

MAP = (0.679 + 0.512 + 0.834 + 0.290) / 4
    = 2.315 / 4
    = 0.579
```

## What MAP Rewards and Punishes

MAP rewards:

- Relevant documents appearing at early ranks
- Consistent performance across all queries
- Retrieving all relevant documents (missing one reduces AP)

MAP punishes:

- Relevant documents buried deep in the ranked list
- Strong performance on easy queries but poor on hard ones
- Failing to retrieve relevant documents entirely (they contribute 0 precision)

## MAP vs Other Metrics

| Property                  | P@K      | MAP            | MRR       | NDCG      |
| ------------------------- | -------- | -------------- | --------- | --------- |
| Ranking sensitivity       | Partial  | Full           | First hit | Full      |
| Recall sensitivity        | No       | Yes            | No        | Partial   |
| Graded relevance          | No       | No             | No        | Yes       |
| Multi-query summary       | Mean P@K | Yes            | Yes       | Yes       |
| Sensitive to missing docs | No       | Yes            | No        | Partial   |
| Standard in               | Web IR   | TREC, academic | QA tasks  | Modern IR |

## Limitations of MAP

### Binary relevance only

MAP treats every relevant document as equally important. A highly relevant document
counts the same as a marginally relevant one. NDCG addresses this with graded
relevance scores.

### Requires complete relevance judgments

AP divides by |Relevant| - the total number of relevant documents in the corpus.
If some relevant documents were never judged (due to pooling), they are assumed
non-relevant, artificially deflating AP scores. This pool bias affects MAP more
than metrics that only look at the top K.

### Sensitive to queries with many relevant documents

A query with 100 relevant documents contributes more to MAP variance than one with 2. Systems optimized for high-recall queries may score well on MAP without being
useful for queries with few relevant documents.

### Less interpretable than P@K

A MAP of 0.42 is harder to interpret intuitively than "60% of the top 10 results
are relevant." MAP is a relative metric - meaningful for comparison between systems
but not as an absolute quality indicator.

## MAP in Practice

Typical MAP scores on standard benchmarks:

| System type                  | Typical MAP range |
| ---------------------------- | ----------------- |
| BM25 (sparse baseline)       | 0.20 - 0.30       |
| Dense retrieval (bi-encoder) | 0.30 - 0.40       |
| Cross-encoder reranking      | 0.40 - 0.50       |
| Hybrid (sparse + dense)      | 0.40 - 0.55       |

These ranges are dataset-dependent. MS MARCO and TREC DL scores are not directly
comparable to BEIR scores.

## My Summary

MAP is the mean of Average Precision scores across all queries in a test collection.
AP for a single query is computed by calculating precision each time a relevant
document is encountered in the ranked list, then averaging those values over all
relevant documents - naturally rewarding systems that rank relevant documents early.
MAP then averages AP across queries, giving a single number that reflects system
quality across diverse information needs. Its main limitation is the assumption of binary relevance - every relevant document is treated equally regardless of how
relevant it actually is. NDCG addresses this by incorporating graded relevance and a logarithmic position discount.
