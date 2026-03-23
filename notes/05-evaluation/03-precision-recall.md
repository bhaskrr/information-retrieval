# Precision and Recall

## What is it?

Precision and recall are the two foundational metrics in IR evaluation. They measure
complementary aspects of retrieval quality — precision measures how much of what was
retrieved is useful, and recall measures how much of what is useful was retrieved.
Every other IR metric is either a combination of these two or an extension of them.

## Intuition

Think of a fishing net:

- **Precision** — of all the fish you caught, what fraction are the ones you wanted?
  A net full of the right fish = high precision.
- **Recall** — of all the fish you wanted, what fraction did you actually catch?
  Catching every fish you wanted = high recall.

A net with tiny holes catches everything (high recall) but is full of unwanted fish
(low precision). A very selective net catches only the right fish (high precision)
but misses many (low recall). This tension is the precision-recall tradeoff.

## The Confusion Matrix

For a single query, every document in the corpus falls into one of four categories:

```
                        Retrieved       Not Retrieved
                   ┌───────────────┬───────────────┐
      Relevant     │      TP        │      FN       │
                   ├───────────────┼───────────────┤
  Not Relevant     │      FP        │      TN       │
                   └───────────────┴───────────────┘
```

- **TP (True Positive)** — retrieved and relevant
- **FP (False Positive)** — retrieved but not relevant
- **FN (False Negative)** — relevant but not retrieved
- **TN (True Negative)** — not retrieved and not relevant

## Precision

Of everything retrieved, what fraction is relevant?

```
Precision = TP / (TP + FP) = TP / |Retrieved|
```

Range: 0 (nothing retrieved is relevant) to 1 (everything retrieved is relevant)

## Recall

Of everything relevant, what fraction was retrieved?

```
Recall = TP / (TP + FN) = TP / |Relevant|
```

Range: 0 (nothing relevant was retrieved) to 1 (all relevant docs were retrieved)

## Worked Example

Corpus: D1 through D10
Relevant documents for query q: {D1, D3, D5, D7, D9} → 5 relevant docs

System retrieves: {D1, D2, D3, D4, D5}

```
TP = {D1, D3, D5}          (retrieved AND relevant)
FP = {D2, D4}              (retrieved but NOT relevant)
FN = {D7, D9}              (relevant but NOT retrieved)

Precision = 3 / 5 = 0.60
Recall    = 3 / 5 = 0.60
```

Now the system retrieves: {D1, D3, D5, D7, D9, D10}

```
TP = {D1, D3, D5, D7, D9}
FP = {D10}
FN = {}

Precision = 5 / 6 = 0.833
Recall    = 5 / 5 = 1.000
```

Now the system retrieves all 10 documents:

```
TP = {D1, D3, D5, D7, D9}
FP = {D2, D4, D6, D8, D10}

Precision = 5 / 10 = 0.50
Recall    = 5 / 5  = 1.00
```

Recall is perfect but precision dropped — retrieving everything guarantees perfect
recall but destroys precision.

## The Precision-Recall Tradeoff

Precision and recall pull in opposite directions:

- **Retrieving more documents** increases recall but decreases precision
- **Retrieving fewer documents** increases precision but decreases recall

Different tasks sit at different points on this curve:

```
Task                    Priority
─────────────────────────────────────────────────
Web search              Precision — top results must be good
Legal discovery         Recall — must not miss any relevant document
Medical diagnosis       Recall — missing a relevant case is dangerous
Spam filtering          Precision — blocking good email hurts trust
Recommendation          Precision — irrelevant recommendations annoy users
```

## The Precision-Recall Curve

As you move down a ranked list, precision and recall both change. Plotting precision
against recall at each rank position gives the precision-recall curve.

```
Ranked list: D1(rel), D2(not), D3(rel), D4(not), D5(rel)
Relevant docs in corpus: {D1, D3, D5}

After D1: P=1.00, R=0.33
After D2: P=0.50, R=0.33
After D3: P=0.67, R=0.67
After D4: P=0.50, R=0.67
After D5: P=0.60, R=1.00
```

A system whose curve stays high across all recall levels is better than one whose
precision drops steeply as recall increases.

## Problems with Precision and Recall

### 1. No ranking sensitivity

Standard precision and recall treat retrieval as an unordered set. A system that
ranks relevant documents first scores identically to one that ranks them last —
as long as the same set of documents is retrieved.

```
System A: [D1(rel), D3(rel), D2(not), D4(not)]
System B: [D2(not), D4(not), D1(rel), D3(rel)]

Both: Precision=0.5, Recall=1.0  — but A is clearly better
```

This is why ranked metrics like MAP, MRR, and NDCG were developed.

### 2. Requires knowing total relevant documents

Recall requires knowing |Relevant| — the total number of relevant documents in the
corpus. At web scale this is essentially unknowable.

### 3. Treats all relevant documents equally

A highly relevant document counts the same as a marginally relevant one. Graded
metrics like NDCG address this.

## My Summary

Precision measures what fraction of retrieved documents are relevant; recall measures
what fraction of relevant documents were retrieved. They capture complementary aspects
of retrieval quality and trade off against each other — retrieving more improves recall
but hurts precision. Both metrics treat retrieval as an unordered set with no
sensitivity to ranking order, which is their main limitation. Every subsequent metric
in this module either combines them, restricts them to a rank cutoff, or extends them
to handle ranking and graded relevance.
