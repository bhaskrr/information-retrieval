# F-Measure

## What is it?

F-Measure (also called F-score) is a single metric that combines precision and recall
into one number. It is the harmonic mean of precision and recall, and summarizes the
tradeoff between the two in a single comparable value.

## Intuition

Precision and recall individually are incomplete. A system that retrieves every document
in the corpus has perfect recall but terrible precision. A system that retrieves exactly
one highly relevant document has perfect precision but terrible recall. Neither extreme
is useful.

F-Measure forces a balance. A system that does well on both precision and recall scores
high. A system that sacrifices one for the other scores lower than it might appear from
either metric alone.

## Why Harmonic Mean and Not Arithmetic Mean

The arithmetic mean of precision and recall is deceiving. Consider:

```
System A: Precision=1.0, Recall=0.0
Arithmetic mean = (1.0 + 0.0) / 2 = 0.50

System B: Precision=0.5, Recall=0.5
Arithmetic mean = (0.5 + 0.5) / 2 = 0.50
```

Both systems score 0.50 arithmetically even though System A is completely useless —
it retrieved nothing relevant at all (or retrieved one perfect result and missed
everything else).

The harmonic mean penalizes extreme imbalance:

```
System A: H-mean = 2 × (1.0 × 0.0) / (1.0 + 0.0) = 0.00
System B: H-mean = 2 × (0.5 × 0.5) / (0.5 + 0.5) = 0.50
```

System A correctly scores 0 — it is useless on one dimension. The harmonic mean pulls
the combined score toward whichever of the two values is lower.

## F1 Score

The most common variant — weights precision and recall equally:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Equivalently expressed using TP, FP, FN:

```
F1 = 2TP / (2TP + FP + FN)
```

Range: 0 (worst) to 1 (best)

## Worked Example

```
Retrieved: [D1, D2, D3, D4, D5, D6]
Relevant:  {D1, D3, D5, D7, D9}

TP = {D1, D3, D5} = 3
FP = {D2, D4, D6} = 3
FN = {D7, D9}     = 2

Precision = 3 / 6 = 0.500
Recall    = 3 / 5 = 0.600

F1 = 2 × (0.500 × 0.600) / (0.500 + 0.600)
   = 2 × 0.300 / 1.100
   = 0.600 / 1.100
   = 0.545
```

F1 = 0.545 sits between precision and recall, pulled slightly toward the lower value
(precision = 0.5).

## F-Beta Score

F1 weights precision and recall equally. Sometimes one matters more than the other.
F-beta introduces a parameter β that controls the weighting:

```
Fβ = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

- **β < 1** — weights precision more heavily (e.g. β=0.5)
- **β = 1** — equal weight, same as F1
- **β > 1** — weights recall more heavily (e.g. β=2)

### When to use F-beta

```bash
Task                        β choice    Reason
──────────────────────────────────────────────────────────────────
Web search                  β=0.5       Precision matters more
Legal discovery             β=2         Recall matters more — cannot miss docs
Medical screening           β=2         Missing a diagnosis is worse than a
                                        false positive
Spam filtering              β=0.5       False positives (blocking real email)
                                        are more costly than false negatives
```

### Example with β=2 (recall-weighted)

```
Precision = 0.500, Recall = 0.600

F2 = (1 + 4) × (0.500 × 0.600) / (4 × 0.500 + 0.600)
   = 5 × 0.300 / (2.000 + 0.600)
   = 1.500 / 2.600
   = 0.577
```

F2 = 0.577 > F1 = 0.545 because recall (0.6) is higher than precision (0.5) and
β=2 rewards recall more heavily.

### Example with β=0.5 (precision-weighted)

```bash
F0.5 = (1 + 0.25) × (0.500 × 0.600) / (0.25 × 0.500 + 0.600)
     = 1.25 × 0.300 / (0.125 + 0.600)
     = 0.375 / 0.725
     = 0.517
```

F0.5 = 0.517 < F1 = 0.545 because precision (0.5) is lower than recall (0.6) and
β=0.5 rewards precision more heavily, pulling the score down.

## When F-Measure is Misleading

### Class imbalance

In IR, the number of non-relevant documents vastly outnumbers relevant ones. F-measure ignores true negatives (TN) entirely. A system that retrieves nothing scores F1=0, but so does a system that retrieves only irrelevant documents — F-measure does not distinguish between these failure modes.

### No ranking sensitivity

Like precision and recall, F-measure treats retrieval as an unordered set. It does not care whether relevant documents appear at rank 1 or rank 1000 as long as they are in the retrieved set. This is its core limitation for ranked retrieval tasks.

### Aggregating across queries

F-measure is computed per query. Averaging F1 across queries is common but hides
variance — a system with F1=0.8 on easy queries and F1=0.2 on hard queries averages
the same as one consistently achieving F1=0.5.

## F-Measure vs MAP vs NDCG

| Property                  | F-Measure   | MAP        | NDCG        |
| ------------------------- | ----------- | ---------- | ----------- |
| Ranking sensitivity       | No          | Yes        | Yes         |
| Graded relevance          | No          | No         | Yes         |
| Single query summary      | Yes         | Yes (AP)   | Yes         |
| Handles missing judgments | Poorly      | Moderately | Moderately  |
| Still used in IR          | Less common | Common     | Most common |

F-measure is more commonly used in NLP classification tasks (NER, relation extraction, text classification) than in ranked IR evaluation. In IR it appears mostly as a theoretical building block and in tasks where ranking does not matter.

## My Summary

F-Measure is the harmonic mean of precision and recall, combining both into a single score that penalizes extreme imbalance between the two. F1 weights them equally; F-beta lets you shift weight toward precision (β < 1) or recall (β > 1) depending on the task. The harmonic mean is used instead of arithmetic mean because it correctly scores zero when either dimension is zero. F-measure is more commonly seen in NLP classification tasks than in ranked IR; its main limitation is no sensitivity to document ranking order, which is why MAP and NDCG are preferred for ranked retrieval evaluation.
