# Extended Boolean Model

The Extended Boolean Model is a retrieval model that generalizes the Boolean model by introducing partial matching and continuous relevance scores. Instead of a document either satisfying a query or not, documents are scored on a spectrum of how well they match, allowing ranking while preserving the familiar AND/OR/NOT query structure.

## Intuition

The core failure of Boolean retrieval is its binary nature. A document matching 9 out of 10 query terms scores identically to one matching 0, both are simply "not retrieved". This feels wrong.

The Extended Boolean Model fixes this by asking: what if AND and OR were not hard
logical operators but soft ones that measured degree of match?

Think of it geometrically. In the Boolean model, a document either is or is not in the result set. In the Extended Boolean Model, every document has a distance from the ideal match and that distance determines its rank.

## From Boolean to Extended Boolean

### Hard AND vs. Soft AND

Boolean AND: retrieved only if ALL terms present

```bash
query: "python AND search AND index"
doc matches "python" and "search" but not "index" -> not retrieved
```

Extended AND: score reflects how close the document is to matching ALL terms

```bash
same doc -> high score, not perfect, but not zero either
```

### Hard OR vs. Soft OR

Boolean OR: retrieved if ANY term present, all equally

```bash
doc matching one term = doc matching all terms, no distinction
```

Extended OR: score reflects how many terms match, not just whether any do

```bash
doc matching 3 terms scores higher than doc matching 1 term
```

## The Formal Model - P-norm

The Extended Boolean Model uses a p-norm distance to measure how far a document is
from the ideal match.

### Setup

Each document d is represented as a vector of term weights:

```bash
d = (w₁, w₂, ..., wₙ)
```

where wᵢ is the TF-IDF weight of term i in document d (between 0 and 1).

### Soft OR score

For a query q = t₁ OR t₂ OR ... OR tₙ:

```
score_OR(d, q) = ( (w₁ᵖ + w₂ᵖ + ... + wₙᵖ) / n ) ^ (1/p)
```

This is the Lp-norm of the weight vector, normalized by n.

### Soft AND score

For a query q = t₁ AND t₂ AND ... AND tₙ:

```
score_AND(d, q) = 1 - ( ((1-w₁)ᵖ + (1-w₂)ᵖ + ... + (1-wₙ)ᵖ) / n ) ^ (1/p)
```

The AND score measures how close the document is to having ALL terms — it penalizes
missing terms via (1 - wᵢ).

### The role of p

The parameter p controls how "hard" the operators are:

```
p = 1  → pure averaging, very soft (OR becomes mean, AND becomes 1 - mean)
p = ∞  → recovers hard Boolean (OR becomes max, AND becomes min)
p = 2  → the standard choice, Euclidean norm, good balance
```

As p increases, the model behaves more like strict Boolean.
As p decreases toward 1, it behaves more like pure averaging.

### Worked Example

Query: "python AND search"
Documents with TF-IDF weights:

```
        python   search
D1:     0.8      0.6
D2:     0.9      0.0
D3:     0.5      0.5
```

Soft AND score with p=2:

```
D1: 1 - sqrt( ((1-0.8)² + (1-0.6)²) / 2 )
  = 1 - sqrt( (0.04 + 0.16) / 2 )
  = 1 - sqrt(0.10)
  = 1 - 0.316 = 0.684

D2: 1 - sqrt( ((1-0.9)² + (1-0.0)²) / 2 )
  = 1 - sqrt( (0.01 + 1.00) / 2 )
  = 1 - sqrt(0.505)
  = 1 - 0.711 = 0.289

D3: 1 - sqrt( ((1-0.5)² + (1-0.5)²) / 2 )
  = 1 - sqrt( (0.25 + 0.25) / 2 )
  = 1 - sqrt(0.25)
  = 1 - 0.5 = 0.5
```

Ranking: D1 (0.684) > D3 (0.5) > D2 (0.289)

D2 scores lowest despite having a strong "python" signal because it completely
misses "search" — the soft AND penalizes missing terms heavily. In Boolean retrieval
D2 would simply not be retrieved at all.

## Strengths

- **Partial matching** — documents that almost satisfy the query are retrieved and
  ranked, not silently dropped
- **Familiar syntax** — users still write AND/OR/NOT queries; no new query language
- **Tunable hardness** — p controls the trade-off between strict and soft matching
- **Bridges Boolean and VSM** — conceptually sits between the two, useful for
  understanding the progression

## Weaknesses

- **TF-IDF dependency** — requires computing term weights, adding complexity over
  pure Boolean
- **p is a hyperparameter** — the right value of p is not obvious and is corpus
  dependent
- **Rarely used in practice** — the Vector Space Model and BM25 superseded it
  quickly; the Extended Boolean Model is primarily of theoretical and historical
  interest today
- **Still no semantics** — like Boolean, it has no understanding of synonyms or
  meaning; "car" and "automobile" are still different terms

## Where this fits in the progression

```bash
Boolean model       → binary match, no ranking, feast or famine
Extended Boolean    → soft match, partial scoring, familiar query syntax  ← you are here
Vector Space Model  → geometric similarity, full continuous scoring
BM25                → probabilistic relevance, length normalization
```

Extended Boolean is the conceptual bridge. It shows that the move from binary to
continuous scoring is not a radical departure — it is a natural generalization of
the same operators users already know. VSM takes this further by abandoning the
Boolean syntax entirely in favor of a geometric framework.

## My Summary

The Extended Boolean Model softens Boolean AND and OR into continuous scoring functions
using a p-norm distance measure. A document is no longer simply in or out — it receives
a score reflecting how closely it matches all query terms (soft AND) or any of them
(soft OR). The parameter p controls hardness: p=1 is pure averaging, p=∞ recovers hard
Boolean, p=2 is the standard choice. It fixes the partial matching problem of Boolean
retrieval while keeping familiar query syntax, but was quickly superseded by the Vector
Space Model in practice. Its main value today is as the conceptual bridge between
Boolean and ranked retrieval.
