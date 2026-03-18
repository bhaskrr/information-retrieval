# Vector Space Model

The Vector Space Model (VSM) is a retrieval model that represents both documents and
queries as vectors in a high-dimensional term space, and ranks documents by their
geometric similarity to the query vector. The closer a document vector is to the query
vector, the more relevant the document is considered to be.

## Intuition

Imagine each unique term in the corpus is an axis in a high-dimensional space. Every
document is a point in that space — its coordinates are determined by how much of each
term it contains. A query is also a point in the same space.

Retrieval becomes a geometric problem: find the documents whose points are closest to
the query point. "Closest" is measured by the angle between vectors, not their length
— because a long document and a short document about the same topic should score
similarly, even though their vectors have very different magnitudes.

This is the first model where retrieval is fully continuous — every document gets a
score, every score is comparable, and ranking falls out naturally.

## From Boolean to VSM

Boolean model: is term t present in document d? → yes/no
VSM: how much does term t contribute to document d? → a real number

This shift from binary to continuous term weights is what makes ranking possible.

## Building the Vector Space

### Step 1 — Define the term space

The vocabulary V = {t₁, t₂, ..., tₙ} defines the dimensions.
For a corpus with 50,000 unique terms, every document and query is a vector in
50,000-dimensional space. Most entries are zero — the vectors are sparse.

### Step 2 — Represent documents as vectors

Each document d is represented as:

```bash
d = (w(t₁,d), w(t₂,d), ..., w(tₙ,d))
```

where w(tᵢ, d) is the weight of term tᵢ in document d.

The simplest weight is raw term frequency. A better weight is TF-IDF
(covered in depth in 03-tf-idf.md). For now, assume weights are TF-IDF scores.

### Step 3 — Represent the query as a vector

The query q = "python search" becomes:

```bash
q = (0, 0, ..., 1, 0, ..., 1, 0, ...)
        ↑ python          ↑ search
```

Query weights are often binary (1 if term present, 0 if not) or also TF-IDF weighted
for longer queries.

## Similarity Measure — Cosine Similarity

### Why not Euclidean distance?

Euclidean distance is sensitive to vector magnitude. A long document about "python"
has a higher raw term count than a short document about "python" — its vector is
further from the origin even if both are equally relevant. We want to compare
direction, not magnitude.

### Cosine similarity

Measures the cosine of the angle between two vectors:

```bash
cos(θ) = (d · q) / (|d| × |q|)
```

Where:

- d · q is the dot product: Σ w(tᵢ,d) × w(tᵢ,q)
- |d| is the magnitude of d: sqrt(Σ w(tᵢ,d)²)
- |q| is the magnitude of q: sqrt(Σ w(tᵢ,q)²)

Range: 0 (no overlap) to 1 (identical direction)

### Why cosine works

Two documents about "python" — one short, one long — point in roughly the same
direction in term space even if their magnitudes differ. Cosine similarity measures
the angle between them, which is small. A document about "cooking" points in a
completely different direction — large angle, low cosine similarity.

## Term Weighting - Why Raw TF is Not Enough

Raw TF has two problems:

1. **Term frequency saturation** — a document mentioning "python" 10 times is not
   10x more relevant than one mentioning it once
2. **No corpus-level signal** — "the" appearing 100 times in a document is treated
   the same as "python" appearing 100 times

This is why TF-IDF weights are used in practice. TF-IDF is covered fully in
03-tf-idf.md — VSM is the framework, TF-IDF is the weighting scheme that makes it
work well.

## The Full VSM Pipeline

```bash

corpus
→ text processing (tokenize, normalize, stem)
→ build inverted index with TF-IDF weights
→ at query time:
→ represent query as weighted vector
→ for each candidate document:
→ compute cosine similarity
→ rank by similarity score
→ return top-k results

```

## Strengths

- **Full ranking**: Every document gets a continuous score; no feast or famine.
- **Partial matching**: Documents matching some but not all query terms are still retrieved and ranked.
- **Simple and interpretable**: The geometric intuition is clean and the math is straightforward.
- **Flexible weighting**: Any term weighting scheme can be plugged in.
- **Foundation**: VSM is the conceptual basis for modern dense retrieval; neural models replace sparse TF-IDF vectors with dense learned embeddings but the geometric similarity idea is identical.

## Weaknesses

- **Term independence assumption**: Each term is an independent dimension; the model has no notion that "python" and "programming" are related.
- **No semantics**: "car" and "automobile" are orthogonal vectors; zero similarity even though they mean the same thing
- **High dimensionality**: vocabulary size can be 100k+; vectors are huge and sparse.
- **Term frequency saturation**: Raw TF does not saturate; BM25 fixes this.
- **No length normalization by default**: Cosine partially addresses this but not perfectly; BM25 handles it more explicitly
- **Query-document asymmetry**: Short queries produce sparse query vectors; a single term query has a very imprecise direction in term space.
