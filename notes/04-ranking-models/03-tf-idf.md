# TF-IDF

TF-IDF (Term Frequency - Inverse Document Frequency) is a numerical weighting scheme that reflects how important a term is to a document relative to the entire corpus. It is the standard term weighting method used inside the Vector Space Model and many other retrieval systems.

## Intuition

Two observations drive TF-IDF:

1. **A term that appears many times in a document is probably important to that
   document** - but the relationship is not linear. A term appearing 10 times is not
   10x more important than one appearing once.

2. **A term that appears in almost every document tells you nothing about which
   documents are relevant** - "the", "is", "a" appear everywhere and discriminate
   nothing. A term like "backpropagation" appearing in only a handful of documents is highly discriminative.

TF-IDF combines both observations into a single weight: high when a term is frequent
in a specific document but rare across the corpus, low when it is common everywhere.

## The Two Components

### Term Frequency (TF)

Measures how often a term appears in a document.

Raw TF:

```bash
TF(t, d) = count of t in d
```

Normalized TF (more common, prevents bias toward longer documents):

```bash
TF(t, d) = count of t in d / total terms in d
```

Log-normalized TF (dampens the effect of very high frequencies):

```bash
TF(t, d) = 1 + log(count of t in d)   if count > 0, else 0
```

### Inverse Document Frequency (IDF)

Measures how rare a term is across the corpus.

```bash
IDF(t) = log( N / df(t) )
```

Where:

- N = total number of documents in the corpus
- df(t) = number of documents containing term t

A term appearing in every document has IDF = log(1) = 0, it contributes nothing.
A term appearing in one document has IDF = log(N), maximum discriminative power.

Smoothed IDF (avoids division by zero for unseen terms):

```bash
IDF(t) = log( N / (df(t) + 1) ) + 1
```

## Putting It Together

```bash
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

High TF-IDF -> term is frequent in this document AND rare in the corpus.  
Low TF-IDF  -> term is either rare in this document OR common everywhere.

## Worked Example

Corpus of 4 documents:
```
D1: "python search index python"
D2: "python tutorial beginners"
D3: "search engine design"
D4: "cooking recipes dinner"
```

N = 4

Computing TF-IDF for "python" in D1:

```
count("python", D1) = 2
total terms in D1   = 4
TF("python", D1)    = 2/4 = 0.5

df("python") = 2  (appears in D1 and D2)
IDF("python") = log(4/2) = log(2) = 0.693

TF-IDF("python", D1) = 0.5 × 0.693 = 0.347
```

Computing TF-IDF for "cooking" in D4:

```
count("cooking", D4) = 1
TF("cooking", D4)    = 1/3 = 0.333

df("cooking") = 1  (appears only in D4)
IDF("cooking") = log(4/1) = log(4) = 1.386

TF-IDF("cooking", D4) = 0.333 × 1.386 = 0.462
```

"cooking" scores higher than "python" because it is rarer across the corpus,
it is more discriminative.

## TF-IDF Variants - SMART Notation

Different combinations of TF and IDF normalization encoded as three letters (TF.IDF.norm):

| TF variant        | Formula                               |
|-------------------|---------------------------------------|
| n (natural)       | tf                                    |
| l (logarithm)     | 1 + log(tf) if tf > 0, else 0        |
| a (augmented)     | 0.5 + 0.5 × tf / max_tf              |
| b (boolean)       | 1 if tf > 0, else 0                  |

| IDF variant       | Formula                               |
|-------------------|---------------------------------------|
| n (none)          | 1                                     |
| t (standard)      | log(N / df)                           |
| p (probabilistic) | log((N - df) / df)                    |

| Normalization     | Formula                               |
|-------------------|---------------------------------------|
| n (none)          | 1                                     |
| c (cosine)        | divide by L2 norm of vector           |

## Strengths

- Simple and interpretable: the formula is transparent
- Effective baseline: hard to beat without more sophisticated methods
- Computationally efficient: weights precomputed at index time
- Language agnostic: works on any language without modification

## Weaknesses

- **TF saturation**: raw TF grows linearly; 100 occurrences is not 100x more relevant than 1. BM25 fixes this.
- **No length normalization**: longer documents accumulate higher raw TF scores. BM25 handles this explicitly.
- **Term independence**: "python" and "programming" have zero relationship in the model.
- **No semantics**: "car" and "automobile" are completely separate terms

## The Bridge to BM25

TF-IDF has two concrete problems that BM25 directly fixes:

**Problem 1 — TF grows without bound**

```
TF-IDF: weight ∝ raw term count -> unbounded growth
BM25:   weight saturates via tf / (tf + k₁) -> bounded
```

**Problem 2 — No explicit length normalization**

```
TF-IDF: cosine similarity partially compensates
BM25:   explicit parameter b compares doc length to average doc length
```

## My Summary

TF-IDF weights a term in a document by multiplying how frequently it appears in that document (TF) by how rarely it appears across the corpus (IDF). The result is high for terms that are locally frequent but globally rare, exactly the terms that are discriminative for that document. It is the standard weighting scheme for the Vector Space Model and a strong baseline. Its two main weaknesses: unbounded TF growth and imprecise length normalization, are directly addressed by BM25, which is why understanding TF-IDF deeply makes BM25's design choices immediately intuitive.
