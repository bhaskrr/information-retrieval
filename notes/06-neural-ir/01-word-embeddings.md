# Word Embeddings

Word embeddings are dense vector representations of words learned from large text
corpora. Each word is mapped to a fixed-size vector of real numbers such that words
with similar meanings or usage patterns are mapped to nearby points in the vector
space. Word embeddings are the foundation of neural IR — they replace the sparse
high-dimensional TF-IDF vectors of classical IR with compact learned representations
that capture semantic meaning.

## Intuition

In TF-IDF and BM25, every word is an independent dimension. "car" and "automobile"
are orthogonal vectors — zero similarity, no relationship. The model has no way to
know they mean the same thing.

Word embeddings fix this by learning from context. The distributional hypothesis
underlies all embedding methods:

> Words that appear in similar contexts tend to have similar meanings.

"car" and "automobile" appear in similar sentences — near words like "drive",
"engine", "road", "passenger". After training on enough text, their vectors end up
close together in the embedding space. This is not hand-coded — it emerges from
the data.

## From Sparse to Dense

Classical IR:

```bash
Vocabulary size: 50,000 terms
Document vector: [0, 0, 0, 1, 0, 0, 2, 0, ..., 0]  ← 50,000 dimensions, mostly zeros
                                                         (sparse)
```

Word embeddings:

```bash
Embedding size: 300 dimensions
Word vector:    [0.23, -0.41, 0.87, 0.12, ..., -0.33]  ← 300 dimensions, all nonzero
                                                            (dense)
```

Dense vectors are:

- Much smaller — 300 vs 50,000 dimensions
- Semantically meaningful — distance reflects meaning
- Generalizable — similar words share similar vectors

## Word2Vec

Introduced by Mikolov et al. at Google in 2013. The most influential word embedding
method and the starting point for understanding all that followed.

### The core idea

Train a shallow neural network to predict context from a word (or word from context).
The network is a means to an end — the learned weights become the word vectors.

### Two architectures

**Skip-gram** — given a center word, predict surrounding context words:

```bash
Input:  "search"
Target: predict ["information", "engine", "query", "index"]
```

**CBOW (Continuous Bag of Words)** — given context words, predict the center word:

```bash
Input:  ["information", "engine", "query", "index"]
Target: predict "search"
```

Skip-gram works better for rare words. CBOW is faster to train.

### Training

The network has one hidden layer. The weight matrix of this layer (vocab_size ×
embedding_dim) becomes the embedding lookup table after training. Each row is the
embedding vector for one word.

Training uses negative sampling — for each positive (word, context) pair, sample
several random negative words and train the model to distinguish real context from
random noise. This makes training feasible at scale.

### What Word2Vec learns

Famous analogies emerge from the geometry:

```bash
king - man + woman ≈ queen
paris - france + italy ≈ rome
walking - walk + swim ≈ swimming
```

These relationships are encoded as linear offsets in the embedding space — a
remarkable emergent property of the distributional training objective.

## GloVe (Global Vectors)

Introduced by Pennington et al. at Stanford in 2014. Rather than training on local
context windows (like Word2Vec), GloVe trains on global word co-occurrence statistics
across the entire corpus.

### The core idea

Build a co-occurrence matrix C where C[i][j] = how many times word i and word j
appear within a window of each other in the corpus. Train embeddings such that
the dot product of two word vectors approximates the log of their co-occurrence
count:

```bash
wᵢ · wⱼ + bᵢ + bⱼ ≈ log(C[i][j])
```

### GloVe vs Word2Vec

| Property             | Word2Vec                | GloVe                         |
| -------------------- | ----------------------- | ----------------------------- |
| Training signal      | Local context windows   | Global co-occurrence matrix   |
| Training speed       | Faster on large corpora | Slower (matrix construction)  |
| Performance          | Strong on analogy tasks | Strong on similarity tasks    |
| Pretrained available | Yes (Google News, 100B) | Yes (Wikipedia, Common Crawl) |
| Still used           | Less common             | Less common                   |

In practice Word2Vec and GloVe perform similarly. Both have been largely superseded
by contextual embeddings (BERT and beyond) but remain useful for lightweight
applications.

## FastText

Introduced by Bojanowski et al. at Facebook AI in 2017. Extends Word2Vec by
representing each word as a sum of its character n-gram vectors.

```bash
"search" → <se, sea, ear, arc, rch, ch>, <sea, ear, arc, rch>, ...
           each n-gram has its own vector → word vector = sum of n-gram vectors
```

### Why character n-grams matter

- **Out-of-vocabulary words** — Word2Vec has no vector for words not seen during
  training. FastText can construct a vector for any word from its character n-grams,
  even entirely new words.
- **Morphological variation** — "search", "searching", "searches" share n-grams and
  thus similar vectors, even without explicit stemming.
- **Misspellings** — "recieve" shares n-grams with "receive" and gets a reasonable
  vector.

FastText is particularly useful for morphologically rich languages and domains with
specialized vocabulary (medical, legal, technical).

## Key Properties of Word Embeddings

### Semantic similarity

```bash
cosine_similarity("dog", "cat")    → high   (both are pets)
cosine_similarity("dog", "car")    → low    (unrelated)
cosine_similarity("king", "queen") → high   (similar roles)
```

### Analogical reasoning

Linear offsets encode relationships:

```bash
embedding("Rome") - embedding("Italy") ≈ embedding("Paris") - embedding("France")
```

### Polysemy problem

Word2Vec, GloVe, and FastText produce one vector per word — regardless of meaning.
"bank" gets one vector even though it can mean a financial institution or a
riverbank. Context is ignored entirely.

This is the fundamental limitation that BERT and contextual embeddings address.

## From Word Embeddings to Document Embeddings

Word embeddings represent individual words. IR needs document representations.
Simple approaches:

### Averaging

Average all word vectors in a document:

```bash
doc_vector = mean([embedding(w) for w in document])
```

Fast and surprisingly effective for short texts. Loses word order and loses the
ability to distinguish "dog bites man" from "man bites dog".

### Weighted averaging (TF-IDF weighted)

Weight each word vector by its TF-IDF score before averaging — rare, important
words contribute more than common ones:

```bash
doc_vector = Σ tfidf(w) × embedding(w) / Σ tfidf(w)
```

Better than simple averaging for retrieval tasks.

### SIF (Smooth Inverse Frequency)

Weight by inverse word frequency, then remove the first principal component
(which captures common discourse patterns). Outperforms simple averaging on
many sentence similarity benchmarks.

Word embeddings are the first step into neural IR. They demonstrate that learned
dense vectors capture meaning in a way sparse TF-IDF vectors never can. Their
limitation — one static vector per word regardless of context — is exactly what
BERT solves.

## My Summary

Word embeddings map words to dense vectors such that semantically similar words
are geometrically close. Word2Vec learns from local context windows, GloVe from
global co-occurrence statistics, and FastText from character n-grams — enabling
robust handling of rare and out-of-vocabulary words. All three produce one static
vector per word regardless of context, which means polysemous words like "bank"
get one vector that blends all their meanings. Document embeddings are constructed
by averaging word vectors, weighted or unweighted. Word embeddings are the
conceptual foundation of neural IR — they establish that dense learned vectors
capture semantic relationships that sparse TF-IDF vectors cannot, setting up the
transition to contextual embeddings via BERT.
