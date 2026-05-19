# SPLADE Deep Dive

SPLADE (SParse Lexical AnD Expansion) is the dominant learned sparse retrieval
model. Introduced by Formal et al. at NAVER Labs Europe in 2021 and significantly
improved in SPLADE-v2 (2022) and SPLADE++ (2023), it uses BERT's Masked Language
Modeling head to produce sparse lexical representations where both the active terms
and their weights are learned from retrieval training data. SPLADE is the model that
made learned sparse retrieval practically competitive with dense retrieval - it
consistently outperforms BM25 by 15-25% on BEIR benchmarks while maintaining
full compatibility with standard inverted index infrastructure. It is the state-of-
the-art answer to the question: what is the best retrieval model I can run on my
existing Elasticsearch cluster?

## Intuition

In the previous note, the idea of vocabulary expansion was introduced abstractly -
a neural model adds related terms to a document's sparse representation. SPLADE
makes this concrete by using a specific mechanism: the MLM head of a pretrained
BERT model.

BERT's MLM head was trained to answer: "given this surrounding context, which
vocabulary token is most likely to fill this masked position?" After training on
hundreds of millions of sentences, it has learned rich associations. Given the
context "the patient experienced a [MASK]", the head assigns high probability
to "stroke", "seizure", "cardiac event", "heart attack". Given "the [MASK] sat
on a mat", it assigns high probability to "cat", "dog", "child".

SPLADE uses this learned knowledge for a completely different purpose: term
generation for retrieval. For every token in a document, it runs the MLM head and
collects the full probability distribution over the vocabulary. It then aggregates
these distributions - summing up which vocabulary terms have high probability
across all token positions - to produce a sparse document vector.

The result: a document about "myocardial infarction" produces high activations for
"heart", "attack", "cardiac", "chest pain", and "coronary" because the MLM head,
given the context of the document, assigns high probability to these terms at
various positions. The sparse vector has learned, in effect, what a relevance-
maximizing term list for this document looks like.

## SPLADE Architecture

### The MLM head as a term generator

BERT architecture with MLM head:

```
Input text: [CLS] document tokens [SEP]
                    ↓
            BERT encoder layers (12)
                    ↓
            Token embeddings: h₁, h₂, ..., hₙ
                    ↓
            MLM head (linear + GELU + LayerNorm + linear)
                    ↓
            Logits: lᵢ ∈ ℝ^|V| for each token position i
                    ↓
            log(1 + ReLU(lᵢ)) - activation function
                    ↓
            Max pooling over positions: max_i log(1 + ReLU(lᵢⱼ)) for each vocab term j
```

The final sparse vector w ∈ ℝ^|V| has:

- w_j = 0 for most vocabulary terms (sparsity)
- w_j > 0 only for terms that the MLM head activates somewhere in the document

### The log(1 + ReLU(x)) activation

The choice of activation is fundamental to SPLADE's properties:

**ReLU(x)** - sets negative logits to zero:

```
If logit for term j is negative at all positions: w_j = 0 (sparse)
If logit for term j is positive at any position: w_j > 0
```

**log(1 + x)** - compresses high activations:

```
log(1 + 0.1) = 0.095   (small activation stays small)
log(1 + 1.0) = 0.693   (moderate activation compressed)
log(1 + 10.) = 2.398   (large activation heavily compressed)
log(1 + 100) = 4.615   (very large activation compressed more)
```

Without log compression, frequently activated terms would dominate the sparse
vector. The logarithm creates a saturation effect similar to TF saturation in
BM25 - the first occurrence of a term adds more weight than the hundredth.

### Max pooling over positions

For each vocabulary term j, the final weight is:

```
w_j = max_i log(1 + ReLU(lᵢⱼ))
```

Taking the maximum over token positions rather than the sum ensures that the
document's sparse vector reflects the most relevant activation of each term -
not a noisy sum that grows with document length. This also provides an implicit
length normalization that BM25 achieves through its b parameter.

### Full SPLADE sparse vector formula

For document d with tokens t₁, ..., tₙ and BERT MLM logits L ∈ ℝⁿˣ|V|:

```
w_j(d) = max_{i=1}^{n} log(1 + ReLU(L_ij))
```

The sparse vector w(d) ∈ ℝ^|V| has at most |V| = 30,522 possible non-zero
entries (BERT vocabulary size), but in practice only 50-200 are non-zero due to
the ReLU zeroing out most logits.

## SPLADE Training

### Training data

SPLADE is trained on MS MARCO - 530K (query, positive passage, hard negative
passage) triples. The same training data used for dense bi-encoders.

### Loss function

SPLADE uses pairwise cross-entropy loss over query-document scores:

```
s(q, d) = w(q) · w(d)   ← sparse dot product

L_retrieval = -log(
    exp(s(q, d⁺)) /
    (exp(s(q, d⁺)) + exp(s(q, d⁻)))
)
```

Where d⁺ is the relevant document and d⁻ is the hard negative.

### FLOPS regularization - the sparsity controller

Without regularization, SPLADE would learn dense representations (every term
has non-zero weight) because more active terms increase dot product scores.
FLOPS regularization penalizes the expected number of floating point operations
at query time:

```
L_FLOPS(d) = Σ_j (mean_i w_ij)²
```

Where the mean is over training batch documents. This penalizes terms that are
consistently active across many documents - the model learns to be selective.

Total SPLADE loss:

```
L = L_retrieval + λ_d × L_FLOPS(document) + λ_q × L_FLOPS(query)
```

Where λ_d and λ_q are hyperparameters controlling document and query sparsity.
Higher λ → sparser representations → faster retrieval, lower quality.

### Hard negative mining

SPLADE training uses hard negatives from BM25 retrieval - documents that BM25
retrieves for the query (thus vocabulary-overlapping and "hard" for a sparse model)
but that are not relevant:

```
Step 1: Index MS MARCO with BM25
Step 2: For each training query, retrieve BM25 top-100
Step 3: Filter out the known positive document
Step 4: Sample hard negatives from remaining BM25 top-100
Step 5: Train SPLADE with (query, positive, BM25_hard_negative)
```

This is particularly important for SPLADE - a model that expands vocabulary must
learn to distinguish documents that share terms but are not relevant, which is
exactly what BM25 hard negatives test.

## SPLADE Variants

### SPLADE (original, 2021)

```
Base model:    BERT-base
Training:      MS MARCO with FLOPS regularization
Avg active:    ~200 terms per document
NDCG@10 BEIR:  ~0.435
Key issue:     Slow indexing and retrieval (too many active terms)
```

### SPLADE-v2 (2022)

Major improvement through ensemble distillation and better training:

```
Base model:    BERT-base
Training:      Distillation from cross-encoder teacher (MarginMSE)
               + hard negative mining
Variants:
  SPLADE-v2-max:    highest quality, ~200 active terms
  SPLADE-v2-distil: balanced, ~100 active terms
  SPLADE-v2-max-distil: best of both, top BEIR scores
NDCG@10 BEIR:  ~0.448 (max-distil)
Key improvement: Cross-encoder distillation boosts quality significantly
```

### SPLADE++ (2023)

Further improvements through better distillation and training data:

```
Base model:    BERT-base or DistilBERT
Training:      Improved distillation, better hard negatives, data augmentation
NDCG@10 BEIR:  ~0.455-0.462
Key improvement: Better teacher, larger training batch
```

### SPLADE-efficient (2022)

Designed specifically for production efficiency:

```
Base model:    DistilBERT (6 layers, 40% faster than BERT-base)
Training:      Aggressive FLOPS regularization (λ = 0.01 vs 0.001)
Avg active:    ~50-70 terms per document (3-4x sparser than SPLADE-v2)
NDCG@10 BEIR:  ~0.428 (8% below max but 3x faster indexing)
Latency:       ~5ms per query on CPU (vs ~20ms for dense on GPU)
```

### CoCondenser + SPLADE (2022)

Uses a better BERT initialization (CoCondenser pre-fine-tuning) before SPLADE training:

```
Initialization: CoCondenser (BERT fine-tuned with contrastive learning
                on Wikipedia before SPLADE training)
Training:       Standard SPLADE training on top of CoCondenser
NDCG@10 BEIR:   ~0.452
Key benefit:    Better starting point for SPLADE training
```

## SPLADE vs Dense Retrieval on BEIR

SPLADE's performance on the 18-dataset BEIR benchmark, compared to BM25 and
strong dense retrievers:

```
Dataset             BM25    E5-base  SPLADE-v2  SPLADE++
──────────────────────────────────────────────────────────
MS MARCO            0.228   0.340    0.322       0.338
TREC-COVID          0.656   0.755    0.711       0.722
NFCorpus            0.325   0.357    0.345       0.352
NQ                  0.329   0.468    0.521       0.534
HotpotQA            0.603   0.668    0.684       0.695
FiQA                0.236   0.404    0.372       0.389
ArguAna             0.315   0.490    0.469       0.484
Touché-2020         0.367   0.210    0.278       0.292
DBPedia             0.313   0.402    0.435       0.448
SCIDOCS             0.158   0.198    0.173       0.179
FEVER               0.753   0.866    0.841       0.853
ClimFEVER           0.213   0.319    0.285       0.301
SciFact             0.665   0.731    0.723       0.741
Quora               0.789   0.882    0.861       0.874
────────────────────────────────────────────────────────────
Average             0.439   0.521    0.508       0.521
```

Key observations:

- SPLADE++ averages equal to E5-base on BEIR despite using inverted index
- SPLADE significantly outperforms BM25 on semantic datasets (NQ, HotpotQA)
- Dense (E5) wins on ArguAna - requires deep semantic understanding
- BM25 surprisingly competitive on Touché (argument retrieval)

## SPLADE in Elasticsearch

SPLADE sparse vectors can be stored and searched in Elasticsearch using the
`sparse_vector` field type (Elasticsearch 8.x):

```json
PUT /splade_index
{
  "mappings": {
    "properties": {
      "text": {"type": "text"},
      "splade_vector": {
        "type": "sparse_vector"
      }
    }
  }
}
```

Indexing a SPLADE-encoded document:

```json
POST /splade_index/_doc/1
{
  "text": "heart attack symptoms and treatment",
  "splade_vector": {
    "heart": 2.1,
    "attack": 1.7,
    "symptoms": 2.0,
    "cardiac": 1.9,
    "treatment": 1.8,
    "arrest": 1.4,
    "myocardial": 0.9
  }
}
```

Querying with SPLADE:

```json
GET /splade_index/_search
{
  "query": {
    "sparse_vector": {
      "field": "splade_vector",
      "query_vector": {
        "cardiac": 1.8,
        "arrest": 1.5,
        "treatment": 1.2
      }
    }
  }
}
```

The `sparse_vector` query computes the dot product between query and document
sparse vectors using the inverted index - identical mechanics to BM25.

## SPLADE Deployment Checklist

```
Step                        Action
──────────────────────────────────────────────────────────────────────
Model selection             Use naver/splade-cocondenser-selfdistil
                            for best quality/speed tradeoff

Document encoding           Encode offline, CPU is sufficient
                            (~1000 docs/min on single CPU core)

Index type                  Standard inverted index (Elasticsearch,
                            Lucene, or custom hash map)

Query encoding              Online at query time (fast, ~5ms on CPU)

Sparsity tuning             Check avg active terms after encoding
                            50-100 = efficient, 200+ = too slow

Batching                    Batch document encoding (batch_size=32)
                            Single query encoding at query time

Token filtering             Filter special tokens ([CLS], [SEP], ##)
                            from sparse vector before indexing

Score threshold             Optional: zero out weights < 0.1
                            to increase sparsity further

Elasticsearch integration   Use sparse_vector field type (ES 8.x)
                            or script_score with dot product
```

## My Summary

SPLADE uses BERT's Masked Language Modeling head to produce sparse lexical
representations by aggregating per-token vocabulary distributions with log(1 +
ReLU(logits)) activation and max pooling over sequence positions. The result is
a sparse vector where both the active terms and their weights are learned from
retrieval training, enabling vocabulary expansion (adding semantically related
terms like "cardiac" to a document about "heart attack") and learned contraction
(zeroing out function words). FLOPS regularization controls sparsity by penalizing
terms with consistently high activation across training documents - higher λ produces
sparser, faster vectors at the cost of some quality. SPLADE-v2 improved quality
through cross-encoder distillation; SPLADE-efficient reduced latency through
DistilBERT and aggressive regularization. On BEIR, SPLADE++ matches E5-base quality
despite using standard inverted index infrastructure rather than ANN search - making
it the optimal choice for organizations that need dense retrieval quality without
replacing Elasticsearch.
