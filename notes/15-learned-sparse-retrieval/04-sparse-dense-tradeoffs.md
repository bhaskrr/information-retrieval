# Sparse-Dense Tradeoffs

Sparse-dense tradeoffs is the systematic analysis of when to use sparse retrieval
(BM25, SPLADE, uniCOIL), dense retrieval (bi-encoders, E5, GTE), or hybrid
combinations - and specifically which configuration to choose for a given
application based on measurable characteristics of the corpus, query distribution,
infrastructure constraints, and performance requirements. The tradeoffs are not
theoretical - they are empirically measurable on benchmarks like BEIR across 18
diverse retrieval tasks - and they are not uniform: sparse retrieval systematically
outperforms dense on some task types while dense systematically outperforms sparse
on others. Understanding these patterns is what allows IR practitioners to make
principled deployment decisions rather than cargo-culting the latest state-of-the-art
system.

## Intuition

After studying BM25, SPLADE, uniCOIL, dense retrieval, and hybrid search across
many notes in this repo, the natural question is: which one do I actually deploy?
The answer is never the same for every application, and the reason is fundamental:
different retrieval methods encode different assumptions about what makes a document
relevant, and those assumptions match or mismatch different query and document
distributions in predictable ways.

Dense retrieval assumes that meaning is fully captured by a single 768-dim vector -
that two semantically equivalent sentences in completely different words will produce
nearby vectors. This assumption holds well for natural language queries against
general web text but breaks down for queries that require exact matching of rare
identifiers, for domains where the vocabulary is highly specialized and
underrepresented in pretraining data, and for queries so short (one or two words)
that the semantic compression into a fixed vector loses critical signal.

Sparse retrieval assumes that relevance is indicated by shared vocabulary - that
the right documents contain some of the same words as the query. BM25 makes this
assumption literally (exact term overlap). SPLADE relaxes it (synonym overlap
through expansion) while maintaining the same inverted index mechanics.

Neither assumption is universally correct. BEIR makes this concrete: looking at
per-dataset scores reveals systematic patterns where each method wins and loses.
The job of an IR practitioner is to recognize which pattern applies to their
application and configure the retrieval stack accordingly.

## The BEIR Evidence

BEIR (Benchmarking Information Retrieval) evaluates retrieval models across 18
diverse datasets. The per-dataset breakdown reveals clear patterns:

### Datasets where BM25 and sparse are competitive or better than dense

**Robust04 (NDCG@10):**

```
BM25:       0.408
SPLADE-v2:  0.417
E5-base:    0.391    ← dense BELOW BM25
```

Robust04 consists of 250 deliberately "difficult" TREC ad-hoc queries designed
to stress-test retrieval systems. Many queries are short and highly specific.
BM25 wins because exact term matching for precise but rare query terms is more
reliable than semantic compression.

**TREC-COVID:**

```
BM25:       0.656
SPLADE-v2:  0.711
E5-base:    0.755    ← dense wins, SPLADE between
```

Biomedical queries - dense wins here because domain terminology is rich
and SPLADE's expansion cannot cover all biomedical synonyms.

**Touché-2020 (argument retrieval):**

```
BM25:       0.367
SPLADE-v2:  0.278    ← SPLADE significantly below BM25
E5-base:    0.210    ← dense significantly below BM25
```

Argument retrieval queries are long and topical. BM25 handles them well
because documents are long and share vocabulary with queries naturally.
Dense and SPLADE over-compress the argument structure into embeddings.

**DBPedia entity:**

```
BM25:       0.313
SPLADE-v2:  0.435    ← SPLADE big win
E5-base:    0.402    ← dense also wins
```

Entity retrieval with DBPedia knowledge base - SPLADE wins because entity
names and their synonyms/aliases are expanded effectively.

### Datasets where dense is clearly better

**FiQA (financial QA):**

```
BM25:       0.236
SPLADE-v2:  0.372
E5-base:    0.404    ← dense clear winner
```

Conversational financial questions require semantic understanding - "how do I
protect my savings from inflation" retrieves documents about Treasury bonds,
which BM25 misses entirely.

**ArguAna (counterargument retrieval):**

```
BM25:       0.315
SPLADE-v2:  0.469
E5-base:    0.490    ← dense narrow winner
```

Finding counterarguments requires understanding the thesis of an argument and
finding documents that oppose it - fundamentally semantic, not lexical.

**Quora (duplicate question detection):**

```
BM25:       0.789
SPLADE-v2:  0.861
E5-base:    0.882    ← dense wins, all methods high
```

Paraphrase matching - dense and SPLADE both excel at finding questions with
the same meaning in different words.

### Summary pattern

```
Dense outperforms sparse when:
  → Queries are natural language questions (conversational)
  → Relevant documents use different vocabulary than queries (synonym-heavy)
  → Task requires understanding of meaning rather than lexical matching
  → Documents are short passages (not long enough for BM25 TF to be reliable)

Sparse competitive or better when:
  → Queries are keyword-style or contain specific identifiers
  → Domain vocabulary is specialized and partially out of dense model's pretraining
  → Queries and relevant documents share natural vocabulary overlap
  → Recall is less critical than precision (BM25's exact matching is precise)
  → Documents are long (TF statistics more reliable over longer text)
```

## Per-Domain Analysis

### Biomedical IR

```
Method                NDCG@10     Strengths                 Weaknesses
──────────────────────────────────────────────────────────────────────────
BM25                  0.45-0.65   Exact acronym matching    Synonym-heavy queries fail
SPLADE                0.50-0.70   Partial synonym expansion Domain terms partially covered
Dense (BioMedBERT)    0.55-0.75   Semantic understanding    Requires domain pretraining
Hybrid                0.60-0.78   Best of both              Complexity

Key insight: domain-adapted dense (BioBERT, BioMedBERT, PubMedBERT) significantly
outperforms general dense models. For biomedical, domain adaptation matters more
than sparse vs dense choice.
```

### Legal IR

```
Method                NDCG@10     Strengths                 Weaknesses
──────────────────────────────────────────────────────────────────────────
BM25                  0.35-0.55   Legal citation matching   Paraphrased statutes fail
SPLADE                0.38-0.58   Citation + some synonyms  Legal jargon expansion limited
Dense (general)       0.30-0.50   Semantic understanding    Legal domain poorly covered
Dense (legal-adapted) 0.42-0.62   Best semantic match       Requires domain training
Hybrid                0.45-0.65   Best overall              Complexity

Key insight: BM25 surprisingly strong in legal IR due to exact citation and
statute matching requirements. Legal document retrieval heavily depends on
precise terminology.
```

### Code Search

```
Method                NDCG@10     Strengths                 Weaknesses
──────────────────────────────────────────────────────────────────────────
BM25                  0.40-0.60   Exact API/function match  NL-to-code semantic gap
SPLADE                0.42-0.62   NL synonym expansion      Code syntax not expanded
Dense (CodeBERT)      0.55-0.72   NL-code alignment         Requires code pretraining
Hybrid                0.58-0.75   Best coverage             Complexity

Key insight: code search is fundamentally cross-modal (NL query → code document).
Dense models pretrained on code (CodeBERT, UniXcoder) are essential.
BM25 and SPLADE handle exact API name queries well.
```

### E-commerce Product Search

```
Method                NDCG@10     Strengths                 Weaknesses
──────────────────────────────────────────────────────────────────────────
BM25                  0.35-0.50   Exact product name match  "comfortable sneakers" fails
SPLADE                0.42-0.58   Attribute expansion       Brand/model expansion noisy
Dense (general)       0.48-0.65   Semantic attribute match  Out-of-vocabulary products
Dense (product-tuned) 0.55-0.72   Best semantic match       Requires training data
Hybrid                0.58-0.75   Full coverage             Complexity

Key insight: user queries in e-commerce are highly diverse - "red running shoes
under $50 wide width" requires attribute filtering (wide width) AND semantic
matching (comfort/running features). Hybrid with attribute filtering beats pure
retrieval.
```

### General Web Search

```
Method                NDCG@10     Strengths                 Weaknesses
──────────────────────────────────────────────────────────────────────────
BM25                  0.25-0.40   Keyword queries           Natural language queries fail
SPLADE                0.32-0.45   Broad synonym coverage    Complex queries limited
Dense (E5, GTE)       0.40-0.55   Semantic understanding    Rare exact queries fail
Hybrid                0.45-0.60   Best overall quality      Latency increase

Key insight: hybrid is the clear winner for general web search.
Neither sparse nor dense alone covers the full diversity of web queries.
```

## Infrastructure Tradeoffs

Quality is only one dimension. Infrastructure cost, latency, and operational
complexity determine which high-quality option is actually deployable:

### Index storage

```
Method           Storage per 1M docs (768-dim, float32)
──────────────────────────────────────────────────────────
BM25              ~500MB (compressed inverted index)
SPLADE            ~800MB-2GB (expanded sparse vectors)
uniCOIL           ~400MB-800MB (similar to BM25)
Dense (flat)      ~3.1GB (768 × 4 bytes × 1M)
Dense (HNSW)      ~3.5GB (flat + graph structure)
Dense (int8)      ~780MB (quantized)
Dense (Matry-128) ~512MB (truncated + quantized)
Hybrid            Dense + Sparse costs combined
```

### Query latency

```
Method           Latency (1M docs, single CPU core)
──────────────────────────────────────────────────────
BM25              2-10ms (inverted index lookup)
SPLADE            5-25ms (encoding + inverted index)
uniCOIL           3-15ms (encoding + inverted index)
Dense (CPU)       50-500ms (ANN search, CPU only)
Dense (GPU)       5-30ms (ANN search with GPU)
Dense (HNSW)      10-50ms (approximate, no GPU)
Hybrid            Max of sparse + dense components
```

### Indexing cost (per document)

```
Method           Indexing time per document
──────────────────────────────────────────────
BM25              Microseconds (tokenize + count)
DeepCT            ~5ms (BERT forward pass, CPU)
uniCOIL           ~5ms (BERT forward pass, CPU)
SPLADE            ~8ms (BERT MLM forward pass, CPU)
Dense (all-MiniLM) ~2ms (fast encoder, CPU)
Dense (E5-base)    ~8ms (larger encoder, CPU)
Dense (E5-large)   ~20ms (large encoder, CPU)
```

For 1M documents at 8ms/doc: ~2.2 GPU hours, or ~22 CPU hours.

### Operational complexity

```
Method           Infrastructure    Monitoring         Updates
────────────────────────────────────────────────────────────────────
BM25              Elasticsearch     Standard ES        Incremental
SPLADE            Elasticsearch     Standard ES        Re-encode new docs
Dense             FAISS/Qdrant/Pin  Custom + ES        Re-encode new docs
Hybrid            Both              Both               Both
```

## The Decision Framework

A complete decision procedure for selecting a retrieval stack:

### Step 1 - Measure baseline

Always start with BM25. It is free to run and provides the baseline against which
everything else is measured:

```python
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
bm25_ndcg = evaluate_ndcg(bm25, test_queries)
```

If BM25 already achieves your quality target: stop here. Do not add complexity
without evidence that it is needed.

### Step 2 - Identify the failure mode

Run BM25 and manually inspect failures. Which failure type dominates?

```
Failure type A - Vocabulary mismatch:
  Query: "cardiac symptoms"
  BM25 top result: document about "heart attack treatments" (should be relevant)
  BM25 score: 0 (no term overlap)
  → Dense retrieval or SPLADE will fix this

Failure type B - Exact match misses:
  Query: "CVE-2021-44228"
  BM25 top result: general log4j security document (misses specific CVE)
  Dense score: high (semantic similarity, but wrong document)
  → BM25 was correct, dense would make this worse
  → Hybrid needed

Failure type C - Ranking errors (correct documents retrieved, wrong rank):
  Query: "python list comprehension performance"
  BM25 retrieves: correct documents at ranks 4-8 instead of 1-3
  → Reranking with cross-encoder fixes this
  → Not a retrieval problem, a ranking problem

Failure type D - Coverage gaps (relevant documents not retrieved at all):
  Query: "convert pandas dataframe to json"
  BM25 Recall@100: 0.45 (relevant document not in top 100)
  → Need better first-stage retrieval
  → Dense or SPLADE + BM25 hybrid for recall improvement
```

### Step 3 - Select based on failure type and constraints

```
Failure type A + GPU available + can tolerate new infrastructure:
  → Hybrid (BM25 + dense) with RRF
  → NDCG@10 improvement: +15-25%

Failure type A + no GPU + must use inverted index:
  → SPLADE or uniCOIL
  → NDCG@10 improvement: +8-15%

Failure type B + already dense:
  → Add BM25 to hybrid
  → NDCG@10 improvement: +5-10%

Failure type C + have GPU:
  → Add cross-encoder reranker to existing retrieval
  → NDCG@10 improvement: +5-15%

Failure type D + have labeled data:
  → Fine-tune bi-encoder on domain data
  → NDCG@10 improvement: +10-20%

All failure types + maximum quality budget:
  → Hybrid (SPLADE + dense) + cross-encoder reranker
  → NDCG@10 improvement: +25-40% over BM25 alone
```

### Step 4 - Validate

After choosing a configuration, validate on your specific data, not on BEIR:

```python
# Always measure on your domain, not just BEIR
domain_ndcg = evaluate_ndcg(model, domain_test_queries)
beir_ndcg   = evaluate_ndcg(model, beir_subset_queries)   # generalization check

# Report both
print(f"Domain NDCG@10: {domain_ndcg:.4f}")
print(f"BEIR avg NDCG@10: {beir_ndcg:.4f}")
```

## Decision Summary Table

```
Situation                                      Recommended stack
──────────────────────────────────────────────────────────────────────────
BM25 already meets quality target              BM25 - do not add complexity
Dense > BM25 by 10%+ on domain test           Dense or hybrid
BM25 > dense on domain test                   Keep BM25, check domain adapt
Both fail on vocabulary mismatch              SPLADE (inverted index) or dense
Both fail on exact match                      BM25 + dense hybrid
Latency budget < 50ms, no GPU                 BM25 or SPLADE (CPU inverted index)
Latency budget 50-200ms, GPU available        Hybrid (BM25 + dense) + reranker
Latency budget > 200ms, max quality wanted    Hybrid + cross-encoder reranking
Corpus < 100K documents                       BM25 is often sufficient
Corpus 100K - 10M, mixed queries              Hybrid is the safe default
Corpus > 10M documents                        Hybrid with ANN (HNSW/IVF)
Existing Elasticsearch, no GPU                SPLADE (drop-in quality upgrade)
New system, GPU available, no constraints     Hybrid + reranker
Domain highly specialized                     BM25 baseline + domain-adapted dense
```

## My Summary

Sparse-dense tradeoffs are not uniform across tasks - dense retrieval systematically
outperforms sparse on semantic and conversational tasks (FiQA, ArguAna, Quora) while
sparse remains competitive or better on keyword-heavy and argument retrieval tasks
(Robust04, Touché-2020). BEIR's 18-dataset benchmark reveals these patterns, but
domain-specific evaluation on your actual data is essential before making deployment
decisions. The decision framework starts with BM25 as a free baseline, identifies
which failure mode dominates through manual inspection, and selects the minimal
additional complexity that addresses that failure mode. For vocabulary mismatch with
existing inverted index infrastructure, SPLADE is the drop-in upgrade. For general
mixed-query applications with GPU budget, hybrid BM25 + dense with RRF is the safe
default. Infrastructure tradeoffs matter as much as quality - dense retrieval requires
3x the storage of BM25 and ANN infrastructure rather than inverted indexes, which can
be the deciding factor for organizations with mature Elasticsearch deployments. The
single most important practical rule: measure on your domain, not just BEIR, because
BEIR averages can hide task-specific patterns that dominate your application.
