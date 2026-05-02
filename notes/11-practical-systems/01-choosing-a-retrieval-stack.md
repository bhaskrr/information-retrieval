# Choosing a Retrieval Stack

Choosing a retrieval stack is the engineering decision of which combination of
retrieval models, indexes, and infrastructure components to use for a specific
application. The decision involves tradeoffs between retrieval quality, latency,
throughput, infrastructure complexity, maintenance burden, cost, and the amount
of labeled data available. There is no universally correct retrieval stack - the
right choice depends entirely on the application's scale, domain, query distribution,
latency budget, and team capabilities. This note provides a structured framework
for making this decision rather than prescribing a single answer.

## Intuition

Every retrieval technique covered in this repo - BM25, dense retrieval, hybrid
search, reranking, SPLADE - is a point on multiple tradeoff curves simultaneously.
A technique that is perfect for one application can be completely wrong for another.

Consider three teams:

**Team A** - building a search feature for an internal knowledge base with 50,000
documents, no labeled data, two engineers, and a 200ms latency budget. The right
answer is almost certainly BM25 with basic query preprocessing. Building a full
dense retrieval pipeline with bi-encoders, FAISS, and cross-encoder reranking would
take weeks, require ongoing maintenance, and produce marginal improvement over BM25
on a well-curated internal corpus.

**Team B** - building a semantic search product over 10 million research papers
where users ask natural language questions and vocabulary mismatch is a critical
problem. BM25 alone fails here. Dense retrieval or hybrid search is necessary.
The team needs to invest in the neural stack.

**Team C** - building a conversational AI assistant that answers questions over
a company's product documentation. They need RAG with hybrid retrieval, query
reformulation for multi-turn conversations, and tight latency control. The full
modern stack is justified by the application requirements.

The framework in this note helps you determine which team's situation matches yours.

## The Decision Framework

### Step 1 - Assess your corpus

| Corpus size    | Implication                                          |
| -------------- | ---------------------------------------------------- |
| < 10,000 docs  | Exact search is fast enough - no ANN needed          |
| 10K - 1M docs  | Standard dense retrieval with FAISS flat index       |
| 1M - 100M docs | ANN required (HNSW or IVF), consider hybrid          |
| > 100M docs    | Distributed index required (Elasticsearch, Weaviate) |

| Corpus type          | Implication                                 |
| -------------------- | ------------------------------------------- |
| Well-structured docs | BM25 strong - exact term matching reliable  |
| Technical/code       | BM25 strong - exact match critical          |
| Conversational text  | Dense retrieval strong - vocabulary varies  |
| Scientific papers    | Hybrid - technical terms + semantic queries |
| Customer support     | Dense retrieval - paraphrase-heavy          |
| Legal/medical        | BM25 + domain-specific expansion            |
| Mixed/general        | Hybrid almost always best                   |

### Step 2 - Assess your query distribution

| Query type                | Best retrieval approach              |
| ------------------------- | ------------------------------------ |
| Short keyword (1-3 words) | BM25 - exact match sufficient        |
| Long natural language     | Dense retrieval - captures semantics |
| Named entities            | BM25 - exact match critical          |
| Ambiguous/semantic        | Dense retrieval - resolves ambiguity |
| Mixed                     | Hybrid - covers both cases           |
| Conversational/multi-turn | Hybrid + query reformulation         |
| Code/technical            | BM25 with domain tokenization        |

### Step 3 - Assess your labeled data availability

| Labeled data         | What it unlocks                                    |
| -------------------- | -------------------------------------------------- |
| None                 | BM25 only (no training needed)                     |
| < 100 examples       | BM25 + basic heuristics                            |
| 100 - 1,000 examples | Fine-tuned reranker (cross-encoder on your domain) |
| 1,000 - 10,000       | Fine-tuned bi-encoder + reranker                   |
| > 10,000 examples    | Full LTR pipeline or end-to-end neural training    |
| MS MARCO (available) | Zero-shot dense retrieval (pretrained models)      |

Dense retrieval models pretrained on MS MARCO transfer reasonably to many
domains - you do not always need domain-specific labeled data. BEIR shows
that models like E5 and GTE generalize well out-of-the-box.

### Step 4 - Assess your latency budget

| Latency budget | Viable options                                          |
| -------------- | ------------------------------------------------------- |
| < 50ms         | BM25 only, or BM25 + very fast reranker (MiniLM int8)   |
| 50 - 100ms     | BM25 + MiniLM reranker, or dense retrieval (HNSW)       |
| 100 - 200ms    | Hybrid search + cross-encoder reranking (top-50)        |
| 200 - 500ms    | Full pipeline: hybrid + reranking + query understanding |
| > 500ms        | RAG with LLM generation, complex multi-stage pipelines  |

### Step 5 - Assess your team and infrastructure

| Team situation               | Recommendation                                       |
| ---------------------------- | ---------------------------------------------------- |
| 1-2 engineers, no ML         | BM25 (Elasticsearch or BM25Okapi)                    |
| Small team, some ML          | BM25 + pretrained bi-encoder (sentence-transformers) |
| ML team, GPU available       | Hybrid + cross-encoder reranking                     |
| ML team, GPU + labeled data  | Full fine-tuned pipeline                             |
| Platform team, scale matters | Managed vector DB (Weaviate, Pinecone, Qdrant)       |

## The Retrieval Stack Options

### Option 1 - BM25 Only

**When to use:**

- Corpus < 500K documents
- Queries are keyword-heavy (technical, legal, code)
- No labeled data
- Tight latency budget (< 50ms)
- Small team or no ML expertise
- Need interpretable ranking

**Implementation:**

```bash
Documents -> tokenization + normalization -> inverted index (Elasticsearch, BM25Okapi)
Query     -> same tokenization + normalization -> BM25 scoring -> ranked results
```

**Typical performance:** NDCG@10 ≈ 0.35-0.45 on standard benchmarks

**Tools:** Elasticsearch, OpenSearch, Apache Solr, rank_bm25 (Python),
Lucene (Java), Whoosh (Python)

**What you give up:** semantic matching, vocabulary mismatch handling,
meaning-based ranking

---

### Option 2 - Dense Retrieval Only

**When to use:**

- Queries are natural language, conversational, or semantic
- Vocabulary mismatch is a known problem
- MS MARCO pretrained models transfer well to your domain
- GPU available for offline encoding
- Corpus < 10M documents

**Implementation:**

```bash
Documents -> bi-encoder -> dense vectors -> FAISS index
Query     -> bi-encoder -> query vector -> ANN search -> ranked results
```

**Typical performance:** NDCG@10 ≈ 0.38-0.48 on standard benchmarks
(weaker than BM25 on keyword queries, stronger on semantic queries)

**Tools:** sentence-transformers, FAISS, Qdrant, Pinecone, Weaviate

**What you give up:** exact keyword matching, rare term handling, low-resource
domain performance, interpretability

---

### Option 3 - Hybrid Search (BM25 + Dense)

**When to use:**

- Mixed query types (some keyword, some semantic)
- Need to improve over BM25 baseline without full neural investment
- Corpus > 100K documents
- Latency budget ≥ 100ms
- This is the **recommended default for most production systems**

**Implementation:**

```bash
Documents -> BM25 index + dense vectors in FAISS
Query     -> BM25 retrieval (top-1000) + dense retrieval (top-1000)
          -> RRF fusion -> merged top-1000
          -> (optional) cross-encoder reranking -> top-10
```

**Typical performance:** NDCG@10 ≈ 0.45-0.55 on standard benchmarks
(consistently outperforms either alone)

**Tools:** Elasticsearch kNN + BM25, Weaviate hybrid, Qdrant hybrid,
sentence-transformers + rank_bm25 + custom RRF

**What you give up:** simplicity of single-system operation, some additional
infrastructure complexity

---

### Option 4 - Hybrid + Cross-Encoder Reranking

**When to use:**

- Retrieval quality is the primary concern
- Latency budget ≥ 150ms
- GPU available for reranking
- This is the **production best-practice for search applications**

**Implementation:**

```bash
Documents -> BM25 index + dense FAISS index
Query     -> hybrid retrieval -> top-100 candidates
          -> cross-encoder reranking -> top-10 results
```

**Typical performance:** NDCG@10 ≈ 0.50-0.60 on standard benchmarks

**Tools:** All hybrid tools + sentence-transformers CrossEncoder,
cross-encoder/ms-marco-MiniLM-L-6-v2

**What you give up:** lower latency options (reranking adds 50-200ms)

---

### Option 5 - RAG Pipeline

**When to use:**

- Application requires generated answers (not just ranked documents)
- Users ask questions expecting prose answers with citations
- Conversational interface with multi-turn interactions
- Long-form document retrieval (chunking and passage retrieval required)

**Implementation:**

```bash
Documents -> chunking -> hybrid index
Query     -> (reformulation if conversational) -> hybrid retrieval
          -> cross-encoder reranking
          -> context assembly -> LLM generation -> answer with citations
```

**Typical performance:** measured by answer quality (faithfulness, relevance)
rather than pure retrieval NDCG

**Tools:** All hybrid + reranking tools + LLM API (Anthropic, OpenAI) +
LangChain / LlamaIndex for orchestration

**What you give up:** determinism, cost control, latency predictability

---

### Option 6 - SPLADE or Learned Sparse

**When to use:**

- Want sparse retrieval infrastructure (inverted index) with neural quality
- Domain has significant vocabulary mismatch
- Cannot afford ANN infrastructure complexity
- Zero-shot generalization across domains is critical

**Implementation:**

```bash
Documents -> SPLADE encoder -> learned sparse vectors -> inverted index
Query     -> SPLADE encoder -> learned sparse query vector -> inverted index lookup
          -> (optional) dense retrieval for hybrid
```

**Typical performance:** NDCG@10 ≈ 0.45-0.52 on BEIR (better generalization
than dense retrieval out-of-domain)

**Tools:** naver/splade models (HuggingFace), custom inverted index,
Elasticsearch with custom similarity

## Cost and Complexity Comparison

| Stack                 | Setup time | Monthly cost (1M docs)   | Maintenance |
| --------------------- | ---------- | ------------------------ | ----------- |
| BM25 only             | 1 day      | $50-200 (Elasticsearch)  | Low         |
| Dense retrieval       | 3-5 days   | $200-500 (GPU + storage) | Medium      |
| Hybrid search         | 1-2 weeks  | $300-700                 | Medium      |
| Hybrid + reranking    | 2-3 weeks  | $500-1000                | Medium-High |
| RAG pipeline          | 3-6 weeks  | $1000+ (LLM API costs)   | High        |
| Fine-tuned full stack | 2-3 months | $1500+                   | High        |

Costs are rough estimates for a startup-scale deployment. Costs scale with
query volume, corpus size, and GPU instance type.

## Common Mistakes

| Mistake                                                                 | Impact                                        | Fix                                           |
| ----------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| Over-engineering early                                                  | Weeks wasted, complexity without benefit      | Start with BM25, add neural only if needed    |
| Under-engineering for scale                                             | System collapses at load                      | Benchmark at target query volume early        |
| Not measuring BM25 baseline                                             | Cannot quantify improvement from neural stack | Always run BM25 first, measure NDCG           |
| Fine-tuning on wrong domain                                             | Model worse than zero-shot                    | Evaluate pretrained models before fine-tuning |
| Ignoring latency until launch                                           | System unusable in prod                       | Profile each stage, set latency budget early  |
| Using FAISS exact search at scale Query latency spikes at corpus growth | Switch to HNSW or IVF before hitting 1M docs  |
| No monitoring post-launch                                               | Drift undetected                              | Track NDCG on sample of production queries    |

## Upgrade Path

Most teams should start simple and upgrade incrementally when they have evidence
that the current stack is the bottleneck:

```bash
Phase 1 - Baseline (days)
  BM25 with basic query preprocessing
  Measure: NDCG on labeled test set, latency, user feedback

Phase 2 - Neural boost (weeks)
  Add dense retrieval with pretrained bi-encoder
  RRF fusion for hybrid search
  Measure: NDCG improvement over BM25, latency delta

Phase 3 - Reranking (weeks)
  Add MiniLM cross-encoder reranking on top-50 candidates
  Measure: NDCG improvement, latency, whether users notice quality difference

Phase 4 - Fine-tuning (months, if justified)
  Collect labeled data from user interactions
  Fine-tune bi-encoder and/or cross-encoder on your domain
  Measure: NDCG improvement over zero-shot, ROI vs engineering cost

Phase 5 - Scale optimization (ongoing)
  Quantization, ONNX export, result caching, ANN index tuning
  Measure: latency reduction, cost reduction, quality preservation
```

The critical discipline: **only advance to the next phase when you have
measured evidence that the current phase is the limiting factor**. Teams
that skip to Phase 4 without measuring Phase 1 often discover that BM25
was good enough and months were wasted.

## My Summary

Choosing a retrieval stack requires balancing corpus size, query type, labeled
data availability, latency budget, and team capabilities. BM25 is the correct
starting point for almost every application - it is fast, interpretable, requires
no training data, and performs competitively on keyword-heavy queries. Dense
retrieval or hybrid search is justified when vocabulary mismatch is a measured
problem, not a hypothesized one. The upgrade path is incremental: measure BM25
baseline, add neural components when the baseline is demonstrably insufficient,
fine-tune only when enough labeled data exists and zero-shot performance is measured
to be inadequate. The most common mistake is over-engineering the initial stack -
building full hybrid + reranking + fine-tuning infrastructure before measuring
whether BM25 alone meets user needs. Always start with the simplest option that
could plausibly work and upgrade based on measured evidence.
