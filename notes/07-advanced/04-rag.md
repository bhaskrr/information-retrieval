# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a framework that combines an information
retrieval system with a large language model (LLM) to answer questions or generate
text grounded in retrieved documents. Instead of relying solely on knowledge encoded
in the LLM's parameters during training, RAG retrieves relevant documents at
inference time and provides them as context to the LLM. The LLM then generates a
response conditioned on both the query and the retrieved documents.

## Intuition

LLMs have two fundamental limitations:

1. **Knowledge cutoff**: LLMs are trained on data up to a fixed date. They cannot
   answer questions about events after that date without external information.

2. **Hallucination**: LLMs generate plausible-sounding text even when they do not
   know the answer. They confabulate facts, cite nonexistent sources, and state
   incorrect information confidently.

RAG addresses both by giving the LLM access to a knowledge base at inference time.
Instead of relying on parametric memory (knowledge baked into weights during
training), the LLM reads retrieved documents and generates answers grounded in
actual source material.

The intuition is simple: asking an LLM to answer from memory is like asking someone
to answer a difficult exam question from memory. RAG is like giving them the textbook
— they can look up the relevant passage and construct a well-grounded answer.

## The RAG Pipeline

```
User query
    ↓
Retriever
  → encodes query (sparse or dense)
  → retrieves top-k documents from knowledge base
    ↓
Context assembly
  → concatenate retrieved documents with query into a prompt
    ↓
LLM generator
  → generates response conditioned on query + retrieved context
    ↓
Response returned to user
```

## Core Components

### 1. Knowledge base

The document collection the retriever searches over. Can be:

- A curated document store (company knowledge base, product documentation)
- A passage-chunked corpus (Wikipedia split into 100-word passages)
- A web index (live retrieval from the web)
- A database of structured records converted to text

### 2. Retriever

The IR system that fetches relevant documents given a query. Any retrieval model
applies here:

```
Sparse (BM25, SPLADE)    → fast, exact match strong
Dense (bi-encoder)       → semantic, handles vocabulary mismatch
Hybrid (sparse + dense)  → best coverage
```

The retriever is the IR component covered throughout this repo. RAG is the
application context that motivates building a good retriever.

### 3. Context assembly

Retrieved documents are formatted into a prompt:

```
System: You are a helpful assistant. Answer the question using only the
provided context. If the context does not contain the answer, say so.

Context:
[1] {retrieved_doc_1}
[2] {retrieved_doc_2}
[3] {retrieved_doc_3}

Question: {user_query}
Answer:
```

Design decisions:

- How many documents to include (top-3 to top-10 typically)
- How to order them (by retrieval score, or reverse — most relevant last)
- Whether to include source citations
- How to handle document length — truncate or summarize

### 4. LLM Generator

Any instruction-following LLM — GPT-4, Claude, Llama, Mistral, etc. The LLM reads
the assembled context and generates a grounded response.

## Naive RAG vs Advanced RAG

### Naive RAG (basic pipeline)

The simplest implementation: retrieve once, generate once.

```
query → retrieve top-k → assemble prompt → generate response
```

Problems:

- Retrieval quality directly caps generation quality
- If relevant documents are not retrieved, the LLM cannot answer correctly
- Long retrieved contexts may confuse the LLM or exceed context window limits
- The retrieval step is not aware of what the LLM needs — it retrieves based on
  the raw query, not on what would be most useful for generation

### Advanced RAG

Improvements over naive RAG:

**Query rewriting**
Rewrite the user's query before retrieval to be more retrieval-friendly:

```
User query:   "what did you just say about transformers"
Rewritten:    "transformer architecture self-attention mechanism"
```

LLMs are used to rewrite queries — HyDE (Hypothetical Document Embeddings) is a
prominent approach: generate a hypothetical answer, embed it, use the embedding
for retrieval instead of the query embedding.

**Multi-step retrieval (iterative RAG)**
Retrieve, generate a partial answer, identify gaps, retrieve again:

```
query → retrieve → partial answer → identify missing info → retrieve again → final answer
```

**Reranking retrieved documents**
Apply a cross-encoder reranker to the retrieved candidates before passing to LLM:

```
query → retrieve top-50 → cross-encoder rerank → top-5 to LLM
```

**Contextual compression**
Instead of passing full retrieved documents, extract only the relevant sentences:

```
retrieved doc (500 words) → compressor → relevant passage (50 words)
```

Reduces context length and noise, improving generation quality.

**Self-RAG**
The LLM learns to decide when to retrieve, what to retrieve, and whether the
retrieved content is relevant. The LLM generates special tokens that trigger
retrieval and assess retrieved document quality.

## Chunking Strategy

How documents are split into retrievable units significantly impacts RAG quality.

### Fixed-size chunking

Split every N tokens with M tokens overlap:

```
chunk_size = 256 tokens
chunk_overlap = 32 tokens
```

Simple, consistent, but may split mid-sentence or mid-concept.

### Sentence-based chunking

Split at sentence boundaries. Preserves semantic units but produces variable-size
chunks.

### Recursive chunking

Split at paragraph → sentence → word boundaries in order, until chunks are below
target size. Preserves document structure.

### Semantic chunking

Use embedding similarity to detect topic shifts — split when adjacent sentences
have low embedding similarity. Produces semantically coherent chunks but is slower.

### Document-specific chunking

For structured documents (PDFs, markdown), split at section headers, list items,
table boundaries. Preserves document semantics.

### The chunking tradeoff

```
Small chunks (64-128 tokens):
  + Higher retrieval precision (less noise in context)
  - May miss cross-sentence context
  - More chunks = larger index

Large chunks (512-1024 tokens):
  + More context per chunk
  - Lower retrieval precision (more irrelevant content passed to LLM)
  - Fewer chunks = smaller index
```

Typical production choice: 256-512 tokens with 10-20% overlap.

## Evaluation of RAG Systems

RAG introduces a two-component evaluation challenge:

### Retrieval evaluation

Standard IR metrics on the retrieval component:

- Recall@k — are relevant documents in the top-k retrieved?
- Precision@k — are retrieved documents relevant?
- MRR, NDCG — ranking quality of retrieved documents

### Generation evaluation

**Faithfulness** — does the generated answer stay grounded in the retrieved context?
Measures hallucination rate.

**Answer relevance** — does the generated answer address the query?

**Context relevance** — are the retrieved documents actually relevant to the query?

**RAGAS** (RAG Assessment) is the standard framework for end-to-end RAG evaluation,
measuring all three dimensions automatically using an LLM judge.

## RAG vs Fine-tuning

A common question: when should you use RAG vs fine-tuning the LLM on your data?

```
Use RAG when:
  - Knowledge changes frequently (news, product updates, live data)
  - You need source citations and grounding
  - Data is too large to fit in context or fine-tuning budget
  - You need to answer questions about specific private documents
  - Factual accuracy is critical

Use fine-tuning when:
  - You need a specific response style or format
  - The task requires specialized reasoning patterns
  - Knowledge is stable and domain-specific
  - Inference latency is critical (no retrieval step)
  - You want to teach the model skills, not facts
```

In practice, RAG and fine-tuning are complementary: fine-tune the LLM on domain-
specific response style while using RAG to inject current factual knowledge.

## Advanced Pattern: HyDE (Hypothetical Document Embeddings)

HyDE improves retrieval quality by generating a hypothetical answer to the query
and using its embedding for retrieval instead of the raw query embedding:

## RAG Failure Modes

Understanding where RAG fails helps build better systems:

| Failure Mode                            | Cause                                       | Fix                                               |
| --------------------------------------- | ------------------------------------------- | ------------------------------------------------- |
| Relevant docs not retrieved             | Poor retrieval quality                      | Improve retriever, use hybrid search              |
| LLM ignores context                     | Context too long or irrelevant noise        | Reduce context, rerank before passing             |
| LLM hallucinates despite good retrieval | Insufficient context for the specific query | Add faithfulness check, improve chunking          |
| Wrong chunk retrieved                   | Chunking splits                             | Semantic chunking, related content larger overlap |
| Slow response                           | Retrieval + LLM latency                     | Cache embeddings,smaller retriever model          |

## Where This Fits in the Progression

```
Dense Retrieval → first-stage neural retrieval
Bi-encoders → efficient first-stage retrieval
Cross-encoders → accurate second-stage reranking
SPLADE → learned sparse retrieval
Reranking → two-stage pipeline
Hybrid Search → combining first-stage signals
ColBERT → late interaction retrieval
RAG → retrieval powering generation ← you are here
Evaluation at Scale → benchmarking advanced systems
```

RAG is the application layer that brings together everything in the IR curriculum.
A complete RAG system uses chunking and indexing (Phase 3), retrieval models (Phase
4-6), evaluation metrics (Phase 5) to measure quality, and hybrid search and
reranking (Phase 7) to maximize retrieval quality before generation.

## My Summary

RAG grounds LLM generation in retrieved documents by retrieving relevant passages
at inference time and assembling them into the LLM prompt. It addresses two core
LLM limitations — knowledge cutoff and hallucination — by providing current,
source-traceable factual context at generation time. The retrieval component is
a full IR pipeline (chunking, indexing, retrieval, optional reranking) and its
quality directly caps the quality of generated answers. Advanced RAG techniques
include query rewriting, iterative retrieval, contextual compression, and HyDE.
RAG is the most practically deployed application of modern IR — virtually every
production LLM system that needs up-to-date or domain-specific knowledge uses
some variant of this architecture.
