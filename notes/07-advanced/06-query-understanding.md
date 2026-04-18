# Query Understanding

Query understanding is the set of techniques applied to a raw user query before
retrieval begins - analyzing what the user is asking, what type of information
they need, and how to best represent that need for the retrieval system. It is
the preprocessing layer that sits between the user interface and the retrieval
pipeline. Good query understanding makes retrieval faster, more accurate, and
more robust to the variability of natural language input.

## Intuition

Users are not information retrieval systems. They type short, ambiguous, often
misspelled fragments of their actual information need. "python tutorial beginner
fast" is not a well-formed retrieval query - it has no punctuation, mixes intent
signals (beginner, fast), and does not say what aspect of Python is wanted.

A retrieval system that treats this raw string as its input is throwing away
signal. A system with query understanding analyzes it first:

- Intent: informational (user wants to learn something)
- Domain: programming, specifically Python
- Difficulty: beginner level
- Constraint: prefers concise content

With this understanding, retrieval can be targeted more precisely - boosting
documents tagged as tutorials, filtering out advanced content, expanding the
query with related terms like "introduction" and "getting started."

In production search engines, query understanding is one of the highest-leverage
components. A 10% improvement in query understanding often produces a larger
uplift in end-to-end retrieval quality than a 10% improvement in the retrieval
model itself.

## The Query Understanding Pipeline

```bash
Raw user query
      ↓
Spell correction          → "pytohn" → "python"
      ↓
Query normalization       → lowercase, punctuation removal
      ↓
Query classification      → intent type, domain, difficulty
      ↓
Named entity recognition  → identify entities in query
      ↓
Query expansion           → add related terms, synonyms
      ↓
Query rewriting           → reformulate for retrieval
      ↓
Refined query / query representation
      ↓
Retrieval system
```

Not every system implements all stages. Simple systems may only do normalization
and expansion. Production search engines implement all stages with learned models.

## Spell Correction

### Why it matters

Studies show 10-15% of web search queries contain spelling errors. A retrieval
system without spell correction completely misses these queries if the correct
term does not appear in the misspelled form.

### Edit distance approach

The simplest approach: find the dictionary word with the smallest edit distance
(Levenshtein distance) from the misspelled query token:

```bash
"pytohn" → candidates within edit distance 2:
  python (distance 2) ← correct
  python3 (distance 3)
  ...
```

Fast but limited - does not use context to disambiguate ("recieve" in a sentence
about mail vs code).

### Language model approach

Use a language model to score candidate corrections in context:

```bash
Query: "pytohn tutorial beginner"
P("python tutorial beginner") >> P("pytohn tutorial beginner")
→ correct to "python"
```

SymSpell and BK-trees are efficient data structures for fast spell correction
lookups. For production systems, a character n-gram language model over a query
log provides better corrections than pure dictionary lookup.

### Noisy channel model

Treat misspelling as a noisy channel problem:

```bash
P(correction | misspelling) ∝ P(misspelling | correction) × P(correction)
```

P(correction) = language model probability of the corrected word in context.
P(misspelling | correction) = keyboard error model (e.g. q and w are adjacent).

## Query Classification

### Intent classification

Classify the user's intent to determine retrieval strategy:

```bash
Navigational  → user wants a specific page/resource
               "python.org download"
               Strategy: exact match, boost official sources

Informational → user wants to learn or understand something
               "what is gradient descent"
               Strategy: broad retrieval, educational content

Transactional → user wants to do something (download, buy, sign up)
               "download pytorch latest version"
               Strategy: action-oriented results

Factual       → user wants a specific fact
               "who created python"
               Strategy: QA-style retrieval, short precise answer

Exploratory   → user is browsing a topic
               "machine learning applications"
               Strategy: diverse results, broad coverage
```

Different intent classes warrant different retrieval strategies - a navigational
query should return one highly specific result, while an exploratory query should
return diverse results covering different aspects.

### Domain classification

Identify the subject domain of the query:

```bash
"gradient descent learning rate"  → domain: machine learning
"python syntax error"             → domain: programming
"transformer architecture paper"  → domain: deep learning research
```

Domain classification enables domain-specific retrieval strategies - boosting
results from authoritative domain sources, applying domain-specific query
expansion vocabularies, routing to specialized indexes.

### Query difficulty prediction

Predict how hard it will be to retrieve relevant documents:

```bash
Easy query:   "python hello world"  → many relevant documents, high confidence
Hard query:   "niche python library for async database connections" → few documents
```

Hard queries benefit from more aggressive retrieval strategies - broader expansion,
lower confidence thresholds, more candidates passed to reranking.

Standard difficulty predictors:

- **SCQ (Simplified Clarity Score)** - measures how well the query terms predict
  relevant documents
- **VAR (Variance of Average Weight)** - variance of IDF scores across query terms
- **MaxIDF** - maximum IDF of any query term (rare terms = hard queries)

## Named Entity Recognition in Queries

Identifying named entities in queries changes how retrieval should work:

```bash
Query: "Karpukhin DPR paper 2020"
Entities: [Karpukhin → PERSON, DPR → MODEL_NAME, 2020 → DATE]
```

Named entities should be treated differently from content words:

- Entity terms should not be stemmed ("BERTs" → do not stem to "bert")
- Entity terms should be matched exactly (BM25 with high weight on exact match)
- Entity terms can be used for knowledge base lookup (fetch entity page directly)

Standard NER models (spaCy, HuggingFace NER) work reasonably well for common
entity types (people, organizations, locations). For domain-specific entities
(model names, paper titles, dataset names) a domain-specific NER model is needed.

## Query Expansion

Query expansion adds terms to the query that were not in the original but are
semantically related - improving recall by retrieving documents that use
different vocabulary to express the same concept.

### Manual thesaurus expansion

Use a curated synonym dictionary:

```bash
"car" → expand to ["car", "automobile", "vehicle", "auto"]
```

Simple and reliable for known synonyms. Does not generalize to new terms.

### Pseudo-Relevance Feedback (PRF)

Retrieve an initial set of documents, extract the most prominent terms, add
them to the query, re-retrieve:

```bash
Original query: "python async programming"
Top-3 retrieved docs contain: asyncio, coroutine, event loop, await
Expanded query: "python async programming asyncio coroutine event loop"
→ re-retrieve with expanded query
```

Classic technique, still effective. Risk: top retrieved documents may not
be relevant (especially for hard queries), leading to query drift.

### Neural query expansion - BERT

Use a BERT masked language model to predict terms that should co-occur with
the query:

```bash
Query: "python [MASK] programming"
BERT predicts: async, web, functional, concurrent, ...
Expand query with top-k predictions
```

### HyDE - Hypothetical Document Embeddings

Generate a hypothetical answer document using an LLM, embed it, use the
embedding for retrieval:

```bash
Query: "how does attention mechanism work in transformers"
LLM generates hypothetical answer: "The attention mechanism computes..."
Embed hypothetical answer → use as query vector for dense retrieval
```

HyDE often outperforms direct query embedding for complex informational queries
because the hypothetical document embedding better represents the expected
relevant document space. Covered in RAG context in 07-advanced/04-rag.md.

### Query expansion with SPLADE

SPLADE performs implicit query expansion automatically - the MLM head predicts
related vocabulary terms and assigns them non-zero weights without explicit
expansion logic. This is one reason SPLADE generalizes better than bi-encoders
across BEIR domains.

## Query Rewriting

Query rewriting transforms the query into a different form that is more
amenable to retrieval - not just adding terms but fundamentally reformulating
the query.

### Abbreviation expansion

```bash
"ML models" → "machine learning models"
"NLP tasks" → "natural language processing tasks"
```

### Conversational query rewriting

In multi-turn conversations, queries often depend on context:

```bash
Turn 1: "What is BERT?"
Turn 2: "How was it trained?"   ← ambiguous without context
         → rewrite to "How was BERT trained?"
```

A rewriting model takes the conversation history and the current query,
produces a standalone query that can be understood without context.

### LLM-based rewriting

Use an LLM to rewrite queries for better retrieval:

```bash
System: "Rewrite this query to be more specific and retrieval-friendly.
         Keep the intent but make it more precise."

User: "python stuff for beginners"
LLM:  "python programming tutorial for beginners introduction"
```

LLM rewriting is powerful but adds latency - only worth it for complex or
ambiguous queries.

## Query Segmentation

Long queries should be segmented into meaningful units before retrieval:

```bash
Query: "best python async web framework 2024 tutorial"
Segments: ["async web framework", "python", "2024", "tutorial"]
→ treat each segment as a retrieval unit with different weights
```

Segmentation prevents over-specification - treating a long query as a single
unit causes BM25 to require all terms to appear, hurting recall.

## Query Performance Prediction

Query Performance Prediction (QPP) estimates retrieval quality for a given
query before or after retrieval - without looking at ground truth relevance.

### Pre-retrieval QPP

Estimate difficulty from the query alone, before retrieval:

```bash
Features:
  Query length              → short queries are often harder
  IDF scores of query terms → rare terms = harder
  Query clarity score       → how specific is the vocabulary?
```

### Post-retrieval QPP

Use retrieval results to estimate performance:

```bash
Features:
  Score variance of top-k results → low variance = uncertain ranking
  Score gap between rank-1 and rank-2 → large gap = confident top result
  Coherence of top results → similar results = confident retrieval
```

QPP is used in production systems to route hard queries to more expensive retrieval strategies (e.g. use cross-encoder for all queries when QPP predicts low retrieval confidence).

## Query Understanding in Production Systems

Real production search systems go far beyond rule-based query understanding.
Here is what major search engines implement:

| Component             | Production approach                                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Spell correction      | Neural sequence-to-sequence correction model trained on query logs with click feedback                               |
| Intent classification | Fine-tuned BERT classifier on thousands of labeled query-intent pairs per domain                                     |
| Entity recognition    | Domain-specific NER trained on annotated query logs; linked to knowledge graph                                       |
| Query expansion       | Learned expansion using query-document co-click data; session-based expansion using queries from same search session |
| Query rewriting       | Seq2seq model trained on (original, rewritten) pairs mined from query reformulation logs                             |
| Difficulty prediction | Gradient boosted trees on IDF, clarity, scope features; calibrated per domain                                        |

The key data source for all of these is the query log - the record of what users
searched, what they clicked, and what they reformulated. Without query logs,
production-quality query understanding is not achievable. This is why open-source
systems rely more on model-based approaches (BERT, T5) than query log mining.

## Impact on Retrieval Quality

Understanding the contribution of each query understanding component helps
prioritize engineering effort:

| Component                | Typical NDCG@10 uplift              |
| ------------------------ | ----------------------------------- |
| Spell correction         | +2-5% (on queries with errors)      |
| Intent-based routing     | +3-8% (correct strategy per intent) |
| Query expansion          | +2-6% (recall improvement)          |
| Conversational rewriting | +5-15% (on multi-turn queries)      |
| Entity-aware retrieval   | +3-10% (on entity-centric queries)  |

Spell correction and conversational rewriting typically give the highest return
per engineering effort - they address systematic failure modes that affect large
fractions of queries.

## Where This Fits in the Progression

```bash
Classical retrieval (BM25, VSM)  → retrieves based on raw query terms
Neural retrieval (dense, sparse) → retrieves based on query embeddings
Query understanding              → transforms raw query before retrieval
                                   ← you are here
Advanced topics (RAG, hybrid)    → use retrieved results for generation
Evaluation at scale              → measure quality across all query types
```

Query understanding is the upstream component that determines the quality of
input to every retrieval model covered in this repo. A neural retriever with
good query understanding consistently outperforms the same retriever without it.

## My Summary

Query understanding transforms raw user input into a form that retrieval systems
can act on precisely. It encompasses spell correction (fixing surface errors),
intent classification (understanding what the user wants to do), named entity
recognition (identifying entities that need exact matching), query expansion
(adding related terms to improve recall), and query rewriting (reformulating for
better retrieval or for conversational context). Production systems implement all
these stages with learned models trained on query logs - a resource unavailable
to most open-source systems, which rely instead on model-based approaches. Query
understanding is one of the highest-leverage components in a production IR pipeline
because it directly determines the quality of input to every downstream retrieval
model - a well-understood query with good expansion often outperforms a better
retrieval model receiving a poorly processed raw query.
