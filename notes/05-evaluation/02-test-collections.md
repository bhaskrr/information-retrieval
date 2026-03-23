# Test Collections and the Cranfield Paradigm

A test collection is a standardized bundle of three components - a document corpus,
a set of queries, and relevance judgments - used to evaluate and compare retrieval
systems in a reproducible way. The Cranfield paradigm is the methodology behind test collections: fix everything about the evaluation environment so that differences in scores reflect differences in systems, not differences in evaluation conditions.

## Intuition

Imagine two researchers each claiming their retrieval system is better. Without a shared
test collection, they are measuring different things on different data with different
notions of relevance. There is no way to compare them.

A test collection solves this by giving everyone the same corpus, the same queries, and
the same relevance judgments. Now scores are comparable. A system that achieves MAP 0.42
on MS MARCO can be directly compared to one that achieves MAP 0.38 on the same
collection — without any ambiguity about what was measured.

## The Three Components

### 1. Document Corpus

The collection of documents being searched.

Properties that matter:

- **Size** — from hundreds (Cranfield) to billions (web corpora)
- **Domain** — newswire, scientific papers, web pages, legal documents, etc.
- **Format** — plain text, passages, full documents, structured fields
- **Temporality** — static snapshot or continuously updated

### 2. Queries (Topics)

The set of information needs used to evaluate the system.

In formal evaluation, queries are called **topics** and are often accompanied by:

- A short keyword query (what a user might type)
- A description (a sentence describing the information need)
- A narrative (detailed specification of what is and is not relevant)

Example TREC topic:

```bash
Number:      301
Title:       International Organized Crime
Description: Identify organizations that participate in international
             criminal activity, the activity, and the countries involved.
Narrative:   Relevant documents will contain information about
             specific organizations, the crimes committed, and countries
             involved. Documents about national crime are not relevant.
```

In modern benchmarks like MS MARCO, topics are just the raw query strings that users
typed into a search engine — no description or narrative.

### 3. Relevance Judgments (Qrels)

Human annotations specifying which documents are relevant to which queries.

Standard format (qrels file):

```bash
query_id  0  doc_id  relevance_score
```

Example:

```bash
301  0  doc1234  1
301  0  doc5678  0
301  0  doc9012  2
```

- Binary judgments: 0 (not relevant) or 1 (relevant)
- Graded judgments: 0 (not relevant), 1 (marginally), 2 (relevant), 3 (highly relevant)

## The Cranfield Paradigm

### Origins

Developed at Cranfield University in the UK in the 1960s by Cyril Cleverdon. The
original Cranfield collection had ~1,400 documents on aeronautics, 225 queries, and
complete relevance judgments. It was the first systematic attempt to evaluate IR
systems empirically rather than anecdotally.

### The core idea

Fix the evaluation environment completely:

- Same corpus for all systems
- Same queries for all systems
- Same relevance judgments for all systems

Then run different systems, compute scores, compare. The only variable is the system.

### Why it was revolutionary

Before Cranfield, IR evaluation was anecdotal - researchers would demonstrate their
system on a few examples and claim it worked well. Cranfield introduced the scientific
method to IR: controlled experiments with reproducible results.

### The paradigm today

Every modern benchmark is a Cranfield-style collection. TREC, MS MARCO, BEIR, MTEB -
all follow the same structure: corpus + queries + qrels. The scale has changed
dramatically but the methodology is identical.

## The Pooling Problem

### The challenge

For a corpus of 1 million documents and 100 queries, exhaustive relevance judgment
requires 100 million human annotations — completely infeasible.

### The solution — pooling

1. Run multiple retrieval systems on all queries
2. Take the top-k results (typically top 100) from each system
3. Pool the results together, removing duplicates
4. Judge only the pooled set

Any document not in the pool is assumed non-relevant. This assumption holds reasonably
well in practice — documents not retrieved by any system in the pool are unlikely to
be relevant.

### Depth-k pooling

The standard approach. Depth-100 pooling (top 100 from each system) is common in TREC.

### The bias problem

Pooling introduces a bias: systems that contributed to the pool are evaluated fairly,
but new systems that retrieve documents not in the pool are penalized — their relevant
documents are marked as non-relevant because they were never judged. This is called
the **pool bias** problem and is an active area of research.

## Major Test Collections

### Cranfield (1960s)

- ~1,400 documents, 225 queries, complete judgments
- Domain: aeronautics
- Significance: the original; established the evaluation paradigm

### TREC (Text REtrieval Conference)

- Run by NIST since 1992
- Multiple tracks per year: Web, News, Deep Learning, Clinical, etc.
- Corpus sizes from thousands to billions of documents
- Gold standard for academic IR evaluation
- Relevance judgments are graded (0-3) in recent tracks

### Reuters / AP News Corpora

- Newswire collections used in early TREC evaluations
- Still used as standard small-scale benchmarks

### MS MARCO

- Released by Microsoft in 2016
- ~8.8 million passages, ~1 million queries from Bing search logs
- Sparse relevance judgments — typically 1 relevant passage per query
- The dominant training and evaluation corpus for modern neural IR
- Two versions: passage retrieval and document retrieval

### BEIR (Benchmarking IR)

- Released 2021 by Thakur et al.
- 18 heterogeneous datasets across different domains and tasks
- Used for zero-shot evaluation — train on MS MARCO, evaluate on BEIR
- Tests generalization, not just in-domain performance
- Domains include: scientific papers, news, financial documents, biomedical, etc.

### Natural Questions (NQ)

- Google search queries with answers from Wikipedia
- Used for open-domain question answering and passage retrieval
- ~300,000 training queries

### MTEB (Massive Text Embedding Benchmark)

- 56 datasets across 8 task types
- The standard benchmark for evaluating embedding models
- Covers retrieval, clustering, classification, reranking, and more

## Relevance Judgment Scales

| Scale      | Values     | Used in                     |
| ---------- | ---------- | --------------------------- |
| Binary     | 0, 1       | Early TREC, most benchmarks |
| 3-point    | 0, 1, 2    | Some TREC tracks            |
| 4-point    | 0, 1, 2, 3 | TREC Deep Learning track    |
| Continuous | 0.0 - 1.0  | Some crowdsourced datasets  |

## The Incomplete Judgments Problem in Practice

Because pooling only judges a subset of documents, computed metrics are approximations
of true performance. This has two implications:

1. **Absolute scores are not meaningful in isolation** — a MAP of 0.35 only means
   something relative to other systems evaluated on the same collection with the same pooling depth

2. **New systems may be underestimated** - if your system retrieves relevant documents that were not in the original pool, those documents are marked non-relevant and your score is artificially deflated

This is why reusable test collections — where the pool is deep and diverse — are more valuable than shallow ones.

## My Summary

A test collection bundles a corpus, a set of queries, and relevance judgments into a fixed evaluation environment. The Cranfield paradigm - established in the 1960s - says: fix everything except the system, so scores reflect system quality alone. Pooling makes large-scale judgment feasible by only judging documents retrieved by existing systems. Modern benchmarks like MS MARCO, BEIR, and TREC follow this paradigm at massive scale. Understanding test collections is the prerequisite for every evaluation metric that follows - metrics are only meaningful relative to the qrels they are computed against.
