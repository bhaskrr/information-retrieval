# Why Evaluation Matters

Evaluation in IR is the systematic measurement of how well a retrieval system satisfies user information needs. It provides the empirical foundation for comparing systems, tuning parameters, and making principled decisions about which approach works better for a given task.

Without evaluation, IR is guesswork. You might build a BM25 system and a TF-IDF system
and believe BM25 is better, but "better" means nothing without a way to measure it
reproducibly. Evaluation gives you a number. Numbers let you compare, iterate, and
improve.

The deeper problem is that relevance is subjective. Two people searching for "python"
may want the programming language or the snake. A document that is relevant to one
user may be irrelevant to another. Evaluation frameworks exist to make this subjectivity manageable by fixing the notion of relevance for a given set of queries and measuring against it consistently.

## The Core Question Evaluation Answers

```bash
Given a retrieval system S and a set of queries Q,
how well does S satisfy the information needs expressed in Q?
```

This decomposes into three sub-questions:

1. **Are the right documents being retrieved?** — relevance
2. **Are they being ranked in the right order?** — ranking quality
3. **How does this system compare to another?** — comparative evaluation

## Offline vs. Online Evaluation

### Offline evaluation

Conducted in a lab setting using a fixed test collection:

- A corpus of documents
- A set of queries
- Relevance judgments: human annotations of which documents are relevant to which queries

Advantages: reproducible, cheap, no real users needed.  
Disadvantages: judgments may not reflect real user behavior, corpus is static.

### Online evaluation

Conducted with real users in a live system:

- A/B testing — split users between two systems, measure which produces better outcomes
- Implicit feedback — clicks, dwell time, reformulation rate as proxies for satisfaction

Advantages: reflects actual user behavior, captures real information needs.  
Disadvantages: expensive, requires live traffic, harder to control.

For learning IR, offline evaluation is what you will work with. Almost all benchmarks
and research papers use it.

## What Makes a Good Evaluation Framework

Three properties matter:

### 1. Reproducibility

Given the same system and the same test collection, two researchers should get the
same score. This requires fixed queries, fixed relevance judgments, and a defined
scoring protocol.

### 2. Validity

The metric should actually measure what you care about. A metric that rewards
retrieving many documents regardless of relevance is not valid. The metric must
correlate with real user satisfaction.

### 3. Discriminability

The metric should be sensitive enough to detect real differences between systems.
A metric where every system scores between 0.48 and 0.52 is not useful for
distinguishing good systems from bad ones.

## The Relevance Judgment Problem

Human relevance judgments are the foundation of offline evaluation. But they are:

- **Expensive** — judging thousands of query-document pairs requires significant
  human effort
- **Incomplete** — for a corpus of 1 million documents and 100 queries, you cannot
  judge every possible pair (100 million judgments)
- **Inconsistent** — different judges often disagree on borderline cases
- **Binary or graded** — is relevance a yes/no decision or a spectrum?

### Pooling

The standard solution to incompleteness. Run multiple retrieval systems, take the
top-k results from each, pool them together, and judge only the pooled set. Any
document not in the pool is assumed non-relevant. This makes large-scale evaluation
feasible without judging every document.

### Binary vs. graded relevance

- **Binary**: a document is relevant (1) or not (0). Simple but loses nuance.
- **Graded**: a document is highly relevant (3), relevant (2), marginally relevant
  (1), or not relevant (0). More informative but harder to collect.

Metrics like precision and recall assume binary relevance. NDCG is designed for
graded relevance. Both are covered in subsequent notes.

## Why Good Metrics Are Hard to Design

A retrieval metric must simultaneously capture:

- **Precision** — not retrieving junk
- **Recall** — not missing relevant documents
- **Ranking** — relevant documents should appear early
- **Graded relevance** — highly relevant documents matter more than marginally relevant
- **User behavior** — users rarely look past the first page

No single metric captures all of these perfectly. This is why there are many metrics
and why different tasks call for different ones:

```bash
Web search           → P@10, NDCG@10  (users look at top 10)
Question answering   → MRR            (one right answer exists)
Legal discovery      → Recall         (must not miss relevant docs)
Recommendation       → NDCG, MAP      (ranked list quality matters)
```

## The Cranfield Paradigm

The foundation of modern IR evaluation, established in the 1960s at Cranfield
University. The idea: fix a test collection (corpus + queries + relevance judgments)
and use it to compare retrieval systems objectively.

This paradigm is still the dominant evaluation methodology in IR today. Every major
benchmark — TREC, MS MARCO, BEIR — is a Cranfield-style test collection.

It is covered in detail in 02-test-collections.md.

## The Gap Between Metrics and Reality

Even good metrics have a gap with real user satisfaction. A system can score well on
MAP and still feel bad to use because:

- The metric uses old relevance judgments that do not reflect current information needs
- The metric does not account for diversity — 10 highly relevant but identical documents
  score perfectly but help nobody
- The metric does not account for latency — a perfect result in 5 seconds is worse
  than a good result in 50ms for most users
- Relevance judgments were made by professional judges, not the actual users

This gap is not a reason to dismiss metrics — it is a reason to understand what they
measure and what they do not.

## Where This Fits in the Learning Path

```bash
Classical models (Boolean → BM25) give you systems to evaluate.
Evaluation gives you the tools to measure how good those systems are.
Neural IR gives you better systems — but you need evaluation to prove they are better.
```

Evaluation is not a detour from learning IR — it is what makes the rest of the
learning meaningful. Without it, you cannot tell whether BM25 is better than TF-IDF,
or whether your neural retriever beats your sparse baseline.

## My Summary

IR evaluation is the systematic measurement of retrieval quality against fixed relevance judgments. It answers whether the right documents are retrieved, whether they are ranked correctly, and how one system compares to another. Offline evaluation using fixed test collections is the standard approach in research, built on the Cranfield paradigm of corpus + queries + relevance judgments. Good metrics must be reproducible, valid, and discriminative — but no single metric captures everything, which is why different tasks use different metrics.
