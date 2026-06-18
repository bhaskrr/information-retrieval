# Chunking Evaluation

Chunking evaluation is the systematic measurement of how chunking strategy
choices affect retrieval and generation quality in a RAG pipeline. It is the
empirical discipline that transforms chunking from a one-time configuration
decision made by intuition into an evidence-based engineering choice calibrated
to the specific corpus, query distribution, and quality requirements of a system.
A chunking evaluation protocol answers specific, measurable questions: Which
chunking strategy produces the highest Recall@100 on this corpus? What chunk size
maximizes answer accuracy for this query type? How much does switching from
fixed-size to semantic chunking improve end-to-end generation quality? Without
evaluation, teams rely on default parameters that may be dramatically suboptimal
for their specific use case, accept poor retrieval quality as an inherent
limitation of their embedding model, and attribute generation failures to the LLM
rather than to the chunking configuration that determined what context it received.

## Intuition

Every engineer who has deployed a RAG system has experienced a variation of the
same failure pattern: the system performs poorly on a set of representative
queries, the embedding model is assumed to be the bottleneck, a larger or better
embedding model is evaluated and produces minimal improvement,
and the real cause - chunking configuration, was never investigated.

The reason chunking evaluation is systematically skipped is that it requires
a labeled test collection: queries with known relevant passages or document
sections. Creating this test collection takes effort, and the payoff is not
immediately obvious. But the evidence from systematically measuring chunking
impact is consistent: for document-heavy corpora, chunking configuration
frequently has larger impact on Recall@100 than switching embedding models.
A poor chunking strategy cannot be saved by a better embedding model. No
matter how powerful the encoder, if the answer to a query is split across two
chunks at an arbitrary token boundary, neither chunk will retrieve highly enough
to surface the answer.

Chunking evaluation operationalizes the claim made throughout this module:
chunking is a hyperparameter. Like learning rate in neural network training,
its optimal value depends on the specific context, and finding that value
requires measurement.

## What to Measure

Chunking quality affects the retrieval pipeline at two distinct stages, each
requiring different metrics:

### Stage 1 - Retrieval quality

Does the relevant chunk appear in the top-K retrieved results for each query?
This is measured before any reranking or generation occurs.

**Primary metric: Recall@K**

```
Recall@K = |{relevant chunks in top-K retrieved}| / |{all relevant chunks}|

Key choices:
  K should match the candidate set size passed to the reranker:
    If reranker receives top-100: measure Recall@100
    If reranker receives top-50: measure Recall@50

  "Relevant chunk" definition:
    Option A (exact): a chunk is relevant if the exact annotated answer span
                      falls within the chunk
    Option B (overlap): a chunk is relevant if it contains a specified minimum
                        overlap with the annotated answer span (e.g., >50% overlap)
    Option C (human): human annotators judge each retrieved chunk's relevance

  Report both:
    Recall@10 (how often is the answer in the very top results)
    Recall@100 (how often is the answer anywhere in the candidate set)
```

**Secondary metric: Mean Reciprocal Rank (MRR@K)**

```
MRR@K = (1/|Q|) × Σ_q (1 / rank_q)

Where rank_q = rank of first relevant chunk for query q

MRR@10 = 0.85 means on average the first relevant chunk appears at rank 1.18
MRR@10 = 0.40 means on average the first relevant chunk appears at rank 2.5
```

MRR is particularly relevant when only one relevant chunk exists per query -
it directly measures whether that chunk appears near the top of the retrieved list.

**Tertiary metric: NDCG@K**

```
NDCG@K rewards placing highly relevant chunks at the very top ranks
Most useful when chunks are graded (highly relevant, relevant, marginally relevant)
rather than binary (relevant or not relevant)
```

### Stage 2 - Generation quality

Given the retrieved chunks as context, does the LLM produce a correct, faithful,
and complete answer? This measures end-to-end RAG quality, not just first-stage
retrieval.

**Accuracy:**

```
Exact match accuracy:
  For factual queries with short expected answers:
    accuracy = fraction of queries where LLM answer exactly matches gold answer

F1 score:
  Token-level F1 between LLM answer and gold answer
  More forgiving than exact match for paraphrase

LLM-as-judge:
  Prompt a capable LLM to judge if the answer is correct:
    "Given the query and gold answer, is the system's answer correct?
     Answer YES, PARTIAL, or NO."
  Report: percentage of YES, PARTIAL, NO judgments
```

**Faithfulness:**

```
Is the answer grounded in the retrieved chunks?
Hallucination detection: does the answer contain claims not supported by chunks?

LLM-as-judge:
  For each claim in the generated answer:
    "Is this claim supported by the provided context? YES or NO"
  Faithfulness = fraction of claims supported by context
```

**Context relevance:**

```
Does the retrieved context actually help answer the query, or is it
noise that the LLM had to ignore?

Measure: fraction of retrieved chunks that the LLM explicitly
         references or quotes in its answer
```

### The two-stage evaluation interaction

A critical insight: improving retrieval quality (Stage 1) does not always
improve generation quality (Stage 2), and improving generation quality does
not require improving retrieval quality.

```
Case A: Retrieval improves, generation improves
  Small chunks → better recall on specific queries
  LLM receives the right information → better answers
  This is the common case and the primary motivation for chunking evaluation

Case B: Retrieval improves, generation does not improve
  Semantic chunking improves Recall@100 from 0.78 to 0.86
  But each chunk is now too short (50-100 tokens) to provide sufficient context
  LLM receives the right information but lacks surrounding context to use it well
  → Chunk size is a local optimum, not a global optimum

Case C: Retrieval does not improve, generation improves
  Smaller chunks produce lower Recall@100 (fewer chunks capture the full answer)
  But those smaller chunks provide a more focused, less noisy context
  → LLM receives less distraction, produces better answers despite lower recall

Case B and C motivate measuring both stages independently
```

## Building the Evaluation Set

A chunking evaluation requires a test collection of queries with known relevant
passages or document sections. There are four practical approaches for building
this collection:

### Approach 1 - Manual annotation

Human annotators review queries and documents, marking which document sections
are relevant for each query.

```
Process:
  1. Select 50-200 representative queries from production logs or user research
  2. For each query, select 5-20 potentially relevant documents (from BM25 results)
  3. Annotators mark relevant passages within those documents
  4. Verify with second annotator for inter-annotator agreement

Quality:    highest (ground truth from human judgment)
Cost:       $500-$5,000 depending on domain expertise required
Time:       1-3 weeks
Ideal for:  high-stakes systems where evaluation quality is critical
```

### Approach 2 - Synthetic query generation

Generate evaluation queries from documents using an LLM, creating (query, source_passage)
pairs without manual annotation:

```
Process:
  1. Select representative documents from the corpus
  2. For each document section (paragraph or heading unit):
     prompt = f"Generate 2-3 specific questions that this passage answers:
               [passage text]"
     questions = LLM(prompt)
  3. Each generated question has the source passage as its known relevant chunk
  4. Validate with automated quality checks (remove yes/no questions, duplicates)

Quality:    moderate (LLM-generated questions may be too easy or too narrow)
Cost:       $50-$500 in API costs
Time:       hours to 1 day
Ideal for:  rapid iteration during development, initial evaluation before
            investing in manual annotation
```

### Approach 3 - Existing QA datasets as a proxy

Use existing question-answering datasets (Natural Questions, SQuAD, TriviaQA)
where questions have known answer passages, as a proxy for domain-specific evaluation:

```
Limitation: the dataset's document collection may not match your corpus
Adaptation: identify which documents in your corpus contain the same information
            as the QA dataset's source documents

Practical use: for general-purpose knowledge retrieval systems, NQ or
               SQuAD-based evaluation provides a reasonable benchmark
               For domain-specific corpora, this approach is a weak proxy
```

### Approach 4 - Implicit feedback from production

Use production interaction signals (clicks, dwell time, absence of reformulation)
as implicit relevance signals:

```
For each query-document click pair:
  Identify which chunk was displayed / scrolled to
  Use as weak relevance signal

Quality:    low-medium (noisy, position-biased, not chunk-level precise)
Data:       requires deployed system with logging
Use:        continuous monitoring of chunking quality in production,
            not initial evaluation during development
```

## The Chunking Evaluation Protocol

A complete protocol for evaluating and selecting a chunking configuration:

### Step 1 - Define the evaluation set

```
Target: 50-200 queries (more for statistical reliability, fewer for speed)
Balance:
  Include queries that are expected to be easy (single factual lookup)
  Include queries that are expected to be hard (complex, cross-document)
  Include queries representative of the production distribution

Define "relevant chunk":
  A chunk is relevant if it contains the correct answer to the query
  OR if human judgment deems it sufficient context to answer the query

Document this definition explicitly - it affects all comparison results
```

### Step 2 - Define the evaluation grid

List all chunking configurations to evaluate. Avoid evaluating too many at once
(combinatorial explosion), but cover the key dimensions:

```
Dimension 1: Chunking strategy
  Fixed-size, recursive, semantic, document-aware

Dimension 2: Chunk size (for fixed-size and recursive)
  [100, 200, 300, 400, 500] tokens

Dimension 3: Overlap (for fixed-size)
  [0%, 10%, 20%, 30%]

Dimension 4: Semantic threshold (for semantic chunking)
  [90th, 95th, 99th percentile]

Practical approach: fix all but one dimension at a time
  Start: recursive, 300 tokens, 20% overlap (strong baseline)
  Then: vary chunk size at [100, 200, 300, 400] with recursive
  Then: compare best recursive vs semantic vs document-aware
```

### Step 3 - Index and retrieve for each configuration

```
For each chunking configuration:
  1. Re-chunk the corpus with this configuration
  2. Embed all chunks and build retrieval index
  3. For each query in the evaluation set:
     a. Retrieve top-100 chunks
     b. Compute Recall@10, Recall@50, Recall@100, MRR@10
  4. Average metrics across all queries
  5. Record: configuration, chunk count, average chunk size, metrics
```

### Step 4 - Evaluate end-to-end generation quality

For the top-3 configurations from Step 3:

```
For each query in the evaluation set:
  1. Retrieve top-5 chunks with this configuration
  2. Construct LLM prompt with retrieved chunks as context
  3. Generate answer
  4. Compare answer to gold answer (exact match, F1, or LLM judge)
  5. Record faithfulness (fraction of answer claims supported by context)

Goal: identify if the configuration with best Recall@100 also produces
      best generation quality, or if there is a divergence (Case B or C
      from the two-stage interaction analysis above)
```

### Step 5 - Select and validate

```
Select: the configuration that maximizes end-to-end generation quality
        (with Recall@100 as a leading indicator)

Validate: measure selected configuration on a held-out test set
          (different queries than the evaluation grid, same document corpus)
          Report: Recall@100 and generation accuracy on held-out set

Document: which configuration was selected and why
          Log the full evaluation results for future reference
```

## The Recall-Chunk Size Curve

One of the most informative visualization tools in chunking evaluation is the
recall-vs-chunk-size curve, which reveals the systematic relationship between
chunk size and retrieval quality:

```
Typical shape:
  Recall@100
    ↑
0.92│                          _______________
0.88│                    _____/
0.84│             ______/
0.80│      ______/
0.76│_____/
    └────┬────┬────┬────┬────┬────→ Chunk size (tokens)
        50  100  200  300  400  500

Interpretation:
  Very small chunks (50): high recall per token but chunks lack context
                           → individual sentences often insufficient
  Medium chunks (200-300): optimal for most factual queries
  Large chunks (400-500): approaching embedding model token limit
                           → semantic diffusion reduces specificity
```

This curve shape is typical for factual retrieval on technical documents. It
will differ for other query types:

```
For complex multi-hop queries requiring broad context:
  Recall@100 may peak at 400-500 tokens (more context needed per chunk)

For very specific single-sentence factual lookups:
  Recall@100 may peak at 50-100 tokens (short focused chunks best)

For conversational queries:
  Flat curve - chunk size matters less than semantic boundaries
```

The curve reveals the optimal chunk size for a specific corpus and query type
without requiring theoretical justification - it is an empirical measurement.

## The Chunk Size vs Overlap Tradeoff Matrix

For fixed-size chunking, visualizing recall across the (chunk_size, overlap)
grid reveals interaction effects that single-dimension variation misses:

```
Recall@100 across (chunk_size, overlap) grid:

              chunk_size
overlap │  100    200    300    400    500
────────┼──────────────────────────────────
  0%    │ 0.74   0.81   0.85   0.86   0.83
 10%    │ 0.76   0.84   0.87   0.88   0.85
 20%    │ 0.78   0.85   0.89   0.88   0.84
 30%    │ 0.79   0.85   0.88   0.87   0.83
```

Key observations from this matrix:

- **Optimal configuration:** chunk_size=300, overlap=20% (Recall = 0.89)
- **Overlap saturation:** overlap beyond 20% provides no benefit at larger chunk sizes
- **Overlap helps more at small chunk sizes:** at chunk_size=100, overlap from 0% to
  30% adds 0.05 to Recall; at chunk_size=400, the same range adds only 0.01-0.02
- **Large chunk size + high overlap:** worst combination for recall at chunk_size=500
  due to semantic diffusion combined with near-duplicate chunks lowering diversity

This kind of grid evaluation typically requires a few hours of compute but can
permanently improve baseline retrieval quality.

## Comparative Evaluation Across Chunking Strategies

When comparing fundamentally different strategies (fixed-size vs semantic vs
document-aware), report both quality metrics and operational characteristics:

```
Strategy              Recall@100  MRR@10  Index Size  Indexing Time  Complexity
──────────────────────────────────────────────────────────────────────────────────
Fixed-size (300, 20%) 0.89        0.71    100%        1×             Very low
Recursive (300, 10%)  0.91        0.73    95%         1×             Low
Semantic (95th pct)   0.93        0.76    92%         8×             Medium
Document-aware        0.95        0.79    90%         15×            High
Late chunking+doc     0.97        0.82    90%         45×            Very high
```

This comparison shows that:

- Fixed-size is the baseline, not the recommended strategy
- Recursive provides a small but free improvement (no extra compute)
- Semantic provides meaningful improvement at 8x indexing cost
- Document-aware provides the best quality at significant infrastructure cost
- Late chunking compounds document-aware's improvement at very high cost

The decision of which strategy to use cannot be made from quality metrics alone -
the operational cost column is equally important.

## Segment-Level Recall Decomposition

Aggregate recall metrics can mask systematic failures on specific query or document
types. Breaking recall down by segment reveals where each strategy fails:

### By query type

```
Query type              Fixed-size  Recursive  Semantic  Doc-aware
──────────────────────────────────────────────────────────────────────
Simple factual lookup   0.91        0.91       0.90      0.92
Definition queries      0.87        0.89       0.93      0.94
Comparative queries     0.83        0.85       0.88      0.91
Procedural queries      0.82        0.86       0.87      0.93
Table-referencing       0.71        0.73       0.72      0.91  ←
Code-referencing        0.68        0.70       0.71      0.88  ←
```

Table-referencing and code-referencing queries show a dramatically larger
improvement from document-aware chunking (20+ Recall points) than other query
types. If these query types are common in the production distribution, the
investment in document-aware chunking is clearly justified. If they are rare,
simpler strategies may suffice.

### By document type

```
Document type           Fixed-size  Recursive  Semantic  Doc-aware
──────────────────────────────────────────────────────────────────────
Wikipedia articles      0.89        0.91       0.93      0.92
Technical manuals       0.82        0.85       0.88      0.94  ←
Legal documents         0.80        0.83       0.85      0.91  ←
Scientific papers       0.84        0.86       0.90      0.93
News articles           0.91        0.91       0.92      0.91
FAQ documents           0.93        0.93       0.93      0.93
```

Technical manuals and legal documents show larger improvements from document-
aware chunking because their structure (heading hierarchies, tables, defined terms)
provides rich structural signals that document-aware parsing can exploit.

## Comparing Chunking Impact to Embedding Model Impact

A frequently useful calibration: how much does chunking strategy affect recall
compared to switching embedding models?

```
Embedding model × chunking strategy comparison:

Strategy           all-MiniLM-L6   E5-base   BGE-large
Fixed-size         0.81            0.89       0.91
Recursive          0.83            0.91       0.93
Semantic           0.85            0.93       0.95
Document-aware     0.88            0.95       0.96
```

Observations:

- Switching from all-MiniLM to E5-base: +0.08 Recall@100 with fixed-size chunking
- Switching from fixed-size to document-aware: +0.07 Recall@100 with all-MiniLM
- Both changes provide similar magnitude improvement
- The best combination (E5-base + document-aware) provides cumulative benefit

The practical implication: teams that skip chunking evaluation while testing
multiple embedding models may be optimizing the wrong dimension. Measuring
chunking impact on the current embedding model first often reveals larger
improvements at lower cost (chunking strategy change is free; larger embedding
model has inference cost implications).

## Continuous Evaluation in Production

Chunking evaluation is not a one-time activity - it should be part of a
continuous quality monitoring system:

### Metrics to track continuously

```
Daily:
  Zero-chunk-recall rate: fraction of queries where no retrieved chunk is relevant
    (equivalent to "complete retrieval failure" rate from 04-recall-oriented)
  Average semantic similarity between query and top-1 retrieved chunk

Weekly:
  Recall@100 on a rotating sample of labeled queries
  Distribution of chunk sizes in retrieved results
  Fraction of retrieved chunks containing answer vs context-only vs irrelevant

Monthly:
  Full evaluation protocol on representative query sample
  Comparison to previous month (detect drift from corpus changes)
```

### Triggers for re-evaluation

```
Trigger immediate re-evaluation when:
  Zero-chunk-recall rate increases > 5% over 2-week rolling average
  New document type added to corpus (different structure, different optimal chunking)
  Corpus size doubles (may affect optimal chunk size at scale)
  Significant change in query distribution (different query types emerge)
  Embedding model changed (optimal chunk size interacts with model capacity)
```

## Common Evaluation Pitfalls

### Pitfall 1 - Evaluating only on easy queries

Test collections built from generated questions or simple factual lookups
systematically underrepresent hard queries where chunking matters most.
Complex queries that span multiple concepts or require cross-document context
are where chunking strategies diverge most significantly.

**Mitigation:** Deliberately include "hard" queries in the evaluation set

- queries that require synthesizing information, queries about relationships
  between concepts, queries that can only be answered from specialized document
  sections.

### Pitfall 2 - Using chunk count as a proxy for quality

A smaller total number of chunks (from larger chunk sizes) is sometimes
mistakenly treated as evidence of better chunking quality ("fewer redundant
chunks"). Chunk count is not a quality metric. What matters is whether the
relevant information is retrievable, regardless of how many chunks exist.

**Mitigation:** Always measure Recall@K, not chunk count.

### Pitfall 3 - Not accounting for the embedding model's token limit

Evaluating chunking strategies with chunk sizes that exceed the embedding
model's token limit (512 tokens for BERT-family models) introduces silent
truncation that biases results. Chunks that appear to perform well because
they are large may actually be performing well on only their first 400 tokens.

**Mitigation:** Ensure all evaluated chunk configurations fit within the
embedding model's actual token limit with a safety margin (≤ 80% of max).

### Pitfall 4 - Optimizing only Recall@100 when the bottleneck is generation

A chunking configuration with high Recall@100 but poor generation quality
(because chunks are too short to provide sufficient context, or because they
contain too much noise relative to the answer) is not better than a configuration
with slightly lower Recall@100 but much better generation quality.

**Mitigation:** Always evaluate end-to-end generation quality for the top
configurations from the Recall@K ranking, not just first-stage retrieval quality.

### Pitfall 5 - Not separating the evaluation of chunking from the evaluation of the retrieval model

If a retrieval model change (embedding model, index type) is made at the same
time as a chunking change, it is impossible to attribute the quality change to
either factor. This makes it impossible to learn from the evaluation.

**Mitigation:** Change one thing at a time. Evaluate chunking with a fixed
embedding model. Evaluate embedding models with a fixed chunking configuration.

## My Summary

Chunking evaluation measures how chunking strategy choices affect retrieval and
generation quality through a structured empirical protocol rather than intuition
or default parameters. The two primary measurement targets are first-stage retrieval
quality - measured by Recall@K (does the relevant chunk appear in the top-K
retrieved results?) and MRR@K - and end-to-end generation quality - measured by
answer accuracy and faithfulness. These two stages must be evaluated separately
because improvements in one do not always correspond to improvements in the other:
better Recall@100 from larger chunks can reduce generation quality if the chunks
lack the focused context the LLM needs. The evaluation set should contain 50-200
representative queries with labeled relevant passages, built through manual
annotation for high-stakes systems, synthetic query generation for rapid iteration,
or implicit feedback from production for continuous monitoring. The evaluation
protocol involves comparing configurations across a systematic grid (varying chunk
size and overlap for fixed-size, thresholds for semantic, separator hierarchy for
recursive), measuring Recall@K for each configuration, selecting the top few for
end-to-end generation evaluation, and validating the winner on a held-out set.
Segment-level recall decomposition by query type and document type reveals which
strategies are best for which content, often showing that document-aware chunking
provides the largest benefit specifically for table-referencing and code-referencing
queries while simpler strategies suffice for general text. Chunking impact is
typically comparable in magnitude to embedding model impact - making systematic
chunking evaluation a high-value, often-overlooked investment.
