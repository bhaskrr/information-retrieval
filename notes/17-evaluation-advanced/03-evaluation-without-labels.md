# Evaluation Without Labels

Evaluation without labels is the set of techniques for estimating retrieval system
quality when no human relevance judgments exist for the queries and documents being
evaluated. Standard offline evaluation requires a test collection with annotated
(query, document, relevance_grade) triples. These annotations are expensive, slow
to produce, and unavailable for new domains, new corpora, or novel query types.
Evaluation without labels encompasses three broad families of approaches: using
Large Language Models as automated relevance judges (LLM-as-judge), generating
evaluation data synthetically from corpus documents without any human annotation,
and using statistical corpus-based signals that can estimate quality without any
per-document judgments. Together these techniques enable retrieval teams to measure
system quality in conditions where traditional evaluation would require weeks of
annotation work and thousands of dollars in annotation costs.

## Intuition

Evaluation without labels sounds like it should not work - how can you measure
whether a system retrieved the right documents when you do not know which documents
are right? The answer lies in two sources of signal that exist without annotation:

**The corpus itself.** Documents in the corpus have semantic content that can be
compared to queries. A document that contains concepts closely related to the query
is more likely to be relevant than one that does not. Statistical similarity measures
can approximate relevance without requiring anyone to explicitly label documents.

**Language model knowledge.** A large pretrained language model has internalized
enormous amounts of information about what answers questions, what constitutes a
good explanation, and what content is topically related to a given query. By asking
an LLM to judge whether a document is relevant to a query, we leverage this implicit
knowledge as a substitute for human annotator judgments.

Neither source is perfect. Corpus-based signals miss relevance that depends on
world knowledge external to the corpus. LLM judges have systematic biases - they
prefer longer documents, well-structured prose, and fluent writing even when a
shorter document is actually more factually accurate. But both are dramatically
cheaper and faster than human annotation, and when their biases are understood and
managed, they produce evaluation signals useful enough to guide system development.

## The Annotation Bottleneck

Standard evaluation pipeline timeline:

```
Week 1: Design annotation guidelines
Week 2: Recruit and train annotators
Week 3-5: Annotation (1000 query-document pairs, 2 annotators each)
Week 6: Adjudication of disagreements
Week 7: Quality control and final dataset assembly
Total: 6-8 weeks, $5,000-$50,000 depending on annotator rates and domain
```

For a production team that ships model updates weekly, six-week evaluation
cycles are completely impractical. For a team exploring a new domain without
labeled data, the annotation cost may exceed the project budget entirely.

Evaluation without labels reduces this to:

```
LLM-as-judge approach:
  Day 1:  Generate candidate query-document pairs
  Day 1:  Run LLM judge on all pairs (hours of API calls)
  Day 2:  Analyze results
  Total:  1-2 days, $50-500 in API costs
```

The output is noisier than human annotation, but noise-with-speed is often more
valuable during development than precision-with-delay.

## LLM-as-Judge for Retrieval Evaluation

The most powerful evaluation-without-labels approach. A capable LLM (GPT-4,
Claude, Gemini) is prompted to play the role of a relevance assessor, judging
whether a retrieved document is relevant to a query.

### Basic relevance judgment

The simplest form: ask the LLM to produce a binary or graded relevance label
for each (query, document) pair:

```
Prompt:
  You are a relevance judge evaluating search results. Given a query and a
  document, assess how relevant the document is to the query.

  Query: {query}

  Document: {document_text}

  Please rate the relevance on the following scale:
    0 - Not relevant: The document does not address the query at all
    1 - Marginally relevant: The document tangentially relates to the query
    2 - Relevant: The document addresses the query
    3 - Highly relevant: The document directly and comprehensively answers the query

  Respond with only the integer score (0, 1, 2, or 3).
```

These LLM-generated relevance labels are then used exactly like human relevance
labels to compute NDCG@K, MAP, MRR, and other standard metrics.

### Quality of LLM judgment

Research shows that GPT-4 and Claude-3 Sonnet achieve inter-annotator agreement
with human judges comparable to agreement between human judges themselves -
approximately 0.7-0.8 Cohen's kappa on binary relevance, 0.5-0.6 on graded
relevance. This is sufficient for many evaluation purposes, particularly for
comparative system evaluation (does System A outperform System B?) even if
absolute score values are noisy.

Key findings from TREC 2023 and related work:

```
Pairwise agreement (LLM vs human):  ~75-85%
Pairwise agreement (human vs human): ~75-85%
System ranking correlation:          Spearman ρ ≈ 0.85-0.95 (system rankings
                                     from LLM labels correlate well with human label rankings)
```

The system ranking correlation is the most practically important finding: even
if individual document relevance labels from LLMs do not perfectly match human
labels, the relative ranking of retrieval systems using LLM labels tends to
match the ranking using human labels. This means LLM evaluation can reliably
identify whether System A is better than System B, even if the absolute NDCG
values are not comparable to human-labeled benchmarks.

### Pairwise comparison evaluation

Instead of absolute labels, ask the LLM to compare two documents and determine
which better answers the query:

```
Prompt:
  Given the following query and two documents, determine which document better
  answers the query. Consider factual accuracy, completeness, and directness
  of the answer.

  Query: {query}

  Document A: {doc_a}

  Document B: {doc_b}

  Which document better answers the query? Answer only "A" or "B" or "Tie".
```

Pairwise comparison is more reliable than absolute scoring because it is a
simpler cognitive task - relative judgments are easier to make consistently than
absolute judgments. Aggregate pairwise preferences using Bradley-Terry or
win-rate scoring to produce a full ranking.

### Chain-of-thought evaluation

Asking the LLM to reason before judging improves calibration and explainability:

```
Prompt:
  You are evaluating whether a document is relevant to a search query.

  Query: {query}

  Document: {document}

  Please follow these steps:
  1. Identify the main information need expressed by the query.
  2. Identify the main topic and key claims of the document.
  3. Determine whether the document addresses the information need directly.
  4. Consider whether any important aspects of the query are NOT addressed.

  After your analysis, provide a relevance score (0-3) and a one-sentence
  justification.

  Format your response as:
  Analysis: [your step-by-step reasoning]
  Score: [0/1/2/3]
  Justification: [one sentence]
```

Chain-of-thought evaluation produces higher quality labels (studies show ~10-15%
improvement in human agreement) and provides interpretable explanations for
why each document was rated as it was - useful for debugging retrieval systems.

### LLM judge biases

LLM judges have systematic biases that must be understood and managed:

**Length bias:** LLMs tend to rate longer documents as more relevant because
they contain more content. A 500-word document about a topic scores higher than
a 100-word document with the same information density.

**Sycophancy:** Some LLMs rate documents that use authoritative language,
academic prose, and well-structured argumentation as more relevant even when
a simpler document better answers the query.

**Position bias in pairwise:** When comparing Document A vs Document B,
the document presented first tends to receive slightly higher ratings due
to primacy effects in the LLM's attention.

**Knowledge boundary:** LLMs may incorrectly judge documents in domains
where their training data was limited or where information has changed since
training.

**Mitigations:**

- Normalize document length before judging (truncate long documents)
- Run pairwise comparisons with both orderings (A vs B and B vs A) and average
- Include diverse judge models to reduce single-model bias
- Compare LLM ratings to human spot-checks to calibrate the bias magnitude

## Synthetic Query Generation for Evaluation

Generate evaluation queries synthetically from the corpus documents, creating
(query, relevant_document) pairs without any human annotation:

### DocTTTTTQuery-based evaluation

Use a T5 model fine-tuned for query generation to create plausible queries for
each document:

```
For document D:
  synthetic_queries = T5_query_generator(D)
  Example: D = "aspirin reduces platelet aggregation and prevents blood clots"
           → q₁ = "how does aspirin prevent heart attacks"
           → q₂ = "what is the mechanism of aspirin anticoagulation"
           → q₃ = "aspirin platelet function"
```

Each document becomes the known-relevant document for its synthetic queries.
This creates an evaluation set where relevance is certain (by construction)
rather than annotated.

**Advantages:**

- No human annotation required
- Can generate evaluation sets for any corpus domain
- Each document gets multiple queries covering different aspects

**Disadvantages:**

- Synthetic queries may not reflect real user query distribution
- Evaluation is biased toward queries that the corpus naturally answers
  (queries whose answers are missing from the corpus cannot be generated)
- Generated queries may be easier or harder than real queries in unpredictable ways

### LLM query generation with diversity constraints

Generate queries that cover diverse aspects and difficulty levels:

```
Prompt:
  Given the following document, generate 5 diverse search queries that this
  document would be the best answer for. Include:
  1. A simple keyword query
  2. A natural language question
  3. A technical/expert query
  4. A query using different vocabulary than the document (synonym queries)
  5. A multi-aspect query that combines multiple concepts from the document

  Document: {document}

  Return each query on a separate line.
```

This produces evaluation sets with controlled query diversity, addressing the
homogeneity problem of simple query generation methods.

### Hard negative generation for evaluation

Beyond generating positive queries, generate hard negative documents - documents
that appear related but do not actually answer the query:

```
For query q with true relevant document D:
  hard_negatives = retrieve_top_k_excluding_D(q, k=5)

Evaluation:
  System quality measured by whether it ranks D above hard_negatives
  NDCG computed with D as relevant, hard_negatives as non-relevant
```

Hard negatives make the evaluation more discriminating - a system that simply
retrieves semantically similar documents without distinguishing relevance
will score poorly on hard negative evaluation.

## ARES: Automated Retrieval Evaluation System

ARES (Saad-Falcon et al., 2023) is a framework for automated RAG evaluation
that combines LLM judges with human calibration:

### ARES components

**Context Relevance:** Does the retrieved document contain information relevant
to the query?

**Answer Faithfulness:** Does the generated answer faithfully reflect the
retrieved document? (for RAG evaluation)

**Answer Relevance:** Does the generated answer address the original query?

### The calibration approach

ARES uses a small number of human annotations (as few as 150-300 examples) to
calibrate the LLM judge, correcting for systematic biases:

```
Step 1: Collect 200 human relevance judgments (small, fast)
Step 2: Run LLM judge on same 200 examples
Step 3: Fit a calibration model:
  calibrated_score = f(llm_score, human_score_regression)
Step 4: Apply calibrated judge to the full evaluation set (thousands of examples)

Result: calibrated scores that approximate human judgments with only 200
        annotations instead of thousands
```

This blending of human calibration with LLM scale is the key ARES contribution -
it gets most of the benefit of full human annotation at a fraction of the cost.

## RAGAS: Evaluation Framework for RAG Systems

RAGAS (Shahul Es et al., 2023) specifically addresses evaluation of retrieval-
augmented generation pipelines. It uses reference-free metrics computed by
LLMs:

### RAGAS metrics

**Faithfulness:** Fraction of claims in the generated answer that are supported
by the retrieved context:

```
Prompt: "Given the following context and answer, identify each factual claim
         in the answer and determine whether it is supported by the context.
         Return the fraction of supported claims."
```

**Answer Relevancy:** How well the answer addresses the original question,
measured by asking the LLM to generate questions from the answer and checking
similarity to the original:

```
Generate k questions from the answer:
  q₁, q₂, ..., qₖ = LLM.generate_questions(answer)
Answer relevancy = mean(cosine_similarity(qᵢ, original_question))
```

**Context Recall:** Fraction of ground truth answer sentences attributable to
the retrieved context (requires a reference answer):

```
For each sentence in reference_answer:
  attributable = LLM.can_be_attributed_to_context(sentence, retrieved_context)
Context recall = mean(attributable)
```

**Context Precision:** Fraction of retrieved context that is relevant to
answering the question:

```
For each sentence in retrieved_context:
  useful = LLM.is_useful_for_answering(sentence, question)
Context precision = mean(useful)
```

RAGAS metrics enable evaluation of the full RAG pipeline - not just whether the
right document was retrieved, but whether the generation step used it correctly.

## Corpus-Based Evaluation Without LLMs

When LLM API costs are prohibitive or data privacy prevents sending documents
to external APIs, corpus-based statistical approaches can estimate quality:

### Expected Reciprocal Rank from click models

If the system has been deployed and user interactions have been logged, click
models can infer relevance probabilities without explicit annotation:

```
Click model assumption:
  P(click | doc, position) = P(examine | position) × P(relevant | doc, query)

Infer P(relevant) by:
  Fitting position-based click model to logged data
  Deconvolving position effects from click rates
  Using result as proxy for relevance
```

The resulting inferred relevance values can be used to compute NDCG and MRR
without any explicit annotation.

### Embedding-based quality estimation

Measure alignment between query embeddings and document embeddings as a proxy
for relevance:

```
Embedding similarity distribution:
  For retrieved results, compute cosine similarity between query and document
  High average similarity → retrieval found relevant content
  Low average similarity → retrieval failed to match the query

Calibration:
  On a small labeled set, establish what similarity score corresponds to
  "relevant" and "not relevant" in your domain
  Apply threshold to unlabeled data
```

This is noisy but fast and requires no human annotation or LLM calls. It is
useful as a cheap monitoring signal rather than a precise quality measurement.

### Cluster coherence evaluation

Measure whether retrieved results cover diverse relevant aspects of the query
or cluster around a single interpretation:

```
For each query and its top-k results:
  Cluster results by topic using k-means on embeddings
  Coherence = fraction of results in the dominant cluster
  Diversity = 1 - coherence

High coherence + high relevance = good, focused retrieval
High coherence + low diversity = potentially missing relevant aspects
```

Cluster coherence requires no labels but requires knowing approximately how
many relevant clusters should exist for a query type.

## Practical Evaluation Protocol for New Domains

When entering a new domain with no labeled data and needing evaluation quickly:

### Phase 1 - Bootstrap evaluation (Day 1-2)

Generate synthetic evaluation data using LLMs:

```
Step 1: Sample 200-500 documents from the corpus
Step 2: Generate 2-3 synthetic queries per document
Step 3: Create evaluation set:
  (synthetic_query, document) pairs with document as positive
  Add BM25-retrieved hard negatives for each pair
Step 4: Run system evaluation on synthetic set
Report: NDCG@10 on synthetic queries (note: biased toward corpus-answerable queries)
```

### Phase 2 - Calibrated LLM evaluation (Week 1)

Add LLM judging to evaluate on more realistic queries:

```
Step 1: Collect 100-200 real user queries (from logs, product team, experts)
Step 2: Run retrieval system, take top-5 results per query
Step 3: LLM judges each (query, document) pair (0-3 scale)
Step 4: Compute NDCG@5 using LLM relevance labels
Report: NDCG@5 on real queries with LLM labels (biased toward LLM preferences)
```

### Phase 3 - Human calibration (Week 2-3)

Add a small human annotation set to calibrate LLM labels:

```
Step 1: Human annotators label 200 (query, document) pairs
Step 2: Compare to LLM labels on the same 200 pairs
Step 3: Calibrate LLM scores using human labels (regression or rescaling)
Step 4: Apply calibrated LLM judge to the full evaluation set
Report: Calibrated NDCG@10 on large query set with human-calibrated LLM labels
```

### Phase 4 - Full annotation (optional, Month 1+)

If the domain is important enough and budget allows, move to standard human
annotation:

```
Full TREC-style annotation: 500-1000 queries, 2 annotators each
Used for: authoritative benchmarks, research publication, long-term monitoring
```

The key insight: phases 1-3 provide actionable evaluation within days to weeks
and at a fraction of the cost of phase 4. For most production teams, phases 1-3
are sufficient for development decisions and phase 4 is reserved for major
architecture choices or publication.

## Validity Threats and How to Manage Them

Every evaluation-without-labels approach has validity threats - ways the
evaluation may not accurately reflect true retrieval quality. Understanding
these threats is essential for interpreting results correctly.

### Threat 1 - Construct validity

Does the LLM judge or synthetic metric actually measure the relevance concept
we care about?

**Evidence it is a threat:** LLM judges rate fluent but inaccurate documents
highly. Users actually care about accuracy, not fluency.

**Mitigation:** Periodically compare LLM labels to human labels on a sample.
If agreement is low, revise the judging prompt or add factual accuracy criteria.

### Threat 2 - Coverage validity

Does the synthetic or LLM-judged evaluation set cover the full range of real
queries users submit?

**Evidence it is a threat:** Synthetic query generation tends to produce queries
that documents directly answer, missing queries where the answer is missing from
the corpus (navigational queries, queries requiring synthesis across documents).

**Mitigation:** Compare synthetic query distribution to real query distribution
using clustering or topic modeling. Supplement with real queries when available.

### Threat 3 - Distribution shift

If the evaluation set was generated from the current corpus, does it reflect
future queries and documents?

**Evidence it is a threat:** System performs well on synthetic evaluation but
poorly on new documents added to the corpus after evaluation set creation.

**Mitigation:** Refresh evaluation sets periodically. Evaluate on time-held-out
document subsets. Include queries about recent or newly added content.

### Threat 4 - Circularity

Does the evaluation approach favor systems that are similar to the LLM judge?

**Evidence it is a threat:** A system that uses GPT-4 for reranking may score
higher under GPT-4 judging simply because the judge prefers its own output style.

**Mitigation:** Use a different LLM for judging than for any component of the
retrieval system. Or validate with human labels that the correlation between
LLM-judged and human-judged performance holds for all systems in the comparison.

## My Summary

Evaluation without labels addresses the annotation bottleneck by estimating
retrieval quality without human relevance judgments. The three main approaches
are: LLM-as-judge (prompt a capable LLM to rate relevance, achieving human-level
inter-annotator agreement at dramatically lower cost), synthetic query generation
(use T5 or LLM query generators to create (query, document) pairs directly from
corpus content), and statistical corpus-based signals (embedding similarity,
click model inference) that approximate relevance without any annotation. LLM
judging is the most powerful: GPT-4 and Claude achieve Spearman ρ ≈ 0.85-0.95
correlation with human-labeled system rankings, making them reliable for comparing
systems even if absolute NDCG values differ from human-labeled benchmarks.
Systematic LLM biases - length preference, sycophancy, position effects in
pairwise comparisons - must be understood and mitigated through prompt design,
calibration on small human-labeled sets (ARES approach), and multi-model judging.
Evaluation frameworks like RAGAS extend label-free evaluation to full RAG
pipelines by measuring faithfulness, context recall, and answer relevancy using
LLM-inferred signals. The practical protocol for new domains is staged: synthetic
evaluation in days (phase 1), LLM-judged real queries in a week (phase 2),
human calibration of LLM labels in two weeks (phase 3), and full human annotation
only if the domain warrants the investment (phase 4).
