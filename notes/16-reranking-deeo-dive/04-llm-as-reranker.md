# LLM as Reranker

LLM-as-reranker is the use of a large language model - GPT-4, Claude, Gemini,
Llama, or similar - as the second-stage relevance scoring component in a two-stage
retrieval pipeline. Instead of a specialized cross-encoder trained on MS MARCO
relevance data, the reranking decision is delegated to a general-purpose language
model through prompting. The LLM reads the query and each candidate document, then
uses its broad world knowledge, language understanding, and reasoning capabilities
to produce relevance scores, pairwise comparisons, or direct rankings. LLM-based
reranking consistently achieves state-of-the-art NDCG@10 on BEIR benchmarks,
outperforming smaller cross-encoders, particularly on specialized domains where
the LLM's pretraining knowledge fills gaps that labeled retrieval data cannot.
The tradeoff is significant latency and cost - making LLM reranking most practical
for offline, high-stakes, or low-volume search applications.

## Intuition

A MS MARCO-trained cross-encoder has learned to score relevance by pattern-matching
against millions of web search query-document pairs. It knows that "symptoms of
diabetes" documents should rank highly for "diabetes symptoms" queries because it
has seen thousands of similar (query, relevant document) pairs during training.

But what about a rare query like "quantum decoherence in open systems at finite
temperature"? The cross-encoder has seen little training signal for this specific
domain intersection. It falls back to surface-level pattern matching - does the
document contain these exact terms? - which may rank a document that uses all the
right words but actually discusses something different.

An LLM has pretraining knowledge about quantum mechanics, decoherence, open quantum
systems, and thermodynamics. It can actually read and understand a candidate document
about this topic - determine whether it answers the specific question being asked,
whether the mathematical treatment is at the right level, whether the finite
temperature constraint is actually addressed. It applies genuine comprehension rather
than pattern matching.

This is the fundamental advantage of LLM-as-reranker: it brings world knowledge
and reading comprehension to relevance judgment, not just statistical patterns from
retrieval training data. For specialized, technical, or knowledge-intensive queries,
this advantage can be dramatic.

## Prompting Strategies

How you prompt the LLM for reranking fundamentally determines quality, latency,
and cost. Four main strategies exist, each with different tradeoffs.

### Strategy 1 - Pointwise scoring

Ask the LLM to score each document independently on a relevance scale:

```
Prompt:
  You are a relevance judge for a search engine. Given a query and a document,
  rate the relevance of the document to the query on a scale from 0 to 10,
  where 0 = completely irrelevant and 10 = perfectly relevant.

  Query: {query}

  Document: {document}

  Relevance score (integer 0-10):
```

The LLM outputs a score for each document. Documents are ranked by score.

**Advantages:** Parallelizable across documents, simple aggregation, interpretable.
**Disadvantages:** Poor calibration across queries (same as cross-encoder pointwise
problem), score distribution varies unpredictably across different LLMs and queries.

### Strategy 2 - Pairwise comparison

Ask the LLM to compare two documents and choose which is more relevant:

```
Prompt:
  Given the following query and two documents, determine which document is more
  relevant to the query. Output only "A" or "B".

  Query: {query}

  Document A: {doc_A}

  Document B: {doc_B}

  Which document is more relevant to the query (A or B)?
```

Run pairwise comparisons for all pairs (or a subset), then aggregate into a ranking
using a sorting algorithm or tournament-style competition.

**Advantages:** Binary decision is more reliable than absolute scoring, pairwise
comparison is a natural task for LLMs.
**Disadvantages:** O(k²) comparisons for k candidates, expensive. Sorting-based
aggregation reduces to O(k log k) if comparison is used as a comparator in
merge sort or quicksort.

**Practical implementation - Tournament ranking:**

```
Round 1: compare [D₁ vs D₂], [D₃ vs D₄], [D₅ vs D₆], ...
Winners advance → Round 2: compare winners of Round 1
Continue until one document remains at top
Repeat for top-k positions
```

Total comparisons: O(k log k) with merge sort using LLM as the comparator.

### Strategy 3 - Listwise ranking (generate permutation)

Ask the LLM to directly produce a ranked ordering of all candidates:

```
Prompt:
  I will provide you with {k} documents and a query. Please rank the documents
  from most to least relevant to the query.

  Query: {query}

  Documents:
  [1] {doc₁}
  [2] {doc₂}
  ...
  [k] {docₖ}

  Rank the documents from most to least relevant. Output only the document
  numbers in ranked order, separated by commas, like: 3, 1, 5, 2, 4
```

The LLM outputs a permutation of document identifiers.

**Advantages:** Single forward pass for all documents, explicitly listwise.
**Disadvantages:** Position bias (documents earlier in list tend to be ranked
higher), context window limits k, generation of long sequences may have errors.

### Strategy 4 - Yes/No relevance judgment

Ask binary yes/no relevance for each document - the simplest LLM-compatible format:

```
Prompt:
  Is the following document relevant to the query? Answer only "Yes" or "No".

  Query: {query}
  Document: {document}

  Is this document relevant?
```

Rank by the log-probability of "Yes" token minus log-probability of "No" token:

```
score(doc) = logprob("Yes") - logprob("No")
```

This approach (similar to monoT5's training objective) is fast, reliable, and
produces scores that can be meaningfully compared across documents when using the
log-probability difference rather than raw text output.

## RankGPT

RankGPT (Sun et al., 2023) is the most studied LLM-as-reranker system. It uses
GPT-4 or GPT-3.5 with a sliding window listwise prompting strategy:

### RankGPT prompt design

```
System: You are RankGPT, an intelligent assistant that can rank passages based
on their relevance to the query.

User: I will provide you with {num} passages, each indicated by number identifier [].
Rank the passages based on their relevance to query: {query}

[1] {passage_1}
[2] {passage_2}
...
[{num}] {passage_num}

Search Query: {query}

Rank the {num} passages above based on their relevance to the search query.
The passages should be listed in descending order using identifiers. The most
relevant passages should be listed first. The output format should be [] > [] > [].
Only respond with the ranking results, do not say any word or explain.
```

### Sliding window algorithm

RankGPT cannot rank 100 candidates in one context window (too many tokens). It
uses a sliding window with bubble-sort aggregation:

```
Initial list: [D₁, D₂, ..., D₁₀₀]
Window size: w = 20
Step size: s = 10

Pass 1 (back to front):
  Window 1: [D₈₁..D₁₀₀] → LLM ranks → reorder these 20
  Window 2: [D₇₁..D₉₀]  → LLM ranks → reorder these 20
  Window 3: [D₆₁..D₈₀]  → LLM ranks → reorder these 20
  ...
  Window 10: [D₁..D₂₀]  → LLM ranks → reorder these 20

After Pass 1: best documents have bubbled toward the front

Pass 2 (optional, front to back):
  Further refinement

2 passes typically sufficient for good quality
```

### RankGPT performance

```
Method                          NDCG@10 (BEIR avg)
──────────────────────────────────────────────────────
BM25 first stage (no reranking)  0.439
MiniLM cross-encoder             0.496
RoBERTa cross-encoder            0.512
RankT5-3b                        0.531
RankGPT-3.5 (20-window)          0.538
RankGPT-4 (20-window)            0.547
```

RankGPT-4 achieves state-of-the-art NDCG on BEIR, outperforming all specialized
cross-encoders by a meaningful margin - at the cost of much higher latency and
API expense.

## Open-Source LLM Rerankers

Commercial LLMs (GPT-4, Claude) produce the highest quality but are expensive
and raise data privacy concerns. Open-source alternatives have emerged:

### RankLLaMA

Fine-tunes Llama-2-7B on MS MARCO with a pointwise relevance scoring objective:

```
Input format:
  <s> [INST] Given the query '{query}' and the document '{document}',
  determine whether the document is relevant to the query.
  Please provide only 'Yes' or 'No'. [/INST]

Score: logprob("Yes") - logprob("No")
```

RankLLaMA achieves quality between MiniLM and RoBERTa cross-encoders with much
larger model size (7B vs 22M-125M parameters). The 7B parameter size makes
inference significantly more expensive than cross-encoders but cheaper than
commercial APIs.

### RankVicuna

Similar to RankLLaMA but using Vicuna (Llama fine-tuned on conversational data).
Listwise ranking with permutation generation.

### Mistral-based rerankers

7B Mistral models fine-tuned for reranking achieve competitive quality with
better inference efficiency than Llama-based models due to grouped-query
attention and sliding window attention in Mistral's architecture.

### BAAI/bge-reranker series

BGE (Beijing Academy of AI) rerankers are cross-encoders fine-tuned on large
multilingual datasets:

```
bge-reranker-base:   Chinese + English, 278M params, fast
bge-reranker-large:  Chinese + English, 560M params, quality
bge-reranker-v2-m3:  Multilingual (100+ languages), very strong
```

BGE rerankers are not LLMs (they are encoder-only cross-encoders) but are
often grouped with LLM rerankers because they match LLM quality at smaller scale.

### Cohere Rerank API

Managed reranking API providing a cross-encoder-quality reranker through a simple
REST interface:

```
POST https://api.cohere.ai/v1/rerank
{
  "model": "rerank-english-v3.0",
  "query": "quantum decoherence finite temperature",
  "documents": [doc₁, doc₂, ..., doc₁₀₀],
  "top_n": 10
}
```

Returns relevance scores for all documents. NDCG@10 quality approaches GPT-3.5
at much lower latency and cost.

## Reasoning-Enhanced Reranking

Beyond direct scoring, LLMs can be prompted to reason about relevance before
scoring - chain-of-thought reranking:

```
Prompt:
  Query: {query}
  Document: {document}

  Think step by step:
  1. What is the main topic of the query?
  2. What is the main topic of the document?
  3. How well does the document address the query?
  4. Are there any important aspects of the query that the document misses?

  Based on your analysis, rate the document's relevance (0-10):
```

Chain-of-thought reranking improves accuracy on complex queries - particularly
multi-hop questions, queries requiring background knowledge, and queries where
semantic similarity is insufficient. The reasoning trace also provides
interpretable explanations for why a document was ranked where it was.

### Self-consistency for reranking

Run the same reranking prompt multiple times with temperature > 0, aggregate
scores:

```
For each document:
  score₁ = LLM_score(query, doc, temp=0.7)
  score₂ = LLM_score(query, doc, temp=0.7)
  score₃ = LLM_score(query, doc, temp=0.7)
  final_score = mean(score₁, score₂, score₃)
```

Self-consistency reduces variance in LLM scoring, particularly for borderline
documents where a single judgment may be unreliable.

## When LLM Reranking Outperforms Cross-Encoders

LLM reranking has the largest advantage in specific scenarios:

### Scenario 1 - Domain knowledge required

Queries that require understanding domain-specific content benefit from the LLM's
pretraining knowledge:

```
Query: "differential diagnosis for Prinzmetal angina vs NSTEMI"
Cross-encoder: pattern-matches medical terms, may rank any cardiology doc highly
LLM: understands that Prinzmetal involves vasospasm, NSTEMI involves thrombosis,
     can correctly rank documents that address this specific distinction
```

### Scenario 2 - Complex multi-hop relevance

When relevance requires connecting multiple facts:

```
Query: "which countries have both universal healthcare and a constitutional ban
        on deficit spending"
Cross-encoder: surface matching on individual terms
LLM: can reason about each document's claims and check both conditions
```

### Scenario 3 - Nuanced exclusion criteria

When relevant documents must satisfy negative constraints:

```
Query: "machine learning papers that do NOT use neural networks"
Cross-encoder: may rank neural network papers highly (heavy co-occurrence)
LLM: explicitly understands the NOT constraint and applies it
```

### Scenario 4 - Zero-shot domain transfer

New domains with no training data for specialized cross-encoders:

```
Domain: legal contract analysis
Cross-encoder (MS MARCO): poor generalization to legal domain
LLM (GPT-4): broad legal knowledge from pretraining, strong zero-shot
```

## When Cross-Encoders Outperform LLM Rerankers

### Scenario 1 - Latency-sensitive applications

Interactive search requiring sub-100ms total response:

```
LLM reranker (GPT-4, k₁=100): 5-30 seconds
Cross-encoder (MiniLM, k₁=100): 70ms on GPU
```

### Scenario 2 - High query volume

At 1000 queries per second, LLM API costs become prohibitive:

```
GPT-4 reranking: ~$0.01-0.05 per query (100 candidates)
Cross-encoder:   ~$0.0001 per query (amortized GPU cost)
```

### Scenario 3 - Simple keyword matching domains

For retrieval tasks where BM25 already performs well (technical documentation,
code search with exact API names), cross-encoders match LLM quality at far
lower cost.

### Scenario 4 - Data privacy constraints

When documents cannot leave the organization's infrastructure, commercial LLM
APIs are not viable. Cross-encoders run locally.

## Practical Deployment Patterns

### Pattern 1 - LLM reranker for high-value queries only

Route high-stakes or low-confidence queries to LLM reranking, standard
cross-encoder for all others:

```
if query.is_high_value or cross_encoder.confidence < threshold:
    final_results = llm_reranker.rank(query, candidates)
else:
    final_results = cross_encoder.rank(query, candidates)
```

### Pattern 2 - Cascade with increasing quality

Three-stage cascade with escalating cost:

```
BM25 → top-1000 → cross-encoder → top-20 → LLM reranker → top-5
```

The LLM sees only 20 documents (much faster than 100), applied after the
cross-encoder has already done most of the precision work.

### Pattern 3 - Offline batch reranking

For applications where results are precomputed (daily news digest, research
recommendation, document curation):

```
Offline:   LLM reranker runs nightly on new documents
Online:    Serve precomputed rankings (no LLM latency)
```

### Pattern 4 - LLM reranking with caching

Cache LLM reranking scores for repeated queries:

```
cache_key = hash(query + sorted(doc_ids))
if cache_key in cache:
    return cache[cache_key]
else:
    scores = llm_reranker.score(query, docs)
    cache[cache_key] = scores
    return scores
```

Particularly effective for enterprise search where the same queries recur
frequently.

## Prompt Engineering for Reranking Quality

Small changes in prompting have large effects on LLM reranking quality:

### Relevance criteria specification

```
Vague (poor quality):
  "Rate how relevant this document is to the query."

Specific (better quality):
  "Rate how well this document would answer the user's question, considering:
   1. Does it directly address the main question?
   2. Is the information accurate and up-to-date?
   3. Is the depth of explanation appropriate for the query?
   4. Does it contain actionable or specific information?"
```

### Role specification

Telling the LLM what role to play improves calibration:

```
Generic: "You are a helpful assistant."

Specific: "You are an expert information retrieval judge with deep knowledge
           of {domain}. Your task is to evaluate the relevance of documents
           to queries with the precision of a domain expert."
```

### Output format constraints

Force parseable output to avoid generation failures:

```
Ambiguous: "What is the relevance score?"
(LLM might output: "I would rate this document a 7 out of 10, because...")

Constrained: "Output ONLY a single integer from 0 to 10. No other text."
(LLM outputs: "7")
```

## Position Bias in LLM Reranking

LLMs exhibit strong position bias in listwise reranking - documents appearing
earlier in the prompt tend to receive higher relevance scores regardless of
actual relevance:

```
Experiment: same 10 documents, two orderings
  Order A: [D₁(irrelevant), D₂(relevant), D₃(irrelevant), ...]
  Order B: [D₂(relevant), D₁(irrelevant), D₃(irrelevant), ...]

LLM ranking from Order A: D₁ ranked 1st  (position bias)
LLM ranking from Order B: D₂ ranked 1st  (correct)
```

Mitigations:

- **Calibration prompting:** explicitly instruct the LLM not to rely on position
- **Multiple shuffles:** run reranking with different orderings, aggregate
- **Pairwise instead of listwise:** compare two documents at a time (no position issue)
- **Pointwise instead of listwise:** score each document independently (no position issue)

## Evaluation of LLM Rerankers

LLM rerankers are evaluated using the same metrics as cross-encoders:

### BEIR benchmark

NDCG@10 averaged over 18 datasets. The gold standard for comparing reranker quality.

### TREC-DL 2019/2020

High-quality graded relevance judgments from TREC. More reliable than automatic
BEIR evaluation for fine-grained quality differences.

### Human preference evaluation

For production systems: compare LLM reranker rankings with cross-encoder rankings
and have human evaluators state which ranked list they prefer. Captures qualities
that NDCG misses - diversity, explanation quality, format.

### Cost-quality tradeoff curve

Plot NDCG@10 vs cost-per-query for different models and configurations:

```
           NDCG@10
              |
         0.55 +         × RankGPT-4
              |       × RankGPT-3.5
         0.52 +     × RankT5-3b
              |   × RoBERTa cross-encoder
         0.49 + × MiniLM cross-encoder
              |
         0.44 +× BM25 only
              |
              ┼────────────────────── cost per query
              $0      $0.01      $0.05
```

The cost-quality frontier determines the optimal choice for each budget.

## The Future of LLM Reranking

Several trends are shaping the evolution of LLM-based reranking:

### Efficient LLM rerankers

7B parameter models fine-tuned specifically for reranking (RankLLaMA, Mistral
rerankers) are closing the gap with GPT-4 quality at a fraction of the cost.
By 2025, 7B models achieve within 2-3 NDCG points of GPT-4 on BEIR.

### Retrieval-augmented LLM reranking

LLMs that retrieve additional evidence to support relevance judgment:

```
Query + Document + Retrieved background knowledge → LLM → Relevance score
```

For highly specialized domains, providing the LLM with domain reference material
during reranking further improves quality.

### Learning from LLM reranking signals

Use LLM reranking output as training data for smaller specialized models:

```
LLM reranker produces rankings for 100K queries
→ Train cross-encoder on LLM-ranked training data
→ Cross-encoder approximates LLM quality at 100x lower cost
```

This distillation pipeline is already used by some production search systems.

LLM reranking represents the quality ceiling of the reranking module - it brings
the full reasoning and world knowledge capabilities of large language models to
relevance judgment. The subsequent note on reranking efficiency addresses how to
bring the cost and latency of these high-quality rerankers into production-viable
ranges.

## My Summary

LLM-as-reranker delegates the second-stage relevance scoring to a large language
model through prompting, leveraging broad world knowledge and reading comprehension
rather than statistical patterns from retrieval training data. Four prompting
strategies exist: pointwise scoring (rate each document independently), pairwise
comparison (compare two documents at a time), listwise ranking (generate a full
permutation), and yes/no judgment (binary relevance with log-probability scoring).
RankGPT using GPT-4 with a sliding window listwise strategy achieves the highest
NDCG@10 on BEIR benchmarks, outperforming all specialized cross-encoders. The
primary advantages over cross-encoders are domain knowledge (the LLM knows the
subject matter), complex reasoning (multi-hop relevance, negation constraints),
and zero-shot domain transfer (no training data needed). The primary disadvantages
are latency (seconds vs milliseconds), cost (orders of magnitude more expensive),
and data privacy (documents sent to external APIs). Practical deployment patterns
include routing only high-value queries to LLM reranking, using a three-stage
cascade where the LLM sees only 20 cross-encoder-filtered candidates rather than
100, and offline batch reranking with result caching. Position bias - where
documents appearing earlier in the prompt receive higher scores - is the central
quality challenge in listwise LLM reranking and requires mitigation through
shuffling or pairwise comparison strategies.
