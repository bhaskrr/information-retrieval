# Language Models for IR

Language models for IR are a family of probabilistic retrieval models that estimate
the relevance of a document to a query by treating retrieval as a language modeling
problem. Instead of computing a relevance score directly, these models ask: what is
the probability that this document generated this query? Or equivalently: how likely
is it that a user who wanted this document would use these query terms? Documents are
ranked by the probability that their underlying language model produced the observed
query. This framework, known as the Query Likelihood Model, was the dominant
probabilistic retrieval paradigm from the late 1990s through the early 2010s and
forms the theoretical bridge between BM25 and neural retrieval.

## Intuition

BM25 is a scoring function built from engineering intuition - term frequency,
inverse document frequency, length normalization - combined by a formula that
works well in practice but has no clean probabilistic interpretation.

Language models for IR provide that clean interpretation. The core idea:

> Each document defines a probability distribution over words - its language model.
> A relevant document is one whose language model assigns high probability to the query terms.

Think of it this way. A document about information retrieval has a high probability of generating the word "retrieval". A document about cooking has a low probability of generating it. If a user queries "retrieval", the IR language model correctly ranks the IR document higher because its language model better explains the observed query.

This probabilistic framing has two major advantages over BM25. First, it connects
retrieval to a rigorous mathematical framework - probability theory - with clear
assumptions that can be examined and improved. Second, it naturally handles term
frequency through probability estimation without requiring the ad-hoc saturation
function that BM25 introduces.

## The Query Likelihood Model (QLM)

### Core formulation

The Query Likelihood Model ranks documents by the probability of generating the
query given the document's language model:

```
score(q, d) = P(q | θ_d)
```

Where θ_d is the language model estimated from document d.

For a query q = (q₁, q₂, ..., qₙ) with terms assumed independent:

```
P(q | θ_d) = Π P(qᵢ | θ_d)
              i=1 to n
```

Taking the log for numerical stability:

```
log P(q | θ_d) = Σ log P(qᵢ | θ_d)
                 i=1 to n
```

### Estimating the document language model

The maximum likelihood estimate (MLE) of the probability of term t in document d:

```
P_MLE(t | d) = tf(t, d) / |d|
```

Where tf(t, d) is the count of term t in document d and |d| is document length.

### The zero probability problem

MLE has a fatal flaw for retrieval: if a query term does not appear in the document,
P_MLE(t | d) = 0, making the entire query probability 0 regardless of how relevant
the document is on all other terms.

```
Document: "information retrieval systems ranking"
Query:    "information retrieval evaluation"

P_MLE("evaluation" | d) = 0
→ P(q | d) = P("information") × P("retrieval") × P("evaluation")
           = 0.25 × 0.25 × 0
           = 0   ← document is ranked last despite being highly relevant
```

This is the motivation for smoothing - the central contribution of language
model approaches to IR.

## Smoothing - The Core Innovation

Smoothing addresses the zero probability problem by interpolating the document
language model with a background collection language model:

```
P_smooth(t | d) = (1 - λ) × P_MLE(t | d) + λ × P(t | C)
```

Where:

- λ ∈ [0,1] is the smoothing weight
- P(t | C) is the probability of term t in the entire corpus C (collection model)
- P_MLE(t | d) is the maximum likelihood estimate from the document alone

The collection model P(t | C) provides a non-zero floor for every term - even
terms not in the document get probability proportional to their corpus frequency.

### Why smoothing connects to IDF

The smoothed log probability decomposes into a term that resembles TF-IDF weighting:

```
log P_smooth(q | d) = Σ log ((1-λ) × tf(t,d)/|d| + λ × P(t|C))
                                                        / λ × P(t|C)
                    + n × log λ × Σ log P(t|C)

The first term: log (1 + (1-λ)/λ × tf(t,d) / (|d| × P(t|C)))
```

The factor 1/P(t|C) is equivalent to IDF - terms rare in the corpus (low P(t|C))
get boosted more. The smoothing parameter λ plays the role of BM25's k₁ parameter
in controlling TF saturation. Language models thus provide a theoretical justification
for the heuristics already embedded in BM25.

## Three Major Smoothing Methods

### 1. Jelinek-Mercer (JM) Smoothing

Linear interpolation between document model and collection model with a fixed λ:

```
P_JM(t | d) = (1 - λ) × tf(t,d)/|d| + λ × P(t | C)
```

Parameters:

- λ ∈ [0, 1] - smoothing weight (typically 0.1-0.5)
- Single global parameter - same λ for all documents and queries

Behavior:

- λ = 0 → pure MLE, zero probability for unseen terms (bad)
- λ = 1 → pure collection model, ignores document content (bad)
- λ = 0.1 → 90% document model, 10% collection model

Best for: long queries where term coverage is high and precision matters.
Empirically: λ ≈ 0.1-0.2 for title queries, λ ≈ 0.5-0.7 for verbose queries.

### 2. Dirichlet Smoothing (Bayesian Smoothing)

Uses a Dirichlet prior over term distributions. The smoothing amount adapts
to document length - shorter documents get more smoothing:

```
P_Dir(t | d) = tf(t,d) + μ × P(t | C)
               ─────────────────────────
               |d| + μ
```

Parameters:

- μ > 0 - Dirichlet prior strength (typically 1000-2500)

Key insight: the effective smoothing weight is:

```
λ_effective = μ / (|d| + μ)
```

For short documents (|d| << μ): λ ≈ 1 → heavily smoothed toward collection
For long documents (|d| >> μ): λ ≈ 0 → mostly document model

This length-adaptive smoothing is Dirichlet's primary advantage over JM -
short documents automatically get more regularization because they have less
evidence for their own language model. This is equivalent to BM25's length
normalization but derived from first principles.

Best for: varying-length document collections, ad-hoc retrieval.
Empirically: μ ≈ 2000 works well for TREC ad-hoc retrieval tasks.

### 3. Absolute Discounting

Subtracts a fixed δ from each observed term count and redistributes the
probability mass to unseen terms:

```
P_AD(t | d) = max(tf(t,d) - δ, 0)  +  δ × |V_d| × P(t | C)
              ─────────────────────────────────────────────────
                              |d|
```

Where:

- δ ∈ [0, 1] - discount parameter (typically 0.5-0.7)
- |V_d| - number of unique terms in document d

Interpretation: discount each observed term by δ, collect the total discounted
probability (δ × |V_d| / |d|), redistribute it to unseen terms proportional
to the collection model.

Best for: documents with many rare terms, when the vocabulary diversity
of the document is a useful signal.

## Smoothing Methods Compared

```
Property                JM Smoothing    Dirichlet       Abs. Discounting
──────────────────────────────────────────────────────────────────────────
Length adaptation       No              Yes             Partial
Parameters              λ (one)         μ (one)         δ (one)
Typical value           λ=0.1-0.5       μ=1000-2500     δ=0.5-0.7
Best for                Long queries    Short docs       Varied vocab
Connects to BM25        λ ≈ b param     Yes directly     Partial
Empirical performance   Good verbose q  Best general     Competitive
```

## Document Expansion and Relevance Models

### Relevance Model (RM3)

An extension of the basic query likelihood model that uses pseudo-relevance
feedback to expand the query:

```
Step 1: Retrieve initial top-k documents using basic QLM

Step 2: Estimate a relevance model from top-k documents:
        P(t | R) = Σ_d P(d | q) × P(t | d)

        Where P(d | q) ∝ P(q | d) (query likelihood of each retrieved doc)

Step 3: Interpolate original query with relevance model:
        P_RM3(t | q) = (1-α) × P(t | q) + α × P(t | R)

Step 4: Re-retrieve using expanded RM3 query
```

RM3 is one of the most effective pseudo-relevance feedback methods. It often
improves NDCG@10 by 3-8 points over the basic QLM on standard TREC benchmarks.

The intuition: the top retrieved documents collectively define what "relevance"
means for this query. Terms that appear prominently in those documents are likely
relevant - add them to the query.

### RM3 parameters:

- k: number of feedback documents (typically 10-20)
- m: number of expansion terms (typically 10-50)
- α: interpolation weight (typically 0.5-0.8)

## From Language Models to Neural IR

The language model framework directly anticipates neural IR in a precise way.
In the QLM:

```
P(q | θ_d) - probability that document's language model generated the query
```

In neural dense retrieval:

```
score(q, d) = encoder_q(q) · encoder_d(d) - learned similarity
```

Both frameworks answer the same question: given a document, how well does it
match the query? The QLM answers it probabilistically using term statistics;
neural retrieval answers it geometrically using learned representations.

The smoothing parameter λ in QLM corresponds roughly to the temperature in
neural contrastive training. The collection model P(t|C) in QLM corresponds
to the negative sampling distribution in neural training. The relevance model
RM3 corresponds to iterative retrieval and hard negative mining in neural systems.

Understanding language models for IR makes every neural IR technique feel
less like magic and more like a learned version of principled probabilistic ideas.

## Language Models for IR vs BM25 - When to Use Which

| Scenario                          | Recommendation              |
| --------------------------------- | --------------------------- |
| Need probabilistic interpretation | Language model (QLM)        |
| Varying document lengths          | Dirichlet smoothing         |
| Short queries (1-2 terms)         | JM smoothing (λ=0.1)        |
| Verbose queries (5+ terms)        | JM smoothing (λ=0.5)        |
| Want query expansion              | RM3 on top of QLM           |
| Production system, no retraining  | BM25 (simpler, competitive) |
| Tight engineering constraints     | BM25 (widely implemented)   |
| Research requiring theory         | QLM (principled framework)  |

In practice, well-tuned BM25 and well-tuned Dirichlet QLM perform similarly
on most standard TREC benchmarks. The choice often comes down to implementation
convenience - BM25 is available in every search engine, while QLM requires
custom implementation or specialized libraries.

## The Legacy of Language Models in Neural IR

Language models for IR have influenced neural IR in three concrete ways:

**Smoothing → temperature in contrastive learning**
The JM λ and Dirichlet μ parameters control how much to rely on the specific
document versus the general corpus. Temperature τ in neural contrastive training
controls the same tradeoff - how peaked versus diffuse the similarity distribution
should be.

**Collection model → negative sampling distribution**
P(t|C) in QLM is the background distribution that provides probability mass for
unseen terms. In neural contrastive learning, the sampling distribution for
negatives serves the same role - it determines the "background" against which
relevant documents must stand out.

**RM3 relevance feedback → iterative retrieval and hard negative mining**
RM3 uses retrieved documents to update the query representation. Neural systems
do the same via iterative retrieval (retrieve → update query → re-retrieve) and
hard negative mining (use retrieved documents as challenging negatives for training).

The terminology changed, the notation changed, the models changed - but the
core statistical ideas from language model IR are alive in every modern neural
retrieval system.

Language models for IR occupy the theoretical high ground between BM25 and
neural IR. They provide the mathematical justification for the heuristics in
BM25 and the conceptual vocabulary that neural IR inherits. Understanding them
makes every model on both sides - earlier BM25, later neural systems - feel
like part of a coherent intellectual progression rather than a collection of
disconnected techniques.

## My Summary

Language models for IR rank documents by the probability that their underlying
word distribution generated the query - P(q | θ_d). Maximum likelihood estimation
creates a zero-probability problem for unseen query terms, which smoothing addresses by interpolating the document model with a collection-wide background model. Jelinek-Mercer uses a fixed interpolation weight $\lambda$, Dirichlet uses a prior strength $\mu$ that automatically adapts smoothing to document length (shorter documents get more smoothing), and Absolute Discounting subtracts a fixed $\delta$ from observed counts.
Dirichlet smoothing is the most theoretically grounded and empirically competitive.
The Relevance Model RM3 extends QLM with pseudo-relevance feedback, estimating
which terms are characteristic of relevant documents and expanding the query
accordingly. Language models for IR directly anticipate neural retrieval -
the smoothing parameter corresponds to contrastive training temperature, the
collection model corresponds to negative sampling distributions, and RM3
corresponds to iterative retrieval. Understanding this lineage makes neural IR
feel like a learned generalization of principled probabilistic ideas.
