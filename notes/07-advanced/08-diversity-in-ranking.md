# Diversity in Ranking

Diversity in ranking is the practice of ensuring that a ranked result list covers
multiple distinct aspects, subtopics, or interpretations of a query rather than
returning many documents that are all relevant but redundant. A diverse result list
maximizes the probability that at least one result satisfies any user's specific
information need, even when that need is ambiguous or multi-faceted. Diversity is
distinct from relevance - a list of ten highly relevant but nearly identical
documents scores perfectly on NDCG but fails users who wanted different perspectives
or who had a different interpretation of the query in mind.

## Intuition

Consider the query "jaguar". Three interpretations exist:

- The car manufacturer
- The animal
- The Apple operating system

A purely relevance-based retrieval system trained on a corpus where most "jaguar"
clicks go to the car manufacturer will return ten car-related documents. Users
asking about the animal or the OS are completely unserved. A diverse system
returns results covering all three interpretations - perhaps four car results,
three animal results, and three OS results - serving the full range of users.

Now consider the query "climate change effects". Even with an unambiguous query,
a purely relevance-based system might return ten documents that all discuss rising
temperatures because that is the most commonly covered effect. A diverse system
ensures the result list also covers sea level rise, extreme weather, biodiversity
loss, and economic impacts - different aspects of the same unambiguous topic.

Diversity matters in two distinct scenarios:

- **Ambiguous queries** - multiple interpretations exist, and it is unclear which
  the user intends
- **Multi-faceted queries** - one interpretation exists but the topic has many
  distinct subtopics that different users care about

## Why Pure Relevance Fails for Diversity

Standard NDCG and MAP treat all relevant documents as independent. Retrieving
document D1 and a near-duplicate D1' both count as separate relevant hits.
This incentivizes systems to retrieve many documents from the same cluster
(they are all highly relevant) rather than covering different subtopics.

```
Query: "machine learning applications"

Result set A (no diversity):
  D1: ML in healthcare (rel=3)
  D2: ML in healthcare - similar paper (rel=3)
  D3: ML in healthcare - another paper (rel=2)
  D4: ML in healthcare - survey (rel=2)
  D5: ML in finance (rel=2)

NDCG@5 = 0.91   ← high score

Result set B (diverse):
  D1: ML in healthcare (rel=3)
  D5: ML in finance (rel=2)
  D6: ML in autonomous vehicles (rel=2)
  D7: ML in natural language processing (rel=2)
  D8: ML in computer vision (rel=2)

NDCG@5 = 0.82   ← lower score but much more useful to users
```

Set A scores higher on NDCG but set B better serves users who came with different
aspects of the query in mind. Diversity metrics exist precisely to capture this gap.

## Diversity Metrics

### α-NDCG (Alpha-NDCG)

The most widely used diversity metric. Extends NDCG by penalizing redundancy -
a document that covers a subtopic already covered by a higher-ranked document
gets a reduced credit.

```
α-NDCG@k = Σᵢ₌₁ᵏ  G(d, c, α) / log₂(i + 1)
```

Where G(d, c, α) is the gain of document d at position i given the context c
of already-retrieved documents:

```
G(d, c, α) = Σⱼ (1 - α)^cⱼ × rⱼ(d)
```

Where:

- rⱼ(d) = relevance of document d to subtopic j (binary: 0 or 1)
- cⱼ = number of documents already retrieved that cover subtopic j
- α = redundancy penalty (typically 0.5)

The factor (1 - α)^cⱼ penalizes documents covering subtopics already covered.
When α = 0, α-NDCG reduces to standard NDCG. When α = 1, second documents on
the same subtopic get zero credit.

### ERR (Expected Reciprocal Rank)

Models user behavior as a cascade - the user scans the ranked list and stops
when satisfied. Documents higher up the list that satisfy the user reduce the
probability of the user continuing to scan:

```
ERR = Σᵣ₌₁ᵏ  (1/r) × Π_{i<r}(1 - Rᵢ) × Rᵣ
```

Where Rᵣ = probability that the user is satisfied by the document at rank r,
derived from its relevance grade.

ERR penalizes redundancy implicitly - if the user was satisfied at rank 1, rank 2
gets zero credit regardless of its relevance.

### D-measure

Explicitly measures both relevance and subtopic coverage:

```
D-measure = Σⱼ  wⱼ × P(subtopic j covered in top-k results)
```

Where wⱼ is the estimated probability that a random user's intent is subtopic j.

Requires explicit subtopic annotations - less commonly used in practice because
obtaining subtopic weights requires human judgment.

### Intent-Aware Metrics (IA-P, IA-NDCG)

Extend standard metrics by computing them per-intent and then averaging weighted
by the probability of each intent:

```
IA-NDCG@k = Σⱼ P(intent j) × NDCG@k(results for intent j)
```

TREC Web Track and NTCIR INTENT tracks use intent-aware metrics as their primary
evaluation framework for diversity.

## Diversification Algorithms

### Maximal Marginal Relevance (MMR)

Introduced by Carbonell and Goldstein (1998). The most widely implemented diversity
algorithm. Greedily builds a diverse result list by iteratively selecting the
document that maximizes a linear combination of relevance and novelty:

```
MMR(dᵢ) = λ × sim(dᵢ, q) - (1 - λ) × max_{dⱼ ∈ S} sim(dᵢ, dⱼ)
```

Where:

- sim(dᵢ, q) = relevance of document i to query
- sim(dᵢ, dⱼ) = similarity between document i and already-selected document j
- S = set of already-selected documents
- λ ∈ [0,1] = tradeoff (1 = pure relevance, 0 = pure diversity)

Algorithm:

```
1. Start with the most relevant document
2. For each remaining position:
   a. Compute MMR score for each candidate
   b. Select candidate with highest MMR score
   c. Add selected document to result list
3. Repeat until k documents selected
```

MMR is simple, fast, and parameter-free beyond λ. Its limitation is that similarity
is computed at the document level - it does not have an explicit model of subtopics.

### xQuAD (eXplicit Query Aspect Diversification)

Explicitly models query aspects (subtopics) and selects documents to cover them:

```
xQuAD(dᵢ) = (1 - λ) × P(dᵢ | q) + λ × Σⱼ P(sⱼ | q) × P(dᵢ | sⱼ) × Π_{dₖ ∈ S}(1 - P(dₖ | sⱼ))
```

Where:

- P(dᵢ | q) = relevance of document to query
- sⱼ = query subtopic j
- P(sⱼ | q) = probability that subtopic j is the user's intent
- P(dᵢ | sⱼ) = how well document i covers subtopic j
- Π (1 - P(dₖ | sⱼ)) = probability that subtopic j is not already covered by S

The product term is the key - it reduces the value of document i for subtopic j
proportionally to how well already-selected documents cover that subtopic.

xQuAD requires subtopic relevance annotations or automatic subtopic discovery
(from query logs, Wikipedia categories, or clustering).

### PM-2 (Proportionality-based Maximal Marginal Relevance)

Extends MMR to explicitly balance coverage proportional to estimated subtopic
probabilities:

```
For each subtopic j, maintain a coverage quota proportional to P(sⱼ | q).
Select documents to fill quota for most under-covered subtopics first.
```

Tends to produce more balanced coverage than MMR or xQuAD for queries with
many subtopics of similar importance.

### Subtopic-Aware Clustering

Pre-cluster candidate documents by topic, then select the top document from
each cluster:

```
1. Retrieve top-50 candidate documents
2. Cluster into k groups by semantic similarity
3. Select top-ranked document from each cluster
4. Rank selected documents by their within-cluster score
```

Simple and effective for queries with clearly separable subtopics. Less effective
for queries where subtopics overlap significantly.

## Subtopic Discovery

Many diversity algorithms require knowing what the subtopics of a query are.
Subtopics can be discovered in several ways:

### Query log mining

Extract frequent query reformulations and related queries from search logs:

```
"jaguar" → query log shows users also search:
  "jaguar car models 2024"     → subtopic: cars
  "jaguar animal facts"        → subtopic: animal
  "jaguar macOS features"      → subtopic: OS
```

Requires large query logs - not available for new systems.

### Wikipedia-based expansion

Use Wikipedia disambiguation pages and categories:

```
"jaguar" Wikipedia disambiguation:
  Jaguar (car brand) → category: automobiles
  Jaguar (animal)    → category: animals
  Jaguar (macOS)     → category: software
```

Available for any query that has a Wikipedia disambiguation page.

### Clustering top retrieved documents

Use semantic clustering to discover subtopics from retrieved documents:

```
Retrieve top-50 documents for "jaguar"
→ k-means clustering on document embeddings
→ cluster 1: car-related documents
→ cluster 2: animal-related documents
→ cluster 3: OS-related documents
```

No external data needed. Quality depends on corpus and embedding quality.

### LLM-based subtopic generation

Prompt an LLM to generate likely subtopics for a query:

```
Prompt: "What are the distinct interpretations or subtopics a user
         might have in mind when searching for '{query}'?
         List them as a JSON array."

Response: ["Jaguar car manufacturer", "Jaguar animal (big cat)",
           "Jaguar macOS operating system"]
```

High quality, no labeled data required, but adds LLM latency.

## Diversity vs Personalization

Diversity and personalization pull in opposite directions and must be balanced:

```
Personalization:  give this user more of what they like
Diversity:        give this user coverage of what they might not know they want

Pure personalization → filter bubble, redundant results
Pure diversity       → ignores known user preferences, wastes result positions

Balance:
  Use personalization to rank subtopics by user preference
  Use diversity to ensure each subtopic gets at least one representative result
```

The practical recipe:

1. Estimate subtopic probabilities adjusted by user interest
2. Allocate result positions to subtopics proportionally
3. Select best document per subtopic

This gives personalization at the subtopic level and diversity at the result level.

## Choosing a Diversification Strategy

```
Scenario                               Strategy          Notes
────────────────────────────────────────────────────────────────────────
General purpose, fast                  MMR               λ=0.5 default
Known subtopics from query logs        xQuAD             Best quality
No subtopic labels available           Clustering-based  Automatic
Short result list (k ≤ 5)             MMR               Simpler wins
Long result list (k ≥ 10)            xQuAD or PM-2     More structure needed
Ambiguous navigational queries         xQuAD             Intent coverage
Multi-faceted informational queries    MMR               Aspect coverage
Real-time, latency-sensitive           MMR               O(n²) but fast
```

## λ Parameter Sensitivity

The diversity-relevance tradeoff is the key engineering decision:

```
λ = 1.0  → pure relevance (standard ranking, no diversity)
λ = 0.7  → relevance-weighted (small diversity boost)
λ = 0.5  → balanced (typical default)
λ = 0.3  → diversity-weighted (strong diversity)
λ = 0.0  → pure diversity (maximum coverage, ignores relevance)
```

In practice, λ between 0.4 and 0.6 works well for most query types. Tune
on a held-out set of ambiguous queries with subtopic annotations.

## My Summary

Diversity in ranking ensures result lists cover multiple distinct aspects or
interpretations of a query rather than returning many redundant relevant documents.
Two scenarios motivate diversity: ambiguous queries (multiple plausible interpretations)
and multi-faceted queries (one interpretation, many relevant subtopics). Standard
NDCG fails to capture diversity because it treats all relevant documents as
independent - α-NDCG and ERR extend it with redundancy penalties. Three main
diversification algorithms exist: MMR greedily selects documents maximizing
relevance minus similarity to already-selected documents; xQuAD explicitly models
query subtopics and selects documents to cover them proportionally; clustering-based
approaches discover subtopics automatically from retrieved documents. MMR is the
default for general use - simple, fast, and parameter-free beyond λ. xQuAD provides
better coverage for queries with known subtopic structure but requires subtopic
annotations or automatic subtopic discovery. Diversity and personalization are
complementary: use personalization to weight subtopics by user preference, and
diversity to ensure coverage of each weighted subtopic.
