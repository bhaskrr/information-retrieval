# Query Segmentation

Query segmentation is the process of identifying meaningful multi-word units within
a tokenized query and grouping them into semantically coherent segments. Where
tokenization splits a query into individual words and normalization standardizes
those words, segmentation identifies which consecutive words form a single semantic
unit that should be treated together rather than independently. "New York Times",
"machine learning", and "support vector machine" are each multi-word expressions
that lose meaning when treated as bags of independent terms.

## Intuition

Consider the query "New York Times best seller list". Tokenization produces six
tokens. But the query actually contains three semantic units:

```
[New York Times] [best seller] [list]
```

A retrieval system that treats these as six independent tokens will retrieve
documents containing "new", "york", "times", "best", "seller", and "list" in
any combination and any positions. A document about "the best time to sell new
items in York" matches perfectly by term overlap but is completely irrelevant.

Segmentation fixes this by identifying that "New York Times" is a named entity,
"best seller" is a compound noun, and "list" is a standalone term. The retrieval
system can then search for documents containing these units as coherent phrases,
not random co-occurrences of the constituent words.

Segmentation is particularly important for:

- Named entities: "Los Angeles", "New York Times", "United States of America"
- Technical compound nouns: "support vector machine", "random forest", "gradient descent"
- Domain-specific phrases: "machine learning", "information retrieval", "deep learning"
- Product names: "Google Search", "GPT-4", "iPhone 15"

## Why Independent Term Matching Fails

Standard BM25 and TF-IDF treat each query term independently:

```
Query:    "support vector machine"
Document: "this machine supports the vector database"

Term matching:
  "support" found ✓
  "vector"  found ✓
  "machine" found ✓
  BM25 score: high  ← but document is completely irrelevant
```

The document scores highly because it contains all three terms, but none of them
appear as the compound concept "support vector machine". Segmentation prevents
this by searching for the phrase as a unit:

```
Segmented query: ["support vector machine"]
Document must contain the phrase → no match → correct ranking
```

## Segmentation Approaches

### Approach 1 - Dictionary-based segmentation

Maintain a dictionary of known multi-word expressions. Greedily match the longest
expression in the dictionary:

```
Dictionary: {"New York": True, "New York Times": True, "machine learning": True, ...}

Query: "New York Times machine learning"

Greedy longest match:
  "New York Times" found → segment
  "machine learning" found → segment
  Result: ["New York Times", "machine learning"]
```

Fast and high-precision for known phrases. Limited to what is in the dictionary -
fails for novel compound terms not in the vocabulary.

Dictionary sources:

- Wikipedia titles and redirects (millions of named entities)
- Domain-specific glossaries (ML terms, medical terms, legal terms)
- Product catalogs (brand names, product lines)
- Query logs (frequently co-occurring terms)

### Approach 2 - Statistical segmentation

Use corpus statistics to score candidate segmentations. A pair of words is a
segment if they co-occur more often than expected by chance:

**Pointwise Mutual Information (PMI):**

```
PMI(w₁, w₂) = log( P(w₁, w₂) / (P(w₁) × P(w₂)) )
```

High PMI → words co-occur much more than expected → likely a meaningful phrase:

```
PMI("machine", "learning") ≈ 4.2  → segment together
PMI("machine", "the")      ≈ -0.3 → do not segment
```

**Normalized PMI (NPMI):**

```
NPMI(w₁, w₂) = PMI(w₁, w₂) / -log(P(w₁, w₂))
```

Normalizes to [-1, 1], making thresholds comparable across different term frequencies.

**Segmentation as a sequence labeling problem:**
Label each token as B (begin segment), I (inside segment), or O (standalone):

```
Query: "New York Times machine learning tutorial"
Labels: B   I    I     B       I        O

Segments: ["New York Times", "machine learning", "tutorial"]
```

Train a sequence labeler (CRF, BiLSTM, BERT) on annotated query-segmentation pairs.

### Approach 3 - Language model scoring

Score candidate segmentations by how natural they sound as phrases:

```
Candidate 1: ["New York", "Times", "machine", "learning"]
Candidate 2: ["New York Times", "machine learning"]

Score via n-gram language model:
  P("New York Times") >> P("New York") × P("Times")
  P("machine learning") >> P("machine") × P("learning")

Candidate 2 scores higher → preferred segmentation
```

Modern approach: use a pre-trained language model (BERT, GPT) to score phrase
naturalness. Phrases that appear frequently in pretraining data get higher scores.

### Approach 4 - Neural segmentation

Fine-tune BERT or similar on a span-detection task to identify segment boundaries:

```
Input:  [CLS] New York Times machine learning tutorial [SEP]
Output: boundary probabilities at each token gap:
         New|York: 0.02  (no boundary - same segment)
         York|Times: 0.05 (no boundary)
         Times|machine: 0.91 (boundary - new segment)
         machine|learning: 0.03 (no boundary)
         learning|tutorial: 0.87 (boundary)
```

Trained on annotated query-segmentation datasets. Highest quality but requires
labeled data and adds latency.

## Segmentation in the IR Pipeline

Segmentation changes how queries are processed in different retrieval systems:

### BM25 with phrase queries

Segmented terms become phrase constraints:

```
Original:   "information retrieval tutorial"
Segmented:  ["information retrieval", "tutorial"]
BM25 query: require phrase "information retrieval" + term "tutorial"
```

Implementation: use Elasticsearch or Lucene phrase queries for identified segments
and term queries for standalone tokens.

### Dense retrieval

Segmentation improves the query text before neural encoding:

```
Without segmentation: "New York Times machine learning"
  → BERT tokenizes as 6 independent tokens
  → embedding averages across all tokens equally

With segmentation: "New York Times, machine learning"
  → Comma or separator signals phrase boundaries to encoder
  → Alternatively: encode each segment separately, combine
```

Simpler approach: the neural encoder often handles multi-word expressions
implicitly through attention - "New York Times" tokens attend strongly to each
other. Explicit segmentation is less critical for dense retrieval than for sparse.

### Query expansion with segmentation

Expand segments as units rather than individual words:

```
Without segmentation: expand "machine" → "machine, device, apparatus"
                       expand "learning" → "learning, education, study"
                       → cross-product explosion, wrong concepts

With segmentation:    "machine learning" → expand as unit
                       → "deep learning, artificial intelligence, neural networks"
                       → correct and relevant expansions
```

Segment-aware expansion dramatically improves expansion quality for technical
queries.

## Segmentation Granularity

Segmentation decisions involve tradeoffs across three levels:

```
Level 1 - No segmentation:
  "support vector machine learning tutorial"
  → ["support", "vector", "machine", "learning", "tutorial"]
  Recall: high (matches any combination)
  Precision: low (matches wrong combinations)

Level 2 - Phrase segmentation:
  → ["support vector machine", "learning", "tutorial"]
  Recall: medium (must match "support vector machine" as phrase)
  Precision: medium (reduces wrong matches)

Level 3 - Maximum segmentation:
  → ["support vector machine", "learning tutorial"]
  Recall: low (must match both phrases exactly)
  Precision: high (only exact phrase matches)
```

The right granularity depends on:

- Query length: longer queries benefit from more segmentation
- Domain specificity: technical domains have more compound terms
- Retrieval task: navigational queries need tight segmentation, exploratory queries need loose

## Evaluation of Segmentation Quality

### Human annotated datasets

**MSST (Microsoft Segmentation Test set):**
500 queries with human-annotated segmentations. Standard benchmark for
query segmentation research.

**AOL query log subsets:**
Sampled queries from the AOL query log annotated with segmentations.
Larger but noisier than MSST.

### Metrics

**Segment-level F1:**
Precision and recall over predicted segments vs gold segments.

```
Gold:      ["New York Times", "machine learning"]
Predicted: ["New York", "Times machine", "learning"]

Precision = matching segments / predicted segments = 0/3 = 0.00
Recall    = matching segments / gold segments     = 0/2 = 0.00
F1 = 0.00  ← both segmentations share tokens but no full match
```

**Boundary F1:**
Precision and recall over predicted boundaries vs gold boundaries.

```
Gold boundaries:      [after "Times", after "learning"]
Predicted boundaries: [after "York", after "machine", after "learning"]

Boundary matches: "after learning"
Precision = 1/3 = 0.33
Recall    = 1/2 = 0.50
F1 = 0.40  ← partial credit for correct boundaries
```

Boundary F1 is more forgiving and better correlates with downstream retrieval quality.

### Extrinsic evaluation

The most meaningful metric: does better segmentation improve retrieval NDCG?

```
System A: no segmentation → NDCG@10 = 0.412
System B: dictionary segmentation → NDCG@10 = 0.438
System C: neural segmentation → NDCG@10 = 0.451

Segmentation improvement: +3.9 points NDCG@10 → significant
```

## Segmentation Strategy Selection

| Query type                    | Segmentation approach                     |
| ----------------------------- | ----------------------------------------- |
| Named entity queries          | Dictionary (high precision)               |
| Technical domain queries      | PMI + dictionary hybrid                   |
| General web queries           | Minimal segmentation (BM25 handles it)    |
| Code search queries           | Special rules (::, ->, camelCase)         |
| Long natural language queries | Neural segmentation (BERT-based)          |
| Short queries (≤ 3 tokens)    | No segmentation (every word matters)      |
| Product search                | Domain dictionary (brand + product names) |

## Segmentation vs Phrase Queries

A common question: why not just use quoted phrase queries instead of segmentation?

```
Quoted phrase query:
  User types: "support vector machine"
  System interprets: must find exact phrase
  Downside: requires user to know phrase boundaries and use quotes

Automatic segmentation:
  User types: support vector machine tutorial
  System automatically identifies: ["support vector machine", "tutorial"]
  Upside: transparent to user, no special syntax required
```

Segmentation provides the benefits of phrase queries without requiring users to
manually quote everything. It works automatically on natural language queries.

## My Summary

Query segmentation identifies meaningful multi-word units within a tokenized query
and groups them into semantically coherent segments. Without segmentation, "support
vector machine" becomes three independent terms that match documents using any
combination of those words - many of which are irrelevant. With segmentation, the
phrase is treated as an atomic unit that must appear together. Three approaches
exist: dictionary-based greedy longest-match (fast, high-precision for known phrases),
PMI-based statistical segmentation (discovers phrases from corpus co-occurrence
statistics), and neural span detection (highest quality, requires labeled data).
The hybrid approach - using dictionary for known phrases and PMI as fallback for
unknown compounds - is the practical default for most systems. Segmentation improves
retrieval most for technical queries with compound terms, named entity queries, and
product search. For dense retrieval, neural encoders handle many multi-word expressions
implicitly through attention, making explicit segmentation less critical but still
beneficial for query preprocessing.
