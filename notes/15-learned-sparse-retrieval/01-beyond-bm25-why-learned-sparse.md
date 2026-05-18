# Beyond BM25 - Why Learned Sparse Retrieval

Learned sparse retrieval is a family of neural retrieval methods that produce
sparse vector representations - vectors where most values are zero and only a
small fraction are non-zero - using a learned neural model rather than classical
term frequency statistics. Unlike BM25, which counts exact term occurrences with
hand-crafted weighting formulas, learned sparse models use a transformer to decide
which terms (including terms not in the original document) should receive non-zero
weight and how much weight each should carry. Unlike dense retrieval, which produces
fully dense 768-dim vectors requiring ANN search, learned sparse retrieval produces
vectors that can be indexed in standard inverted indexes - the same infrastructure
BM25 runs on. The result is a family of models that approaches dense retrieval
quality while maintaining the exact-match efficiency and infrastructure compatibility
of BM25.

## Intuition

BM25 has three interconnected limitations that motivate the shift to learned sparse
representations:

**Limitation 1 - Vocabulary mismatch**
BM25 scores a document zero for a query term if that exact term does not appear.
"cardiac arrest" and "heart attack" have no BM25 overlap despite meaning the same
thing. "automobile" and "car" are identical in meaning but score zero for each
other's queries. BM25 sees strings, not meaning.

**Limitation 2 - IDF is corpus-global, not query-aware**
BM25's IDF weight for a term is fixed - computed once from corpus statistics and
applied uniformly regardless of the query context. The word "bank" has one IDF
score whether the query is about finance or rivers. BM25 cannot distinguish that
"bank" is more important in a finance context than a geography context.

**Limitation 3 - Term weight ceiling**
TF saturation via (k₁ + 1) × tf / (tf + k₁ × ...) is a fixed mathematical curve.
It cannot learn that some terms saturate quickly (function words) and others scale
linearly with frequency (rare technical terms).

Dense retrieval solves all three limitations - the neural encoder learns semantic
similarity directly, ignoring surface form. But it introduces its own limitations:

**Dense limitation 1 - Infrastructure incompatibility**
Dense retrieval requires FAISS or a vector database. BM25 runs on Elasticsearch
or any inverted index. Replacing BM25 with dense retrieval means replacing the
entire search infrastructure - a major engineering investment.

**Dense limitation 2 - Poor exact match for rare terms**
Dense models generalize well but can miss exact keyword matches for rare proper
nouns, version numbers, and technical identifiers:

```
Query:    "pytorch 2.0.1 installation error"
BM25:     finds exact "pytorch 2.0.1" matches reliably
Dense:    may find semantically similar documents but miss the specific version
```

**Dense limitation 3 - Black box representations**
Dense vectors are uninterpretable. You cannot explain why a document was retrieved
or debug retrieval failures by inspecting the representation.

Learned sparse retrieval is the middle path: neural quality with inverted index
infrastructure. The sparse vector looks like a weighted vocabulary - you can read
which terms are active and why a document matched - while using a neural model
to determine which terms carry meaning in context.

## The Core Idea

A learned sparse model maps a text to a sparse vector over the full vocabulary:

```
Standard document: "heart attack symptoms and treatment"
BM25 sparse vector: {"heart": 2.3, "attack": 1.8, "symptoms": 2.1,
                     "treatment": 1.9}
                     (only exact terms from the document)

Learned sparse vector: {"heart": 2.1, "attack": 1.7, "symptoms": 2.0,
                        "treatment": 1.8, "cardiac": 1.9, "chest": 1.3,
                        "arrest": 1.4, "pain": 1.1, "myocardial": 0.9,
                        "coronary": 0.8}
                        (expansion: related terms added by neural model)
```

Two key differences:

1. **Term expansion** - terms semantically related to the document's content are
   added with non-zero weights ("cardiac", "myocardial", "coronary") enabling
   cross-vocabulary matching
2. **Learned weights** - term weights reflect semantic importance in context,
   not raw frequency statistics

The inverted index stores these expanded, learned-weight representations. At query
time, the query terms (also expanded by the neural model) look up their posting
lists and compute dot product scores - identical to BM25 retrieval mechanics.

## BM25 vs Dense vs Learned Sparse: The Triangle

```
Model family     Quality        Infrastructure    Vocab mismatch    Interpretable
──────────────────────────────────────────────────────────────────────────────────
BM25             Medium         Inverted index    None handled      Yes
Learned sparse   Medium-high    Inverted index    Partially handled Yes
Dense retrieval  High           ANN / vector DB   Fully handled     No
Hybrid           Highest        Both              Fully handled     Partial
```

## Why Infrastructure Compatibility Matters

The practical importance of inverted index compatibility is often underestimated
by researchers but is the primary reason learned sparse retrieval matters to
production engineers:

**Scenario: company has Elasticsearch cluster serving 500M documents**

```
Switching to dense retrieval:
  - Compute embeddings for 500M documents (GPU hours, significant cost)
  - Deploy FAISS cluster or vector database (new infrastructure)
  - Re-implement filtering, faceting, and metadata queries
  - New monitoring, operations, incident response procedures
  - Migration risk: weeks of parallel running both systems

Switching to learned sparse (SPLADE):
  - Compute sparse representations for 500M documents (can use CPU)
  - Index in existing Elasticsearch cluster (inverted index unchanged)
  - All filtering, faceting, metadata queries unchanged
  - Existing monitoring and operations unchanged
  - Migration risk: days of parallel running
```

For organizations with mature BM25 infrastructure, learned sparse retrieval
delivers 80-90% of the quality improvement of dense retrieval at 20-30% of
the migration cost.

## The Vocabulary Expansion Mechanism

The central innovation of learned sparse models is vocabulary expansion -
adding terms to a document's representation that did not appear in the original
text but are semantically relevant.

### How expansion works in SPLADE

SPLADE uses BERT's Masked Language Modeling (MLM) head for expansion. The MLM
head produces a probability distribution over the full vocabulary for every input
token position. SPLADE aggregates these distributions:

```
Input: "heart attack symptoms"

At token "heart":
  MLM predicts high probability for: cardiac, coronary, myocardial, chest
  → these terms contribute to the document's sparse vector

At token "attack":
  MLM predicts high probability for: arrest, failure, event, onset
  → these terms contribute to the document's sparse vector

At token "symptoms":
  MLM predicts high probability for: signs, presentation, manifestations
  → these terms contribute to the document's sparse vector

Combined sparse vector: weighted sum of per-token MLM distributions
```

The MLM head was pretrained to understand language - it knows "cardiac" is
related to "heart" because it learned this from millions of documents. SPLADE
harnesses this knowledge for retrieval.

### Expansion vs contraction

Learned sparse models do both:

- **Expansion** - add semantically related terms not in the original text
- **Contraction** - reduce or zero out weights for common low-information terms

```
Original BM25 terms: "the cat sat on the mat"
  Expanded:  {"feline": 0.8, "rested": 0.6, "surface": 0.4, ...}
  Contracted: {"the": 0.0, "on": 0.0}   ← function words zeroed out
```

Contraction gives sparse models an advantage over BM25 even without expansion:
function words that pollute BM25 scores are automatically downweighted based on
learned importance rather than fixed stopword lists.

## When Vocabulary Expansion Helps and Hurts

Expansion is not universally beneficial. Understanding when it helps and when
it hurts guides deployment decisions:

### Expansion helps

**Synonym-heavy queries:**

```
Query:    "automobile safety features"
Document: "car crash protection systems"
BM25:     score = 0 (no overlap)
SPLADE:   score > 0 (automobile ↔ car, safety ↔ protection learned)
```

**Abbreviation/full-form mismatches:**

```
Query:    "MI treatment protocol"   (MI = myocardial infarction)
Document: "management of heart attacks in emergency settings"
SPLADE:   expansion links MI → myocardial → heart attack
```

**Technical jargon to plain language:**

```
Query:    "acetylsalicylic acid uses"
Document: "aspirin for pain relief and heart disease"
SPLADE:   expansion links acetylsalicylic → aspirin
```

### Expansion hurts

**Precise identifier queries:**

```
Query:    "CVE-2021-44228 log4shell"   (specific vulnerability ID)
Document: "log4j logging framework configuration"
SPLADE:   may expand to general security terms, missing the specific CVE ID
BM25:     exact "CVE-2021-44228" match if document contains it
```

**Version-specific queries:**

```
Query:    "python 3.11 walrus operator"
SPLADE:   may match "python 3.9 walrus operator" after expansion
BM25:     distinguishes "3.11" from "3.9" through exact matching
```

**Proper noun disambiguation:**

```
Query:    "Jordan Peterson book"   (person)
Document: "travel guide to Jordan"  (country)
SPLADE expansion:  might connect Jordan (person) → Jordan (country)
BM25:     both contain "Jordan" but at least does not expand confusingly
```

This is precisely why hybrid search (BM25 or dense + SPLADE) outperforms either
alone - SPLADE handles vocabulary mismatch, BM25 handles exact identifiers.

## The Sparsity-Quality Tradeoff

Learned sparse models have a regularization hyperparameter controlling how sparse
the representations are:

```
High sparsity (aggressive regularization):
  → Very few non-zero dimensions (~10-30 per document)
  → Faster indexing and retrieval
  → Lower quality due to information loss

Low sparsity (weak regularization):
  → Many non-zero dimensions (~100-200 per document)
  → Slower indexing and retrieval
  → Higher quality but closer to dense in computational cost
```

SPLADE uses FLOPS regularization to control sparsity:

```
L_sparsity = λ × Σ_j log(1 + Σ_i h_ij²)
```

Where h_ij is the weight for term j at position i. Higher λ → more sparsity.

Practical SPLADE configurations:

```
Configuration        Avg active terms    NDCG@10 (BEIR)    Latency
────────────────────────────────────────────────────────────────────
SPLADE-v2-max        ~200               0.451              Slow
SPLADE-v2-distil     ~100               0.438              Medium
SPLADE-efficient     ~60                0.428              Fast
CoCondenser+SPLADE   ~50                0.440              Fast
```

## Learned Sparse vs Dense: When to Use Which

```
Choose learned sparse when:
  ✓ Existing Elasticsearch/OpenSearch infrastructure
  ✓ Interpretability requirements (compliance, debugging)
  ✓ GPU not available for dense indexing at scale
  ✓ Vocabulary mismatch is moderate (not extreme)
  ✓ Need exact-match as backup for technical identifiers
  ✓ Storage constraints favor sparse representations

Choose dense retrieval when:
  ✓ Semantic understanding is critical
  ✓ Queries are conversational, not keyword-heavy
  ✓ Cross-lingual retrieval required
  ✓ GPU infrastructure already in place
  ✓ Maximum retrieval quality is the priority

Choose hybrid (sparse + dense) when:
  ✓ Maximum quality is needed and infrastructure allows
  ✓ Mixed query types (keyword + semantic)
  ✓ All production search applications where latency ≥ 100ms
```

## The Learned Sparse Family

The subsequent notes in this module cover the major learned sparse models:

```
02-splade-deep-dive.md
  SPLADE, SPLADE-v2, SPLADE-distil
  The dominant learned sparse model - MLM head + sparsity regularization

03-unicoil-and-deepct.md
  uniCOIL: per-token importance estimation
  DeepCT:  context-sensitive term weighting
  Simpler alternatives to SPLADE, different tradeoffs

04-sparse-dense-tradeoffs.md
  BEIR benchmark results
  Per-domain analysis
  Practical deployment decisions
```

## Where Learned Sparse Sits in the Broader IR Landscape

```
Year    Development
────────────────────────────────────────────────────────────────────
1994    BM25 introduced (Robertson et al., TREC-3)
2013    Word2Vec (Mikolov et al.) - first neural word representations
2019    BERT (Devlin et al.) - contextual representations
2020    Dense Retrieval - DPR (Karpukhin et al.) - bi-encoders for retrieval
2021    SPLADE (Formal et al.) - learned sparse via MLM head
2021    uniCOIL (Lin & Ma) - per-token importance estimation
2021    DeepCT (Dai & Callan) - context-sensitive term weighting
2022    SPLADE-v2, efficient variants - production-ready sparse models
2023    SPLADE++ - improved distillation, better BEIR performance
2024    Hybrid sparse+dense becomes production standard
```

## My Summary

Learned sparse retrieval bridges BM25 and dense retrieval by using a neural model
to produce sparse vectors over the vocabulary - representing documents as weighted
term lists where weights are learned rather than frequency-counted and the term list
can include words not in the original document. The key innovation is vocabulary
expansion: a biomedical document about "heart attack" gets non-zero weights for
"cardiac", "myocardial", and "coronary" through the neural encoder's understanding
of semantic relationships. These expanded sparse vectors are stored in standard
inverted indexes - the same Elasticsearch or Lucene infrastructure that BM25 runs
on - making learned sparse retrieval a drop-in quality upgrade for organizations
with existing search infrastructure. BM25 fails completely (score = 0) for synonym
and paraphrase queries due to vocabulary mismatch; learned sparse handles these
while still supporting the exact-match strength of inverted indexes. The tradeoff
versus dense retrieval is infrastructure simplicity and interpretability (learned
sparse wins) against cross-lingual retrieval and maximum semantic generalization
(dense wins). SPLADE is the dominant learned sparse model and is covered in depth
in the next note.
