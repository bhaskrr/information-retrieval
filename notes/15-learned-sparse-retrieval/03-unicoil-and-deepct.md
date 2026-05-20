# uniCOIL and DeepCT

uniCOIL (unified COntextual Inverted List) and DeepCT (Deep Contextualized Term
Weighting) are two learned sparse retrieval models that take a simpler and more
interpretable approach than SPLADE. Rather than using the full MLM head to expand
vocabulary across the entire vocabulary space, both models focus on estimating
importance weights for terms that are already present in the document or query.
DeepCT (Dai and Callan, 2019) replaces BM25 term frequencies with context-sensitive
importance scores from BERT, without any vocabulary expansion. uniCOIL
extends this idea to also score query terms per-passage and introduces a
unified training approach that works for both document and query encoding. They sit
between BM25 and SPLADE on the quality-complexity spectrum: simpler than SPLADE,
stronger than BM25, and with different failure modes that make them useful complements
to SPLADE in production hybrid systems.

## Intuition

### The problem DeepCT and uniCOIL solve

BM25 weights terms by their raw frequency in the document and their inverse
document frequency in the corpus. A term appearing 3 times in a document scores
higher than one appearing once, regardless of whether those occurrences are
informationally redundant.

Consider: "The database stores data in a database table. The database is optimized."

BM25 gives "database" a high TF-based weight because it appears 3 times. But
do those three occurrences add 3x the information? The second and third mentions
are largely redundant - a skilled reader would not weigh them equally.

BERT, by contrast, understands context. In this passage, each occurrence of
"database" carries different contextual meaning: first it is the subject of storage,
second it is a specific table context, third it is being described as optimized.
DeepCT uses BERT's contextual understanding to estimate how much each token
occurrence contributes to the passage's meaning - and uses this as a replacement
for raw term frequency.

### The conceptual difference from SPLADE

```
DeepCT approach:
  Original terms only - no expansion
  Question: "how important is this term in this context?"
  Output:   {term: importance_score} only for terms in the document

uniCOIL approach:
  Largely original terms - minimal expansion via [MASK] token
  Question: "what is the relevance weight for this passage term?"
  Output:   {term: relevance_weight} for passage terms

SPLADE approach:
  Full vocabulary expansion - many terms not in original document
  Question: "what should this document rank for?"
  Output:   {term: weight} for many vocabulary terms including expansions
```

The analogy: DeepCT and uniCOIL are like a skilled editor who improves BM25's
term weighting without changing which words are present. SPLADE is like a skilled
paraphraser who also suggests related vocabulary that captures the same meaning
differently.

## DeepCT in Depth

### Architecture

DeepCT uses BERT as a contextualized term importance estimator:

```
Input:  Document passage tokens [CLS] t₁ t₂ ... tₙ [SEP]
            ↓
        BERT encoder (12 layers)
            ↓
        Token embeddings h₁ h₂ ... hₙ ∈ ℝ^768
            ↓
        Linear regression head: w = Linear(h) ∈ ℝ
            ↓
        Scalar importance score for each token: s₁ s₂ ... sₙ ∈ ℝ
```

The linear regression head maps each 768-dim token embedding to a single
scalar - the estimated importance of that token in the context of the full passage.

### Training objective

DeepCT is trained with a regression objective on MS MARCO passage-level term
importance labels:

```
Ground truth: for each (passage, term) pair, the importance score is derived from
              downstream retrieval performance - how much does removing this term
              from the passage hurt retrieval?

Loss: MSE(predicted_importance, ground_truth_importance)
```

Because ground truth importance labels require running full retrieval experiments
to measure, DeepCT uses a proxy: the relevance label of the passage (0/1) provides
the training signal, and the model learns to concentrate importance scores on terms
that make the passage more relevant to likely queries.

### Indexing with DeepCT

Instead of indexing raw term frequencies, DeepCT indexes the learned importance
scores:

```
Standard BM25 index:
  term → [(doc_id, tf), ...]    ← raw frequency

DeepCT index:
  term → [(doc_id, importance_score), ...]   ← learned importance
```

At query time, the BM25 scoring formula uses DeepCT importance scores as a
replacement for term frequency. The IDF component and other BM25 components
remain unchanged.

### DeepCT limitations

DeepCT's regression approach has two significant limitations:

**1. No vocabulary expansion**
DeepCT only weights existing terms. A document about "heart attack" scores zero
for queries containing "cardiac arrest" because DeepCT never adds "cardiac" to
the document's representation.

**2. Training target mismatch**
The regression target (passage relevance) is a coarse proxy for term importance.
A passage can be relevant for reasons unrelated to a specific term's presence,
making the regression signal noisy.

These limitations directly motivated uniCOIL's different approach.

## uniCOIL in Depth

### Architecture

uniCOIL is conceptually simpler than both DeepCT and SPLADE. It uses a BERT-based
encoder with a single linear layer that outputs a scalar weight per input token:

```
Input:   passage tokens [CLS] t₁ t₂ ... tₙ [SEP]
              ↓
         BERT encoder
              ↓
         Token embeddings h₁ h₂ ... hₙ
              ↓
         Linear(hᵢ) → scalar weight wᵢ ∈ ℝ
              ↓
         ReLU(wᵢ) → non-negative weight (ensure non-negative sparse vector)
```

For each unique term t in the passage, the final weight is the maximum scalar
across all its occurrences (similar to SPLADE's max pooling):

```
w(t, passage) = max_{i: tokenᵢ = t} ReLU(Linear(hᵢ))
```

### Minimal vocabulary expansion via [MASK]

uniCOIL optionally adds a single [MASK] token to the input and uses the MLM head
to predict one expansion term:

```
Input:   [CLS] t₁ t₂ ... tₙ [MASK] [SEP]
         ↑
         optional expansion token

The [MASK] position produces one additional term to add to the sparse vector.
```

This is much more conservative than SPLADE (which expands to potentially hundreds
of additional vocabulary terms) while still providing some cross-vocabulary matching.

### Training objective

uniCOIL is trained with a pairwise cross-entropy loss directly on retrieval:

```
Score:  s(q, d) = Σ_{t in q ∩ d} w_q(t) × w_d(t)
         ↑ dot product over shared terms

Loss:   L = -log(sigmoid(s(q, d⁺) - s(q, d⁻)))
```

Where w_q(t) is the query term weight and w_d(t) is the document term weight,
both estimated by the same uniCOIL encoder applied to query and document separately.

This end-to-end training on retrieval directly optimizes the final scoring function
- unlike DeepCT which trains a regression proxy. This is why uniCOIL significantly
outperforms DeepCT despite having a simpler architecture.

### uniCOIL with DocTTTTTQuery expansion

The strongest uniCOIL variant combines uniCOIL weights with DocTTTTTQuery (doc2query)
document expansion. DocTTTTTQuery uses a T5 model to predict queries for a document,
then appends those predicted queries to the document before indexing:

```
Step 1 - T5 query generation:
  Document: "heart attack symptoms and treatment"
  T5 generates: "what are symptoms of cardiac arrest?"
                "how to treat myocardial infarction"
                "cardiac event emergency response"
  Append to document: "heart attack symptoms and treatment
                       cardiac arrest symptoms myocardial infarction"

Step 2 - uniCOIL weighting:
  Apply uniCOIL to expanded document
  → sparse vector now includes "cardiac", "myocardial" with learned weights
```

This combination achieves vocabulary expansion (via T5) with learned weighting
(via uniCOIL), approaching SPLADE quality with more controllable expansion.

## DeepCT vs uniCOIL vs SPLADE

```
Property                DeepCT          uniCOIL         SPLADE
──────────────────────────────────────────────────────────────────────────
Vocabulary expansion    None            Minimal (1 term) Full (50-200 terms)
Base architecture       BERT + Linear   BERT + Linear    BERT + MLM head
Training objective      Regression      Pairwise CE      Pairwise CE + FLOPS
Avg active terms/doc    Same as input   Same as input    50-200 (expanded)
Index infrastructure    Inverted index  Inverted index   Inverted index
BEIR avg NDCG@10        ~0.390          ~0.420           ~0.450
Interpretability        High            High             Medium
Training complexity     Simple          Simple           Complex (FLOPS reg.)
Inference speed         Fast            Fast             Fast (similar)
Implementation effort   Low             Low              Medium
Best for                Reweighting     Balanced CLIR    Maximum quality
```

### When to choose uniCOIL over SPLADE

**Interpretability is required**
uniCOIL weights correspond directly to original document terms - you can explain
why a document was retrieved by inspecting which original terms had high weights.
SPLADE's expanded terms are harder to trace back to original content.

**Domain-specific controlled vocabulary**
In domains like legal or medical retrieval where exact terminology matters,
uncontrolled vocabulary expansion can introduce noise. uniCOIL's minimal or no
expansion is preferable for precision-critical applications.

**Simpler implementation and maintenance**
uniCOIL has a simpler training procedure (no FLOPS regularization), simpler
inference (linear head rather than full MLM head), and simpler debugging.
For teams without deep IR expertise, uniCOIL is more maintainable.

**When document expansion is handled separately**
If you already run DocTTTTTQuery or another document expansion method as a
preprocessing step, adding SPLADE's expansion on top is redundant. uniCOIL +
DocTTTTTQuery is cleaner than SPLADE in this scenario.

### When to choose SPLADE over uniCOIL

**Maximum quality is needed**
SPLADE consistently outperforms uniCOIL on BEIR by ~5-8% NDCG@10.

**Strong vocabulary mismatch is the primary problem**
SPLADE's rich expansion handles synonym-heavy queries better than uniCOIL's
minimal or no expansion.

**Multilingual retrieval**
Multilingual SPLADE variants exist (mSPLADE). uniCOIL is primarily English.

## DocTTTTTQuery + uniCOIL Pipeline

The strongest uniCOIL deployment pattern:

```
Step 1 - Document expansion (offline, once)
  For each document chunk:
    input = "Generate queries for: {document_text}"
    5 synthetic queries = T5_base_msmarco(input)
    expanded_doc = document_text + " " + " ".join(synthetic_queries)

Step 2 - uniCOIL encoding (offline, once)
  sparse_vec = unicoil_encoder(expanded_doc)
  index(doc_id, sparse_vec)

Step 3 - Query encoding (online, per query)
  query_vec = unicoil_encoder(query)

Step 4 - Inverted index retrieval (standard BM25 mechanics)
  results = inverted_index.search(query_vec)
```

This pipeline approaches SPLADE quality because:

- DocTTTTTQuery handles vocabulary expansion (what SPLADE handles with MLM head)
- uniCOIL handles learned weighting of expanded terms (what SPLADE handles with weights)
- Both steps can be swapped independently for maintenance

## Choosing Between DeepCT, uniCOIL, and SPLADE

```
Requirement                         Best choice
──────────────────────────────────────────────────────────────────────
Maximum out-of-box quality          SPLADE++
Best quality with expansion control uniCOIL + DocTTTTTQuery
Simplest implementation             DeepCT
Vocabulary expansion not wanted     DeepCT or uniCOIL (no expansion)
Existing doc expansion pipeline     uniCOIL (weights only)
Need interpretable term weights     DeepCT or uniCOIL
Multilingual retrieval              mSPLADE (uniCOIL is primarily EN)
Fast iteration / experimentation    uniCOIL (simpler training)
```

## My Summary

DeepCT and uniCOIL are simpler learned sparse alternatives to SPLADE that focus
on learning better term weights rather than expanding the vocabulary. DeepCT trains
a linear regression head on BERT token embeddings to predict term importance scores
as a replacement for raw BM25 term frequencies - same terms, better weights. uniCOIL
trains a similar scalar weight head end-to-end with pairwise retrieval loss rather
than a regression proxy, producing stronger results with a simpler architecture.
Both produce interpretable sparse vectors containing only terms from the original
document, making them preferable to SPLADE when interpretability is required or when
vocabulary expansion is handled by a separate step like DocTTTTTQuery. On BEIR,
uniCOIL without expansion achieves approximately NDCG@10 ≈ 0.42 - between BM25
(0.44) and SPLADE++ (0.46) - with uniCOIL + DocTTTTTQuery approaching SPLADE quality.
DeepCT is the simplest starting point for contextual term weighting; uniCOIL is
the production recommendation when SPLADE's complexity is undesirable.
