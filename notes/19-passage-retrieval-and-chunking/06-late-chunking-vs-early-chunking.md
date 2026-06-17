# Late Chunking vs Early Chunking

Late chunking and early chunking are two fundamentally different orderings of
the encoding and splitting operations in a retrieval pipeline. Early chunking -
the approach used in all preceding notes in this module - splits a document into
chunks first and then encodes each chunk independently. Late chunking - introduced
by Jina AI in 2024 - encodes the full document (or as much as fits in the model's
context window) first and then pools the resulting token-level representations into
chunk-level embeddings by averaging over the token positions that belong to each
chunk. The critical difference is that in late chunking, every token's representation
is computed with full document context available through the transformer's self-
attention mechanism before the chunking operation that produces retrievable
embeddings. This resolves a fundamental limitation of early chunking: chunks that
contain pronouns, references, or concepts whose meaning depends on earlier text
lose that context when encoded in isolation. Late chunking preserves it.

## Intuition

Consider a technical document with the following structure:

```
Paragraph 1: "The transformer architecture was introduced in the paper
              'Attention Is All You Need' by Vaswani et al. in 2017."

Paragraph 2: "It relies on self-attention mechanisms that compute pairwise
              relationships between all positions in the input sequence."

Paragraph 3: "This allows it to capture long-range dependencies without the
              sequential bottleneck of recurrent networks."
```

In early chunking, each paragraph becomes an independent chunk encoded in isolation:

```
Chunk 2 (encoded alone):
  "It relies on self-attention mechanisms that compute pairwise relationships
   between all positions in the input sequence."
  Problem: "It" refers to "the transformer architecture" - context is in Chunk 1
           The encoder sees "It" without knowing what "It" is
           The embedding is slightly less precise about what this describes

Chunk 3 (encoded alone):
  "This allows it to capture long-range dependencies..."
  Problem: "This" and "it" are both undefined pronouns without prior context
           The embedding must guess the referent from the chunk text alone
```

In late chunking, the full document is encoded first:

```
Full encoding:
  Token[1-20]:  "The transformer architecture was introduced..." → emb[1-20]
  Token[21-45]: "It relies on self-attention..."                 → emb[21-45]
  Token[46-70]: "This allows it to capture..."                   → emb[46-70]

At token 22 ("It"), the self-attention mechanism has access to tokens 1-21,
  including "transformer architecture" - "It" is resolved in context.
At token 47 ("This"), the self-attention has access to tokens 1-46,
  including the self-attention description - "This" is resolved.

Chunk 2 embedding = mean(emb[21-45])
  These embeddings were computed with full context → no pronoun ambiguity

Chunk 3 embedding = mean(emb[46-70])
  These embeddings were computed with full context → referents are resolved
```

The mathematical operation (mean pooling over token positions) is the same
in both approaches. The difference is when context was available to the model
that produced those token embeddings.

## The Formal Distinction

### Early chunking (standard approach)

```
Document D is split into chunks: C₁, C₂, ..., Cₙ
Each chunk Cᵢ is encoded independently:
  emb(Cᵢ) = pool(encode(Cᵢ))    ← encode takes only Cᵢ as input

encode(Cᵢ) sees only the tokens in Cᵢ
Self-attention within encode(Cᵢ) is restricted to positions within Cᵢ
```

### Late chunking

```
Document D is encoded as a whole:
  token_embs = encode(D₁:Dₘ)    ← encode takes the full document as input
                                   (up to the model's max context window)

token_embs[i] sees the full document context through self-attention
token_embs[i] ← f(all tokens in D through attention mechanism)

Chunk boundaries are determined separately (by any chunking strategy):
  positions(Cᵢ) = {token positions in D that belong to Cᵢ}

Late chunk embeddings are formed by pooling over chunk token positions:
  emb_late(Cᵢ) = mean(token_embs[positions(Cᵢ)])
```

## The Context Loss Problem in Early Chunking

### Types of context dependency

Early chunking loses context for several categories of cross-boundary
semantic dependency:

**Pronoun and coreference resolution:**

```
"The algorithm was first described by Smith et al. in 2015. Since then,
it has been applied to..."
→ Early chunk: "Since then, it has been applied to..."
   "it" is ambiguous without the preceding sentence
→ Late chunk: token[it] has attention to token[algorithm] → resolved
```

**Implicit subject continuation:**

```
"Retrieval systems face two challenges: vocabulary mismatch and latency.
The first challenge is addressed by vocabulary expansion techniques."
→ Early chunk starting at "The first challenge..."
   "The first challenge" refers to "vocabulary mismatch" from the prior chunk
→ Late chunk: full context available, "first challenge" correctly resolved
```

**Comparative and contrastive references:**

```
"BM25 achieves high precision on exact-match queries. Dense retrieval,
in contrast, generalizes better to synonym queries."
→ Early chunk: "Dense retrieval, in contrast, generalizes better..."
   "in contrast" implies a comparison - with what? (BM25, from prior chunk)
→ Late chunk: contrastive context fully visible
```

**Running definitions and terminology:**

```
"We define relevance score (RS) as the product of precision and recall.
RS is used throughout this paper as the primary evaluation metric."
→ Early chunk: "RS is used throughout this paper..."
   "RS" is undefined - it was defined in the previous chunk
→ Late chunk: "RS" token has attention to its definition → full meaning preserved
```

### When context loss matters most

Context dependency across chunk boundaries varies dramatically by document type:

```
High context dependency:
  Scientific papers with running notation and forward references
  Technical manuals with defined terminology used throughout
  Narrative text with pronouns and coreference chains
  Legal documents with defined terms and cross-references

Low context dependency:
  FAQ documents where each question-answer pair is self-contained
  Product descriptions that are individually complete
  Short documents that fit in a single chunk
  Documents with explicit section headers that reset context
```

For self-contained documents or documents where each natural section introduces
its own context, early and late chunking produce similar quality. For
cross-referential documents, late chunking provides a meaningful advantage.

## How Late Chunking Works in Practice

### The token position pooling mechanism

After the full document is encoded, the chunk boundaries define which token
positions to pool:

```
Document: "word₁ word₂ ... wordₙ"
Tokenization: [token₁, token₂, ..., tokenₘ]  (m ≥ n due to subword tokenization)
Full encoding: token_embs ∈ ℝ^{m × d}

Chunk 1 covers words 1-50 → tokens 1-65 (subword expansion)
Chunk 2 covers words 51-100 → tokens 66-132
Chunk 3 covers words 101-150 → tokens 133-198

Late chunk embeddings:
  emb_late(C₁) = mean(token_embs[1:65])     ∈ ℝ^d
  emb_late(C₂) = mean(token_embs[66:132])   ∈ ℝ^d
  emb_late(C₃) = mean(token_embs[133:198])  ∈ ℝ^d
```

This is exactly the same mean pooling used in standard bi-encoder encoding -
the difference is that the token embeddings being pooled were produced with
full document context rather than chunk-only context.

### The chunk boundary determination

Late chunking does not change how chunk boundaries are determined - it changes
when encoding happens relative to those boundaries. The chunk boundaries can be
determined by any of the strategies covered in previous notes:

```
Option 1: Fixed-size late chunking
  Determine chunk boundaries at fixed token intervals (e.g., every 200 tokens)
  Encode the full document
  Pool token embeddings for each 200-token span

Option 2: Semantic late chunking
  Determine chunk boundaries by semantic similarity on the raw text
    (using preliminary lightweight embeddings or sentence boundaries)
  Encode the full document with a long-context model
  Pool token embeddings for each semantic chunk span

Option 3: Recursive/document-aware late chunking
  Determine chunk boundaries using structural signals (headings, paragraphs)
  Encode the full document
  Pool token embeddings for each structural chunk span
```

In all cases, the encoding happens after chunk boundaries are determined but
before the pooling that produces retrievable chunk embeddings.

## The Long-Context Requirement

Late chunking requires encoding the full document (or large document sections)
as a single input. This creates a fundamental requirement: the embedding model
must have a sufficiently large context window to process the entire document or
a meaningful section of it.

### Model context window requirements

```
Standard BERT-family models:
  Max context: 512 tokens (~380 words)
  Late chunking scope: limited to 512 tokens - documents must be split
                       into 512-token segments that are then late-chunked
                       independently
  Effectiveness: limited - only cross-chunk context within a 512-token window
                 is preserved; inter-window context is still lost

Long-context embedding models:
  jina-embeddings-v2-base:    Max 8192 tokens
  jina-embeddings-v3:         Max 8192 tokens (native late chunking support)
  text-embedding-3-large:     Max 8191 tokens
  E5-mistral-7b-instruct:     Max 4096 tokens
  GTE-Qwen2-7B-instruct:      Max 32768 tokens

Late chunking effectiveness:
  For most documents (<8192 tokens): full document context available
  For long documents (>8192 tokens): split into overlapping windows,
                                      apply late chunking within each window
```

The emergence of long-context embedding models is what made late chunking
practically useful - with 512-token models, the benefit is marginal because
document segments are already small enough that much of the inter-chunk context
is within the model's window anyway.

### The computational cost of long-context encoding

Encoding a full document with a long-context model is more expensive than
encoding multiple small chunks:

```
Early chunking (200-token chunks, 1000-token document, 512-token model):
  5 chunks × 200 tokens = 5 forward passes of 200 tokens each
  Attention computation: 5 × O(200²) = 5 × 40,000 = 200,000 operations

Late chunking (1000-token document, 8192-token model):
  1 forward pass of 1000 tokens
  Attention computation: O(1000²) = 1,000,000 operations

For 1000-token document:
  Late chunking ≈ 5× more expensive than early chunking
  (scales quadratically with context length for standard attention)

For 5000-token document:
  Late chunking: O(5000²) = 25,000,000 operations
  Early chunking (5 × 1000): 5 × O(1000²) = 5,000,000 operations
  Late chunking ≈ 5× more expensive

The cost penalty grows quadratically with document length.
```

Flash Attention and other efficient attention implementations reduce but do not
eliminate this quadratic scaling. Late chunking is most cost-effective for:

- Documents shorter than the model's context window (full document fits in one pass)
- Documents where context dependency is high (benefit justifies cost)
- Indexing pipelines where compute cost is a one-time expense

## Empirical Quality Comparison

Jina AI's original late chunking paper (2024) and subsequent evaluations report:

```
Benchmark: MTEB ChunkBenchmark
(synthetic benchmark comparing early vs late chunking quality)

Early chunking (jina-embeddings-v2, 256-token chunks):   NDCG@10 = 0.631
Late chunking  (jina-embeddings-v2, same chunk boundaries): NDCG@10 = 0.674

Improvement: +6.8% NDCG@10

By document type:
  Scientific papers:          +9.2% (high cross-reference density)
  Technical documentation:    +7.1% (terminology defined in prior sections)
  News articles:              +3.1% (lower cross-reference density)
  FAQ collections:            +0.8% (near-self-contained question-answer units)
```

The pattern matches the theoretical prediction: documents with high cross-chunk
context dependency benefit most; self-contained document collections benefit least.

## Connecting to Late Interaction Models

Late chunking shares its name structure (and philosophical approach) with late
interaction retrieval models like ColBERT, but they are distinct concepts addressing
different stages of the retrieval pipeline:

```
ColBERT (late interaction at retrieval time):
  Encodes query and document independently at indexing time
  Stores per-token document embeddings in the index
  At query time: computes token-level query×document interactions
  "Late" refers to when the query-document interaction happens
  → Richer interaction than bi-encoder at retrieval time

Late chunking (late pooling at indexing time):
  Encodes the full document with global context at indexing time
  Pools token embeddings into chunk embeddings after encoding
  At query time: standard bi-encoder retrieval (query embedding vs chunk embedding)
  "Late" refers to when the pooling into chunk-level embedding happens
  → Richer encoding context than early chunking at indexing time
```

Both approaches delay a simplifying operation (interaction for ColBERT, pooling
for late chunking) to a later stage where more information is available, trading
computational cost for representation quality. They are complementary: late
chunking improves chunk embedding quality, while ColBERT improves query-chunk
matching quality. In theory, late chunking could be combined with ColBERT-style
late interaction, though this combination is not yet commonly explored.

## Relationship to the Prior Notes in This Module

Late chunking is not an alternative to the chunking strategies covered in notes
02-05 - it is an orthogonal dimension that can be applied on top of any of them:

```
Note 02: Fixed-size chunking determines chunk boundaries
  → Early mode: encode each fixed-size chunk independently
  → Late mode: encode full document, pool over fixed-size positions

Note 03: Semantic chunking determines chunk boundaries
  → Early mode: encode each semantic chunk independently
  → Late mode: encode full document, pool over semantic chunk positions

Note 04: Recursive chunking determines chunk boundaries
  → Early mode: encode each recursively-determined chunk independently
  → Late mode: encode full document, pool over recursive chunk positions

Note 05: Document-aware chunking determines chunk boundaries
  → Early mode: encode each structural chunk independently
  → Late mode: encode full document, pool over structural chunk positions
```

Late chunking is best understood as a modifier applied to any boundary-determination
strategy - it changes the encoding procedure (full document vs chunk-by-chunk)
while leaving the boundary determination procedure unchanged.

## When to Use Late Chunking

### Use late chunking when

**Document has high cross-chunk context dependency:**
Running definitions, pronoun chains, numbered references, forward references
to later sections - any structure where understanding a passage requires
information from a previous passage.

**Long-context embedding model is available:**
Late chunking's benefit is limited to the context window of the model. With
a 512-token model, the benefit is minimal. With an 8192-token model, most
documents can be encoded in full.

**Indexing cost is acceptable as a one-time expense:**
Late chunking increases indexing compute cost by roughly 3-10x compared to early
chunking of the same documents. For large corpora, this is significant. For
smaller corpora or high-value documents (legal, medical, technical), the quality
improvement typically justifies the cost.

**Chunk coherence matters more than indexing speed:**
Systems where retrieval quality is the primary constraint rather than indexing
throughput.

### Use early chunking when

**Documents are largely self-contained at the chunk level:**
FAQs, product descriptions, short articles, structured databases where each
record is independently meaningful.

**Large corpora where indexing cost is the binding constraint:**
Tens of millions of documents where even a 2x increase in indexing cost is
prohibitive.

**512-token embedding models are required:**
Infrastructure or cost constraints require BERT-base class models that make
late chunking's benefit minimal.

**Simplicity is valued:**
Early chunking is simpler to implement, debug, and reason about. For teams
without infrastructure for long-context model deployment, early chunking with
good boundary determination is usually the right starting point.

## The Practical Implementation Decision

For most practitioners, the decision should be structured as follows:

### Step 1 - Evaluate your corpus's context dependency

Sample 20-50 documents from your corpus. Manually inspect how often chunk
boundaries fall mid-context (mid-pronoun-chain, mid-definition, mid-comparison).
If this happens frequently (>30% of boundaries), late chunking is likely to help.

### Step 2 - Evaluate your current retrieval quality

If early chunking with good boundaries already achieves Recall@100 > 0.85 and
the downstream generation quality is adequate, the benefit of late chunking may
not justify its cost. If Recall@100 is poor despite good boundary placement,
late chunking is worth evaluating.

### Step 3 - Check model availability and cost

If your deployment requires BERT-base class models or has strict latency constraints
on indexing, late chunking may be impractical. If long-context models are available
and indexing is an offline, one-time process, late chunking is a viable option.

### Step 4 - A/B evaluate on representative queries

If steps 1-3 suggest late chunking may help, run both approaches on your corpus
and query set, measure Recall@K and end-to-end generation quality, and make
the decision based on measured improvement rather than theoretical expectation.

## My Summary

Late chunking encodes the full document (or the largest document segment that
fits in the model's context window) before splitting into retrievable chunk
embeddings, which are formed by mean-pooling over the token positions belonging
to each chunk. This preserves cross-chunk context in token representations -
pronouns, running terminology, and comparative references are all resolved through
self-attention on the full document before pooling produces the chunk-level
embeddings used for retrieval. Early chunking - the standard approach - splits
first and encodes each chunk independently, losing this cross-chunk context.
The practical benefit of late chunking scales with the density of cross-chunk
context dependencies in the corpus: scientific papers and technical documentation
with running notation see ~7-9% NDCG improvement; self-contained FAQ collections
see ~1% improvement. Late chunking requires long-context embedding models -
with 512-token BERT-family models the benefit is minimal since most documents
exceed the context window and must be segmented anyway. The primary cost is
quadratically higher encoding computation (one long forward pass vs many short
ones), which is typically acceptable for offline indexing pipelines but may be
prohibitive for real-time or massive-corpus applications. Late chunking is an
orthogonal modification applicable on top of any boundary-determination strategy
from the prior notes - it changes the encoding procedure, not the boundary
determination procedure. ColBERT-style late interaction is a distinct concept
that delays query-document interaction to retrieval time rather than delaying
pooling to post-encoding time.
