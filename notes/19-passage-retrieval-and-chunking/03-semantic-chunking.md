# Semantic Chunking

Semantic chunking is a chunking strategy that determines chunk boundaries based
on shifts in meaning detected through embedding similarity, rather than fixed
token or character counts. Instead of cutting text every N tokens regardless of
content, semantic chunking measures how similar adjacent sentences or text spans
are to each other and places chunk boundaries at points where similarity drops
sharply - points where the discourse moves from one topic or idea to another.
The result is chunks of variable length that correspond to coherent semantic
units: a chunk ends not because it reached an arbitrary token count, but because
the content shifted to a different topic. Semantic chunking trades computational
cost (every sentence must be embedded before chunking decisions can be made) for
chunk quality (boundaries align with actual shifts in meaning rather than
arbitrary positions).

## Intuition

Imagine reading a long article and being asked to mark where one idea ends and
the next begins - not by counting words, but by noticing when the conversation
"changes direction." A paragraph about the history of a technology flows into a
paragraph about its technical mechanism, which flows into a paragraph about its
limitations. A human reader naturally senses these transitions through shifts in
vocabulary, focus, and rhetorical purpose, even without explicit headers.

Fixed-size chunking cannot sense these transitions - it only counts tokens. If
the transition from "history" to "mechanism" happens to fall at token 287, but
your chunk size is 300, the transition gets split awkwardly: chunk 1 ends with
"...and this set the stage for later developments" and chunk 2 begins with "The
underlying mechanism works as follows..." Neither chunk is fully coherent - chunk
1 is missing its conclusion's context, and chunk 2 lacks its introduction.

Semantic chunking approximates the human reader's sense of topic transitions by
measuring embedding similarity between adjacent text spans. When two consecutive
sentences are embedded and found to be highly similar, they likely discuss the
same topic and belong in the same chunk. When similarity drops sharply, a topic
shift has likely occurred, and that point becomes a natural chunk boundary.

## The Core Algorithm

Semantic chunking operates through a sliding window comparison of embedding
similarity between adjacent text units (sentences or small groups of sentences):

```
Step 1: Split document into atomic units (sentences)
  units = [s₁, s₂, s₃, ..., sₙ]

Step 2: Embed each unit (or a sliding window of units)
  embeddings = [embed(s₁), embed(s₂), ..., embed(sₙ)]

Step 3: Compute similarity between adjacent units
  sim_i = cosine_similarity(embeddings[i], embeddings[i+1])
  similarities = [sim_1, sim_2, ..., sim_{n-1}]

Step 4: Identify breakpoints where similarity drops below a threshold
  breakpoints = {i : sim_i < threshold}

Step 5: Form chunks by grouping consecutive units between breakpoints
  chunk_1 = [s_1, ..., s_{breakpoint_1}]
  chunk_2 = [s_{breakpoint_1+1}, ..., s_{breakpoint_2}]
  ...
```

The output is a sequence of variable-length chunks, each corresponding to a
span of text where adjacent sentences remained semantically similar - a
coherent discussion of one topic.

## Sliding Window Similarity Comparison

The naive version compares each sentence only to its immediate neighbor. This
is noisy - individual sentences can vary in embedding similarity for reasons
unrelated to topic shifts (sentence length, syntactic structure, emphasis).
A more robust approach uses windows of multiple sentences:

```
Window-based comparison:
  For position i, compare:
    window_before = embed(concat(s_{i-k}, ..., s_i))
    window_after  = embed(concat(s_{i+1}, ..., s_{i+k}))
    sim_i = cosine_similarity(window_before, window_after)

Where k = window size (typically 1-3 sentences)
```

Larger windows smooth out sentence-level noise and detect broader topic shifts.
Smaller windows are more sensitive to fine-grained transitions but noisier.

### The combined-sentence approach

A common practical implementation (used in LlamaIndex's SemanticSplitterNodeParser
and similar tools) embeds groups of sentences with a buffer:

```
For each sentence sᵢ, create a "combined sentence":
  combined_i = concat(s_{i-buffer}, ..., s_i, ..., s_{i+buffer})

Embed each combined sentence:
  emb_i = embed(combined_i)

Compute similarity between consecutive combined embeddings:
  distance_i = 1 - cosine_similarity(emb_i, emb_{i+1})

Breakpoints occur where distance_i is anomalously high
```

The buffer (typically 1 sentence on each side) provides local context for each
sentence's embedding, making the similarity comparison more robust to sentence-level
noise while still detecting topic-level shifts.

## Breakpoint Selection Methods

How the similarity drop threshold is chosen significantly affects chunk
granularity. Three common approaches:

### Percentile-based thresholding

Compute the distribution of all adjacent similarities (or distances) across
the document, and set the breakpoint threshold at a percentile of this
distribution:

```
distances = [1 - sim_i for all i]
threshold = percentile(distances, 95)   # top 5% of distances become breakpoints

Breakpoints = {i : distance_i > threshold}
```

```
Percentile setting    Effect
──────────────────────────────────────────────────────────
90th percentile       More breakpoints → smaller, more numerous chunks
                      Sensitive to minor topic variations

95th percentile       Moderate breakpoints (common default)
                      Balances chunk count and coherence

99th percentile       Fewer breakpoints → larger, fewer chunks
                      Only major topic shifts trigger splits
```

Percentile-based thresholding is adaptive to each document's overall
"semantic volatility" - a document that frequently shifts topics produces
more breakpoints at the same percentile than a document with sustained
single-topic discussion.

### Standard deviation thresholding

Set the breakpoint threshold based on how many standard deviations above
the mean a distance must be:

```
mean_distance = mean(distances)
std_distance  = std(distances)
threshold     = mean_distance + (n_std × std_distance)

Breakpoints = {i : distance_i > threshold}

n_std = 1: more breakpoints (1 standard deviation above mean)
n_std = 2: fewer breakpoints (rare, only large shifts)
n_std = 3: very few breakpoints (only extreme shifts)
```

Standard deviation thresholding is sensitive to the variance of the distance
distribution - documents with highly variable topic flow produce different
absolute thresholds than documents with steady, gradual topic evolution.

### Absolute threshold

A fixed similarity threshold applied uniformly across all documents:

```
threshold = 0.75   (similarity below this triggers a breakpoint)

Breakpoints = {i : sim_i < 0.75}
```

Absolute thresholds are simple but require calibration for the specific
embedding model and domain - different embedding models produce different
similarity distributions, so a threshold calibrated for one model may not
transfer to another.

## Combining Semantic Boundaries with Size Constraints

Pure semantic chunking can produce chunks of wildly varying sizes - a document
section with consistent vocabulary might produce a single 2000-token chunk,
while a section with rapid topic shifts might produce many 30-token chunks.
Both extremes can be problematic.

### Minimum and maximum size constraints

Practical semantic chunking implementations enforce size bounds:

```
For each semantic chunk candidate:
  If chunk_length < min_size (e.g., 50 tokens):
    Merge with adjacent chunk (prefer the more similar neighbor)

  If chunk_length > max_size (e.g., 500 tokens):
    Apply fixed-size or recursive splitting within the semantic chunk
    (fall back to secondary splitting strategy for oversized semantic units)
```

This hybrid approach uses semantic boundaries as the primary signal but
applies fixed-size logic as a safety net - semantic chunking determines
_where_ natural breaks occur, while size constraints ensure chunks remain
within usable bounds for embedding and LLM context.

### Greedy merging algorithm

A common implementation pattern:

```
1. Compute all candidate breakpoints from similarity analysis
2. Form initial chunks at every candidate breakpoint
3. Iterate through chunks:
     If current chunk + next chunk < max_size AND
        similarity(current, next) > merge_threshold:
       Merge current and next chunk
4. Repeat until no more merges possible or max_size constraint reached
```

This produces chunks that respect both the semantic structure (similar
adjacent content merged together) and practical size limits.

## Computational Cost

The defining tradeoff of semantic chunking is computational cost versus fixed-size
chunking:

```
Fixed-size chunking:
  Cost: O(document_length) - simple string/token operations
  No embedding calls required during chunking

Semantic chunking:
  Cost: O(n_sentences) embedding calls (or O(n_sentences / buffer_size)
        for combined-sentence approach)
  Each sentence (or sentence group) requires an embedding computation

For a 10,000-word document (~500 sentences):
  Fixed-size: negligible cost, milliseconds
  Semantic:   500 embedding calls (batched, but still substantial)
              On CPU with all-MiniLM: ~5-15 seconds per document
              On GPU: ~1-2 seconds per document
```

For large corpora (millions of documents), the embedding cost of semantic
chunking is a significant one-time indexing expense. For corpora that are
indexed once and queried many times, this cost is usually acceptable - it
is paid once at indexing time, not at query time.

### Cost mitigation strategies

**Batch processing:** Embed all sentences from a document (or batch of
documents) in a single forward pass rather than one sentence at a time.

**Smaller embedding models for chunking:** Use a lightweight model (e.g.,
all-MiniLM-L6, 384-dim) for the chunking decision, even if a larger model
(e.g., E5-large) is used for the final chunk embeddings used in retrieval.
The chunking embedding only needs to capture topic-level similarity, not
fine-grained retrieval-quality semantics.

**Sentence sampling:** For very long documents, compute similarity on every
2nd or 3rd sentence rather than every sentence, interpolating breakpoints
between sampled positions.

## Semantic Chunking vs Fixed-Size: Empirical Comparison

The quality difference between semantic and fixed-size chunking depends
heavily on document characteristics:

```
Document type              Fixed-size Recall@10   Semantic Recall@10   Improvement
──────────────────────────────────────────────────────────────────────────────────
Narrative text (articles)   0.68                   0.74                 +0.06
                            (smooth topic transitions, semantic helps moderately)

Technical documentation     0.71                   0.79                 +0.08
                            (distinct sections, semantic captures section
                             boundaries better than arbitrary token cuts)

FAQ / structured Q&A        0.82                   0.83                 +0.01
                            (each Q&A pair is already a natural unit,
                             semantic chunking provides little extra benefit)

Mixed-topic compilations    0.61                   0.75                 +0.14
                            (large benefit - topic shifts are frequent and
                             unpredictable, fixed-size frequently misaligns)

Legal contracts             0.65                   0.70                 +0.05
                            (moderate benefit - clause structure provides
                             some natural alignment even with fixed-size)
```

The pattern: semantic chunking provides the largest improvements for documents
with frequent, unpredictable topic shifts where fixed-size chunking has high
probability of misaligned boundaries. For documents that are already well-
structured into natural units close to the target chunk size (FAQs, short
sections), semantic chunking provides smaller marginal benefit.

## When Semantic Chunking's Variable Sizes Help

### Help case 1 - Topic-dense sections produce small, focused chunks

A section that rapidly covers multiple distinct points (a list of short
definitions, a series of brief examples) produces several small semantic
chunks, each focused on one definition or example:

```
Source text:
  "Precision measures the fraction of retrieved documents that are relevant.
   Recall measures the fraction of relevant documents that are retrieved.
   F1 score is the harmonic mean of precision and recall."

Semantic chunking (high topic density):
  Chunk A: "Precision measures the fraction of retrieved documents that are relevant."
  Chunk B: "Recall measures the fraction of relevant documents that are retrieved."
  Chunk C: "F1 score is the harmonic mean of precision and recall."

Fixed-size chunking (300 tokens, this entire passage fits in one chunk):
  Chunk: [all three sentences together]

Query: "what does precision mean in IR"
  Semantic Chunk A: highly focused, high similarity to query
  Fixed-size chunk: diluted by recall and F1 content, lower similarity
```

For dense factual content, smaller semantic chunks improve retrieval precision.

### Help case 2 - Topic-sparse sections produce large, complete chunks

A section that develops a single argument across many sentences (a detailed
explanation, an extended example, a step-by-step derivation) produces one
large semantic chunk that preserves the complete argument:

```
Source text: [800 words explaining the mathematical derivation of gradient
              descent, building each step on the previous]

Semantic chunking: One large chunk (the entire derivation, ~800 tokens)
  → Preserves the complete logical chain

Fixed-size chunking (300 tokens): Three separate chunks
  → Each chunk contains part of the derivation, none complete
  → A query about "why does the gradient descent update rule work"
    cannot be answered by any single chunk
```

For content requiring sustained context (derivations, multi-step arguments,
extended narratives), larger semantic chunks preserve the coherence necessary
for the LLM to use the content effectively.

## When Semantic Chunking's Variable Sizes Hurt

### Hurt case 1 - Inconsistent embedding-to-LLM-context ratios

If the RAG pipeline retrieves a fixed number of chunks (e.g., top-5), and
chunk sizes vary from 50 to 1500 tokens, the total context size varies
unpredictably:

```
Retrieval result with semantic chunking:
  Chunk 1: 80 tokens
  Chunk 2: 1400 tokens
  Chunk 3: 120 tokens
  Chunk 4: 95 tokens
  Chunk 5: 1100 tokens
  Total: 2795 tokens

Versus fixed-size (300 tokens each):
  5 × 300 = 1500 tokens (predictable)
```

Unpredictable context size complicates LLM context budget management -
sometimes the retrieved context is far larger than expected, consuming
generation budget; sometimes it is much smaller than expected, wasting
context window space that could have included more chunks.

### Hurt case 2 - Threshold sensitivity across domains

A breakpoint threshold tuned for one document type may not transfer to
another. A percentile threshold of 95% on technical documentation (where
topic shifts are relatively rare and large) might produce very different
chunk granularity than the same threshold applied to conversational
transcripts (where topic shifts are frequent and small).

```
Same 95th-percentile threshold applied to:
  Technical manual: average chunk size = 380 tokens (few large topic shifts)
  Forum discussion: average chunk size = 45 tokens (many small topic shifts)
```

If a corpus contains heterogeneous document types, a single threshold may
produce wildly inconsistent chunk sizes across document types, which can
create the same downstream context-budget issues as case 1, now compounded
by domain variation.

### Hurt case 3 - Embedding model sensitivity to local context

Some embedding models are more sensitive to local lexical patterns than
global topic - two sentences about the same topic but using different
vocabulary or sentence structure might receive a lower similarity score
than two sentences about different topics that happen to share vocabulary.

```
Sentence A: "The algorithm converges after 100 iterations."
Sentence B: "Convergence is achieved once the loss plateaus."

Both about convergence, but:
  - Different vocabulary ("converges/iterations" vs "convergence/loss/plateaus")
  - Different sentence structure
  - May receive lower similarity than expected for "same topic" sentences

This can cause semantic chunking to insert breakpoints within a coherent
topic discussion simply because the embedding model is sensitive to
lexical variation rather than purely topical content.
```

This failure mode is mitigated by using embedding models specifically
evaluated for sentence-level semantic similarity tasks (STS benchmarks)
rather than retrieval-optimized models, since STS-tuned models tend to
be more robust to lexical variation when judging topical similarity.

## Practical Implementation Patterns

### LlamaIndex SemanticSplitterNodeParser

A widely used implementation following the combined-sentence approach:

```
Configuration:
  buffer_size: number of sentences to group for each comparison (default 1)
  breakpoint_percentile_threshold: percentile for breakpoint detection (default 95)
  embed_model: embedding model used for chunking decisions

Process:
  1. Split document into sentences
  2. Group sentences with buffer (sentence i with i-1 and i+1 if buffer=1)
  3. Embed each group
  4. Compute cosine distances between consecutive group embeddings
  5. Identify breakpoints at the specified percentile
  6. Form final chunks between breakpoints
```

### Two-pass semantic chunking

A more sophisticated pattern: first pass identifies coarse topic boundaries
using a lightweight model; second pass refines boundaries within large
sections using a higher-quality model:

```
Pass 1 (coarse): Use fast model (all-MiniLM) on full document
  → Identify major section boundaries (large topic shifts)

Pass 2 (fine): Within each major section, use higher-quality model
  → Identify finer boundaries within the section if section exceeds max_size
```

This two-pass approach balances cost (most of the document only needs
fast, coarse analysis) with quality (only oversized sections receive
expensive fine-grained analysis).

## My Summary

Semantic chunking determines chunk boundaries by measuring embedding similarity
between adjacent sentences or sentence groups, placing breaks at points where
similarity drops sharply - indicating a shift in topic or discourse direction.
The core algorithm embeds each sentence (often with a buffer of neighboring
sentences for context), computes pairwise distances between consecutive embeddings,
and identifies breakpoints using percentile-based, standard-deviation-based, or
absolute thresholds. The result is variable-length chunks that correspond to
coherent semantic units rather than arbitrary token positions. Practical
implementations combine semantic boundaries with minimum and maximum size
constraints, merging undersized chunks and applying fallback splitting to
oversized ones. The primary cost is computational: every sentence requires an
embedding call during chunking, a one-time indexing expense that is usually
acceptable for corpora indexed once and queried many times. Semantic chunking
provides the largest quality improvements for documents with frequent, unpredictable
topic shifts (mixed-topic compilations showed +0.14 Recall@10 in comparative
evaluation) and smaller improvements for already well-structured content like
FAQs. Variable chunk sizes help when topic-dense sections produce small focused
chunks and topic-sparse sections (extended derivations, multi-step arguments)
produce large chunks that preserve complete logical chains - but they hurt when
downstream LLM context budgets expect predictable chunk sizes, or when breakpoint
thresholds tuned for one document type transfer poorly to heterogeneous corpora.
