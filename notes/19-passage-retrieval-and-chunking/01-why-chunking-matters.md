# Why Chunking Matters

Chunking is the process of dividing long documents into smaller, retrievable
units - passages, paragraphs, or spans - that can be individually indexed,
embedded, and retrieved. It is the preprocessing decision that determines what
the retrieval system treats as a single retrievable item. Documents are typically
too long to embed meaningfully as single vectors - a 10,000-word research paper
cannot be compressed into a 768-dimensional vector without severe information
loss. Chunking converts long documents into shorter units where each chunk
ideally captures a single cohesive topic or information unit that can be
meaningfully embedded and retrieved independently. Chunking strategy is one of
the highest-leverage decisions in a RAG pipeline: the same embedding model and
retrieval architecture can produce dramatically different quality results depending
solely on how documents are divided into chunks. It is also one of the most
underexplored decisions in practice - most teams use fixed-size chunking with
default parameters without measuring the impact of their choice.

## Intuition

Imagine a 200-page textbook on information retrieval. A user asks: "What is
the difference between precision and recall?" The relevant answer is in one
paragraph on page 47. If you embed the entire textbook as a single vector, that
embedding represents the average meaning of 200 pages - it captures broad themes
but loses the specific answer on page 47. A retrieval query about precision and
recall produces a low similarity score because the textbook embedding is diffused
across hundreds of other topics.

Now imagine embedding each paragraph separately. The paragraph about precision
and recall produces a vector strongly in the direction of evaluation metrics,
recall definitions, and precision-recall tradeoffs. The same query produces a
high similarity score precisely because the chunk's embedding is concentrated
on the relevant topic.

The fundamental constraint: embedding models compress variable-length text into
a fixed-size vector. Every additional sentence in the input adds more context
but also more noise - other topics and concepts that dilute the signal. At
some optimal length, the embedding captures enough context to be meaningful
while remaining focused enough to be discriminative. Beyond that length, adding
more text progressively dilutes the signal.

Chunking is how you manage this compression-discrimination tradeoff.

## The Information Compression Problem

Dense retrieval embeds text into fixed-size vectors using mean pooling or CLS
token representations. The fixed size is independent of input length. This
creates a fundamental information bottleneck:

```
Short passage (50 words):
  768-dim vector encodes ~50 words of information
  Information density: 768/50 ≈ 15 dimensions per word
  High specificity: vector strongly represents the specific content

Long document (5000 words):
  768-dim vector encodes ~5000 words of information
  Information density: 768/5000 ≈ 0.15 dimensions per word
  Low specificity: vector represents average of all content
```

As document length increases, each additional word contributes less to the
embedding's meaning. The embedding increasingly represents the document's
general topic rather than any specific claim or answer.

This is not a limitation of particular embedding models - it is a fundamental
consequence of encoding variable-length information into fixed-size vectors.
Even the most powerful embedding model cannot encode a 10,000-word document
into 768 dimensions without losing specificity. Chunking is the architectural
solution to this constraint.

## Retrieval Granularity and User Needs

Different information needs require different granularities of retrieval:

### Query types and their optimal chunk sizes

**Specific factual questions** require short, precise chunks:

```
Query: "What year was the Transformer architecture introduced?"
Optimal chunk: Single sentence or short paragraph containing "2017" and "Attention
               Is All You Need"
Too large: Entire paper section about transformer history (answer buried in context)
Too small: Single token (no context to determine relevance)
```

**Conceptual explanations** require medium chunks:

```
Query: "How does self-attention work?"
Optimal chunk: A paragraph explaining the query-key-value mechanism
Too small: Individual sentences (lack the explanatory context)
Too large: Full chapter on attention (contains much more than needed)
```

**Comparative or analytical queries** may need larger chunks:

```
Query: "What are the tradeoffs between BM25 and dense retrieval?"
Optimal chunk: Multi-paragraph comparison or summary section
Too small: Individual claims about one method lose comparative context
Too large: Entire textbook chapter on retrieval loses focus
```

No single chunk size serves all query types. This motivates multi-granularity
indexing - maintaining chunks at multiple levels and selecting the right
granularity at query time.

## The Chunk-Answer Alignment Problem

A critical but often overlooked constraint: the answer to a query must fit
within a single chunk to be retrievable. If the answer spans multiple chunks,
no single chunk will have high enough similarity to the query, and the relevant
content will be missed.

Consider chunking by paragraph in a document where the relevant answer spans
two paragraphs:

```
Paragraph A (chunk 1):
  "Gradient descent is an optimization algorithm that iteratively updates
   model parameters. The update rule is θ ← θ - α × ∇L."

Paragraph B (chunk 2):
  "The learning rate α controls the step size. Too large and the optimizer
   oscillates; too small and convergence is slow."

Query: "How does learning rate affect gradient descent?"

Chunk 1 embedding: focuses on parameter update rule, gradient
Chunk 2 embedding: focuses on learning rate, convergence behavior

Both chunks are partially relevant but neither is highly relevant.
A single chunk containing both paragraphs would be highly relevant.
```

The answer is distributed across two chunks. Neither individual chunk scores
highly enough to be retrieved reliably. This is the chunk-answer alignment
failure mode - one of the most common reasons for RAG pipelines to fail on
queries where the answer exists in the corpus.

## Chunking Failures and Their Consequences

Understanding why chunking fails helps in choosing the right strategy:

### Failure mode 1 - Chunk too large (semantic diffusion)

Chunks contain multiple distinct topics. The embedding averages across topics,
reducing specificity for any particular query.

```
Symptoms:
  Recall@10 is low despite relevant content existing in the corpus
  Retrieved chunks contain the right answer but also lots of irrelevant content
  LLM generation quality is poor because context contains contradictory information

Diagnosis:
  Average chunk length > 500 tokens
  Chunks regularly span section boundaries in structured documents
  Multiple distinct topics appear in the same chunk
```

### Failure mode 2 - Chunk too small (context loss)

Chunks lack sufficient context for the embedding to capture the meaning of the
text. Isolated sentences often contain pronouns, references, or domain terms
that are only interpretable with surrounding context.

```
Symptoms:
  Retrieved chunks appear to be on the right topic but are uninformative
  Short chunks contain pronouns ("it", "they", "this") without referents
  LLM cannot generate a useful answer from the retrieved context

Diagnosis:
  Average chunk length < 50 tokens
  Chunks regularly break mid-sentence or mid-argument
  High proportion of chunks that reference external content not in the chunk

Example:
  "This approach, however, has significant limitations."
  → What approach? What limitations? Context has been chunked away.
```

### Failure mode 3 - Chunk boundary misalignment

Chunk boundaries fall in the middle of coherent units - mid-sentence, mid-list,
mid-table, mid-code-block.

```
Symptoms:
  Retrieved chunks begin with continuation phrases ("Furthermore...", "On the other hand...")
  Retrieved chunks end with incomplete sentences or half-explained concepts
  Tables and code blocks split across multiple chunks

Impact:
  Each split chunk has lower information quality than the pre-split unit
  LLM receives incomplete context that cannot be understood independently
```

### Failure mode 4 - Inconsistent chunk size (length bias)

When chunks vary widely in length, similarity scores become length-biased. Short
chunks get high scores for specific queries (high information density) while long
chunks get inflated scores for general queries (many matching terms).

```
Comparison:
  Short chunk (20 tokens): "Precision is TP / (TP + FP)"
  Long chunk (400 tokens): [long section about evaluation including precision definition]

Query: "precision formula"

Short chunk: highly specific, high similarity (correct)
Long chunk: contains the formula but also many other things
            → similarity may be lower despite containing the answer
```

Inconsistent chunk lengths make it hard to set useful similarity thresholds
and can cause important short-but-specific chunks to compete poorly against
long-but-general chunks.

## The Token Limit Interaction

Chunking strategy must account for the embedding model's token limit:

### BERT-family models (most common)

```
Max input length: 512 tokens
Typical behavior beyond 512: truncated silently

At 512 tokens (≈ 380 words):
  The model processes all content within the limit
  Content is well-represented

Above 512 tokens:
  Content beyond token 512 is silently dropped
  Only the first 512 tokens influence the embedding
  Common but underappreciated source of retrieval quality loss
```

A surprising number of production RAG systems chunk at 1000 characters or
200 words without realizing that their 512-token embedding model is silently
truncating every chunk beyond 400 words, creating systematic bias toward
information in the early part of each chunk.

### Long-context embedding models

```
E5-mistral-7b-instruct:   max 4096 tokens
jina-embeddings-v2-base:  max 8192 tokens
text-embedding-3-large:   max 8191 tokens
GTE-Qwen2-7B-instruct:   max 32768 tokens
```

Long-context models allow larger chunks but do not eliminate the information
compression problem - they just shift it to longer lengths. A 8192-token
chunk still compresses into a single 768-dim or 4096-dim vector, with the
same dilution effect at long contexts.

## Chunking as a Hyperparameter

Unlike embedding model choice or retrieval algorithm selection, chunking is
almost never treated as a hyperparameter to be tuned on held-out evaluation
data. This is a significant gap in practice.

### What to measure

The effect of chunking strategy should be measured along two dimensions:

**Retrieval quality (first-stage recall):**
Does the relevant chunk appear in the top-K retrieved results?

```
Metric: Recall@10, Recall@100 on a labeled test set
Vary: chunk size, overlap, splitting strategy
Report: Recall@K for each configuration
```

**Generation quality (end-to-end RAG):**
Does the LLM produce a correct, grounded answer from the retrieved chunks?

```
Metric: answer accuracy, faithfulness, completeness
Vary: same chunking configurations
Report: end-to-end answer quality for each configuration
```

The optimal configuration on retrieval quality and generation quality may differ -
smaller chunks improve retrieval precision but may reduce generation quality by
providing insufficient context.

### The evaluation gap

Most practitioners do not measure chunking impact before deployment. The typical
workflow is:

1. Choose chunk size based on intuition (often 200-500 tokens)
2. Deploy RAG system
3. Attribute quality issues to the embedding model or LLM rather than chunking

A structured chunking evaluation would reveal that for many applications,
chunking strategy has a larger impact on end-to-end quality than switching
embedding models or reranking architectures.

## Document Structure and Chunking

Document structure provides natural chunk boundaries that should be exploited
rather than ignored:

### Structured documents

PDFs, HTML, and Markdown documents contain explicit structural markers:

```
Markdown:
  # Headers → natural chunk boundaries
  ## Subheadings → sub-chunk boundaries
  Paragraphs (blank line separation) → minimum chunk unit

HTML:
  <h1>, <h2>, <h3> → section boundaries
  <p> → paragraph boundaries
  <table>, <code> → special handling required

PDF:
  Font size changes → likely heading
  Whitespace patterns → paragraph and section breaks
  Page breaks → sometimes meaningful, sometimes arbitrary
```

Structure-aware chunking respects these boundaries, producing chunks that
correspond to coherent semantic units defined by the document author rather
than arbitrary token counts.

### Unstructured documents

Some documents lack explicit structure - transcripts, scanned text, informal
writing. These require learned or statistical boundary detection:

```
Sentence boundaries:     spaCy, NLTK sentence tokenizers
Topic boundaries:        semantic similarity-based segmentation (TextTiling)
                        embedding shift detection
Paragraph heuristics:    blank lines, indentation changes, sentence
                        patterns (short sentences often end paragraphs)
```

The absence of explicit structure does not mean there is no structure - it
means the structure must be inferred rather than read from markup.

## The Chunking Landscape

The subsequent notes in this module cover the full range of chunking strategies
with increasing sophistication:

```
02-fixed-size-chunking.md
  Splitting by fixed token or character count
  The simplest approach and the most common default
  When it works, when it fails, how to tune it

03-semantic-chunking.md
  Splitting at points of semantic shift detected by embeddings
  Produces semantically coherent chunks at variable sizes
  More expensive to compute but better chunk quality

04-recursive-chunking.md
  Hierarchically splits using multiple separators
  Respects document structure by trying natural boundaries first
  LangChain's default approach, practical balance

05-document-aware-chunking.md
  Structure-aware chunking that reads document format signals
  Heading-aware, table-aware, code-block-aware
  Highest quality for structured documents

06-late-chunking-vs-early-chunking.md
  Encoding first, chunking later (Jina's late chunking)
  Preserves cross-chunk context in embeddings
  Connects to 06-neural-ir/07-late-chunking.md

07-chunking-evaluation.md
  How to measure the impact of chunking choices
  Test collections, metrics, and evaluation protocols
```

## Principles That Apply Across All Chunking Strategies

Regardless of which specific chunking approach is used, several principles hold:

**Principle 1 - Chunks should be retrievable independently**
Each chunk must contain enough context to be useful when retrieved in isolation.
A retriever surfaces individual chunks; the LLM does not have access to the chunks
before and after the retrieved one (unless explicitly added to the context window).

**Principle 2 - Chunks should be embeddable meaningfully**
The content of each chunk should express a coherent semantic unit. If the embedding
model cannot understand what a chunk is about, neither can the retriever.

**Principle 3 - Chunk size should match query specificity**
Short, specific queries benefit from smaller chunks (high information density).
Broad conceptual queries benefit from larger chunks (sufficient context).

**Principle 4 - Chunk boundaries should not break semantic units**
Sentences, named entities, claims, and arguments should not be split mid-unit.
A sentence about "the relationship between X and Y" should appear in one chunk.

**Principle 5 - Chunks should not span topic boundaries unnecessarily**
A chunk discussing two unrelated topics produces an averaged embedding that
represents neither topic well.

**Principle 6 - Evaluate before deploying**
Measure Recall@K on a representative query set with labeled chunks. Do not
assume that default chunk parameters are optimal for your specific corpus and
query distribution.

## My Summary

Chunking divides long documents into smaller units for individual embedding and
retrieval, addressing the fundamental information compression problem: fixed-size
embedding vectors cannot meaningfully represent arbitrarily long documents without
severe specificity loss. The core quality drivers are chunk size (too large creates
semantic diffusion, too small creates context loss), chunk boundary placement (split
mid-sentence or mid-argument causes coherence failures), and chunk-answer alignment
(the answer must fit within a single chunk to be retrievable). The embedding model's
token limit creates an additional constraint - BERT-family models silently truncate
at 512 tokens, making chunks longer than 400 words systematically biased toward
their early content. Document structure (headings, paragraphs, section breaks)
provides natural chunk boundaries that should be exploited rather than ignored.
Chunking strategy is almost never treated as a hyperparameter to be tuned on held-out
evaluation data in practice, despite having a larger impact on end-to-end RAG quality
than many other choices that receive more attention. The subsequent notes cover
the full spectrum from fixed-size splitting (simplest, most common) through semantic
boundary detection (best quality, most expensive) to evaluation protocols that enable
principled strategy selection.
