# Fixed-Size Chunking

Fixed-size chunking is the practice of dividing documents into chunks of a
predetermined maximum size - typically measured in tokens, characters, or words -
optionally with an overlap between consecutive chunks so that content near chunk
boundaries appears in multiple chunks. It is the simplest, fastest, and most
widely deployed chunking strategy: no linguistic analysis, no document parsing,
no embedding computation during chunking - just a sliding window across the text.
Despite its simplicity, fixed-size chunking is the default in most RAG frameworks
(LangChain, LlamaIndex, Haystack), and for many document types and query
distributions it produces acceptable retrieval quality. Understanding its
mechanics, failure modes, and tuning parameters is foundational before considering
more sophisticated alternatives, because the cost of a bad fixed-size chunking
configuration is often larger than the benefit of switching to a more complex
strategy with better default parameters.

## Intuition

Fixed-size chunking is the typist's approach to document splitting: count to N
characters (or tokens), cut, count again, cut again. It makes no judgment about
whether the cut falls at the end of a sentence, between a claim and its
justification, or in the middle of a table. It simply enforces a maximum size
and moves on.

The appeal is its predictability. Every chunk is at most N tokens. Memory usage,
embedding time, and retrieval behavior are all predictable. No chunk will exceed
the embedding model's context window. No chunk will be 50x larger than another.
The simplicity makes fixed-size chunking easy to implement, easy to reason about,
and easy to debug.

The weakness is its indifference to content. The same fixed window applied to a
philosophy essay, a Python code file, an HTML page, and a legal contract will
produce chunks of uniform size regardless of whether those chunks make semantic
sense. The assumption that "N tokens" corresponds to a coherent information unit
is often false - some N-token windows span three complete thoughts, others land
in the middle of one.

Understanding when this indifference matters and when it does not is the core
of fixed-size chunking wisdom.

## The Two Parameters: Size and Overlap

Fixed-size chunking has exactly two tuning parameters. Everything else is a
consequence of these choices.

### Chunk size

The maximum size of each chunk, typically measured in tokens for embedding-model
compatibility:

```
Chunk size controls:
  Information density per chunk:  small → high density, large → low density
  Context availability:           small → less context, large → more context
  Embedding specificity:          small → specific, large → general
  Number of chunks:               small → many, large → few
  Retrieval granularity:          small → precise, large → coarse
```

The chunk size determines what kind of information need each chunk can serve:

```
50-100 tokens (~35-75 words):
  Appropriate for: specific factual questions, precise code snippets, formulas
  Concern:         often too little context, sentence fragments common

100-200 tokens (~75-150 words):
  Appropriate for: most factual retrieval, short explanations
  Sweet spot for: precise question answering

200-400 tokens (~150-300 words):
  Appropriate for: conceptual explanations, procedure descriptions
  Sweet spot for: general knowledge retrieval

400-600 tokens (~300-450 words):
  Approaching embedding model token limit (512 for BERT-family)
  Appropriate for: comprehensive topic coverage
  Risk:           content near token 512 may be truncated silently

600+ tokens:
  Requires long-context embedding models (E5-mistral, jina-v2)
  Appropriate for: context-heavy retrieval where breadth matters
  Risk:           semantic diffusion at very long lengths
```

### Overlap

The number of tokens shared between consecutive chunks:

```
Chunk 1: [token 1 ... token 512]
Chunk 2: [token 385 ... token 896]   (128 token overlap)
Chunk 3: [token 769 ... token 1280]  (128 token overlap)

Overlap fraction = overlap / chunk_size
  128 / 512 ≈ 25% overlap (common default)
```

Overlap addresses the boundary problem: without overlap, content near chunk
boundaries appears in only one chunk and may be split mid-sentence or
mid-argument. With overlap, boundary content appears in two adjacent chunks,
improving the probability that it is retrievable.

**The overlap-redundancy tradeoff:**

```
No overlap (0%):
  Advantages: No redundant content, smaller total index size
  Risk:       Content at boundaries may be incomplete or unsatisfying

Low overlap (10-15%):
  Provides minimal boundary coverage
  Small index size increase
  Good when documents have strong paragraph structure

Standard overlap (20-25%):
  The most common default
  Reasonable boundary coverage
  ~25% increase in index size

High overlap (40-50%):
  Strong boundary coverage
  ~100% increase in index size (doubled chunks)
  Risk: Many near-duplicate chunks pollute retrieval results
        Same content retrieved multiple times reduces diversity
```

## How Fixed-Size Chunking Splits Text

The mechanics of splitting affect quality as much as the size and overlap
parameters:

### Naive character splitting

The simplest possible implementation: split every N characters:

```
Text: "The cat sat on the mat. The dog sat on the rug."
N = 20 characters

Chunk 1: "The cat sat on the m"
Chunk 2: "at. The dog sat on t"
Chunk 3: "he rug."
```

Naive character splitting creates fragments mid-word and mid-sentence.
Essentially never appropriate for retrieval. Mentioned only to identify
what to avoid.

### Token-based splitting with sentence boundary preference

A substantial improvement: split at token N, but prefer to split at the
nearest sentence boundary within the last M tokens:

```
Text: [sentence_1] [sentence_2] [sentence_3] [sentence_4] ...
Target: 256 tokens

Process:
  1. Tokenize text
  2. Find the 256-token position
  3. Search backward for a sentence boundary within the last 50 tokens
  4. If found: split there (some chunks may be < 256 tokens)
  5. If not found: split at 256 tokens (hard split)
```

This produces chunks that almost always start and end at sentence boundaries,
dramatically improving coherence compared to hard character or token splits.
This is what LangChain's RecursiveCharacterTextSplitter does internally
when no separator hierarchy is provided.

### Whitespace-preferring splits

For documents without clear sentence structure (code, data files, logs):

```
Priority of split points:
  1. Double newline (\n\n): paragraph boundary
  2. Single newline (\n): line boundary
  3. Period followed by space: sentence boundary
  4. Comma followed by space: clause boundary
  5. Space: word boundary
  6. Any character: hard split (last resort)
```

This hierarchy ensures splits happen at the most natural available boundary.

## Tuning Chunk Size for Specific Document Types

Different document types have different optimal chunk sizes because they have
different natural unit sizes and information density:

### Technical documentation

```
Structure: clearly delineated sections, subsections, code examples
Typical information unit: procedure (3-6 steps, 100-200 words)
Recommended chunk size: 200-300 tokens
Rationale: Captures one procedure or concept completely
Note:      Identify code blocks separately - chunk code differently than prose
```

### Scientific papers

```
Structure: abstract, introduction, methods, results, discussion
Typical information unit: paragraph making one argument (100-200 words)
Recommended chunk size: 150-250 tokens
Rationale: Each paragraph in a scientific paper typically makes one claim
Note:      Abstract and conclusion often deserve their own chunks (high information density)
```

### Legal documents

```
Structure: sections, subsections, clauses, subclauses
Typical information unit: clause (variable, 50-500 words)
Challenge: Legal meaning often depends on clause hierarchy context
Recommended chunk size: 200-400 tokens with metadata indicating parent section
Rationale: Individual clauses need context of parent section to be interpreted correctly
```

### Conversational transcripts

```
Structure: turns (speaker, utterance)
Typical information unit: exchange (question + answer, 2-5 turns)
Challenge: Single utterances often lack context; full transcripts are too long
Recommended chunk size: 200-400 tokens centered on speaker turns
Rationale: Preserves question-answer structure that defines meaning
```

### Code files

```
Structure: functions, classes, methods, comments
Typical information unit: function (5-50 lines)
Challenge: Fixed token chunking splits functions mid-body
Recommended chunk size: function-aware chunking (not true fixed-size)
If fixed-size is required: 300-400 tokens, no overlap (overlap creates duplicate code)
Note:      For code, structure-aware chunking is strongly preferred over fixed-size
```

### Product descriptions / e-commerce

```
Structure: title, attributes, description, specifications
Typical information unit: complete product description (50-200 words)
Challenge: Very short documents may not need chunking
Recommended: Index entire description as one chunk if < 300 tokens
             If > 300 tokens: split by attribute section (title, specs, description separate)
```

## The Overlap Failure Mode: Redundant Retrieval

A frequently overlooked problem with high overlap: the same information can
appear in many chunks, and all of them may score highly for the same query,
causing multiple nearly-identical chunks to be retrieved:

```
Document: "Information retrieval is the process of obtaining relevant information
          from a collection. The most basic IR systems use keyword matching. Modern
          systems use neural embeddings to capture semantic similarity."

With 50% overlap and chunk_size=50 tokens:
  Chunk 1: "Information retrieval is the process of obtaining relevant information"
  Chunk 2: "of obtaining relevant information from a collection. The most basic"
  Chunk 3: "from a collection. The most basic IR systems use keyword matching."
  Chunk 4: "IR systems use keyword matching. Modern systems use neural embeddings"
  Chunk 5: "Modern systems use neural embeddings to capture semantic similarity."

Query: "what is information retrieval"

Chunk 1: high similarity (directly defines IR)
Chunk 2: high similarity (contains definition fragment)
Chunk 3: moderate similarity (mentions IR)
Chunk 4: moderate similarity (mentions IR systems)

Top-4 retrieved: Chunks 1, 2, 3, 4 - nearly redundant
```

High overlap causes the retrieval results list to be dominated by adjacent
chunks from the same document. The LLM receives four versions of the same
content rather than four diverse relevant passages. This reduces both retrieval
diversity and generation quality.

### Deduplication as a mitigation

Post-retrieval deduplication removes near-duplicate chunks from the retrieved
set before passing to the LLM:

```
For each pair of retrieved chunks:
  if cosine_similarity(chunk_i_embedding, chunk_j_embedding) > threshold:
    keep the chunk with higher query similarity, discard the other

Threshold: 0.85-0.95 depending on acceptable similarity level
```

Deduplication adds a small overhead but substantially improves the diversity
of the retrieved context when high overlap is used.

## Chunking and the Embedding Model Token Limit

The interaction between chunk size and the embedding model's maximum token
length is one of the most commonly misconfigured aspects of RAG systems:

### The silent truncation problem

Most BERT-family embedding models (all-MiniLM, BGE, E5-base, GTE-base) have
a 512-token maximum. When text exceeds 512 tokens, the excess is silently
truncated - the model processes only the first 512 tokens without warning.

```
Chunk size = 1000 tokens (characters, not tokens - common confusion)
1000 characters ≈ 200-300 tokens for English text

In this case, 1000 characters fits within 512 tokens.
No truncation.

Chunk size = 1000 tokens (actual tokens - uncommon but should be explicit)
All content beyond token 512 is silently discarded.
The embedding only represents the first 512 tokens.
```

A common mistake: configuring chunk_size in characters (e.g., 1000 characters)
without realizing that 1000 characters equals approximately 200-250 tokens
for English, well within the 512-token limit. The system appears to work
correctly for English text but fails silently for languages with different
character-to-token ratios.

Another common mistake: configuring chunk_size in words (e.g., 200 words)
without realizing that tokenization can produce 1.2-1.5 tokens per word,
making "200 words" approximately 240-300 tokens - still within limits but
creating variable effective chunk sizes across different text styles.

### Matching chunk size to model token limit

```
Embedding model        Max tokens    Safe chunk size (80% of limit)
─────────────────────────────────────────────────────────────────────────
all-MiniLM-L6-v2       512           ~400 tokens
BAAI/bge-base-en       512           ~400 tokens
intfloat/e5-base-v2    512           ~400 tokens
text-embedding-3-small 8191          ~6500 tokens
jina-embeddings-v2     8192          ~6500 tokens
intfloat/e5-mistral    4096          ~3200 tokens
```

The 80% rule leaves headroom for the model's internal representation
(special tokens like [CLS] and [SEP] consume 2-3 positions) and prevents
edge cases where slightly larger chunks due to tokenization variation
cause unexpected truncation.

## Chunk Size and Retrieval Architecture Interaction

Chunk size affects not just embedding quality but also how the retrieval
pipeline works downstream:

### Effect on reranking

Rerankers (cross-encoders) have their own token limit. The combined length
of query + chunk must fit within the cross-encoder's context:

```
Cross-encoder max: 512 tokens
Query length: 20 tokens
Available for chunk: 512 - 20 - 3 (special tokens) = 489 tokens

If chunks are 400 tokens: fine, fits within cross-encoder limit
If chunks are 600 tokens: cross-encoder must truncate the chunk
→ Reranker loses the end of every chunk during scoring
→ Reranking quality degrades for content in the latter part of chunks
```

Chunk size must be coordinated with reranker context limits, not just
embedding model limits.

### Effect on LLM context window

The LLM that generates answers from retrieved chunks has its own context limit.
With k retrieved chunks of size S tokens each, the total context is:

```
Total context = k × S + query + system_prompt + generation_budget

Example:
  k = 5 chunks, S = 400 tokens, query = 50 tokens, system = 200 tokens, generation = 500
  Total = 5 × 400 + 50 + 200 + 500 = 2,750 tokens → fits in most LLMs

Example with small chunk size:
  k = 15 chunks, S = 100 tokens, query = 50 tokens
  Total = 15 × 100 + 50 + 200 + 500 = 2,250 tokens → fits, and more diverse context

Example with large chunks:
  k = 5 chunks, S = 1000 tokens, query = 50 tokens
  Total = 5 × 1000 + 50 + 200 + 500 = 5,750 tokens → requires larger context window
```

Smaller chunks enable retrieving more diverse context within the same LLM
context budget. Larger chunks provide more complete information per retrieved
item but at higher context cost.

## Common Defaults and When They Work

Most RAG frameworks use specific default values for chunk size and overlap.
Understanding what these defaults optimize for helps in deciding when to
accept them and when to override them:

### LangChain default

```
chunk_size = 1000 (characters)
chunk_overlap = 200 (characters)

1000 characters ≈ 150-200 tokens (English)
200 characters overlap ≈ 30-40 tokens
Overlap fraction ≈ 20%

Optimized for: general-purpose English text, moderate-length documents
Works well for: web pages, articles, general knowledge bases
Suboptimal for: very long documents (may need smaller chunks for specificity),
                code (character splitting breaks code structure),
                short documents (may not need chunking at all)
```

### LlamaIndex default

```
chunk_size = 1024 (tokens)
chunk_overlap = 20 (tokens)

1024 tokens exceeds BERT-family limits → intended for long-context models
20 token overlap ≈ 2% (very low)

Optimized for: long-context embedding models, comprehensive retrieval
Works well for: models with 4K+ token limits, complex documents
Suboptimal for: BERT-family embedding models (silent truncation at 512)
```

### OpenAI recommended

```
chunk_size = 256-512 tokens
chunk_overlap = 10-20% of chunk_size

Optimized for: text-embedding-3 family (8191 token limit)
Works well for: precise factual retrieval, QA applications
```

## Fixed-Size Chunking in the Decision Hierarchy

Fixed-size chunking is the starting point, not the endpoint:

### When fixed-size is sufficient

```
Document type: plain text articles, web pages, unstructured notes
Query type: factual questions, keyword-heavy searches
Quality requirement: acceptable (not demanding production quality)
Team size: small, needs simple maintainable solution
Iteration speed: fast, willing to tune parameters on evaluation feedback
```

Fixed-size chunking with sentence-boundary preference, 200-300 token chunks,
and 15-20% overlap will perform adequately for most straightforward retrieval
applications.

### When to move beyond fixed-size

```
Document type: PDFs with complex layouts, code files, structured documents with tables
Query type: complex multi-hop, context-dependent, comparative
Quality requirement: production-grade, user-facing with high expectations
Evaluation shows: Recall@10 < 0.70 with best fixed-size parameters
```

If evaluating fixed-size chunking on a representative query set shows poor
recall, and tuning size and overlap does not improve it sufficiently, the
problem is likely semantic in nature - content is being split across boundaries
in ways that fixed-size cannot detect. Semantic or structure-aware chunking
will typically improve this.

## The Evaluation Imperative

The single most important practice for fixed-size chunking is evaluation:

```
Minimum evaluation protocol:
  1. Select 50-100 representative queries with known relevant documents
  2. Chunk the corpus with your chosen parameters
  3. Measure Recall@10, Recall@100
  4. Vary chunk_size over [100, 200, 300, 400] tokens
  5. Vary overlap over [0%, 10%, 20%, 30%]
  6. Select the configuration with highest Recall@100 on the query set
  7. Re-measure Recall on a held-out test set to confirm

Common finding:
  The default parameters almost never produce the best retrieval quality
  for a specific corpus and query distribution
  A 2-hour evaluation sweep often finds 15-25% Recall improvement over defaults
```

This evaluation takes a few hours of engineering time and is almost always
worth it. Most teams skip this step, deploying with defaults and attributing
the resulting mediocre quality to the embedding model or LLM.

## My Summary

Fixed-size chunking splits documents into chunks of a predetermined maximum token
or character count, optionally with overlap between consecutive chunks. It is the
default in LangChain, LlamaIndex, and most RAG frameworks due to its simplicity
and predictability. The two tuning parameters - chunk size and overlap - have large
effects on retrieval quality: chunk size controls the information density and
context availability of each chunk (200-300 tokens is the typical sweet spot for
BERT-family models), while overlap (10-25%) reduces boundary fragmentation at the
cost of redundant content. The most critical practical consideration is aligning
chunk size with the embedding model's token limit - BERT-family models silently
truncate at 512 tokens, making chunks longer than 400 tokens systematically biased
toward early content. High overlap (>30%) creates redundant retrieval results where
multiple adjacent chunks from the same passage are retrieved for the same query,
reducing diversity and generation quality. Fixed-size chunking works well for
unstructured plain-text documents with factual retrieval requirements and small
teams needing maintainable solutions. It underperforms for structured documents
(PDFs with tables, code files), complex queries requiring broad context, and
applications demanding production-grade retrieval quality without evaluation-driven
parameter tuning. The single most important practice is evaluating multiple
chunk-size and overlap configurations on a representative query set before deployment -
a 2-hour evaluation sweep almost always finds 15-25% Recall improvement over defaults.
