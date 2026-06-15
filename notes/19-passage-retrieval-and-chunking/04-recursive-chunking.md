# Recursive Chunking

Recursive chunking is a chunking strategy that splits text using a prioritized
hierarchy of separators, attempting the most semantically meaningful separator
first and recursively falling back to less meaningful separators only when
necessary to satisfy a maximum chunk size constraint. Rather than choosing
between "respect document structure" (semantic chunking) and "guarantee size
limits" (fixed-size chunking), recursive chunking does both: it tries to split
on paragraph breaks first, and only if a resulting paragraph still exceeds the
size limit does it fall back to splitting on sentence breaks, and only if a
sentence still exceeds the limit does it fall back to splitting on words, and
so on. This is the default chunking strategy in LangChain (RecursiveCharacterTextSplitter)
and represents the most widely deployed chunking approach in production RAG
systems - a pragmatic middle ground that respects natural document structure
when possible while guaranteeing no chunk exceeds the configured maximum size.

## Intuition

Imagine you are asked to divide a book into sections of at most 500 words each,
while preserving as much natural structure as possible. Your natural approach
would not be to count exactly 500 words and cut - it would be to look for chapter
breaks first. If a chapter is under 500 words, keep it whole. If a chapter exceeds
500 words, look for section breaks within it. If a section still exceeds 500 words,
look for paragraph breaks. If a paragraph (rare, but possible) exceeds 500 words,
look for sentence breaks. Only in the rare case where a single sentence exceeds
500 words would you resort to cutting at an arbitrary word position.

This is exactly the recursive chunking algorithm. It encodes the intuition that
document structure provides a natural hierarchy of meaningful boundaries -
sections contain paragraphs, paragraphs contain sentences, sentences contain
words - and that splitting should happen at the highest level of this hierarchy
that satisfies the size constraint. The "recursive" name comes from the algorithm's
structure: it recursively applies the same splitting logic to oversized chunks,
descending through the separator hierarchy until all pieces fit within the
size limit.

The key insight that distinguishes recursive chunking from semantic chunking:
recursive chunking uses structural signals (where do paragraphs/sentences/words
begin and end) rather than semantic signals (where does the topic shift). It is
much cheaper to compute - no embeddings required - while still producing
meaningfully better boundaries than naive fixed-size chunking.

## The Separator Hierarchy

The core of recursive chunking is an ordered list of separators, from most to
least semantically meaningful:

```
Default separator hierarchy (LangChain RecursiveCharacterTextSplitter):
  1. "\n\n"  (double newline - paragraph boundary)
  2. "\n"    (single newline - line boundary)
  3. " "     (space - word boundary)
  4. ""      (empty string - character boundary, last resort)
```

The algorithm processes text by attempting to split on the first separator in
the hierarchy. Each resulting piece is checked against the size limit. Pieces
that fit are kept as-is. Pieces that exceed the limit are recursively split
using the next separator in the hierarchy.

### Step-by-step algorithm

```
function recursive_split(text, separators, chunk_size):
    if length(text) <= chunk_size:
        return [text]

    if separators is empty:
        # Last resort: hard split at chunk_size
        return hard_split(text, chunk_size)

    separator = separators[0]
    remaining_separators = separators[1:]

    pieces = text.split(separator)

    result = []
    current_chunk = ""

    for piece in pieces:
        if length(current_chunk + separator + piece) <= chunk_size:
            current_chunk += separator + piece
        else:
            if current_chunk:
                result.append(current_chunk)
            if length(piece) > chunk_size:
                # Piece itself is too large - recurse with next separator
                result.extend(recursive_split(piece, remaining_separators, chunk_size))
                current_chunk = ""
            else:
                current_chunk = piece

    if current_chunk:
        result.append(current_chunk)

    return result
```

The algorithm has two distinct behaviors:

**Merging:** When a separator produces many small pieces (e.g., many short
paragraphs), adjacent pieces are merged together up to the size limit -
maximizing chunk utilization without exceeding limits.

**Recursing:** When a separator produces a piece that is itself too large
(e.g., one very long paragraph), that piece is recursively split using the
next separator in the hierarchy.

## Worked Example

Consider a document with this structure:

```
Section A is a short paragraph. It contains two sentences.

Section B is a much longer paragraph that goes on at considerable length,
discussing many different aspects of the topic in a single continuous block
of text without any internal paragraph breaks, making it difficult to split
at the paragraph level alone because the entire paragraph exceeds our chunk
size limit of 100 characters by a significant margin.

Section C is short.
```

With chunk_size = 100 characters and the default separator hierarchy:

```
Step 1: Split on "\n\n" (paragraph boundary)
  Piece 1: "Section A is a short paragraph. It contains two sentences." (60 chars)
  Piece 2: "Section B is a much longer paragraph..." (320 chars)
  Piece 3: "Section C is short." (19 chars)

Step 2: Process each piece against chunk_size = 100

  Piece 1 (60 chars) ≤ 100: fits
  Try merging with Piece 3: 60 + 2 (separator) + 19 = 81 ≤ 100
    → tentatively merge, but Piece 2 comes between them in document order
    → Piece 1 becomes its own chunk (cannot skip Piece 2)

  Piece 2 (320 chars) > 100: too large, recurse with next separator "\n"
    → Piece 2 has no internal "\n" (single paragraph, no line breaks)
    → recurse again with " " (space)
    → split on spaces, merge words up to 100 chars per chunk
    → produces 4 chunks of approximately 80-100 chars each

  Piece 3 (19 chars) ≤ 100: fits
    → too small alone, but it's the last piece - becomes its own chunk
    (or could be merged backward with the last sub-chunk of Piece 2
     if the implementation supports cross-piece merging)

Final chunks:
  Chunk 1: "Section A is a short paragraph. It contains two sentences." (60 chars)
  Chunk 2: "Section B is a much longer paragraph that goes on at considerable
            length, discussing many" (~95 chars)
  Chunk 3: "different aspects of the topic in a single continuous block of
            text without any internal" (~95 chars)
  Chunk 4: "paragraph breaks, making it difficult to split at the paragraph
            level alone because the" (~95 chars)
  Chunk 5: "entire paragraph exceeds our chunk size limit of 100 characters
            by a significant margin." (~90 chars)
  Chunk 6: "Section C is short." (19 chars)
```

Note that Section A (a complete, coherent paragraph) remains intact as Chunk 1
because it fits within the size limit - recursive chunking did not need to
descend the separator hierarchy for it. Section B, which exceeds the limit,
gets recursively split - first attempting line breaks (none present), then
falling back to word boundaries, producing chunks that respect word boundaries
even though paragraph and sentence structure could not be preserved for this
oversized paragraph.

## Why the Hierarchy Order Matters

The order of separators in the hierarchy encodes assumptions about which
boundaries are more likely to correspond to semantic units:

```
"\n\n" (paragraph): Authors use paragraph breaks to separate distinct ideas.
                    A paragraph is usually a self-contained unit of thought.
                    Highest priority - most likely to align with topic boundaries.

"\n" (line):        In structured text (lists, code, poetry), line breaks
                    separate distinct items even within a "paragraph."
                    Second priority - useful when paragraph breaks are absent
                    or paragraphs are too large.

". " (sentence):    A sentence is a complete grammatical unit, usually
                    expressing one claim or idea.
                    Useful fallback when paragraphs lack internal line breaks.

" " (word):         Word boundaries preserve readability (no broken words)
                    but provide no semantic structure.
                    Near-last resort - guarantees readable text but no
                    meaningful boundary alignment.

"" (character):     Absolute last resort. Only used if a single "word"
                    (e.g., a very long URL, hash, or non-whitespace string)
                    exceeds the chunk size.
```

The hierarchy represents a falling scale of "semantic meaningfulness per
boundary type" - and the algorithm always tries to use the most meaningful
boundary that satisfies the size constraint.

## Customizing the Separator Hierarchy

The default hierarchy (paragraph, line, word, character) is designed for
generic prose. Different document types benefit from customized hierarchies
that reflect their specific structural conventions.

### Markdown documents

```
Markdown-aware hierarchy:
  1. "\n## "    (level-2 heading - major section boundary)
  2. "\n### "   (level-3 heading - subsection boundary)
  3. "\n\n"     (paragraph boundary)
  4. "\n"       (line boundary - useful for lists)
  5. ". "       (sentence boundary)
  6. " "        (word boundary)
  7. ""         (character, last resort)
```

By placing heading markers at the top of the hierarchy, recursive chunking
respects the document's authored structure - major sections are kept separate
from each other even if individually they could fit in a larger chunk, because
the algorithm tries to split at headings first and only merges across heading
boundaries if absolutely necessary (which, with headings at the top of the
hierarchy, essentially never happens unless a single section vastly exceeds
the size limit).

### Code files

```
Python code hierarchy:
  1. "\nclass "    (class definition boundary)
  2. "\ndef "      (function definition boundary)
  3. "\n\n"        (blank line - often separates logical blocks)
  4. "\n"          (line boundary)
  5. " "           (word/token boundary, last resort for code)
```

Code-aware hierarchies attempt to keep function and class definitions intact
as single chunks when possible, falling back to line-level splitting only for
very long functions. Note that even this customization is fragile for code -
this is why dedicated code-aware chunking (covered in the next note on
document-aware chunking) typically uses AST-based parsing rather than
separator-based recursion for code files.

### HTML documents

```
HTML-aware hierarchy (after HTML-to-text conversion):
  1. "\n\n\n"   (multiple line breaks - often indicates major section breaks
                 after HTML tag stripping)
  2. "\n\n"     (paragraph boundary)
  3. "\n"       (line boundary, e.g., from <br> or list items)
  4. ". "       (sentence boundary)
  5. " "        (word boundary)
```

For HTML, the separator hierarchy is typically applied after converting HTML
to plain text (stripping tags), since the recursive character splitter operates
on text, not markup. True structure-aware HTML chunking parses the DOM directly
(covered in the document-aware chunking note).

## Recursive Chunking with Overlap

Recursive chunking can be combined with overlap, similar to fixed-size chunking,
but the overlap interacts with the separator-based merging logic in a way that
requires care:

```
With chunk_size = 200, chunk_overlap = 50:

After recursive splitting produces a sequence of chunks:
  Chunk 1: [text ending at position 195]
  Chunk 2: should start ~50 characters before position 195
           → ideally at a natural boundary near position 145

Implementations typically:
  1. Perform recursive splitting to get base chunks
  2. For each chunk after the first, prepend the last `overlap` characters
     of the previous chunk
  3. Optionally, trim the prepended overlap to the nearest separator boundary
     within the overlap region (avoiding mid-word overlap starts)
```

The overlap with recursive chunking is generally less critical than with
fixed-size chunking, because recursive chunking's structure-respecting
boundaries already reduce the frequency of mid-thought splits. Overlap of
0-10% is often sufficient with recursive chunking, compared to the 15-25%
commonly recommended for pure fixed-size chunking.

## Recursive Chunking vs Pure Fixed-Size: What Changes

The practical difference between recursive chunking and pure fixed-size
chunking (with sentence-boundary preference, as described in note 02) can
seem subtle, but the hierarchical fallback produces systematically different
results:

```
Pure fixed-size (sentence-boundary preference):
  - Tries to find a sentence boundary near the target size
  - If no sentence boundary is "near enough," may still cut mid-sentence
  - Operates with a single level of "preferred boundary"

Recursive chunking:
  - Tries paragraph boundaries first across the ENTIRE document
  - Only descends to sentence-level for paragraphs that don't fit
  - Never cuts mid-sentence unless even single sentences exceed the limit
  - Operates with multiple ordered levels of "preferred boundary,"
    applied recursively rather than just "nearest acceptable point"
```

The practical consequence: with recursive chunking, short paragraphs remain
completely intact as single chunks (possibly merged with adjacent short
paragraphs), while long paragraphs get cleanly divided at sentence boundaries.
Pure fixed-size chunking with sentence-boundary preference might still split
a short paragraph awkwardly if the target chunk size falls partway through it
and merging logic is absent.

## Chunk Size Variability in Recursive Chunking

Unlike fixed-size chunking, where every chunk (except possibly the last) is
exactly chunk_size, recursive chunking produces chunks with sizes ranging from
very small to chunk_size:

```
Distribution of chunk sizes with recursive chunking, chunk_size = 500:
  Many chunks: 400-500 tokens (paragraphs that nearly fill the limit,
               or merged short paragraphs)
  Some chunks: 100-400 tokens (individual paragraphs of moderate length)
  Few chunks:  10-100 tokens (very short paragraphs - e.g., a single-line
               heading or a short transitional sentence that didn't merge
               with neighbors due to document ordering constraints)
```

This variability is generally smaller than with full semantic chunking
(where chunks could range from 20 to 2000+ tokens), but larger than pure
fixed-size chunking (where nearly all chunks are exactly chunk_size).

### Handling very small chunks

Recursive chunking can produce undesirably small chunks - for example, a
single-sentence paragraph that sits between two longer paragraphs and cannot
be merged with either without exceeding the size limit:

```
Paragraph A (480 tokens) | Single sentence (15 tokens) | Paragraph B (470 tokens)

With chunk_size = 500:
  Paragraph A: 480 tokens - fits, becomes its own chunk
  Single sentence: 15 tokens - cannot merge with A (480+15=495, fits actually)
                    or with B (15+470=485, fits actually)
    → depends on implementation: greedy merging might attach it to
      whichever neighbor is processed first
```

Most implementations use greedy left-to-right merging: small pieces merge
with the preceding chunk if there is room, otherwise they become their own
(small) chunk or merge with the following chunk. The specific behavior depends
on implementation details and can produce edge cases worth spot-checking
during evaluation.

## When Recursive Chunking Is the Right Default

Recursive chunking represents a strong default choice for several reasons:

```
Advantages:
  ✓ No embedding computation required (fast, cheap, no GPU needed)
  ✓ Respects document structure when present (paragraphs, lines)
  ✓ Guarantees a maximum chunk size (compatible with any embedding model limit)
  ✓ Customizable separator hierarchy for different document types
  ✓ Simple to implement, debug, and reason about
  ✓ Deterministic - same input always produces same output

Limitations:
  ✗ Structural separators (paragraphs, lines) may not align with semantic
    topic boundaries - a long paragraph covering three distinct ideas still
    gets treated as one unit until it exceeds chunk_size
  ✗ Does not detect semantic topic shifts within a single paragraph
  ✗ For documents with poor structural formatting (no paragraph breaks,
    walls of text), recursive chunking degrades toward word-level splitting,
    losing most of its advantage over fixed-size
  ✗ For highly structured documents (tables, code, nested lists), separator-
    based splitting can still produce awkward breaks that document-aware
    chunking would avoid
```

### Decision guide

```
Use recursive chunking when:
  - Documents have reasonable paragraph/section structure
  - No GPU/embedding budget available for chunking-time computation
  - A simple, fast, deterministic baseline is needed
  - Document types are heterogeneous (customize separator hierarchy per type)

Move to semantic chunking when:
  - Documents lack clear structural markers (transcripts, OCR output,
    walls of text)
  - Paragraphs frequently contain multiple distinct topics
  - Evaluation shows recursive chunking's structural boundaries don't
    align with semantic boundaries for your corpus

Move to document-aware chunking when:
  - Documents have rich structure beyond plain text (tables, code blocks,
    nested headings, forms)
  - Separator-based splitting produces broken tables or code blocks
  - Document format (PDF, HTML, DOCX) provides structural metadata that
    plain-text separators cannot capture
```

## Recursive Chunking as a Foundation for Other Strategies

Recursive chunking is rarely used in complete isolation in sophisticated
pipelines - it frequently serves as the fallback mechanism for other strategies:

```
Semantic chunking + recursive fallback:
  Semantic boundaries determine primary chunk divisions
  If a semantic chunk exceeds max_size: apply recursive chunking within it

Document-aware chunking + recursive fallback:
  Document structure (headings, sections) determines primary divisions
  If a section exceeds max_size: apply recursive chunking within the section

Late chunking + recursive boundaries:
  The document is encoded with full context first (late chunking)
  Recursive separator boundaries determine where to "cut" the
  resulting token-level representations into retrievable chunks
```

This compositional pattern - using recursive chunking as the size-safety
mechanism within a higher-level strategy - is extremely common. Recursive
chunking's guarantee of respecting a maximum size while preferring meaningful
boundaries makes it an ideal "inner loop" for more sophisticated outer
strategies that determine primary chunk boundaries through other means.

## My Summary

Recursive chunking splits text using a prioritized hierarchy of separators -
typically paragraph breaks, then line breaks, then sentence breaks, then word
breaks, then character breaks as a last resort - recursively descending the
hierarchy only for pieces that exceed the maximum chunk size. This produces
chunks that respect natural document structure (short, coherent paragraphs
remain intact, possibly merged with neighbors) while guaranteeing no chunk
exceeds the configured size limit (long paragraphs get cleanly subdivided at
the most meaningful available boundary). It is the default strategy in LangChain's
RecursiveCharacterTextSplitter and the most widely deployed chunking approach in
production because it requires no embedding computation, respects structure
when present, and provides hard size guarantees compatible with any embedding
model's token limit. The separator hierarchy is customizable per document type -
Markdown documents benefit from heading-level separators at the top of the
hierarchy, code files from class/function boundaries, and HTML from line-break
patterns after tag stripping. Recursive chunking produces variable chunk sizes
(unlike pure fixed-size) but with less variance than full semantic chunking,
since structural boundaries (paragraphs, sentences) provide intermediate-granularity
splitting points. Its main limitation is that structural separators do not always
align with semantic topic boundaries - a long paragraph covering multiple distinct
ideas is treated as one unit until it exceeds the size limit. In practice, recursive
chunking frequently serves as the size-safety fallback mechanism within more
sophisticated strategies like semantic or document-aware chunking, applied only
to oversized chunks that the primary strategy could not subdivide meaningfully.
