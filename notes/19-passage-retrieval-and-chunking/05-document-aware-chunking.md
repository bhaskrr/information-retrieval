# Document-Aware Chunking

Document-aware chunking is a family of chunking strategies that parse the
specific structural and formatting conventions of a document's native format -
PDF, HTML, Markdown, DOCX, code files, spreadsheets - to produce chunks that
respect the semantic organization the document author encoded in its structure.
Where fixed-size and recursive chunking treat documents as flat streams of text
and where semantic chunking treats them as streams of sentences to compare,
document-aware chunking reads the document as a structured object: headings
define section hierarchies, tables are discrete data units, code blocks are
syntactic structures, list items are parallel semantic units, footnotes relate
to specific claims. By parsing these structural signals rather than inferring
them from text alone, document-aware chunking can produce chunks that correspond
to the document's own organizational units - a single function definition, a
complete table, a headed subsection - rather than arbitrary size-constrained
windows across content that happens to have been flattened to plain text.

## Intuition

Consider a technical reference manual in PDF format. A human reading this manual
naturally navigates it through its structure: the table of contents names
sections; each section has a heading and self-contained content; tables present
structured data as units; code blocks show complete examples that must be read
together; figures have captions that provide their interpretation.

When this manual is chunked using fixed-size or recursive approaches on the
extracted plain text, something unfortunate happens: the extraction process
destroys most of the structural information. A table becomes a sequence of
numbers separated by whitespace. A heading becomes a short line that might get
merged with the paragraph below it or cut off from it. A code block might be
split across two chunks at an arbitrary line. A bulleted list might be split
mid-way through its items.

The retrieval system then tries to find relevant chunks by embedding these
structurally damaged fragments. A query about "the performance benchmarks for
System X" might need to retrieve a table of benchmark results, but the table
has been shredded into three fixed-size chunks of numbers without column headers
repeated - none of which is retrievable in isolation.

Document-aware chunking works from the other direction: before extraction and
splitting, it identifies the structural elements in the original format and uses
them to define chunk boundaries and chunk composition. A table becomes one chunk
(with its header row repeated if necessary). A code block becomes one chunk. A
headed section becomes one chunk (or multiple structured sub-chunks if it exceeds
the size limit). The embedding of each chunk represents the content of a
meaningful structural unit, not an arbitrary slice of flattened text.

## Why Format-Specific Parsing Matters

### The structure loss problem in text extraction

The fundamental problem is that most retrieval pipelines convert documents to
plain text as their first processing step, losing structural information that
cannot be recovered:

```
Original PDF: structured table with headers and data cells
Plain text extraction: "Metric System A System B BM25 0.45 0.38 NDCG@10
                        0.52 0.47 Latency 12ms 8ms"

Fixed-size chunk: "Metric System A System B BM25 0.45 0.38 NDCG@10"
                  (table split mid-row, headers separated from data)

Document-aware chunk:
  "Metric     | System A | System B
   BM25       | 0.45     | 0.38
   NDCG@10    | 0.52     | 0.47
   Latency    | 12ms     | 8ms"
  (complete table preserved as single chunk with consistent formatting)
```

The structural loss is not recoverable from the extracted text alone. Once
the original format signals are discarded, no amount of clever separator
hierarchies or semantic similarity detection can reconstruct which numbers
belonged to which column or which rows were part of the same table.

### The structured element problem

Many document elements have specific semantics that are only meaningful as
complete units:

**Tables:** Individual rows or cells are not retrievable in isolation -
understanding a data cell requires knowing its column header. A query
about "NDCG@10 for System A" cannot be answered by a chunk containing
only "0.52" without the header context "System A" and the metric label.

**Code blocks:** A function call is not interpretable without the function
definition context. An import statement is not useful without the code it
enables. Code examples are designed to be complete working units.

**Equations:** A mathematical expression mid-derivation is not independently
meaningful. The complete derivation from premise to conclusion is the unit.

**Footnotes and citations:** A footnote number in isolation is meaningless.
The footnote text connected to the location it annotates is the coherent unit.

**List items:** A bulleted or numbered list presents parallel items at the
same semantic level. Splitting a list mid-way leaves orphaned items without
the context of the full list structure.

## PDF Chunking

PDF is the most challenging common document format because PDFs are designed
for printing, not for content extraction. A PDF encodes character positions on
a page, not semantic structure - there is no explicit indication of where
paragraphs begin or end, where headings are, or how columns relate to each other.

### The PDF extraction challenge

```
PDF content representation:
  Sequence of drawing instructions: "draw character 'H' at (72, 720)"
  No explicit paragraph markers, section markers, or heading markers
  Column layout creates interleaved text when extracted linearly

Multi-column PDF extraction naive approach:
  Column 1, line 1: "The quick brown fox"
  Column 2, line 1: "jumped over the lazy"
  Column 1, line 2: "dog."
  Column 2, line 2: "cat sat on the mat."

  Naive extraction: "The quick brown foxjumped over the lazydog.cat sat on the mat."
  Column-aware extraction: "The quick brown fox dog. [COLUMN 2] jumped over the lazy
                             cat sat on the mat."
```

Effective PDF extraction requires heuristic or ML-based layout analysis to
identify columns, headings, tables, figures, and reading order before text
can be extracted in a meaningful sequence.

### PDF extraction quality spectrum

```
Low quality (most common in practice):
  Simple text extraction (pdfminer, PyPDF2 basic mode)
  Produces: interleaved column text, broken tables, missing structure
  Impact: chunking on this output produces structurally incoherent chunks

Medium quality:
  Layout-aware extraction with column detection
  Produces: correct reading order, paragraph detection, basic table extraction
  Tools: pdfplumber, PyMuPDF with layout analysis
  Impact: significantly better chunk quality than naive extraction

High quality:
  Document AI models with structure understanding
  Produces: headings identified and labeled, tables structured, reading order correct
  Tools: Azure Document Intelligence, AWS Textract, Google Document AI
  Impact: enables section-based chunking close to what the document author intended

Multimodal approach:
  Render each page as an image, send to vision-language model for structured extraction
  Produces: best structural understanding, handles unusual layouts
  Tools: GPT-4V, Claude with vision, LlamaParse
  Impact: highest quality extraction but significant cost per document
```

### PDF chunking strategies by document type

**Dense reference manuals and textbooks:**
These typically have clear heading hierarchies, numbered sections, and cross-
references. High-quality extraction preserving heading levels enables
section-based chunking:

```
Identified structure:
  Chapter 3: Information Retrieval (heading level 1)
    Section 3.1: Boolean Retrieval (heading level 2)
      [content: 400 tokens - keep as single chunk]
    Section 3.2: Vector Space Models (heading level 2)
      [content: 1200 tokens - too large]
      Subsection 3.2.1: TF-IDF Weighting (heading level 3)
        [content: 380 tokens - keep as single chunk]
      Subsection 3.2.2: Cosine Similarity (heading level 3)
        [content: 290 tokens - keep as single chunk]
```

**Academic papers:**
Key structural elements are abstract (always a distinct chunk), sections by heading,
and figures/tables (keep caption with table/figure content):

```
Priority chunks:
  Abstract: always a standalone chunk (high-density summary)
  Introduction, Methods, Results, Discussion: section-level chunks
  Each table: complete table + caption as one chunk
  Each figure: caption + any inline description (image sent separately if multimodal)
```

**Financial reports and forms:**
Highly structured with specific sections, tables of figures, and standardized layouts:

```
Structured elements:
  Income statement table: complete table as single chunk
  Balance sheet: complete table as single chunk
  Notes to financial statements: one chunk per note
  Management discussion: paragraph-level chunks
```

## HTML Chunking

HTML provides explicit semantic structure through its tag hierarchy. Unlike
PDFs, HTML documents encode structure directly in the markup - headings, sections,
tables, lists, and code blocks are all explicitly labeled.

### DOM-based chunking

Parse the HTML into a Document Object Model (DOM) tree and traverse it to
identify chunk boundaries:

```
HTML structure:
  <article>
    <h1>Introduction to Retrieval</h1>
    <p>Information retrieval is the process...</p>
    <p>Modern systems use neural encoders...</p>
    <h2>BM25</h2>
    <p>BM25 is a probabilistic ranking function...</p>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>k1</td><td>1.2</td></tr>
      <tr><td>b</td><td>0.75</td></tr>
    </table>
    <h2>Dense Retrieval</h2>
    <p>Dense retrieval uses bi-encoders...</p>
  </article>

DOM-based chunks:
  Chunk 1: [h1 text + following paragraphs until next heading]
           "Introduction to Retrieval\n\nInformation retrieval is the process...
            Modern systems use neural encoders..."
  Chunk 2: [h2 + following content until next h2]
           "BM25\n\nBM25 is a probabilistic ranking function..."
  Chunk 3: [table as structured unit]
           "Parameter | Value\nk1 | 1.2\nb | 0.75"
  Chunk 4: [h2 + following content]
           "Dense Retrieval\n\nDense retrieval uses bi-encoders..."
```

### Handling HTML tables

HTML tables present specific chunking challenges - cells reference column headers
that appear only in the first row:

```
Original HTML table:
  <table>
    <tr><th>Model</th><th>NDCG@10</th><th>Latency</th></tr>
    <tr><td>BM25</td><td>0.45</td><td>2ms</td></tr>
    <tr><td>E5-base</td><td>0.52</td><td>30ms</td></tr>
    <tr><td>SPLADE</td><td>0.48</td><td>15ms</td></tr>
  </table>

Single-chunk approach (table fits within size limit):
  All rows as one chunk with headers included
  Best for: most tables where complete context is needed

Row-expansion approach (table too large for one chunk):
  For each data row, prepend the header row:
  Chunk A: "Model | NDCG@10 | Latency\nBM25 | 0.45 | 2ms"
  Chunk B: "Model | NDCG@10 | Latency\nE5-base | 0.52 | 30ms"
  Chunk C: "Model | NDCG@10 | Latency\nSPLADE | 0.48 | 15ms"

  Each row is independently retrievable with full column context
```

### Handling nested HTML structure

Complex HTML pages may have deeply nested structures (navigation menus,
sidebars, footers) mixed with the main content. Document-aware chunking
should identify and exclude boilerplate structure:

```
Identify main content:
  Look for <main>, <article>, or the largest <div> by text content
  Exclude: <nav>, <header>, <footer>, <aside>, <script>, <style>

Within main content:
  Chunk by heading hierarchy as above
  Tables: keep as single chunks or row-expanded if oversized
  Code blocks (<pre>, <code>): keep as single chunks regardless of size
  Lists (<ul>, <ol>): keep complete list as single chunk if fits;
                      split at list item boundaries if too large
```

## Markdown Chunking

Markdown documents provide the most explicit structure of any common plain-text
format. The heading hierarchy is encoded directly in # markers, creating an
unambiguous section tree.

### Heading-based section chunking

```
Markdown document:
  # Introduction
  This section introduces the topic...

  ## Background
  Prior work has shown...

  ### Key Contributions
  Our contributions are:
  1. First contribution
  2. Second contribution

  ## Methods
  We propose a new approach...

Heading hierarchy:
  # Introduction (depth 1)
    ## Background (depth 2)
      ### Key Contributions (depth 3)
    ## Methods (depth 2)
```

Strategy: form chunks at a specified heading depth level. Chunks at depth 2
(h2 sections) include the h2 heading and all content until the next h2 or h1.

### Inline code and code blocks

Markdown code blocks (fenced with ``` or indented) should always be preserved
as single chunks, regardless of size:

````
```python
def retrieve(query, index, k=10):
    query_embedding = encoder.encode(query)
    scores, indices = index.search(query_embedding, k)
    return [(docs[i], scores[i]) for i in indices[0]]
```
````

Splitting this code block mid-function produces a syntactically invalid,
non-retrievable fragment. Even if the function is 300 lines and exceeds the
chunk size, it should be kept as a single chunk (with metadata indicating it
is a code block) or split at function-level boundaries using code-aware parsing.

### Front matter handling

Many Markdown documents (Jekyll, Hugo, Sphinx) contain YAML or TOML front
matter with metadata:

```
---
title: "Dense Retrieval Overview"
author: "Jane Smith"
date: "2024-01-15"
tags: ["retrieval", "embedding", "IR"]
---

# Introduction
...
```

Front matter should be extracted as metadata and attached to all chunks from
the document rather than included as a separate text chunk - the YAML is not
useful as a retrieved passage but is valuable as document-level context.

## Code File Chunking

Code files have a fundamentally different structure from prose documents.
The semantic unit in code is the function or class definition, not the paragraph.
Separator-based recursive chunking is inadequate for code because:

```
Separator-based problem:
  "\n\n" splits at blank lines between functions - often works for simple files
  "\n" splits at individual lines - loses function context

  Problem: blank lines appear inside functions too (separating logical blocks)
           and the separator-based approach cannot distinguish between
           blank line within a function and blank line between functions
```

### AST-based code chunking

Abstract Syntax Tree (AST) parsing provides precise function and class boundaries:

```
Python file structure (via AST):
  Import statements (lines 1-5)
  Class definition: DataProcessor (lines 7-45)
    __init__ method (lines 8-15)
    process method (lines 17-30)
    helper method (lines 32-45)
  Function definition: main (lines 47-60)
  if __name__ == "__main__" block (lines 62-65)

AST-based chunks:
  Chunk 1: Import statements + docstring
  Chunk 2: Complete DataProcessor class (if fits) or each method as separate chunk
  Chunk 3: main function
  Chunk 4: __main__ block
```

When a class exceeds the chunk size, AST chunking splits at method boundaries -
each method is a complete, independently callable unit. This is the lowest level
of semantically meaningful splitting for code.

### Code chunking with signature context

A key challenge: functions reference class attributes and other functions defined
elsewhere in the file. To make each code chunk independently interpretable,
prepend function signatures and class context:

```
Method chunk with context:
  "class DataProcessor:
     def __init__(self, corpus): ...
     def process(self, query): ...  # (signature only)

   def process(self, query):
     '''Full method implementation here...'''
     [complete method body]"
```

By including the class signature and abbreviated signatures of sibling methods,
the chunk provides enough context for an LLM to understand the code's structure
without requiring access to the full class definition.

## DOCX Chunking

Microsoft Word documents encode structure through styles - paragraphs are tagged
with style names like "Heading 1", "Heading 2", "Body Text", "Caption", "Code".
Document-aware chunking reads these style tags rather than inferring structure
from the text content.

```
DOCX structure via styles:
  Paragraph style="Heading 1": "Chapter 3: Retrieval Systems"
  Paragraph style="Heading 2": "Section 3.1: Boolean Retrieval"
  Paragraph style="Body Text": "Boolean retrieval systems..."
  Paragraph style="Body Text": "The simplest form of boolean..."
  Paragraph style="Heading 2": "Section 3.2: Vector Space Models"
  Table: comparison of retrieval methods
    Table Caption: "Table 3.1: Comparison of retrieval methods"

Document-aware chunks:
  Chunk 1: [H1 + H2 "Section 3.1" + following Body Text until next H2]
  Chunk 2: [H2 "Section 3.2" + Table + Caption as structured unit]
```

Style-based DOCX chunking is highly reliable when the document was created with
consistent style application - the most common source of chunking quality issues
in DOCX is inconsistent style application (body text formatted with Heading styles,
headings formatted as bold body text rather than using the Heading style).

## Chunk Metadata for Document-Aware Chunking

A key benefit of document-aware chunking that is often underutilized: structural
information can be preserved as chunk metadata, enabling metadata-filtered retrieval:

```
Chunk metadata example:
  {
    "chunk_id": "doc_001_chunk_15",
    "document_title": "Information Retrieval Textbook",
    "document_type": "PDF",
    "section_path": ["Chapter 3", "Section 3.2", "Subsection 3.2.1"],
    "section_heading": "TF-IDF Weighting",
    "chunk_type": "body_text",
    "page_range": [47, 48],
    "contains_table": False,
    "contains_code": False,
    "word_count": 245,
    "parent_chunk_id": "doc_001_chunk_14"   # preceding section heading chunk
  }
```

This metadata enables:

- **Filtered retrieval:** "Only search within Chapter 3" → filter by section_path
- **Contextual augmentation:** When a chunk is retrieved, automatically prepend
  the section heading for context
- **Citation generation:** The page_range enables precise document citations
- **Structural navigation:** parent_chunk_id enables retrieving adjacent chunks
  when a query needs broader context

## The Hierarchical Chunk Strategy

Document-aware chunking naturally enables a hierarchical indexing strategy:
maintain chunks at multiple levels of granularity from the same document and
retrieve at the appropriate level for each query:

```
Levels of granularity from a technical document:
  Level 1 - Document:   Single embedding per document (or abstract)
                        Use for: broad topic matching, document discovery
  Level 2 - Section:    One chunk per major section (h2 level)
                        Use for: topic-level retrieval
  Level 3 - Subsection: One chunk per subsection (h3 level)
                        Use for: specific concept retrieval
  Level 4 - Paragraph:  One chunk per paragraph
                        Use for: precise factual retrieval
```

At query time, the retrieval system can use the most appropriate granularity:
broad queries retrieve at section level, specific factual queries retrieve at
paragraph level, and the document title/abstract provides the coarse-grained
matching layer.

This hierarchical structure also enables the "small-to-big retrieval" pattern:
retrieve small precise chunks (paragraph level) for initial matching, then
expand the context by fetching the parent section chunk for LLM generation.

## Quality and Cost Tradeoffs

Document-aware chunking requires significantly more infrastructure than simpler
approaches:

```
Fixed-size chunking:
  Infrastructure: none (string splitting)
  Cost: negligible
  Time per document: milliseconds

Recursive chunking:
  Infrastructure: none (string splitting with separator hierarchy)
  Cost: negligible
  Time per document: milliseconds

Semantic chunking:
  Infrastructure: embedding model
  Cost: proportional to document length × embedding cost
  Time per document: seconds (CPU) to sub-second (GPU)

Document-aware chunking:
  Infrastructure: format-specific parsers, possibly OCR/vision models
  Cost: varies widely by approach (PDF extraction: seconds; Document AI: dollars)
  Time per document: seconds (heuristic) to minutes (AI-based extraction)
```

For PDFs requiring AI-based extraction (complex layouts, scanned documents),
the cost can be $0.01-0.10 per page with commercial APIs. For a corpus of
100,000 PDF pages, this is $1,000-$10,000 just for extraction - a significant
budget item that must be justified by corresponding retrieval quality improvement.

### When the investment is justified

```
High value:
  Legal documents: structure is legally meaningful, incorrect chunking
                   could cause wrong case law retrieval with serious consequences
  Medical documents: clinical decision support requires precise section retrieval
  Technical manuals: incorrect code examples or specification tables are harmful
  Financial reports: table integrity is critical for accurate data retrieval

Lower value:
  News articles: relatively flat structure, recursive chunking performs well
  Social media: no structure to preserve
  Short documents (<500 words): structure is simple, any strategy works
  Documents that will be read in full by the LLM: chunking quality matters less
                                                   when the whole document fits in context
```

## My Summary

Document-aware chunking parses the native format structure of documents - PDF,
HTML, Markdown, DOCX, code - to produce chunks that correspond to the semantic
organizational units the document author encoded in the structure rather than
arbitrary size windows across flattened text. The core insight is that format
signals carry semantic information that separator-based and semantic-similarity
approaches cannot recover after text extraction: a HTML table tag explicitly
marks a tabular data unit, a Markdown heading marker explicitly identifies
section boundaries, a Python AST node explicitly identifies function boundaries.
PDF is the most challenging format because PDFs encode character positions rather
than semantic structure, requiring heuristic or AI-based layout analysis to
recover headings, tables, and reading order. HTML provides the richest explicit
structure for chunking, enabling DOM-based section and table extraction. Code
files require AST-based parsing to identify function and class boundaries rather
than the separator-based approaches appropriate for prose. Document-aware
chunking produces structured metadata (section hierarchy, page range, element type)
as a side effect, enabling metadata-filtered retrieval and hierarchical indexing
that supports "small-to-big" retrieval patterns. The investment in format-specific
parsing is justified for high-value corpora where structural integrity matters -
legal, medical, technical - and where incorrect chunking of tables, code, or
clinical data would cause harmful retrieval failures. For simpler corpora, the
additional cost of format-specific parsing may not be justified over well-tuned
recursive chunking.
