# Late Chunking

Late chunking is a document embedding technique introduced by Jina AI in 2024 that
produces chunk-level embeddings while preserving the full document context. In
standard chunking, a long document is split into smaller passages and each passage
is embedded independently losing all context from surrounding text. Late chunking
instead runs the entire document through the transformer encoder first, then pools
the resulting token embeddings into chunk representations. Each chunk's embedding
reflects the context of the whole document rather than the isolated passage alone.

## Intuition

Standard chunking has a fundamental problem it is amnesiac. When you split a
document into 256-token chunks and embed each independently, chunk 3 has no idea
what was discussed in chunk 1. Consider this document:

```bash
Chunk 1: "Alan Turing was a British mathematician and computer scientist."
Chunk 2: "He developed the theoretical foundations of modern computing."
Chunk 3: "His work on codebreaking during World War II was pivotal."
```

Embedded independently:

- Chunk 2 contains "He" with no reference to who "He" is
- Chunk 3 contains "His work" with no reference to whose work
- Both embeddings are impoverished because the pronoun resolution is lost

A query for "Turing contributions to computing" should match chunk 2, but the
embedding of chunk 2 does not know "He" refers to Turing. The match is weakened.

Late chunking fixes this by running the full document through BERT first. BERT's
self-attention resolves "He" in chunk 2 to "Alan Turing" from chunk 1 before any
chunking occurs. The token embeddings at positions corresponding to chunk 2 already
encode the Turing context. Pooling these token embeddings produces a chunk
embedding that reflects both the local text and the broader document context.

## Standard Chunking vs Late Chunking

### Standard chunking pipeline

```bash
Document
    ↓
Split into chunks: [chunk1, chunk2, chunk3, ...]
    ↓
For each chunk independently:
    tokens = tokenize(chunk)
    embedding = encoder(tokens)   ← only sees chunk tokens
    ↓
chunk_embeddings = [emb1, emb2, emb3, ...]
```

Each chunk is encoded in isolation. Cross-chunk context is completely lost.

### Late chunking pipeline

```bash
Document
    ↓
Tokenize entire document:
    tokens = tokenize(document)   ← full document tokens
    ↓
Encode entire document:
    all_token_embeddings = encoder(tokens)   ← all tokens attend to all tokens
    ↓
Map tokens back to original chunk boundaries
    ↓
For each chunk: pool its token embeddings
    chunk_embedding_i = mean_pool(all_token_embeddings[chunk_i_start:chunk_i_end])
    ↓
chunk_embeddings = [emb1, emb2, emb3, ...]
```

The pooling step happens after encoding late pooling hence "late chunking."
Each chunk embedding benefits from the full document context established during
encoding.

## Why Self-Attention Makes This Work

Transformer self-attention is the mechanism that makes late chunking valuable.
In standard BERT attention, every token attends to every other token in the
input sequence:

```bash
Token "He" (position 20) attends to:
  "Alan"    (position 1)  → attention weight: 0.31
  "Turing"  (position 2)  → attention weight: 0.28
  "British" (position 4)  → attention weight: 0.05
  "He"      (position 20) → attention weight: 0.12
  ...
```

By the time all transformer layers have processed, the representation of "He"
at position 20 incorporates information from "Alan Turing" at positions 1-2.
This cross-token information flow is what resolves coreference, maintains topic
coherence, and allows later chunks to carry forward context from earlier ones.

When you pool the token embeddings for chunk 2 after this full-document encoding,
the pooled vector already contains "who is He" information that standard
per-chunk encoding completely misses.

## The Token Limit Problem

Late chunking requires encoding the full document in a single forward pass. This
immediately runs into BERT's 512-token input limit. For longer documents this is
a hard constraint.

Solutions:

### Long-context models

Models like Longformer (4096 tokens), BigBird (4096 tokens), and Jina Embeddings
v2 (8192 tokens) extend the context window. Late chunking with an 8192-token
model can handle documents of roughly 6000 words sufficient for most articles,
papers, and documentation pages.

### Sliding window approach

Process the document in overlapping windows, apply late chunking within each
window, and stitch the chunk embeddings together:

```bash
Window 1: tokens[0:512]   → late chunk embeddings for this window
Window 2: tokens[256:768] → late chunk embeddings (256-token overlap)
Window 3: tokens[512:1024] → ...
```

The overlap ensures boundary chunks have sufficient context. Less elegant than
full-document encoding but handles arbitrarily long documents.

### Hierarchical late chunking

Split documents into sections first, apply late chunking within each section.
Works well for structured documents (research papers, books) where section
boundaries correspond to natural topic shifts.

## When Late Chunking Helps Most

Late chunking provides the largest benefit for documents with:

### Coreference and anaphora

```bash
"Dr. Smith published a landmark paper. She argued that..."
                                       ↑ late chunking resolves "She"
```

### Entity-dense text

```bash
"The company was founded in 1994. Its CEO later went on to..."
                                   ↑ "Its" resolved to the company
```

### Structured technical documents

```bash
"The function takes three parameters. The first parameter controls..."
                                         ↑ "first parameter" needs function context
```

### Academic papers

```bash
Abstract: "We propose a new method FLAIR for semantic retrieval."
Section 2: "FLAIR uses a dual-encoder architecture..."
Section 3: "We evaluated it on BEIR..."
                               ↑ "it" = FLAIR from abstract
```

Late chunking provides less benefit for:

- Documents where chunks are already self-contained (news articles with
  independent paragraphs, FAQ documents)
- Very short documents where standard chunking already captures full context
- Documents with little coreference or cross-chunk dependency

## Late Chunking vs Contextual Retrieval

Anthropic's Contextual Retrieval (2024) addresses the same problem with a
different approach prepending chunk-specific context generated by an LLM:

```bash
Standard chunk: "The revenue increased by 15% year-over-year."

Contextual chunk: "This chunk is from the Q3 2024 earnings report for Apple Inc.
                   discussing iPhone revenue. The revenue increased by 15%
                   year-over-year."
```

| Property               | Late Chunking             | Contextual Retrieval          |
| ---------------------- | ------------------------- | ----------------------------- |
| Mechanism              | Full-doc transformer attn | LLM-generated context prefix  |
| Computation cost       | One encoder forward pass  | One LLM call per chunk        |
| Context quality        | Implicit, from attention  | Explicit, from LLM            |
| Requires LLM           | No                        | Yes                           |
| Works with any encoder | Yes (if long-ctx support) | Yes                           |
| Chunk storage          | Standard                  | Larger (context prefix added) |
| Best for               | Coreference, pronouns     | Missing entity context        |

Both approaches improve retrieval quality over standard chunking. They are
complementary you can apply both (generate contextual prefix + late chunk
the contextualized document).

## Late Chunking in a RAG Pipeline

Late chunking replaces the embedding step in a standard RAG indexing pipeline:

```bash
Standard RAG indexing:
  Document → split → [chunk1, chunk2, ...] → encode each → store embeddings

Late chunking RAG indexing:
  Document → encode full doc → pool by chunk boundaries → store embeddings

Query time (identical for both):
  Query → encode → ANN search → retrieve chunks → assemble context → LLM
```

The retrieval and generation stages are unchanged. Only the indexing step
differs. This makes late chunking a drop-in improvement for existing RAG systems.

## Late Chunking in Practice When to Use It

| Scenario                            | Use late chunking?                    |
| ----------------------------------- | ------------------------------------- |
| Technical documentation             | Yes lots of coreference               |
| Research papers                     | Yes entity-dense, cross-section refs  |
| Books / long narratives             | Yes characters, themes span chapters  |
| News articles                       | Maybe often self-contained paragraphs |
| FAQ documents                       | No questions are independent          |
| Product descriptions                | No usually self-contained chunks      |
| Short documents (< 512 tokens)      | No standard chunking works fine       |
| Real-time indexing (speed critical) | Maybe adds latency per document       |

## Performance Comparison

Empirical results from the Jina AI late chunking paper on retrieval benchmarks:

| Dataset      | Standard chunking | Late chunking | Improvement |
| ------------ | ----------------- | ------------- | ----------- |
| SummScreenFD | 0.421             | 0.468         | +4.7 points |
| QASPER       | 0.312             | 0.341         | +2.9 points |
| NarrativeQA  | 0.189             | 0.218         | +2.9 points |
| QuALITY      | 0.445             | 0.471         | +2.6 points |

Improvements are largest on datasets with long documents containing significant
cross-chunk context dependencies exactly the use cases late chunking is designed
for.

Late chunking sits at the intersection of chunking strategy (Phase 3 indexing
concepts) and neural encoding (Phase 6). It is the practical answer to the
question every RAG practitioner eventually asks: why does my system struggle
with documents that have pronouns and cross-references? Standard chunking is
the answer late chunking is the fix.

## My Summary

Late chunking encodes the full document through a transformer before splitting into
chunks, allowing self-attention to resolve coreference and preserve cross-chunk
context. Each chunk embedding is produced by mean-pooling the corresponding token
embeddings from the full-document encoding rather than encoding the chunk in
isolation. The technique addresses the core weakness of standard RAG chunking
that pronouns, co-references, and context-dependent phrases lose their meaning
when extracted from surrounding text. Late chunking works best for long technical
documents, research papers, and narratives where chunks are interdependent.
It requires a long-context encoder model to handle full documents beyond 512 tokens.
The technique is a drop-in improvement for RAG indexing pipelines only the
embedding step changes, query time retrieval remains identical.
