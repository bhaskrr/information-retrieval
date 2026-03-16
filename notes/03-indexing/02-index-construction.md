# Index Construction

Index construction is the process of building an inverted index over a large document collection. The naive approach, holding everything in memory, breaks down at scale.  
Index construction algorithms are designed to handle corpora that are too large to fit in RAM by using disk-based sorting and merging strategies.

## Why does it matter?

The web has billions of documents. Even a modest enterprise corpus can have millions.
Building an inverted index over this requires careful management of memory, disk I/O,
and merge operations. The algorithms here are what make large-scale search engines
possible.

## The Scale Problem

Recall the naive build process from 01-inverted-index.md:

1. Process documents into (term, doc_id) pairs
2. Sort by term
3. Merge into postings lists

This works fine in memory for small corpora. For large ones:

- The full list of (term, doc_id) pairs may be tens of gigabytes
- Sorting that in memory is impossible
- You need algorithms that work in chunks, using disk as overflow

## Algorithm 1 - BSBI (Blocked Sort-Based Indexing)

### Idea

Process the corpus in fixed-size blocks that fit in memory. Build a partial index for
each block, write it to disk, then merge all partial indexes at the end.

### Steps

1. Read documents until a memory block is full
2. Convert to (termID, docID) pairs — use integer IDs, not strings, to save space
3. Sort the pairs by termID in memory
4. Write the sorted block to disk as a partial index
5. Repeat for all blocks
6. Merge all partial indexes from disk into the final index

```bash
Block 1 → partial_index_1.bin
Block 2 → partial_index_2.bin
Block 3 → partial_index_3.bin
         ↓ merge
    final_index.bin
```

### Merge step

Use an n-way merge — open all partial index files simultaneously, always write the
smallest termID next. This is the same as merging k sorted lists.

### Limitation

BSBI requires a global term → termID mapping that must fit in memory. For very large
vocabularies this becomes a bottleneck.

## Algorithm 2 — SPIMI (Single-Pass In-Memory Indexing)

### Idea

Instead of collecting all (term, doc_id) pairs and sorting them, build postings lists
directly as you stream through documents. When memory fills up, write the current
partial index to disk and start fresh. No global term ID dictionary needed.

### Steps

1. Stream documents one by one
2. For each token, look it up in the current in-memory index:
   - If it exists, append the doc_id to its postings list
   - If not, create a new entry
3. When memory is full, sort the current index by term and write to disk
4. Clear memory and continue
5. After all documents, merge all partial indexes

### Why SPIMI is better than BSBI

- No need for a global termID dictionary
- Postings lists are built directly — no need to sort (term, docID) pairs
- Each token is processed exactly once — single pass
- More memory efficient since postings lists grow in place

```bash
Stream docs → build partial index in memory
           → memory full? sort terms, flush to disk
           → repeat
           → merge all partial indexes
```

## Algorithm 3 — Distributed Indexing (MapReduce)

For web-scale corpora (billions of documents), even SPIMI on a single machine is too
slow. The solution is to distribute the work across many machines.

### MapReduce approach

**Map phase** — each machine processes a subset of documents:

```bash
Input:  (docID, document text)
Output: (term, docID) pairs
```

**Reduce phase** — each machine handles a subset of terms:

```bash
Input:  all (term, docID) pairs for a set of terms
Output: term → sorted postings list
```

The key insight: the reduce phase partitions by term, so each reducer independently
builds the postings lists for its assigned terms. No single machine needs to see
the full index.

```bash
Documents → Map workers → (term, docID) pairs
                        → shuffle by term
                        → Reduce workers → partial final index per term range
                        → combine → complete index
```

This is roughly how Google's early indexing pipeline worked.

## Dynamic Indexing — Handling New Documents

The algorithms above assume a static corpus. In practice, new documents arrive
continuously. Two strategies:

### Auxiliary index

- Maintain a small in-memory index for new documents
- At query time, search both the main index and auxiliary index, merge results
- Periodically merge the auxiliary index into the main index

### Logarithmic merge

- Maintain a series of indexes of exponentially increasing size: I₀, I₁, I₂...
- New documents go into the smallest index I₀
- When I₀ fills up, merge it into I₁; when I₁ fills up, merge into I₂, etc.
- Bounds the number of merges while keeping query time manageable

## Comparison

| Algorithm | Memory use  | Disk I/O  | Scalability        | Complexity |
| --------- | ----------- | --------- | ------------------ | ---------- |
| Naive     | High        | None      | Small corpora only | Low        |
| BSBI      | Bounded     | High      | Single machine     | Medium     |
| SPIMI     | Bounded     | High      | Single machine     | Medium     |
| MapReduce | Distributed | Very high | Web scale          | High       |

## My Summary

Naive in-memory index construction fails at scale because the corpus doesn't fit in RAM.
BSBI and SPIMI solve this by processing documents in chunks, flushing partial indexes to disk, and merging at the end. SPIMI is preferred because it avoids a global term ID dictionary and builds postings lists directly. At web scale, distributed indexing via MapReduce parallelizes the work across many machines. Real search engines also need dynamic indexing strategies to handle continuously arriving new documents.
