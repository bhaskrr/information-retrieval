# Index Compression

Index compression is the application of encoding techniques to reduce the storage size of an inverted index, specifically the postings lists, without losing any information.
The goal is to make the index smaller so it fits in memory (or closer to it), and faster to read from disk.

## Why does it matter?

Postings lists can be enormous. A term like "the" in a web-scale corpus has a postings list with billions of entries. Even after stopword removal, common terms have postings lists that span gigabytes. Compression reduces storage costs and counterintuitively often speeds up retrieval, because reading fewer bytes from disk is faster than reading more bytes even if decompression takes some CPU time.

## What gets compressed?

Two things in an inverted index are worth compressing separately:

- **The dictionary**: the set of all unique terms
- **The postings lists**: the lists of doc IDs (and frequencies, positions)

Postings list compression is the more impactful of the two.

## Step 1 - Gap Encoding (Delta Encoding)

### The problem

Postings lists store doc IDs as integers. For a term that appears in many documents, these are large numbers:

```bash
"python" -> [183, 3724, 8201, 8203, 19000, 19001]
```

### The idea

Instead of storing absolute doc IDs, store the gaps between consecutive IDs:

```bash
[183, 3724, 8201, 8203, 19000, 19001]
-> gaps: [183, 3541, 4477, 2, 10797, 1]
```

Gaps are almost always smaller than the original IDs, especially for frequent terms where documents are close together in the ID space. Smaller numbers compress better.

### Why it works

For a term appearing in many documents, consecutive doc IDs tend to be close together hence the gaps are small integers, often 1 or 2. Small integers need fewer bits to encode.

## Step 2 - Variable Byte Encoding (VByte)

### The problem

Fixed-width integers waste space. Storing every doc ID as a 4-byte int uses 4 bytes even for the gap value "1", which needs only 1 bit.

### The idea

Use a variable number of bytes per integer. Small numbers use 1 byte, larger ones use 2, 3, or 4 bytes. Encode the integer 7 bits at a time, using the 8th bit as a continuation flag:

- If the 8th bit is 1 → this is the last byte of the integer
- If the 8th bit is 0 → more bytes follow

```bash
Encoding the number 5:
5 in binary: 0000101
VByte:       10000101  ← 8th bit set to 1 (last byte)
= 1 byte

Encoding the number 214577:
214577 in binary: 110 1001100 0110001
Split into 7-bit groups: 0001101  0011000  0110001
VByte (reverse, last gets continuation=1):
  00001101 00011000 10110001
= 3 bytes
```

### Why it is widely used

VByte is simple to implement, fast to decode, and achieves good compression for the
small gap values that are common in postings lists. It is the default compression in
many real IR systems including Lucene (used by Elasticsearch).

## Step 3 - Bitwise Encoding (Gamma and Delta Codes)

For even better compression at the cost of more complexity:

### Unary code

Encode integer n as (n-1) ones followed by a zero.

- 1 → 0
- 2 → 10
- 3 → 110
- 5 → 11110

Compact for very small numbers but grows linearly, bad for large gaps.

### Elias Gamma code

Encode integer n as: unary code of (1 + floor(log₂n)) followed by the binary
representation of n minus the leading bit.

```bash
n=1  → 0
n=2  → 10 0
n=3  → 10 1
n=6  → 110 10
n=9  → 1110 001
```

Gamma codes are optimal when the distribution of gap sizes is unknown but assumed
to follow a power law - which postings lists generally do.

### Elias Delta code

An extension of gamma that handles larger numbers more efficiently. Used less
commonly than gamma in practice.

## Dictionary Compression

The dictionary itself (the set of all unique terms) also takes up space - typically around 500KB–5MB for a large corpus, small relative to postings lists but worth optimizing.

### Front coding

Terms in a sorted dictionary share common prefixes. Store the prefix once and encode only the suffix for each term:

```bash
Without front coding:        With front coding:
"automate"                   8automate
"automatic"                  9automat*ic
"automatically"              *ically
"automation"                 *ion
```

Each entry stores how many characters it shares with the previous term, then the
differing suffix. Achieves 20-30% compression on typical English dictionaries.

## Compression Tradeoffs

| Technique    | Compression ratio | Decode speed | Complexity |
| ------------ | ----------------- | ------------ | ---------- |
| Gap encoding | Low (enabler)     | Very fast    | Very low   |
| VByte        | Medium            | Fast         | Low        |
| Gamma codes  | High              | Slower       | Medium     |
| Delta codes  | Higher            | Slowest      | High       |
| Front coding | Medium (dict)     | Fast         | Low        |

In practice: gap encoding is always applied first, then VByte on top. Gamma/delta
codes are used when storage is the primary constraint and decode speed is secondary.

## Where compression fits in the bigger picture

```bash
Build index (SPIMI/BSBI)
    → apply gap encoding to postings lists
    → apply VByte (or gamma) to compressed gaps
    → apply front coding to dictionary
    → write compressed index to disk

Query time:
    → look up term in compressed dictionary
    → read compressed postings list from disk
    → decode VByte → decode gaps → reconstruct doc IDs
    → proceed with query processing
```

## My Summary

Index compression reduces postings list size by first converting absolute doc IDs to smaller gap values (delta encoding), then encoding those gaps with variable-width schemes like VByte that use fewer bytes for smaller numbers. The result is an index that is 3-10x smaller than uncompressed, fits more easily in memory, and is often faster to retrieve because less data is read from disk. VByte is the practical default - simple, fast, and effective. Gamma codes squeeze more compression out at the cost of slower decoding. Dictionary compression via front coding adds modest savings on top.
