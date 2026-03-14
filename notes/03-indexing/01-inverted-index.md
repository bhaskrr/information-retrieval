# Inverted Index

An inverted index is a data structure that maps each unique term in a corpus to the list of documents that contain it. It is called "inverted" because it inverts the natural document -> words direction into terms -> documents.

It is the core data structure behind virtually every search engine ever built.

## Intuition

Imagine you have 1 million documents and a user queries "python". Without an index, you would scan every document for the word "python", that is linear search over the entire corpus, which is unacceptably slow.

An inverted index flips this: at index build time, you do the expensive work once. At query time, you just look up "python" in the index and instantly get the list of documents that contain it. Query time becomes a dictionary lookup instead of a full corpus scan.

## Anatomy of an Inverted Index

```bash
Term          Postings List
─────────────────────────────────────────
"python"   -> [doc3, doc7, doc12, doc19]
"java"     -> [doc1, doc3, doc8]
"search"   -> [doc3, doc5, doc12]
```

Each entry in a postings list is called a **posting**. At minimum it stores the document ID. In richer implementations it also stores:

- **Term frequency (TF)** - how many times the term appears in that document
- **Positions** - which word positions the term occupies (needed for phrase queries)

```bash
"python" -> [(doc3, tf=2, pos=[4,17]), (doc7, tf=1, pos=[2]), ...]
```

## Building an Inverted Index - Step by Step

Given these three documents:

- D1: "the cat sat on the mat"
- D2: "the cat and the dog played"
- D3: "the dog sat quietly"

### Step 1 - Text processing

Tokenize, normalize, remove stopwords:

- D1: [cat, sat, mat]
- D2: [cat, dog, played]
- D3: [dog, sat, quietly]

### Step 2 - Build term -> doc pairs

```bash
(cat, D1), (sat, D1), (mat, D1)
(cat, D2), (dog, D2), (played, D2)
(dog, D3), (sat, D3), (quietly, D3)
```

### Step 3 - Sort by term

```bash
(cat, D1), (cat, D2)
(dog, D2), (dog, D3)
(mat, D1)
(played, D2)
(quietly, D3)
(sat, D1), (sat, D3)
```

### Step 4 - Merge into postings lists

```bash
cat     → [D1, D2]
dog     → [D2, D3]
mat     → [D1]
played  → [D2]
quietly → [D3]
sat     → [D1, D3]
```

This is our inverted index.

## Querying the Index

### AND query — "cat AND dog"

Fetch postings lists for both terms and intersect:

- cat -> [D1, D2]
- dog -> [D2, D3]
- result -> [D2] <- only D2 contains both

Intersection is efficient when postings lists are sorted by doc ID - use a merge
algorithm similar to merge sort, O(m + n) where m and n are list lengths.

### OR query - "cat OR dog"

Union of both lists:

- result -> [D1, D2, D3]

### NOT query - "cat AND NOT dog"

- cat -> [D1, D2]
- NOT dog -> all docs - [D2, D3] = [D1]
- intersection -> [D1]

NOT requires knowing the full set of document IDs.

### Optimization - process shortest list first

For AND queries, always start with the term that has the smallest postings list. This
minimizes the number of comparisons at each merge step.

Query: "cat AND mat AND dog"

- mat -> [D1] <- shortest, start here
- cat -> [D1, D2]
- dog -> [D2, D3]
- mat ∩ cat = [D1], then [D1] ∩ dog = [] -> done early

## What this index does not handle yet

- **Phrase queries**: "cat sat" requires positional information, not just doc IDs
- **Ranking**: all matching documents are treated equally; no scoring
- **Scale**: for millions of documents, this in-memory approach breaks down
- **Updates**: adding new documents requires rebuilding or maintaining a separate
  delta index

## Tradeoffs

| Property    | Detail                                                |
| ----------- | ----------------------------------------------------- |
| Query speed | Very fast, O(1) lookup + O(m+n) merge                 |
| Build time  | Expensive upfront, done offline                       |
| Storage     | Postings lists can be large; compression helps        |
| Flexibility | Boolean queries are easy; ranking requires extensions |

## My Summary

An inverted index maps terms to the documents that contain them, turning a slow full-corpus scan into a fast dictionary lookup. It is built once at index time by processing all documents into (term, doc_id) pairs, sorting them, and merging into postings lists. Query processing becomes set operations - AND is intersection, OR is union, NOT is complement. Almost everything in IR is built on top of this structure.
