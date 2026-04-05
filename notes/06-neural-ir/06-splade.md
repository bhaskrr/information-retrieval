# SPLADE

SPLADE (Sparse Lexical and Expansion Model) is a neural retrieval model that uses
a BERT encoder to produce learned sparse representations of queries and documents.
Instead of dense vectors like bi-encoders, SPLADE produces high-dimensional sparse
vectors over the vocabulary — similar in structure to BM25's term weights but learned
end-to-end from relevance data. It combines the semantic understanding of neural
models with the efficiency and interpretability of sparse retrieval.

## Intuition

Dense retrieval is semantically powerful but has two practical problems:

- It requires ANN infrastructure (FAISS) for efficient search
- It struggles with exact keyword matching and rare terms
- Its representations are uninterpretable — you cannot see why a document was retrieved

BM25 is efficient and exact-match strong but has one fundamental problem:

- It has no semantic understanding — "car" and "automobile" are unrelated terms

SPLADE asks: what if we could learn term weights the way BERT understands language,
but keep the sparse structure of an inverted index?

The answer is to use BERT's masked language modeling head — the component that
predicts missing words — to generate a weight for every vocabulary term given an
input text. Most weights are zero (sparse). Non-zero weights represent the terms
that BERT considers important for representing the input, including terms that do
not literally appear in the text (query expansion).

This is the key insight: BERT can predict that a document about "heart attack
prevention" is relevant to the term "cardiovascular" even if that exact word is
absent — and SPLADE encodes this as a non-zero weight on "cardiovascular" in the
document's sparse vector.

## From BM25 to SPLADE

BM25 weights terms based on:

- How often the term appears in the document (TF)
- How rare the term is across the corpus (IDF)

SPLADE weights terms based on:

- What BERT predicts is important for representing the document
- Including terms not present in the document (expansion)
- Learned end-to-end from query-document relevance signals

```bash
BM25 document vector:
  {python: 2.1, search: 1.4, index: 0.8, ...}   ← only terms in document

SPLADE document vector:
  {python: 1.8, search: 1.6, index: 0.9,
   retrieval: 0.7, programming: 0.5, ...}         ← includes expanded terms
```

## Architecture

SPLADE uses the BERT MLM (Masked Language Modeling) head to produce sparse vectors:

```bash
Input text → BERT encoder → token representations
                          → MLM head (linear + GELU + layer norm)
                          → logits over full vocabulary (vocab_size ≈ 30,000)
                          → ReLU (set negatives to zero)
                          → log(1 + x) (saturate large values)
                          → max pooling over all token positions
                          → sparse vector over vocabulary
```

### Step by step

1. **BERT encoding** — run input text through BERT, get contextual token
   representations of shape (seq_len, hidden_size)

2. **MLM projection** — project each token representation to vocabulary logits
   of shape (seq_len, vocab_size) using the pretrained MLM head

3. **ReLU activation** — zero out negative logits. Only positive activations
   contribute to the sparse vector.

4. **Log saturation** — apply log(1 + x) to prevent any single term from
   dominating:

```bash
   weight(t) = log(1 + ReLU(logit(t)))
```

5. **Max pooling over tokens** — for each vocabulary term, take the maximum
   weight across all token positions. This gives one weight per vocabulary term
   per input text.

6. **Result** — a sparse vector of dimension vocab_size (~30,000) where most
   entries are zero.

### SPLADE score for a (query, document) pair

```bash
score(q, d) = Σ_t  w_q(t) × w_d(t)
```

This is a dot product between two sparse vectors — identical in structure to
BM25 scoring but with learned weights instead of TF-IDF weights. This means
SPLADE can be evaluated using a standard inverted index — no ANN infrastructure
required.

## Query Expansion — The Key Advantage

SPLADE's most important property is implicit query and document expansion.

Example:

```bash
Query: "python programming"

SPLADE query vector (top terms by weight):
  python: 2.1
  programming: 1.8
  code: 1.4
  language: 1.2
  developer: 0.9
  software: 0.7
  ...

Terms not in query but added by expansion: code, language, developer, software
```

A BM25 index only matches "python" and "programming" exactly. SPLADE also matches
documents containing "code", "language", "developer" — capturing semantically
related documents without changing the inverted index infrastructure.

This is sometimes called **neural term weighting + expansion** — SPLADE does both
simultaneously.

## Training SPLADE

SPLADE is trained with two objectives:

### 1. Ranking loss

Contrastive loss to maximize score for relevant (query, document) pairs over
irrelevant ones:

```bash
L_rank = max(0, margin - score(q, d+) + score(q, d-))
```

Or InfoNCE with in-batch negatives, same as bi-encoder training.

### 2. FLOPS regularization (sparsity constraint)

Without regularization, SPLADE would learn dense vectors — every vocabulary term
gets a non-zero weight and efficiency is lost. A FLOPS (floating point operations)
regularizer penalizes the number of non-zero weights:

```bash
L_FLOPS = λ × Σ_t ( mean_over_batch(w(t)) )²
```

This encourages the model to represent each input with as few non-zero terms as
possible while still ranking relevant documents correctly. The λ hyperparameter
controls the sparsity-performance tradeoff.

Total training loss:

```bash
L = L_rank + λ_q × L_FLOPS(query) + λ_d × L_FLOPS(document)
```

### SPLADE variants

| Variant            | Description                                     |
| ------------------ | ----------------------------------------------- |
| SPLADE             | Original, separate query and document expansion |
| SPLADE-v2          | Improved training with hard negatives           |
| SPLADE++           | Better regularization, stronger performance     |
| SPLADE-CoCondenser | Initialized from CoCondenser pretrained model   |
| distilSPLADE       | Smaller, faster, distilled from SPLADE++        |

## SPLADE vs BM25 vs Dense Retrieval

| Property                | BM25           | Dense (bi-encoder) | SPLADE              |
| ----------------------- | -------------- | ------------------ | ------------------- |
| Representation          | Sparse, TF-IDF | Dense, learned     | Sparse, learned     |
| Query expansion         | No             | Implicit           | Explicit + implicit |
| Vocabulary mismatch     | Fails          | Handles well       | Handles well        |
| Exact keyword match     | Strong         | Weaker             | Strong              |
| Rare terms              | Strong         | Weaker             | Strong              |
| Index type              | Inverted index | ANN (FAISS)        | Inverted index      |
| Infrastructure          | Simple         | Complex            | Simple              |
| Interpretability        | High           | None               | High                |
| Training data needed    | None           | Yes                | Yes                 |
| Out-of-domain           | Robust         | Can degrade        | More robust than BE |
| Performance (in-domain) | Baseline       | Better than BM25   | Better than BM25    |

SPLADE typically outperforms BM25 and matches or slightly exceeds bi-encoders on
in-domain benchmarks, while being more robust out-of-domain than bi-encoders.

## Inverted Index Compatibility

SPLADE's sparse output means it is fully compatible with standard inverted index
infrastructure:

```bash
At index time:
    for each document d:
        splade_vec = SPLADE_encoder(d)    → sparse vector
        for each term t with nonzero weight:
            add (doc_id, weight) to postings list of t

At query time:
    splade_q = SPLADE_encoder(query)      → sparse vector
    for each term t in splade_q:
        fetch postings list of t from inverted index
    compute dot product scores
    rank by score
```

This is identical to BM25 query processing — just with learned weights instead
of TF-IDF weights. Existing search infrastructure (Elasticsearch, Lucene) can
serve SPLADE indices with minor modifications.

## Interpretability

Unlike dense retrieval where vectors are opaque 768-dimensional arrays, SPLADE
vectors are readable:

```bash
Document: "Information retrieval is the task of finding relevant documents."

SPLADE document vector top terms:
  retrieval:    2.34
  information:  2.18
  documents:    1.92
  finding:      1.67
  relevant:     1.45
  search:       1.23    ← not in document, expansion term
  query:        0.98    ← not in document, expansion term
  indexing:     0.76    ← not in document, expansion term
```

You can directly inspect which terms SPLADE considers important and which terms
it expanded to. This interpretability is valuable for debugging and for building
trust in the retrieval system.

## Code

```python
# pip install transformers torch

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from collections import defaultdict


class SPLADEEncoder:
    def __init__(self, model_name: str = "naver/splade-cocondenser-selfdistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text: str) -> dict[str, float]:
        """
        Encode text into a SPLADE sparse vector.
        Returns a dict of {term: weight} for non-zero terms only.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits

        # ReLU + log saturation
        activated = torch.log(1 + torch.relu(logits))

        # Max pooling over sequence length dimension
        # shape: (1, vocab_size)
        sparse_vec = torch.max(activated, dim=1).values.squeeze(0)

        # Convert to dict of non-zero terms
        nonzero_indices = sparse_vec.nonzero(as_tuple=True)[0]
        sparse_dict = {}
        for idx in nonzero_indices:
            term = self.tokenizer.decode([idx])
            weight = sparse_vec[idx].item()
            sparse_dict[term.strip()] = weight

        return sparse_dict

    def score(self, query_vec: dict[str, float],
              doc_vec: dict[str, float]) -> float:
        """Compute dot product between two sparse vectors."""
        score = 0.0
        for term, q_weight in query_vec.items():
            if term in doc_vec:
                score += q_weight * doc_vec[term]
        return score


class SPLADEIndex:
    def __init__(self, encoder: SPLADEEncoder):
        self.encoder = encoder
        self.inverted_index = defaultdict(dict)   # term → {doc_id: weight}
        self.doc_ids = []

    def add_documents(self, documents: dict[str, str]):
        """Encode and index all documents."""
        for doc_id, text in documents.items():
            print(f"  Indexing {doc_id}...")
            sparse_vec = self.encoder.encode(text)
            self.doc_ids.append(doc_id)
            for term, weight in sparse_vec.items():
                self.inverted_index[term][doc_id] = weight

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Retrieve top-k documents for a query."""
        query_vec = self.encoder.encode(query)

        # Score only candidate documents (those sharing at least one term)
        candidate_scores = defaultdict(float)
        for term, q_weight in query_vec.items():
            if term in self.inverted_index:
                for doc_id, d_weight in self.inverted_index[term].items():
                    candidate_scores[doc_id] += q_weight * d_weight

        ranked = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]

    def inspect_query_expansion(self, query: str, top_n: int = 10):
        """Show which terms SPLADE expands a query to."""
        query_vec = self.encoder.encode(query)
        top_terms = sorted(query_vec.items(), key=lambda x: x[1], reverse=True)
        print(f"SPLADE expansion for '{query}':")
        for term, weight in top_terms[:top_n]:
            in_query = "✓" if term.lower() in query.lower() else "→"
            print(f"  {in_query} {term}: {weight:.4f}")


# Example
documents = {
    "D1": "Information retrieval is the task of finding relevant documents.",
    "D2": "Python is a popular programming language for data science.",
    "D3": "Search engines use inverted indexes to retrieve documents efficiently.",
    "D4": "Deep learning models learn representations from large amounts of data.",
    "D5": "Heart attack prevention involves lifestyle changes and medication.",
    "D6": "Cardiovascular disease is a leading cause of mortality worldwide.",
    "D7": "BERT is a transformer model pretrained on masked language modeling.",
    "D8": "Dense retrieval encodes queries and documents as dense neural vectors.",
}

print("Loading SPLADE model...")
encoder = SPLADEEncoder()
index = SPLADEIndex(encoder)

print("\nIndexing documents...")
index.add_documents(documents)

queries = [
    "cardiovascular disease risk factors",
    "neural document retrieval",
    "python data analysis tools",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    index.inspect_query_expansion(query, top_n=8)
    results = index.retrieve(query, top_k=3)
    print(f"Top-3 results:")
    for doc_id, score in results:
        print(f"  {doc_id} ({score:.4f}): {documents[doc_id][:60]}...")
```

## SPLADE in Hybrid Search

SPLADE is a natural component in hybrid search pipelines. Its sparse output
makes it compatible with the same infrastructure as BM25:

```bash
Query
  ↓
Parallel retrieval:
  BM25 sparse retrieval   → top-1000 candidates (exact match strength)
  SPLADE sparse retrieval → top-1000 candidates (semantic + expansion)
  Dense bi-encoder        → top-1000 candidates (pure semantic)
  ↓
Score fusion (RRF or weighted sum)
  ↓
Cross-encoder reranking → top-10 final results
```

In practice, SPLADE alone often performs comparably to the BM25 + dense retrieval
combination, making it a compelling single-model alternative to full hybrid search.

## Where This Fits in the Progression

```bash
Word Embeddings     → static dense vectors
BERT for IR         → contextual dense vectors
Dense Retrieval     → applying vectors to first-stage retrieval
Bi-encoders         → fast first-stage neural retrieval
Cross-encoders      → accurate second-stage reranking
SPLADE              → learned sparse retrieval  ← you are here
```

SPLADE completes the neural IR module by closing the loop back to sparse retrieval.
The progression through this module tells a coherent story:

- Word embeddings taught us that words have geometry
- BERT taught us that context changes meaning
- Dense retrieval applied this to first-stage retrieval
- Bi-encoders made it fast
- Cross-encoders made reranking accurate
- SPLADE asked: can we get BERT's semantic understanding while keeping sparse
  retrieval's efficiency and interpretability?

The answer is yes — and that is why SPLADE represents the current frontier of
practical neural IR for many production systems.

## My Summary

SPLADE uses BERT's masked language modeling head to produce learned sparse vectors
over the full vocabulary — applying ReLU activation, log saturation, and max pooling
over token positions to generate a weight for each vocabulary term. Non-zero weights
represent both the terms in the input and expanded terms that BERT predicts are
semantically relevant, solving vocabulary mismatch without dense vector infrastructure.
SPLADE vectors are compatible with standard inverted indexes, interpretable as a
list of weighted terms, and can be trained end-to-end with a FLOPS regularizer to
control sparsity. It combines BM25's efficiency and exact-match strength with neural
semantic understanding — making it one of the most practically useful models in
the neural IR toolkit.
