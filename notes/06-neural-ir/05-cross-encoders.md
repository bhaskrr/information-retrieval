# Cross-encoders

A cross-encoder is a neural relevance scoring architecture that jointly encodes a
query and a document together as a single input sequence and produces a relevance
score from their combined representation. Unlike a bi-encoder that encodes query
and document independently, a cross-encoder allows every query token to attend to
every document token through self-attention — producing a more accurate relevance
judgment at the cost of significantly higher computational expense.

## Intuition

The bi-encoder's weakness is independence. It encodes query and document separately
and never lets them interact. This means the model cannot directly compare query
terms against document terms — it must compress each into a single vector and hope
the vectors end up close if the content is relevant.

A cross-encoder does what a human reader does: it reads the query and the document
together, paying attention to how specific query terms relate to specific document
passages. This direct interaction is why cross-encoders are significantly more
accurate than bi-encoders.

The cost is that cross-encoders cannot precompute document representations. Every
time a new query arrives, the cross-encoder must process the full (query, document)
pair from scratch — making it far too slow for first-stage retrieval over large
corpora. This is why cross-encoders are used exclusively for reranking a small
candidate set produced by a faster first-stage retriever.

## Architecture

```
Input:  [CLS] query tokens [SEP] document tokens [SEP]
Model:  BERT or similar transformer encoder
Output: [CLS] token representation → linear layer → scalar relevance score
```

The entire (query, document) pair is processed as one sequence. Self-attention
operates over all tokens jointly — every query token can directly attend to every
document token and vice versa.

```
Query:    "what causes cardiovascular disease"
Document: "Heart disease risk factors include high blood pressure and smoking."

BERT self-attention:
  "cardiovascular" ←→ "heart", "disease"    (high attention weight)
  "causes"         ←→ "risk factors"         (high attention weight)
  "what"           ←→ "include"              (lower attention weight)
```

The [CLS] token aggregates this joint representation and is passed to a linear
classification or regression head to produce the relevance score.

## Cross-encoder vs Bi-encoder — The Full Comparison

| Property                   | Bi-encoder            | Cross-encoder          |
| -------------------------- | --------------------- | ---------------------- | ---------- | -------- |
| Encoding                   | Independent           | Joint                  |
| Query-document interaction | None during encoding  | Full self-attention    |
| Document precomputation    | Yes — offline         | No — online per query  |
| Query time complexity      | O(1) + ANN search     | O(                     | candidates | ) × BERT |
| Accuracy                   | Lower                 | Higher                 |
| Latency (100 candidates)   | ~1ms                  | ~200-500ms             |
| Corpus size supported      | Millions to billions  | Dozens to thousands    |
| Role in pipeline           | First-stage retrieval | Second-stage reranking |

## Training Cross-encoders

### Binary classification

Train on (query, document, label) triples where label = 1 (relevant) or 0 (not):

```
Input:  [CLS] query [SEP] document [SEP]
Output: sigmoid(linear([CLS])) → probability of relevance

Loss:   binary cross-entropy
```

### Pointwise regression

Train to predict a continuous relevance score directly:

```
Loss: MSE between predicted score and human relevance grade (0, 1, 2, 3)
```

### Pairwise ranking loss

Given a query and a relevant/irrelevant document pair, train the model to score
the relevant document higher:

```
L = max(0, margin - score(q, d+) + score(q, d-))
```

This is more directly aligned with the ranking objective than binary classification.

### Listwise training

Train on the full ranked list simultaneously, optimizing a list-level objective
like NDCG or softmax cross-entropy over the full list of candidates. More complex
but produces stronger models.

### Fine-tuning on MS MARCO

The standard starting point:

- Dataset: MS MARCO passage ranking (~500K training triples)
- Base model: BERT-base or RoBERTa-base
- Training: pairwise loss with hard negatives
- Result: MonoBERT — the original strong cross-encoder baseline

## MonoBERT and MonoT5

### MonoBERT (Nogueira & Cho, 2019)

The first demonstration that BERT cross-encoders dramatically improve over BM25
for passage reranking:

```
BM25 alone:                     MRR@10 = 0.184
BM25 + MonoBERT reranking:      MRR@10 = 0.365
```

Architecture: BERT-base fine-tuned on MS MARCO with binary classification head.

### MonoT5 (Nogueira et al., 2020)

Uses a T5 sequence-to-sequence model as the cross-encoder. The model is trained
to generate the word "true" for relevant pairs and "false" for irrelevant ones.
Relevance score = probability of generating "true".

```
Input:  "Query: {query} Document: {document} Relevant:"
Output: "true" or "false"
Score:  P("true")
```

MonoT5-3B outperforms MonoBERT significantly. T5-based rerankers are now common
in production pipelines.

## The Reranking Pipeline

Cross-encoders are always used as the second stage after a fast first-stage
retriever:

```
Query
  ↓
Stage 1 — BM25 or Bi-encoder
  → retrieves top-1000 candidates in ~10ms
  ↓
Stage 2 — Cross-encoder reranker
  → scores each of 1000 candidates
  → 1000 × BERT forward passes
  → ~2-5 seconds on GPU (too slow for 1000)

Practical approach: rerank only top-100
  → 100 × BERT forward passes
  → ~200-500ms on GPU
  ↓
Final top-10 results
```

The tradeoff between how many candidates to rerank and latency is a core
engineering decision. Typical production systems rerank top-50 to top-200
candidates.

## Cascade Reranking

For very large candidate sets or strict latency requirements, use a cascade:

```
Stage 1:  BM25 → top-1000
Stage 2:  lightweight bi-encoder reranker → top-100
Stage 3:  full cross-encoder → top-10
```

Each stage is more accurate and slower than the previous. The cascade allows
cross-encoder accuracy with manageable latency by reducing the candidate set
before the expensive stage.

## Cross-encoder for Score Distillation

Cross-encoders are not only used for reranking — they are used to generate
training signal for bi-encoders:

```
1. Train a cross-encoder on labeled relevance data
2. Use cross-encoder to score (query, document) pairs at scale
3. Use cross-encoder scores as soft labels to train bi-encoder
   (knowledge distillation)
```

This allows the bi-encoder to approach cross-encoder accuracy while maintaining
retrieval speed. Several strong bi-encoders (TAS-B, ANCE) use this approach.

## Limitations of Cross-encoders

### Cannot precompute document representations

Every query requires re-encoding every candidate document. There is no way to
cache or precompute anything — the joint encoding means representations change
with every new query.

### Input length limit

BERT's 512 token limit applies to the concatenated (query + document) sequence.
Long documents must be split into passages. Typical approach: score each passage
independently, take the maximum or average score as the document score.

### Computational cost

100 BERT forward passes per query is feasible with a GPU. 10,000 is not. This
hard constraint determines how many candidates can be reranked within latency
budgets.

### Domain sensitivity

Cross-encoders fine-tuned on MS MARCO may not generalize well to specialized
domains. Domain-specific fine-tuning is often needed for production deployments
in legal, medical, or technical settings.

## Practical Cross-encoder Models

| Model                               | Base       | Notes                     |
| ----------------------------------- | ---------- | ------------------------- |
| cross-encoder/ms-marco-MiniLM-L-6   | MiniLM     | Fast, lightweight, strong |
| cross-encoder/ms-marco-MiniLM-L-12  | MiniLM-L12 | Stronger, slightly slower |
| cross-encoder/ms-marco-electra-base | ELECTRA    | Efficient pretraining     |
| MonoT5-base                         | T5-base    | Generative reranker       |
| MonoT5-3B                           | T5-3B      | Strongest, very slow      |
| RankLLaMA                           | LLaMA      | LLM-based reranking       |

For most applications, `cross-encoder/ms-marco-MiniLM-L-6` from sentence-
transformers offers the best speed-accuracy tradeoff.

## Code

```python
# pip install sentence-transformers

from sentence_transformers import CrossEncoder
import numpy as np

# Load a pretrained cross-encoder fine-tuned on MS MARCO
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)

# Corpus
documents = {
    "D1": "Information retrieval finds relevant documents satisfying a query.",
    "D2": "Python matplotlib and seaborn are popular data visualization tools.",
    "D3": "BERT is a transformer model pretrained on masked language modeling.",
    "D4": "Dense retrieval encodes queries and documents as dense neural vectors.",
    "D5": "Heart disease risk factors include high blood pressure and smoking.",
    "D6": "Cardiovascular disease is a leading cause of death worldwide.",
    "D7": "Search engines build inverted indexes for efficient document retrieval.",
    "D8": "Bi-encoders precompute document vectors for fast approximate retrieval.",
}

doc_ids   = list(documents.keys())
doc_texts = list(documents.values())


def crossencoder_rerank(query: str,
                         candidates: list[tuple[str, str]],
                         top_k: int = 5) -> list[tuple[str, float, str]]:
    """
    Rerank a list of candidate (doc_id, doc_text) pairs using a cross-encoder.

    Args:
        query:      the query string
        candidates: list of (doc_id, doc_text) tuples
        top_k:      number of results to return

    Returns:
        list of (doc_id, score, doc_text) sorted by score descending
    """
    pairs = [(query, text) for _, text in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip([doc_id for doc_id, _ in candidates], scores,
            [text for _, text in candidates]),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]


# Full two-stage pipeline: BM25 → cross-encoder reranking
from rank_bm25 import BM25Okapi

tokenized_docs = [doc.lower().split() for doc in doc_texts]
bm25 = BM25Okapi(tokenized_docs)

def two_stage_retrieve(query: str,
                        first_stage_k: int = 5,
                        final_k: int = 3) -> list[tuple[str, float, str]]:
    """
    Two-stage retrieval: BM25 first stage + cross-encoder reranking.
    """
    # Stage 1 — BM25 retrieval
    bm25_scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(bm25_scores)[::-1][:first_stage_k]
    candidates = [(doc_ids[i], doc_texts[i]) for i in top_indices]

    print(f"  Stage 1 BM25 candidates:")
    for doc_id, text in candidates:
        score = bm25_scores[doc_ids.index(doc_id)]
        print(f"    {doc_id} ({score:.4f}): {text[:55]}...")

    # Stage 2 — cross-encoder reranking
    reranked = crossencoder_rerank(query, candidates, top_k=final_k)

    return reranked


queries = [
    "cardiovascular disease causes",
    "fast document retrieval neural",
    "how does information retrieval work",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = two_stage_retrieve(query, first_stage_k=5, final_k=3)
    print(f"  Final reranked top-3:")
    for doc_id, score, text in results:
        print(f"    {doc_id} ({score:.4f}): {text[:55]}...")


# Direct cross-encoder scoring (without first stage)
query = "neural dense retrieval systems"
pairs = [(query, text) for text in doc_texts]
scores = model.predict(pairs)

print(f"\nCross-encoder scores for '{query}':")
ranked = sorted(zip(doc_ids, scores, doc_texts),
                key=lambda x: x[1], reverse=True)
for doc_id, score, text in ranked:
    print(f"  {doc_id} ({score:.4f}): {text[:60]}...")


# Compare bi-encoder vs cross-encoder on vocabulary mismatch
from sentence_transformers import SentenceTransformer
import faiss

bi_model = SentenceTransformer("msmarco-distilbert-base-v4")

doc_vecs = bi_model.encode(doc_texts, normalize_embeddings=True)
doc_vecs = np.array(doc_vecs, dtype=np.float32)
index = faiss.IndexFlatIP(doc_vecs.shape[1])
index.add(doc_vecs)

query = "cardiovascular disease risk factors"
q_vec = bi_model.encode([query], normalize_embeddings=True)
q_vec = np.array(q_vec, dtype=np.float32)

be_scores, be_indices = index.search(q_vec, len(doc_ids))
ce_scores = model.predict([(query, text) for text in doc_texts])

print(f"\n--- Bi-encoder vs Cross-encoder comparison ---")
print(f"Query: '{query}'\n")

print("Bi-encoder ranking:")
for idx, score in zip(be_indices[0], be_scores[0]):
    print(f"  {doc_ids[idx]} ({score:.4f}): {doc_texts[idx][:55]}...")

print("\nCross-encoder ranking:")
ce_ranked = sorted(zip(doc_ids, ce_scores, doc_texts),
                   key=lambda x: x[1], reverse=True)
for doc_id, score, text in ce_ranked:
    print(f"  {doc_id} ({score:.4f}): {text[:55]}...")
```

## Where This Fits in the Progression

```
Word Embeddings     → static dense vectors
BERT for IR         → contextual dense vectors
Dense Retrieval     → applying vectors to first-stage retrieval
Bi-encoders         → fast first-stage neural retrieval
Cross-encoders      → accurate second-stage reranking  ← you are here
SPLADE              → learned sparse retrieval
```

The bi-encoder and cross-encoder together define the standard two-stage neural
IR pipeline. Every modern production search system uses some variant of this
architecture. SPLADE is the next evolution — it asks whether we can get semantic
understanding while keeping the efficiency of sparse retrieval.

## My Summary

A cross-encoder jointly encodes query and document in a single BERT forward pass,
allowing full self-attention between all query and document tokens. This direct
interaction produces significantly more accurate relevance scores than bi-encoders
but cannot precompute document representations — every new query requires re-encoding
all candidates. This constrains cross-encoders to reranking small candidate sets
(typically top-50 to top-100) produced by a faster first-stage retriever. MonoBERT
and MonoT5 are the standard cross-encoder baselines; MiniLM-based cross-encoders
offer the best practical speed-accuracy tradeoff. Cross-encoders also serve as
teachers for bi-encoder distillation — their accurate relevance scores train
bi-encoders to be more precise without sacrificing retrieval speed.
