# BERT for IR

BERT (Bidirectional Encoder Representations from Transformers) is a pretrained
transformer-based language model introduced by Devlin et al. at Google in 2018.
In IR, BERT fundamentally changed how text is represented and how query-document
relevance is modeled — replacing static word vectors with deep contextual
representations that understand the meaning of a word in its specific context.

## Intuition

Word2Vec gives "bank" the same vector whether the sentence is "I deposited money
at the bank" or "the boat drifted near the bank of the river." The vector is a
static average of all contexts the word appears in during training.

BERT fixes this. It reads the entire sentence bidirectionally and produces a
different vector for "bank" depending on the surrounding words. The representation
is contextual — the same word gets a different embedding in every sentence it
appears in.

For IR this matters enormously. Query terms and document terms are ambiguous.
"python tutorial" could mean the programming language or the snake. BERT can
distinguish these based on context in a way that static embeddings never could.

## The Transformer Architecture — Key Ideas

BERT is built on the transformer encoder. Understanding three core ideas is enough
to work with BERT in an IR context:

### 1. Self-attention

Every token in a sequence attends to every other token. For each token, self-
attention computes a weighted sum of all other token representations — the weights
reflect how relevant each other token is to understanding the current one.

```bash
Input: "the bank by the river"

For token "bank":
  attends strongly to → "river" (disambiguates meaning)
  attends weakly to   → "the"
```

This is what makes BERT contextual — every token's representation is influenced
by every other token in the sequence.

### 2. Bidirectionality

BERT reads text in both directions simultaneously. Unlike earlier models (GPT,
ELMo) that read left-to-right or combined two unidirectional models, BERT's
self-attention operates over the full sequence at once — left context and right
context are both available when representing any token.

```bash
"the [bank] by the river"
         ↑
sees both "the" (left) and "by the river" (right) simultaneously
```

### 3. Pretraining + Fine-tuning

BERT is pretrained on two tasks using massive text corpora:

**Masked Language Modeling (MLM)** — randomly mask 15% of tokens, train the
model to predict the masked tokens from context:

```bash
Input:  "information [MASK] is the task of finding relevant documents"
Target: predict "retrieval"
```

**Next Sentence Prediction (NSP)** — given two sentences, predict whether the
second follows the first in the original text.

After pretraining on Wikipedia and BookCorpus (~3.3 billion words), BERT learns
deep language representations. Fine-tuning adapts these representations to specific
downstream tasks with relatively little task-specific data.

## BERT Architecture Details

```bash
Input:    [CLS] query or document tokens [SEP]
Layers:   12 transformer encoder layers (BERT-base) or 24 (BERT-large)
Hidden:   768 dimensions (base) or 1024 (large)
Heads:    12 attention heads (base) or 16 (large)
Params:   110M (base) or 340M (large)
Output:   one 768-dim contextual vector per input token
```

The [CLS] token is prepended to every input. Its output representation aggregates
information from the entire sequence and is used as the sequence-level representation
for classification tasks.

## How BERT Changed IR

Before BERT, the best IR systems used:

- BM25 for retrieval (sparse, term-based)
- Hand-crafted features for reranking (query length, BM25 score, PageRank, etc.)
- Simple neural models trained from scratch (limited by small labeled datasets)

BERT enabled two major shifts:

### 1. Better relevance modeling

Fine-tune BERT on query-document pairs labeled for relevance. The model learns
to assess relevance from rich contextual representations rather than term overlap.

```bash
Input:  [CLS] query [SEP] document [SEP]
Output: [CLS] representation → linear layer → relevance score
```

This is the cross-encoder architecture — covered in depth in 05-cross-encoders.md.

### 2. Transfer learning from massive pretraining

BERT pretrained on billions of words already understands language deeply. Fine-tuning
on even a few thousand query-document relevance pairs produces strong results because
the model starts from a rich language understanding rather than random weights.

MS MARCO (1 million labeled query-passage pairs) combined with BERT pretraining
produced models that dramatically outperformed BM25 on passage retrieval — this
was the moment neural IR became the dominant research direction.

## BERT for Query-Document Relevance

### Cross-encoder (accurate, slow)

Concatenate query and document as a single BERT input:

```bash
[CLS] what is information retrieval [SEP] Information retrieval is the task
of finding material that satisfies an information need [SEP]

→ BERT processes jointly
→ [CLS] vector → linear layer → relevance score
```

Advantages: rich query-document interaction, highest accuracy
Disadvantages: must run BERT for every (query, document) pair at query time —
O(|corpus|) BERT forward passes per query — completely infeasible for first-stage
retrieval over large corpora.

Solution: use cross-encoders only for reranking a small shortlist (e.g. top 100
from BM25), not for full corpus retrieval.

### Bi-encoder (fast, approximate)

Encode query and document separately with BERT, then compute similarity:

```bash
query_vec    = BERT([CLS] query [SEP])              → 768-dim vector
document_vec = BERT([CLS] document [SEP])           → 768-dim vector
score        = cosine_similarity(query_vec, doc_vec)
```

Advantages: document vectors precomputed offline — query time is one BERT forward
pass + fast vector search
Disadvantages: query and document never interact during encoding — less accurate
than cross-encoder

This is the dense retrieval paradigm. Covered fully in 03-dense-retrieval.md and
04-bi-encoders.md.

## BERT Variants Relevant to IR

| Model      | Description                                        | IR relevance         |
| ---------- | -------------------------------------------------- | -------------------- |
| BERT-base  | Original, 110M params                              | Foundation           |
| BERT-large | 340M params, stronger but slower                   | Reranking            |
| RoBERTa    | Better pretraining, no NSP, more data              | Stronger baseline    |
| DistilBERT | 66M params, 60% faster, 97% of BERT performance    | Efficiency           |
| ELECTRA    | Replaced token detection pretraining               | More efficient       |
| DPR        | BERT fine-tuned specifically for dense retrieval   | Bi-encoder retrieval |
| MonoBERT   | BERT fine-tuned for passage reranking on MS MARCO  | Cross-encoder        |
| ColBERT    | BERT with late interaction for efficient reranking | Advanced retrieval   |
| SPLADE     | BERT producing learned sparse representations      | Neural sparse IR     |

## Fine-tuning BERT for Relevance — The MS MARCO Moment

The combination of MS MARCO (large labeled dataset) + BERT (pretrained language
model) in 2019 was the inflection point for neural IR:

```bash
BM25 on MS MARCO Dev:           MRR@10 ≈ 0.184
BERT cross-encoder reranking:   MRR@10 ≈ 0.365
```

Nearly double the performance. This result triggered an explosion of neural IR
research that continues today. Every model in 03 through 06 of this module is a
direct descendant of this moment.

## Limitations of BERT in IR

### Computational cost

A full BERT forward pass takes ~5ms on GPU. For a corpus of 10 million documents,
full cross-encoder scoring at query time would take 50,000 seconds — completely
infeasible. This is why the bi-encoder / cross-encoder split exists.

### Input length limit

BERT processes at most 512 tokens. Long documents must be split into passages,
each scored independently — losing document-level context.

### Static after fine-tuning

Once fine-tuned, BERT produces fixed representations for new queries and documents.
It does not update with new information without retraining.

### Domain shift

BERT pretrained on Wikipedia and books may struggle on specialized domains (legal,
biomedical, code) without domain-specific pretraining. Variants like BioBERT,
LegalBERT, and CodeBERT address this.

## The Neural IR Stack

BERT sits at the center of the modern neural IR stack:

```bash
User query
    ↓
First-stage retrieval (BM25 or bi-encoder)
    → fast, approximate, retrieves top-1000 candidates
    ↓
Second-stage reranking (BERT cross-encoder)
    → slow, accurate, reranks top-100 candidates
    ↓
Final top-10 results to user
```

This two-stage pipeline is the standard architecture in production neural IR systems.
Understanding BERT is the prerequisite for understanding both stages.

## Where This Fits in the Progression

```bash
Word Embeddings     → static dense vectors, no context
BERT for IR         → contextual dense vectors  ← you are here
Dense Retrieval     → using BERT for first-stage retrieval
Bi-encoders         → efficient BERT-based retrieval
Cross-encoders      → accurate BERT-based reranking
SPLADE              → BERT producing learned sparse vectors
```

## My Summary

BERT produces deep contextual token representations by running bidirectional self-
attention over the full input sequence — the same word gets a different vector
depending on its context, resolving polysemy in a way static embeddings cannot.
Pretrained on billions of words via masked language modeling, BERT fine-tuned on
MS MARCO passage pairs nearly doubled BM25 performance and triggered the neural IR
revolution. In IR it is used in two ways: as a cross-encoder that jointly encodes
query and document for accurate reranking, and as a bi-encoder that encodes them
separately for efficient dense retrieval. Every subsequent model in this module —
dense retrieval, bi-encoders, cross-encoders, SPLADE, ColBERT — is a direct
extension or adaptation of the BERT architecture for specific IR constraints.
