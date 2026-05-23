# Cross-Encoder Training

Cross-encoder training is the process of fine-tuning a transformer model to
accurately score the relevance of a document given a query by jointly encoding
both as a single concatenated input sequence. Unlike bi-encoders that encode
query and document independently and compare their representations, a cross-encoder
processes the query-document pair together - every query token attends to every
document token through full self-attention - producing a single scalar relevance
score. This joint encoding is what makes cross-encoders the most accurate retrieval
scoring component available, and training them on task-specific data is what makes
them far more accurate than zero-shot scoring. A well-trained cross-encoder is the
quality ceiling of any two-stage retrieval system.

## Intuition

The fundamental difference between bi-encoders and cross-encoders is when the
query and document meet:

```
Bi-encoder:
  query  → encoder → q_vec ─────────────────┐
                                              ├─ cosine_sim(q_vec, d_vec)
  document → encoder → d_vec ────────────────┘
  Query and document never interact during encoding
  → 768-dim bottleneck compresses all relevant information

Cross-encoder:
  [CLS] query [SEP] document [SEP] → transformer → [CLS] hidden state → score
  Query and document interact at every attention layer
  → full attention allows precise relevance judgment
```

Consider the query "Java performance optimization" and two documents:

```
D1: "Java memory management and garbage collection optimization techniques"
D2: "Performance optimization strategies for JavaScript applications"
```

A bi-encoder might score both documents similarly - both contain "performance",
"optimization", and one of "Java" or "JavaScript". The vector representations
of "Java optimization" and "JavaScript optimization" are close in the embedding
space because they share context.

A cross-encoder sees both documents in full context alongside the query. The
attention mechanism directly compares "Java" in the query to "Java" vs
"JavaScript" in each document, and "performance optimization" in the context
of garbage collection vs JavaScript frameworks. It correctly identifies D1 as
much more relevant.

Cross-encoder training teaches the model to make these precise relevance
distinctions on your specific domain and query distribution.

## Architecture

### Input format

```
Standard BERT cross-encoder input:

[CLS] query_tokens [SEP] document_tokens [SEP]

Example:
  [CLS] what is information retrieval [SEP] Information retrieval is the
  process of finding relevant documents from a large collection [SEP]
```

The full sequence is processed by a BERT-base or similar transformer.
The [CLS] token aggregates the full cross-attention between query and document.
A linear layer on the [CLS] representation produces a scalar score.

### Architecture variants

**Pointwise cross-encoder (standard)**
Input: (query, document)
Output: scalar relevance score
Training: binary cross-entropy or MSE on relevance labels

**Pairwise cross-encoder**
Input: (query, document_positive, document_negative) - processed as two separate pairs
Output: two scores, trained to prefer positive
Training: pairwise ranking loss

**Listwise cross-encoder**
Input: (query, [doc₁, doc₂, ..., docₖ]) - all candidates together
Output: relevance scores for all candidates simultaneously
Training: listwise ranking loss
See: 03-listwise-reranking.md for full treatment

## Training Data Formats

### MS MARCO for reranker training

MS MARCO provides the standard training data for passage reranking:

```
BM25 negatives file: 530K triples of (query, positive_passage, negative_passage)
Hard negative file:  additional hard negatives mined with DPR/BM25

Training format:
  query:    "what is the capital of france"
  positive: "Paris is the capital and most populous city of France."
  negative: "France is a country in western Europe."
  label:    positive=1, negative=0
```

### TREC relevance judgments

For high-quality training on specific domains:

```
TREC qrels format:
  query_id  0  doc_id  relevance_grade

  Where relevance_grade ∈ {0, 1, 2, 3}:
    0 = not relevant
    1 = marginally relevant
    2 = relevant
    3 = highly relevant

Training format:
  Convert to (query, doc, grade) triplets
  Use graded labels with MSE loss or convert to binary pairs
```

### Domain-specific pairs

For production deployment, domain-specific training data is essential:

```
Sources:
  User click logs (implicit feedback)
  Expert relevance annotations
  Synthetic pairs from LLM judgment
  Existing QA datasets in the domain
```

## Loss Functions for Cross-Encoder Training

### Binary cross-entropy (pointwise)

Each training example is an independent (query, document, label) triple:

```
Input:   (query, document)
Label:   1 = relevant, 0 = not relevant
Output:  probability of relevance via sigmoid

L = -[y × log(σ(score)) + (1-y) × log(1 - σ(score))]
```

Simple but has a disadvantage: treats all positives equally and all negatives
equally. A document that is "somewhat relevant" gets the same label=1 as a
"highly relevant" document.

### Pairwise ranking loss

Train on (query, positive, negative) triplets. Force the positive to score
higher than the negative:

```
score_pos = cross_encoder(query, positive)
score_neg = cross_encoder(query, negative)

L_pairwise = -log(σ(score_pos - score_neg))
   or
L_margin   = max(0, margin - (score_pos - score_neg))
```

More directly optimizes ranking - which is the actual retrieval objective -
compared to pointwise classification.

### Graded relevance with MSE

Use continuous relevance labels from human annotation:

```
L = MSE(score, relevance_grade)
```

Captures fine-grained relevance distinctions that binary labels miss.
Requires graded annotations (expensive but high quality).

### Knowledge distillation from ensemble teacher

Train a student cross-encoder to match the scores of a stronger teacher
(ensemble of multiple cross-encoders or a larger model):

```
teacher_score = ensemble_score(query, document)
student_score = student_cross_encoder(query, document)

L = MSE(student_score, teacher_score)
```

This approach (used in MiniLM-L-6-v2 training) produces smaller, faster
students that match or approach the teacher's accuracy. The teacher
provides soft labels that carry more information than binary labels.

## Hard Negative Mining for Cross-Encoders

Hard negatives are essential for cross-encoder training. A cross-encoder that
only trains on random negatives learns to distinguish obviously irrelevant
documents - which is easy. It needs to learn to distinguish hard cases.

### BM25 hard negatives

Retrieve BM25 top-k results for each query, exclude the known positive:

```
For query "machine learning optimization":
  BM25 top-5: [D_pos, D_hard1, D_hard2, D_hard3, D_hard4]
  Remove D_pos → hard negatives: [D_hard1, D_hard2, D_hard3, D_hard4]

D_hard1: "gradient descent optimization algorithm"  ← very hard (relevant-looking)
D_hard4: "machine learning tutorial for beginners"  ← easier (less specific)
```

BM25 hard negatives are those that lexically resemble the query but are not
actually relevant. They train the cross-encoder's most important skill: precise
semantic discrimination.

### Dense retrieval hard negatives

Mine hard negatives using the current bi-encoder:

```
Dense top-k minus positive → hard negatives
```

These are semantically similar to the positive but not relevant - harder than
BM25 negatives for a semantically-aware model.

### Adversarial hard negatives

Mine negatives that a previous version of the cross-encoder incorrectly scores
highly:

```
Step 1: Train initial cross-encoder on easy negatives
Step 2: Run cross-encoder on a large set of candidates
Step 3: Documents scored high by cross-encoder but not labeled relevant = adversarial negatives
Step 4: Retrain with adversarial negatives added
```

This iterative adversarial mining produces the strongest cross-encoders.

### Mixing negatives

Best practice: mix different negative types for robust training:

```
For each positive:
  1 × BM25 hard negative    (lexical resemblance)
  1 × Dense hard negative   (semantic resemblance)
  1 × Random negative       (easy, provides stable gradient)
```

## Training Recipe for MS MARCO Cross-Encoder

The standard recipe that produces production-ready cross-encoders:

### Stage 1 - Base training on MS MARCO

```python
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import torch

# Load MS MARCO training triplets
# Available at: https://huggingface.co/datasets/microsoft/ms_marco
train_examples = [
    InputExample(texts=["query", "positive passage", "negative passage"], label=1),
    # 530K examples total
]

model = CrossEncoder("bert-base-uncased", num_labels=1)

# Train with pairwise loss
model.fit(
    train_dataloader=DataLoader(train_examples, batch_size=32, shuffle=True),
    epochs=3,
    optimizer_params={"lr": 2e-5},
    warmup_steps=1000,
    output_path="./ms-marco-cross-encoder"
)
```

### Stage 2 - Hard negative augmentation

After base training, mine hard negatives using the trained model:

```python
# Use trained model to score all BM25 top-100 candidates
# Documents with high model scores but not in qrels = hard negatives
# Retrain with these harder examples
```

### Stage 3 - Knowledge distillation (optional)

Distill from a large ensemble to a fast small model:

```python
# Teacher: ensemble of multiple cross-encoders
# Student: MiniLM-L-6 (fast inference)
# Loss: MSE(student_score, teacher_ensemble_score)
```

## Pretrained Cross-Encoders

Rather than training from scratch, fine-tuning a pretrained MS MARCO
cross-encoder on your domain is the standard approach:

```
Model                                       Params  Speed   NDCG@10
──────────────────────────────────────────────────────────────────────
cross-encoder/ms-marco-MiniLM-L-6-v2       22M     Fast    0.390
cross-encoder/ms-marco-MiniLM-L-12-v2      33M     Medium  0.395
cross-encoder/ms-marco-TinyBERT-L-2-v2     4M      Very fast 0.372
cross-encoder/ms-marco-roberta-base-v2     123M    Slow    0.418
monoT5-base-msmarco-10k                    250M    Slow    0.408
monoT5-3b-msmarco-10k                      3B      Very slow 0.439
```

For production with GPU: `ms-marco-MiniLM-L-6-v2` or `MiniLM-L-12-v2`
For research / offline: `ms-marco-roberta-base-v2` or `monoT5-3b`
For CPU deployment: `TinyBERT-L-2-v2`

## Domain Fine-Tuning Protocol

Fine-tuning a pretrained MS MARCO cross-encoder on domain data:

```
Step 1: Collect domain pairs
  100-500 (query, positive_doc, hard_negative) triplets

Step 2: Start from pretrained MS MARCO model
  model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

Step 3: Fine-tune with low learning rate
  lr = 1e-5 to 2e-5 (lower than initial training)
  epochs = 2-5
  warmup = 10% of steps

Step 4: Evaluate on domain test set
  NDCG@10 should improve 5-15% over zero-shot
```

The key principle: start from the MS MARCO pretrained model, not from BERT.
The MS MARCO model already understands relevance scoring - domain fine-tuning
adjusts it to your domain's relevance criteria, not learns relevance from scratch.

## Evaluation Metrics for Cross-Encoders

Cross-encoders are evaluated as part of a retrieval pipeline:

### NDCG@10 (primary)

The standard metric - measures the quality of the final top-10 ranking
after reranking a fixed set of candidates.

### MRR@10 (Mean Reciprocal Rank)

Measures how quickly the first relevant document appears in the ranked list:

```
MRR@10 = (1/|Q|) × Σ_q 1/rank_q
```

### MS MARCO Dev Set

The standard evaluation for comparing cross-encoders:

```
~7000 queries with relevance judgments
First-stage retrieval: BM25 top-100
Metric: MRR@10 on dev set
```

```
Model                           MRR@10
────────────────────────────────────────
BM25 (no reranking)             0.187
MiniLM-L-6 cross-encoder        0.390
MiniLM-L-12 cross-encoder       0.395
RoBERTa-base cross-encoder      0.418
monoT5-3b                       0.439
```

## Fine-Tuning Checklist

```
Step                          Action
──────────────────────────────────────────────────────────────────────
Data collection               100-500 (query, positive, negative) triplets
                               from domain-specific data

Negative selection             Use BM25 hard negatives (vocabulary similar)
                               + dense hard negatives (semantically similar)
                               + random negatives (training stability)

Base model selection           cross-encoder/ms-marco-MiniLM-L-6-v2
                               for speed or ms-marco-roberta-base for quality

Learning rate                  2e-5 or lower (1e-5 to preserve MS MARCO knowledge)

Warmup                         10% of total steps

Validation                     Evaluate NDCG@10 on held-out domain queries
                               every epoch; early stop on plateau

General performance check      Run on MS MARCO dev set - should not drop
                               more than 2-3 MRR@10 points vs pretrained

Output                         Save both the fine-tuned model and the
                               base model for AB testing
```

## My Summary

Cross-encoder training fine-tunes a transformer to produce accurate relevance
scores by jointly encoding query and document as a single sequence, allowing every
query token to attend to every document token through full self-attention. The
joint encoding is what makes cross-encoders more accurate than bi-encoders but also
more expensive - they cannot be precomputed offline since the score depends on both
query and document together. Training uses pairwise ranking loss on (query, positive,
negative) triplets, with hard negatives being the most critical training factor:
BM25 hard negatives teach lexical discrimination, dense hard negatives teach semantic
discrimination, and mixing both produces robust models. For most applications,
fine-tuning from a pretrained MS MARCO cross-encoder (MiniLM-L-6-v2 for speed,
RoBERTa-base for quality) on 100-500 domain-specific triplets produces 5-15% NDCG
improvement over zero-shot. The standard evaluation protocol is NDCG@10 after
reranking BM25 top-100 candidates on a domain-specific test set, with a concurrent
check on MS MARCO dev MRR@10 to ensure the fine-tuning has not catastrophically
forgotten general relevance judgment.
