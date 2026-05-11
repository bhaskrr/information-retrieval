# Few-Shot Retrieval

Few-shot retrieval is the practice of adapting a pretrained retrieval model to a
target domain using a small number of labeled (query, relevant document) examples -
typically 100 to 1000 pairs - rather than the tens of thousands required for full
fine-tuning. It leverages the rich language representations already learned during
pretraining and uses a small domain-specific signal to steer those representations
toward domain-appropriate retrieval behavior. Few-shot retrieval is the most
practical domain adaptation technique for most real-world IR applications because
labeled data is always scarce, expensive to collect, and domain experts' time is
limited.

## Intuition

A pretrained model like E5-base or all-MiniLM already understands language deeply.
It knows that "neural" and "network" often co-occur with "deep learning". It knows
that "receptor" and "binding" suggest a biological context. It knows grammatical
structure, semantic relationships, and hundreds of millions of linguistic patterns.

What it does not know is your domain's specific relevance criteria. It does not know
that in biomedical retrieval, a paper about "ACE2 receptor binding" is relevant to
a query about "SARS-CoV-2 entry mechanism" - because it has never been trained on
examples that reward this connection. It does not know that in legal retrieval,
a case about "reasonable expectation of privacy in digital communications" is
relevant to a query about "Fourth Amendment smartphone searches".

A small number of domain-specific labeled pairs teaches the model these connections.
100 examples is enough to dramatically shift the model's behavior on the domain -
not because the model learns new vocabulary from scratch, but because it learns which
existing representations are relevant to each other in the domain context.

The analogy: a brilliant generalist linguist can learn to work in a new technical
domain with a short glossary and a few example questions with answers. They already
know language - they just need to learn which concepts are related in this specific
field. Few-shot fine-tuning is giving that glossary and examples to the retrieval
model.

## The Sample Efficiency of Fine-Tuning

Why does fine-tuning work with so few examples? Three reasons:

### 1. Pretrained representations are already informative

The model's encoder already produces meaningful embeddings. Two semantically related
documents are already closer in embedding space than two unrelated documents - just
not calibrated optimally for the target domain. Fine-tuning adjusts the geometric
structure, not the language understanding itself.

### 2. Contrastive loss is data-efficient

Each training example contributes multiple gradient signals. In a batch of 32
(query, positive_doc) pairs with in-batch negatives, each example generates 31
negative examples automatically. 100 labeled pairs generate 3,200 effective
training signals - 32x data multiplication.

### 3. Catastrophic forgetting is limited with careful training

Small learning rates and short training runs update the representations just enough
to capture domain signal without destroying the general language understanding.
The model shifts within the space of valid representations rather than learning
an entirely new space.

## Data Collection Strategies

The quality and collection method of few-shot training data significantly affects
the outcome.

### Strategy 1 - Human annotation (gold standard)

Ask domain experts to judge relevance of document-query pairs:

```
For each query:
  1. Run BM25 and/or neural retrieval → top-20 candidates
  2. Domain expert judges each candidate: relevant (1) or not relevant (0)
  3. Store (query, positive_docs, hard_negative_docs)
```

Gold standard quality but expensive. Budget for:

- 30-60 minutes per annotator per 10 queries
- At least 2 annotators per query for inter-annotator agreement
- Domain experts are expensive - often $50-200/hour

Cost for 200 annotated queries: 200 queries × 30 min × $100/hour = ~$1,000-$3,000.
This is often worth it for a high-value retrieval application.

### Strategy 2 - Clickthrough data (implicit feedback)

If users interact with search results, their clicks are implicit relevance signals:

```
User query → system returns documents → user clicks D3 and D7 → clicks are positive
```

Clickthrough data is abundant but noisy:

- Position bias: documents at rank 1 are clicked more regardless of relevance
- Navigational bias: users click specific pages they already know
- Trust bias: users may click the first result even if it is suboptimal

Debiasing techniques (IPW, DLA) are needed for production click-based fine-tuning.

### Strategy 3 - Synthetic data with LLMs (no human annotation)

Use an LLM to generate synthetic queries for existing domain documents:

```
For each document chunk d in corpus:
  prompt = "Generate 3 questions a user might ask that this document answers:
            Document: {d}"
  synthetic_queries = llm(prompt)
  store (synthetic_query, d) as positive pair
```

No human annotation required. Cost: LLM API calls (~$0.001-0.01 per document).
Quality: lower than human annotation but sufficient for many applications.
Covered in depth in the GPL (Generative Pseudo-Labeling) section below.

### Strategy 4 - Mining from existing resources

Find naturally occurring query-document pairs in your domain:

```
FAQ pages:        question + answer paragraph → (question, paragraph) pairs
Stack Overflow:   question + accepted answer  → (question, answer) pairs
PubMed:          title + abstract            → (title, abstract) pairs
Legal databases: case summary + full text    → (summary, full text) pairs
Product manuals: section heading + content  → (heading, content) pairs
```

Free and abundant for many domains. Quality varies - titles are often poor
query proxies but can be combined with human curation.

## Few-Shot Fine-Tuning Recipes

### Recipe 1 - Bi-encoder fine-tuning (most common)

Fine-tune sentence-transformers with in-batch negatives and hard negatives:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load few-shot training data
train_examples = [
    InputExample(texts=["query 1", "relevant doc 1"]),
    InputExample(texts=["query 2", "relevant doc 2"]),
    # ... 100-500 examples
]

# Load pretrained model
model = SentenceTransformer("intfloat/e5-base-v2")

# Fine-tune with MultipleNegativesRankingLoss (InfoNCE with in-batch negatives)
train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)
train_loss       = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
    show_progress_bar=True
)
```

**Critical parameters:**

- Batch size ≥ 16 (more in-batch negatives = better signal)
- Learning rate: 2e-5 to 5e-5 (lower than standard training to avoid forgetting)
- Epochs: 1-5 (more risks overfitting to small dataset)
- Warmup: ~10% of steps

### Recipe 2 - Hard negative mining

Random in-batch negatives are easy - the model quickly learns to separate clearly
irrelevant documents. Hard negatives are documents that look relevant but are not:

```
Easy negative: "a recipe for chocolate cake" for query "biomedical receptor binding"
Hard negative: "ACE2 receptor in cardiac tissue" for query "ACE2 receptor SARS-CoV-2"
               (related to ACE2 receptors but not about SARS-CoV-2)
```

Mining hard negatives:

1. Train initial model on easy negatives
2. Retrieve top-20 candidates for each training query
3. Documents retrieved but not labeled as relevant = hard negatives
4. Retrain with hard negatives included

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def mine_hard_negatives(model, corpus, train_pairs, top_k=20):
    """
    Mine hard negatives from BM25 or neural retrieval top-k results.
    Documents retrieved but not in the positive set are hard negatives.
    """
    doc_ids   = list(corpus.keys())
    doc_texts = list(corpus.values())
    doc_embs  = model.encode(doc_texts, normalize_embeddings=True)

    d     = doc_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(doc_embs.astype("float32"))

    hard_negative_pairs = []

    for query, pos_doc_id in train_pairs:
        q_emb    = model.encode([query], normalize_embeddings=True)
        _, idxs  = index.search(q_emb.astype("float32"), top_k)
        top_docs = [doc_ids[i] for i in idxs[0]]

        # Hard negatives: retrieved but not positive
        hard_negs = [d for d in top_docs if d != pos_doc_id][:3]

        for neg_doc_id in hard_negs:
            hard_negative_pairs.append(
                InputExample(texts=[
                    query,
                    corpus[pos_doc_id],
                    corpus[neg_doc_id]
                ])
            )

    return hard_negative_pairs
```

### Recipe 3 - Margin MSE distillation from cross-encoder

Use a cross-encoder as a teacher to generate soft labels for the bi-encoder:

```python
from sentence_transformers import CrossEncoder, losses

# Cross-encoder teacher (already strong due to joint encoding)
teacher = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# For each (query, pos, neg) triple:
teacher_scores = teacher.predict([
    (query, pos_doc),
    (query, neg_doc)
])
teacher_margin = teacher_scores[0] - teacher_scores[1]

# Train bi-encoder to match teacher margins
train_examples = [
    InputExample(
        texts=[query, pos_doc, neg_doc],
        label=teacher_margin
    )
    # ...
]

train_loss = losses.MarginMSELoss(student_model)
```

This technique from TAS-B produces the strongest few-shot results when a
good cross-encoder is available. The cross-encoder's relevance judgments
serve as dense soft labels rather than binary 0/1 annotations.

### Recipe 4 - GPL (Generative Pseudo-Labeling)

The most powerful no-annotation technique. Uses an LLM to generate synthetic
queries, a retriever to mine negatives, and a cross-encoder to score pairs:

```
Step 1 - Generate synthetic queries with LLM
  for each document chunk d:
    synthetic_queries = llm.generate(f"What questions does this answer? {d}")

Step 2 - Mine negatives with BM25/dense retrieval
  for each (synthetic_query, positive_doc):
    negatives = retriever.search(synthetic_query, exclude=positive_doc)[:3]

Step 3 - Score with cross-encoder (teacher)
  for each (query, positive, negative):
    pos_score = cross_encoder(query, positive)
    neg_score = cross_encoder(query, negative)
    margin    = pos_score - neg_score

Step 4 - Fine-tune bi-encoder on scored pairs
  train_loss = MarginMSELoss(student_model)
```

GPL requires no human annotation and typically outperforms standard fine-tuning
with small labeled datasets. Cost is primarily LLM API calls for query generation.

## Domain-Specific Evaluation Protocol

Fine-tuning without evaluation is dangerous - you can overfit to the small
training set and degrade general performance. Always evaluate on a held-out set:

```
Data split for few-shot adaptation:
  Total labeled examples: 200-500
  Training set:           80%  (160-400 examples)
  Validation set:         10%  (20-50 examples)  ← for early stopping
  Test set:               10%  (20-50 examples)  ← for final evaluation
```

Additionally, evaluate on a held-out general benchmark (BEIR subset) to confirm
you have not sacrificed general performance:

```
Report:
  Domain NDCG@10 (before): 0.31   → after fine-tuning: 0.42   (+35%)
  BEIR avg NDCG@10 (before): 0.43 → after fine-tuning: 0.41   (-5%)
```

A 35% domain improvement at the cost of 5% general performance is usually
acceptable. A 10% domain improvement at the cost of 20% general performance
is not.

## Practical Guidelines

### How many examples do you need?

```
Labeled examples    Expected NDCG improvement     Notes
───────────────────────────────────────────────────────────────────
10-50               5-15%                         Useful with hard negatives
50-200              15-30%                        Sweet spot for most domains
200-500             25-40%                        Approaching full fine-tuning
500-1000            30-45%                        Most of the benefit captured
> 1000              35-50%                        Diminishing returns
```

The returns are heavily front-loaded. Going from 0 to 100 examples gives much
larger improvement than going from 900 to 1000.

### Learning rate selection

```
Base model         Recommended LR     Rationale
────────────────────────────────────────────────────────────────────
all-MiniLM-L6-v2   2e-5 to 5e-5       Smaller model, more capacity to update
e5-base-v2         1e-5 to 3e-5       Larger model, more careful updating
e5-large-v2        5e-6 to 2e-5       Largest model, conservative LR
```

Lower LR prevents catastrophic forgetting of general representations.

### Catastrophic forgetting mitigation

```python
# Option 1: elastic weight consolidation (EWC)
# Penalize large changes to weights that are important for general tasks

# Option 2: learning rate warmup + small LR (most practical)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=50,           # gradual warmup
    optimizer_params={"lr": 2e-5},  # conservative LR
)

# Option 3: parameter-efficient fine-tuning (LoRA)
# Only fine-tune a small number of adapter parameters
# Preserves base model weights entirely
```

## When Few-Shot is Not Enough

Few-shot fine-tuning with 100-500 examples closes most of the domain gap for
domains where the vocabulary and concepts are partially covered by the pretraining
distribution. It may be insufficient when:

```
Situation                           Recommendation
────────────────────────────────────────────────────────────────────
Completely novel vocabulary         Domain-specific pretraining first
                                    (UDAP/TSDAE), then few-shot FT

Very different document structure   Structure-aware fine-tuning or
                                    document-specific encoders

Retrieval quality plateaus at       More labeled data or
low performance despite FT          domain-specific pretraining

Domain shifts over time             Continual learning or periodic
                                    retraining with new examples
```

These cases are addressed by the domain-specific datasets covered in the next note.

## Where This Fits in the Progression

```
01-why-domain-adaptation  → measure the gap
02-few-shot-retrieval     → close the gap with small data  ← you are here
03-domain-specific-datasets → close the gap with large data
```

## My Summary

Few-shot retrieval adapts pretrained bi-encoders to target domains using 100-500
labeled (query, positive document) pairs - dramatically less than the tens of
thousands required for training from scratch. The technique works because pretrained
models already capture rich language representations; few-shot examples merely
recalibrate which representations are relevant in the domain context. Training uses
contrastive loss with in-batch negatives, which multiplies each labeled pair into
31 effective training signals in a batch of 32. Hard negative mining - using the
current model to retrieve documents that look relevant but are not - provides the
strongest training signal and should always be applied. GPL extends this to zero
annotation by using LLMs to generate synthetic queries for corpus documents.
The learning curve is front-loaded: 50 examples often provides 60-70% of the
improvement achievable with 500 examples. Always evaluate on a held-out set and
a general benchmark to detect catastrophic forgetting. Few-shot fine-tuning is the
default first-choice domain adaptation technique for most production IR applications.
