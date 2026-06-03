# Contrastive Learning for Retrieval

Contrastive learning for retrieval is the training paradigm that teaches a
bi-encoder to produce embedding spaces where semantically similar texts are
geometrically close and semantically dissimilar texts are far apart, by directly
optimizing a loss function that contrasts positive pairs against negative pairs.
It is the foundational training objective behind every dense retrieval model - from
the original DPR to modern E5, GTE, and Cohere Embed. Understanding contrastive
learning at the mathematical level is what allows IR practitioners to make principled
decisions about training data collection, batch size selection, negative mining
strategy, temperature tuning, and loss function choice - all of which have large
and systematic effects on the quality of the resulting retrieval model. This note
covers the theory, variants, and practical mechanics of contrastive learning as
applied specifically to the retrieval setting.

## Intuition

A newly initialized bi-encoder assigns essentially random embeddings to all texts.
The embedding space has no semantic structure - texts about dogs and texts about
astrophysics might accidentally be nearby, and two near-synonyms might be far apart.
Contrastive learning builds semantic structure into this space by applying a simple
pressure: relevant query-document pairs should attract each other, and irrelevant
pairs should repel each other.

The learning signal is entirely relational. The model is never told "the embedding
for 'cardiac arrest' should be the vector [0.23, -0.41, ...]". It is told "the
embedding for 'cardiac arrest' should be closer to 'myocardial infarction' than
to 'automotive repair manual'." The model learns from these relative comparisons
and gradually organizes the embedding space so that semantic similarity corresponds
to geometric proximity.

The challenge is making this learning efficient. There are an astronomical number
of possible positive and negative pairs in any text corpus. Contrastive learning
addresses this through clever sampling - specifically by exploiting the structure
of training batches to get multiple negative examples for free from each positive
example.

## The Contrastive Learning Setup

### The canonical retrieval formulation

Given a corpus of queries Q and documents D with a known set of positive pairs
P ⊆ Q × D:

```
Training objective: learn encoder functions f_q and f_d such that:
  similarity(f_q(q), f_d(d⁺)) >> similarity(f_q(q), f_d(d⁻))
  for all (q, d⁺) ∈ P and d⁻ ∉ P_q
```

Where P_q is the set of positive documents for query q, and d⁻ is any document
not relevant to q.

The encoder functions f_q and f_d can be:

- **Shared encoder:** same BERT-base model processes both query and document
  (used in bi-encoder architecture with mean pooling)
- **Separate encoders:** different models for query and document (used in
  DPR - one BERT-base for queries, one for passages)
- **Asymmetric encoders:** different sizes (small query encoder for speed,
  large document encoder for quality)

In practice, shared encoders (same weights for both) or shared with separate
projection heads are most common because they are simpler and generalize better
with limited training data.

### In-batch negative sampling

The key efficiency insight: within a training batch of B positive pairs
[(q₁, d₁⁺), (q₂, d₂⁺), ..., (qB, dB⁺)], every document dⱼ⁺ is a negative
example for all queries qᵢ where i ≠ j.

```
Batch of B = 4 pairs:
  (q₁, d₁⁺): "cardiac arrest treatment" → "CPR should be performed immediately"
  (q₂, d₂⁺): "python list sorting"      → "Use sorted() or .sort() method"
  (q₃, d₃⁺): "coffee brewing methods"   → "French press extracts oils and sediment"
  (q₄, d₄⁺): "quantum entanglement"     → "Correlated quantum states across distance"

For query q₁:
  Positive: d₁⁺ (CPR document)
  Negatives: d₂⁺, d₃⁺, d₄⁺ (python, coffee, quantum - clearly irrelevant)

Total negative pairs created: B × (B-1) = 4 × 3 = 12 from just 4 labeled positives
```

With B = 256 pairs in a batch, you get 256 positives and 256 × 255 = 65,280 negative
pairs automatically. This is the fundamental efficiency advantage of contrastive
learning - exponential negative examples from linear labeled data.

## InfoNCE Loss (Noise Contrastive Estimation)

The standard loss function for contrastive retrieval training. Also called
MultipleNegativesRankingLoss in sentence-transformers.

### Formula

For a batch of B positive (query, document) pairs:

```
L_InfoNCE = -(1/B) × Σᵢ log(
    exp(sim(qᵢ, dᵢ⁺) / τ) /
    Σⱼ exp(sim(qᵢ, dⱼ) / τ)
)
```

Where:

- sim(q, d) = cosine similarity between normalized embeddings
- τ = temperature hyperparameter (typically 0.01-0.1)
- dᵢ⁺ = positive document for query i
- dⱼ = all documents in the batch (including dᵢ⁺ itself)

The denominator is the softmax normalization over all B documents in the batch.
The numerator is the contribution from the positive pair. The loss minimizes the
negative log-likelihood that the positive document is selected from the batch.

### Geometric interpretation

InfoNCE pushes the query embedding toward the positive document while
simultaneously pushing it away from all in-batch negatives:

```
Before training:
  q₁ ────────────────────────────── d₁⁺  (far apart, high loss)
  q₁ ─────────────────── d₂⁻  (may be nearby, if unlucky random init)

After training:
  q₁ ──── d₁⁺  (close together, low loss contribution)
  q₁ ──────────────────────────── d₂⁻  (far apart, repelled)
```

The temperature τ controls the sharpness of the softmax:

```
Low τ (e.g., 0.01):
  Softmax is very peaked → model focuses on the hardest in-batch negative
  High gradient on the closest negative
  Risk: unstable training if many near-duplicate positives in batch

High τ (e.g., 0.1):
  Softmax is more uniform → gradient spread across all negatives
  Smoother training signal
  Risk: insufficient learning pressure from easy negatives
```

## Temperature - The Critical Hyperparameter

Temperature τ in InfoNCE is arguably the most important hyperparameter for
contrastive retrieval training. It controls how the model treats the difficulty
distribution of negatives.

### Temperature and the concentration property

At low temperature, the loss is dominated by the hardest (closest) negative:

```
sim(q, d⁺) = 0.85
sim(q, d₁⁻) = 0.83   ← very hard negative
sim(q, d₂⁻) = 0.60
sim(q, d₃⁻) = 0.40

Low τ = 0.02:
  softmax scores: [exp(0.85/0.02), exp(0.83/0.02), exp(0.60/0.02), exp(0.40/0.02)]
                  [e^42.5, e^41.5, e^30, e^20]
  Dominated by comparison between 0.85 and 0.83 (hard negative)
  Gradient almost entirely from pushing d₁⁻ away from q

High τ = 0.10:
  softmax scores: [exp(8.5), exp(8.3), exp(6.0), exp(4.0)]
  All negatives contribute meaningfully to the gradient
  Smoother learning signal
```

### Temperature and uniformity-alignment tradeoff

Contrastive learning optimizes two competing objectives:

```
Alignment:   positive pairs should be close (minimize sim(q, d⁺) distance)
Uniformity:  embeddings should spread uniformly over the sphere
             (prevent all embeddings from collapsing to the same point)
```

Temperature regulates this tradeoff:

- Very low temperature → extreme alignment, poor uniformity (collapse risk)
- Very high temperature → good uniformity, poor alignment

Optimal temperature places the model at the right point on this tradeoff.
For dense retrieval on MS MARCO, τ ≈ 0.05 is commonly used. For sentence
similarity tasks where fine-grained discrimination matters, τ ≈ 0.02-0.05.

## Batch Size and Its Effect on Training Quality

Batch size is the second most important hyperparameter for contrastive retrieval
training, and it interacts critically with temperature.

### Effective number of negatives

With B pairs per batch and no additional hard negatives:

```
Negatives per query = B - 1
```

The model's discrimination ability is fundamentally limited by the number of
negatives it sees simultaneously. More negatives → stronger training signal.

### Large batch training regimes

The evolution of dense retrieval training shows a clear trend toward larger
batches:

```
DPR (2020):    Batch = 64,  additional hard negs = 7 per positive
               Effective negatives per query ≈ 63 + 7 = 70

ANCE (2021):   Batch = 128, hard negatives refreshed periodically
               Effective negatives ≈ 127

RocketQA (2021): Batch = 4096 (with distributed training across GPUs)
               Effective negatives ≈ 4095

E5 (2023):     Batch = 16,384 (across many GPUs)
               Effective negatives ≈ 16,383
```

The performance improvement from DPR to E5 is partially attributable to
this dramatic increase in effective batch size - more negatives means
better-calibrated embeddings.

### Gradient accumulation as a substitute

For researchers without access to large GPU clusters:

```
Effective batch size = actual_batch_size × gradient_accumulation_steps

Example:
  GPU memory allows: batch_size = 32
  Gradient accumulation: steps = 16
  Effective batch size: 512
  Negatives per query: 511

Trade-off: gradient accumulation is mathematically not identical to
           large batch training for InfoNCE (gradients accumulated across
           micro-batches miss cross-micro-batch negatives)
```

True cross-batch negatives (where negatives from previous micro-batches are
stored and reused) are available through techniques like MoCo (momentum
contrast), which maintains a queue of recent embeddings as negatives.

### The degenerate solution problem

A naive contrastive learner can cheat: if all positive pairs in a batch come
from the same domain and all in-batch negatives come from different domains,
the model learns to encode domain rather than relevance. Any query in domain A
gets high similarity with any document in domain A, regardless of actual
relevance.

Mitigation: ensure diverse query-document topic pairs within each batch.
For MS MARCO training, ensure queries from different topics appear in each batch.

## Symmetric vs Asymmetric Losses

Standard InfoNCE as described above is applied in one direction: for each query,
find the positive document among batch documents. But we can also apply it in the
reverse direction.

### Bidirectional InfoNCE

```
L_symmetric = L_query→doc + L_doc→query

L_query→doc: for each query, find its document in the batch
L_doc→query: for each document, find its query in the batch

Combined:
  L = L_query→doc + λ × L_doc→query
  Where λ controls the weight of the reverse direction
```

Symmetric loss produces more calibrated embeddings because both query and
document encoders receive direct gradient signal. Used in CLIP and many
modern embedding models.

### Asymmetric cases

Sometimes asymmetric loss is preferable:

```
When queries and documents have very different lengths:
  Long documents contain many concepts → a document may match many queries
  Short queries are specific → most documents do not match most queries

For retrieval: query→doc loss is primary (retrieval is query-driven)
              doc→query loss provides regularization but should be weighted less
```

## The False Negative Problem

In-batch negatives can accidentally be true positives - documents that genuinely
answer the query but happen to appear in the same batch. Training the model to
push away these false negatives produces incorrect gradient signals.

### Prevalence in large corpora

For MS MARCO with 8.8M passages and queries with 1-3 relevant passages each:

```
In a batch of B = 256 queries:
  Each query has ~1 positive passage out of 8.8M
  P(false negative in batch) = B × P(any document is positive for query i)
                              ≈ 256 × (1/8.8M) × 256
                              ≈ very low (~0.001)
```

For smaller corpora or queries with many positive documents, false negatives
become a significant problem:

```
FAQ corpus with 10,000 documents, queries answered by many documents:
  P(false negative in batch of 256) could be 0.1 or higher
  10% false negatives in training batches → significant incorrect gradient
```

### Mitigation strategies

**BM25 negative filtering:**
Before training, check each in-batch negative against BM25:
if BM25 score(query, in-batch-negative) is very high, flag as potential
false negative and remove from the training signal.

**Ground truth filtering:**
If you have complete relevance judgments, filter out known positives
from the negative set at training time.

**Denoised InfoNCE:**
A variant that estimates the false negative probability and corrects
the loss:

```
L_denoised = L_InfoNCE - λ × mean(P(false_negative | query, negative) × sim(q, d⁻))
```

## Augmentation-Based Contrastive Learning

Rather than relying only on labeled (query, document) pairs, augmentation-based
methods generate positive pairs from unlabeled text:

### TSDAE (Denoising Autoencoder)

Creates positive pairs by corrupting sentences and training the encoder to
produce similar representations for corrupted and original text:

```
Original:   "dense retrieval uses neural encoders"
Corrupted:  "dense encoders retrieval uses neural" (word shuffle)

Positive pair: (corrupted, original)
Training: corrupted sentence should encode similarly to original
```

TSDAE is used for domain adaptive pretraining - adapting a general model
to a new domain using unlabeled text before supervised fine-tuning.

### SimCSE

Generates positive pairs from the same sentence by running it through dropout
twice - two forward passes with different dropout masks produce slightly
different embeddings from the same input:

```
sentence = "heart attack emergency treatment"
pass_1 = encoder(sentence, dropout=mask_1)  → embedding_1
pass_2 = encoder(sentence, dropout=mask_2)  → embedding_2

Positive pair: (embedding_1, embedding_2)
  These should be similar (same text, different dropout)

Negatives: all other sentences in the batch
```

SimCSE requires no labels and dramatically improves embedding quality over
raw BERT. It is often used as an initialization step before supervised contrastive
training on labeled retrieval data.

### Cropping and span selection

For long documents, create positive pairs by treating two spans from the
same document as semantically related:

```
Document: "Paris is the capital of France. It is known for the Eiffel Tower,
           the Louvre museum, and its cuisine. The city has 2.1 million inhabitants."

Span 1: "Paris is the capital of France."
Span 2: "The city has 2.1 million inhabitants."

Positive pair: (span_1, span_2)
  Both describe the same entity (Paris), should be similar
```

This technique, used in ICT (Inverse Cloze Task) and REALM, creates millions
of positive pairs from raw document corpora.

## The Full Training Objective for Dense Retrieval

In practice, state-of-the-art dense retrieval training combines multiple
components:

```
L = L_InfoNCE(in-batch negatives)
  + λ_hn × L_InfoNCE(hard negatives)
  + λ_kl × L_KL(knowledge distillation from cross-encoder teacher)
  + λ_reg × L_regularization(embedding uniformity)
```

### Component 1 - In-batch negatives

Efficiency: provides many negatives cheaply.
Weakness: negatives are random and easy (different topic).

### Component 2 - Hard negatives

Quality: provides the most informative training signal.
Source: BM25 top-k or previous model top-k, minus known positives.
Weakness: expensive to mine, requires an existing retrieval system.

### Component 3 - Knowledge distillation

Quality ceiling: uses cross-encoder to provide soft labels for each
(query, document) pair, capturing fine-grained relevance signal beyond
binary positive/negative.

```
teacher_score = cross_encoder(query, document)   ← precise but slow
student_score = bi_encoder(query) · bi_encoder(document)   ← fast

L_KL = KL_divergence(softmax(student_scores/τ), softmax(teacher_scores/τ))

Distillation teaches the bi-encoder to approximate the cross-encoder's
ranking even though it cannot see the full cross-attention.
```

### Component 4 - Uniformity regularization

Prevents embedding collapse by ensuring embeddings are spread uniformly:

```
L_uniform = log E[exp(-2 × ||f(x) - f(y)||²)]
  where x, y are random samples from the training distribution

Minimizing this loss pushes embeddings toward uniform distribution on hypersphere
```

## Curriculum Learning for Retrieval Training

Start with easy negatives and progressively introduce harder ones:

```
Phase 1 (epochs 1-2):  In-batch negatives only
                        Easy random negatives, model learns basic semantic structure

Phase 2 (epoch 3):     Add BM25 hard negatives
                        Lexically similar but non-relevant negatives
                        Model learns to go beyond lexical matching

Phase 3 (epoch 4+):    Add model-mined hard negatives (ANCE strategy)
                        Use current model to retrieve hard negatives
                        Hardest possible negatives given current model state
```

Curriculum learning improves final model quality because:

1. Early easy negatives establish the basic embedding structure
2. Later hard negatives refine discrimination at the semantic boundary

Starting immediately with hard negatives on a randomly initialized model
often leads to training instability.

## Practical Training Configuration

A complete configuration for training a competitive dense retrieval model
on MS MARCO:

```
Base model:           bert-base-uncased or roberta-base
Pooling:              mean pooling of last layer (L2 normalized)
Batch size:           256-512 per GPU (use gradient accumulation to reach 1024+)
Learning rate:        1e-5 to 3e-5 (lower for larger models)
Warmup:               5-10% of training steps
Training epochs:      3-5 on full MS MARCO (stop early if validation plateaus)
Temperature τ:        0.02-0.05
Negatives per query:  7-15 BM25 hard negatives + in-batch
Loss:                 InfoNCE on (positives + hard negatives + in-batch)
Evaluation:           Recall@100 on MS MARCO dev set every 500 steps
Optimizer:            AdamW with weight decay 0.01
```

## Common Training Failures and Diagnostics

### Embedding collapse

Symptom: all embeddings converge to the same or a few points in space.
All similarities become close to 0 or 1.

Diagnostic: compute pairwise similarities for 1000 random embeddings.
If std(similarities) < 0.05, collapse is occurring.

Causes:

- Learning rate too high
- Temperature too low (creates extreme gradients)
- Batch lacks diversity (all queries from same domain)

Fix: add uniformity regularization, increase temperature, ensure batch diversity.

### Slow convergence / plateauing early

Symptom: Recall@100 stops improving after 1 epoch.

Diagnostic: check if in-batch negatives are too easy (accuracy on in-batch
discrimination reaches 0.99 within one epoch).

Fix: add hard negatives, reduce temperature to create harder in-batch contrasts.

### Negative mining false positives contaminating training

Symptom: training loss decreases but Recall@100 on validation set degrades
after adding hard negatives.

Diagnostic: inspect mined hard negatives manually. Are any actually relevant?

Fix: filter hard negatives through relevance judgment or BM25 threshold check.

## My Summary

Contrastive learning trains bi-encoders by optimizing the InfoNCE loss - a
softmax-based objective that maximizes the probability of identifying the positive
document among all documents in a training batch. The central efficiency insight
is in-batch negative sampling: a batch of B positive pairs automatically creates
B × (B-1) negative pairs, producing exponential negative signal from linear labeled
data. Temperature τ is the most critical hyperparameter: low temperature
concentrates learning on the hardest in-batch negative, while high temperature
distributes gradient across all negatives - typical values range from 0.02 to 0.05
for retrieval. Batch size is the second most critical hyperparameter: larger batches
provide more negatives per query and consistently improve model quality, explaining
why E5 and GTE significantly outperform DPR despite similar architectures. The full
training objective for state-of-the-art models combines in-batch negatives (for
efficiency) with hard negatives (for quality) and optionally knowledge distillation
from a cross-encoder teacher (for maximum quality). False negatives - in-batch
documents that are actually relevant to the query - corrupt the training signal
and must be filtered. Curriculum learning (easy negatives first, then hard negatives)
stabilizes training and improves final model quality compared to starting with hard
negatives immediately.
