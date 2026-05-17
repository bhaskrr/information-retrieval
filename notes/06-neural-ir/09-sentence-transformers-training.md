# Sentence Transformers Training

Sentence Transformers training is the process of fine-tuning or training from
scratch a transformer model to produce high-quality sentence embeddings optimized
for semantic similarity, retrieval, or classification tasks. The sentence-transformers
library (by UKPLab/Hugging Face) provides the standard framework for this - wrapping
HuggingFace models with pooling layers, loss functions designed for embedding tasks,
and training utilities specific to learning dense representations. While much of this
repo has used pretrained sentence-transformer models as black boxes, understanding
how they are trained is what allows you to adapt them to new domains, improve their
performance on specific tasks, and understand why certain retrieval systems fail.
Training your own sentence transformer is the step between using existing retrieval
models and building retrieval systems that genuinely understand your domain.

## Intuition

Standard BERT pretraining produces powerful contextual representations but they are
not sentence embeddings in any useful sense. Taking the [CLS] token or averaging all
token embeddings from a pretrained BERT model and calling it a sentence embedding
produces representations that do not cluster semantically similar sentences together.

The reason is objective mismatch. BERT was trained to predict masked tokens and
detect next sentence relationships - objectives that produce good token-level
representations but not sentence-level geometric structure. Two semantically identical
sentences like "the dog chased the cat" and "the feline was pursued by the canine"
produce very different BERT representations because BERT has never been explicitly
trained to make semantically similar sentences produce similar vectors.

Sentence Transformers training addresses this by fine-tuning on data that explicitly
teaches sentence-level similarity. The training signal is pairs or triplets of
sentences with known similarity relationships. The loss function pushes similar
sentences together and dissimilar sentences apart in embedding space. After training
for even a few epochs on the right data, semantically similar sentences cluster
together and the representations become useful for retrieval.

The key insight connecting this to IR: every bi-encoder used for dense retrieval
is a sentence transformer trained on (query, relevant passage) pairs with a
retrieval-appropriate loss. Understanding sentence transformer training means
understanding how dense retrieval models are built.

## Data Formats

Sentence transformer training requires paired or grouped data with known
relationship labels. Five data formats are used depending on the training objective:

### 1. Positive pairs (query, relevant_document)

The most common format for retrieval. Each example is a pair where both texts
are semantically related:

```python
InputExample(texts=["What is information retrieval?",
                     "Information retrieval is the process of finding relevant documents."])
```

Used with: MultipleNegativesRankingLoss, MegaBatchMarginLoss

### 2. Triplets (anchor, positive, negative)

Each example has an anchor text, a related positive, and an unrelated negative:

```python
InputExample(texts=["What is BM25?",
                     "BM25 is a probabilistic ranking function.",
                     "A recipe for chocolate cake with frosting."])
```

Used with: TripletLoss, BatchHardTripletLoss

### 3. Pairs with scores (text1, text2, similarity_score)

Each example is a pair with a continuous similarity score in [0, 1]:

```python
InputExample(texts=["The cat sat on the mat.", "A feline rested on the rug."],
             label=0.85)   # high similarity
InputExample(texts=["The cat sat on the mat.", "Stock markets fell sharply."],
             label=0.02)   # low similarity
```

Used with: CosineSimilarityLoss, ContrastiveLoss

### 4. Pairs with binary labels (text1, text2, is_duplicate)

Each example is a pair with a binary label (similar/not similar):

```python
InputExample(texts=["How do I reset my password?",
                     "I forgot my password, what do I do?"],
             label=1)   # duplicate
InputExample(texts=["How do I reset my password?",
                     "What is the weather forecast for tomorrow?"],
             label=0)   # not duplicate
```

Used with: ContrastiveLoss, OnlineContrastiveLoss

### 5. Groups (query, positive_1, positive_2, ..., negative_1, negative_2, ...)

Each query has multiple positives and/or hard negatives:

```python
# Handled in custom collators or with InputExample + loss combinations
```

Used with: MultipleNegativesRankingLoss over groups, custom training loops

## Core Loss Functions

Loss function selection is the most important training decision. Each loss
function encodes different assumptions about the training data and produces
embeddings with different geometric properties.

### MultipleNegativesRankingLoss (MNRL)

The most important loss for retrieval. Given a batch of (query, positive_doc)
pairs, each positive document serves as a negative for all other queries in the
batch - in-batch negatives:

```
Batch: [(q1, d1), (q2, d2), (q3, d3)]

For q1: d1 is positive, d2 and d3 are negatives
For q2: d2 is positive, d1 and d3 are negatives
For q3: d3 is positive, d1 and d2 are negatives
```

Loss:

$$L = \frac{1}{B} \times \sum_{i}^{B} - \log (\frac{\exp(\frac{similarity(q_i, d_i)}{\tau})}{\sum_{j=1}^{B} \exp(\frac{similarity(q_i, d_j)}{\tau})})$$

Where B = batch size, τ = temperature (typically 0.05).

**Why this works for retrieval:** It directly optimizes the query-document
similarity relative to all other documents in the batch. At inference time,
retrieval does exactly this - rank the relevant document above all others.

**Critical property:** Batch size is the number of negatives. Larger batches
create more informative training signal. This is why dense retrieval training
uses very large batches (512-2048) and why the original DPR paper used 7 hard
negatives per positive in addition to in-batch negatives.

**Data requirement:** Positive pairs only - no explicit negatives needed.
The loss creates negatives automatically from within-batch non-pairs.

### TripletLoss

Directly optimizes that the positive is closer than the negative by at least
a margin δ:

$L = \max(0, sim(q, neg)) - sim(q, pos) + \delta$

Where δ is the margin hyperparameter (typically 0.5).

**Geometric interpretation:** For each anchor, the positive must be at least
δ closer than the negative. The loss is zero when this is already satisfied.

**Data requirement:** Explicit triplets (anchor, positive, negative).

**When to use:** When you have explicit hard negatives identified through
BM25 or neural retrieval mining. The explicit negatives make training more
efficient than relying on random in-batch negatives.

### CosineSimilarityLoss

Minimizes the mean squared error between predicted cosine similarity and
a target similarity score:

$L = MSE(\cos(emb1, emb2), target\_score)$

Where target_score $\in [0, 1]$ is the human-labeled similarity.

**When to use:** STS (Semantic Textual Similarity) tasks where you have
continuous similarity judgments from human annotators. Not ideal for
retrieval where relevance is binary or graded differently.

### ContrastiveLoss

Pulls positive pairs together and pushes negative pairs apart with a margin:

```
L_pos = sim_distance(emb1, emb2)² × label
L_neg = max(0, margin - sim_distance(emb1, emb2))² × (1 - label)
L = (L_pos + L_neg) / 2
```

Where label=1 for positives, label=0 for negatives, and sim_distance is
Euclidean distance on normalized embeddings.

**When to use:** Binary labeled pairs, especially for duplicate detection
and question-answer matching tasks.

### MarginMSELoss (for distillation)

Trains a bi-encoder student to match the score margins of a cross-encoder teacher:

$L = MSE(sim(q, pos) - sim(q, neg), teacher\_score(q, pos) - teacher\_score(q, neg))$

The student learns to match the relative ordering of the teacher for each
(query, positive, negative) triple - not the absolute scores, just the margins.

**Why margin and not score:** Cross-encoder scores are not directly comparable
to bi-encoder cosine similarities (different scale and distribution). Margin
matching is scale-invariant and learns the relative ranking signal.

**When to use:** Knowledge distillation from a cross-encoder teacher to produce
a stronger bi-encoder. Used in TAS-B (Task-Aware Sampling for Bi-encoders).

### CachedMultipleNegativesRankingLoss

A memory-efficient variant of MNRL that caches document embeddings during
training, enabling very large effective batch sizes:

```
Standard MNRL:          batch_size negatives per query
CachedMNRL (cache=8):   batch_size × 8 negatives per query

Memory cost:  same as standard MNRL (cached embs don't require gradients)
Quality:      significantly better than standard MNRL at same batch size
```

This is the recommended loss for retrieval training when GPU memory is limited.

## Pooling Strategies

After the transformer produces contextual token embeddings, a pooling strategy
converts the variable-length sequence into a fixed-size sentence embedding:

### Mean pooling (most common)

Average all token embeddings (excluding padding):

```python
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings      = (token_embeddings * input_mask_expanded).sum(dim=1)
    sum_mask            = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask
```

Works well because it incorporates information from all tokens equally. The
standard for most sentence transformer models.

### CLS token

Use only the [CLS] token embedding (first token). Simple but often worse than
mean pooling because [CLS] does not always capture the full sentence meaning
after pretraining.

### Max pooling

Take the elementwise maximum across token embeddings. Captures the most
prominent feature per dimension. Less common, sometimes useful for short texts.

### Weighted mean (attention-weighted)

Weight token embeddings by their attention scores to the CLS token. Can
capture importance of different words but adds complexity.

## Training Data Sources

### NLI (Natural Language Inference)

Stanford NLI (SNLI) and Multi-Genre NLI (MNLI) provide (premise, hypothesis,
label) triples where label ∈ {entailment, contradiction, neutral}.

```
Entailment  → strong positive pair (premise and hypothesis are semantically similar)
Contradiction → strong negative pair
Neutral     → weak positive or ignored
```

Used in the original SBERT paper. Good for learning general semantic similarity.
Not ideal for retrieval - NLI relationships are not the same as document relevance.

### MS MARCO

The standard training data for retrieval bi-encoders:

```
Format: (query, positive_passage, hard_negative_passage)
Size:   530K training triples
Source: Real Bing queries + relevant/irrelevant passages from Bing results
```

Every major pretrained bi-encoder (DPR, E5, GTE, all-MiniLM) used MS MARCO.

### STS (Semantic Textual Similarity)

STS-B, SICK, STS12-STS16 provide human-labeled sentence pairs with similarity
scores from 0 to 5. Used for evaluation more than training, but useful for
STS fine-tuning.

### Domain-specific training data

Custom (query, relevant_document) pairs from your application. The most valuable
training data for domain adaptation - covered in detail in 13-domain-adaptation/.

## Training Hyperparameters

### Learning rate

Critical and often mistuned. Recommendations by base model size:

```
BERT-small / MiniLM:   2e-5 to 5e-5
BERT-base:             2e-5 to 3e-5
BERT-large / RoBERTa:  1e-5 to 2e-5
LLM-based (7B+):       1e-6 to 5e-6
```

Too high: catastrophic forgetting of pretrained representations.
Too low: training too slow, may not converge.

### Batch size

For MNRL, larger is better - more in-batch negatives = stronger training signal:

```
Minimum viable:    16-32 pairs
Good:              64-128 pairs
Standard for SOTA: 512-2048 pairs (requires gradient accumulation on consumer GPU)
```

To simulate large batch size with limited GPU memory:

```python
model.fit(
    ...,
    gradient_accumulation_steps=8,   # effective batch = batch_size × 8
)
```

### Warmup

Linear warmup for first 5-10% of training steps prevents instability:

```python
warmup_steps = int(0.1 * total_steps)   # 10% warmup
```

### Epochs

For fine-tuning on a small domain dataset (100-1000 pairs): 3-10 epochs.
For training on MS MARCO (530K pairs): 1-3 epochs.
For NLI training (550K pairs): 1-4 epochs.

Overfitting on small datasets is a significant risk - use validation loss
or NDCG@10 on a held-out set for early stopping.

## The Full Training Pipeline

A complete sentence transformer training pipeline for retrieval:

```
1. Data preparation
   ├── Collect (query, positive_doc) pairs from domain
   ├── Mine hard negatives using BM25 or weak neural model
   └── Split into train/val (90/10)

2. Model initialization
   ├── Load pretrained base: BERT, RoBERTa, or domain-specific
   └── Add pooling layer (mean pooling)

3. Loss configuration
   ├── MNRL for retrieval (in-batch negatives)
   └── Optional: TripletLoss if hard negatives are available

4. Training
   ├── Optimizer: AdamW with linear warmup + cosine decay
   ├── Batch size: as large as GPU allows
   └── Epochs: 3-5 with early stopping on val NDCG@10

5. Evaluation
   ├── NDCG@10 on domain test set
   ├── BEIR subset for general performance
   └── Latency benchmarking

6. Optional: Matryoshka fine-tuning
   └── Wrap MNRL in MRL objective for adaptive dimension support
```

## Loss Function Selection Guide

| Data you have                   | Task              | Best loss                          |
| ------------------------------- | ----------------- | ---------------------------------- |
| (query, doc) positive pairs     | Retrieval         | MultipleNegativesRankingLoss       |
| (anchor, pos, neg) triplets     | Retrieval         | TripletLoss                        |
| (text1, text2, 0-1 score)       | STS               | CosineSimilarityLoss               |
| (text1, text2, 0/1 binary)      | Duplicate detect  | ContrastiveLoss                    |
| (anchor, pos, neg) + teacher    | Retrieval+distill | MarginMSELoss                      |
| Large batch, GPU memory limited | Retrieval         | CachedMultipleNegativesRankingLoss |
| MRL objective wanted            | Any retrieval     | Wrap any loss in MatryoshkaLoss    |

## Common Training Mistakes

| Mistake                                        | Impact                     | Fix                                             |
| ---------------------------------------------- | -------------------------- | ----------------------------------------------- |
| Too-small batch size with MNRL                 | Weak negatives, slow conv. | batch_size >= 32 or accumulate gradients        |
| Not normalizing embeddings                     | Wrong similarity metric    | Always add Normalize() layer or normalize after |
| Using same examples as positives and negatives | Degenerate solutions       | Ensure no duplicate (qi, di) in same batch      |
| Learning rate too high                         | Catastrophic forgetting    | Use 2e-5 or lower with warmup                   |
| No validation during training                  | Overfit to train set       | Early stop on val NDCG                          |
| Not mining hard negatives                      | Easy negatives, weak model | Mine after epoch 1 then retrain                 |
| Wrong model for base (English-only for CLIR)   | Poor generalization        | Use multilingual base for CLIR tasks            |

## My Summary

Sentence transformers training fine-tunes transformer models to produce useful
sentence-level embeddings by optimizing task-specific loss functions on paired
data. The framework adds a pooling layer (mean pooling is standard) and optionally
a normalization layer to a pretrained transformer, then trains on (query, document)
pairs using retrieval-appropriate losses. MultipleNegativesRankingLoss (MNRL) is the
standard for retrieval - it creates in-batch negatives automatically, so positive
pairs alone are sufficient training data, and larger batches produce stronger
training signal through more negatives per query. TripletLoss is preferred when
explicit hard negatives are available. MarginMSELoss enables knowledge distillation
from a cross-encoder teacher to a bi-encoder student. Hard negative mining - using
the current model to retrieve documents that look relevant but are not - is the
single highest-impact improvement over random or in-batch negatives. The complete
training recipe cycles between training with MNRL, mining hard negatives with the
current model, and retraining on the harder examples. Every pretrained bi-encoder
model used in this repo - from all-MiniLM to E5 to GTE - was built using exactly
these components and this training loop.
