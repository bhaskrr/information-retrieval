# Batch Size and Temperature

Batch size and temperature are the two hyperparameters with the largest systematic
effect on contrastive retrieval training quality, yet they are often treated as
incidental choices set by GPU memory constraints and default values rather than
as first-class design decisions. Batch size determines how many negatives each
query faces simultaneously during training - directly controlling the difficulty
and richness of the contrastive signal. Temperature scales the similarity scores
in the InfoNCE softmax - controlling how sharply the model discriminates between
positive and negative pairs and how the learning signal is distributed across
negatives of varying difficulty. Understanding how these two hyperparameters
interact with each other, with hard negative mining, and with the scale of the
training data is what allows practitioners to make principled choices that maximize
model quality within the constraints of their hardware and training budget.

## Intuition

### Batch size intuition

Contrastive learning asks: "given this query and these candidates, can you identify
the relevant document?" The difficulty and informativeness of this question scales
with the number of candidates - the size of the choice set.

With batch size 8, a query faces 7 in-batch negatives. If they are all from
completely different topics, the task is trivially easy: the relevant document
stands out immediately by topic alone. The model learns very little from getting
this easy question right over and over.

With batch size 512, a query faces 511 in-batch negatives. Even with random
sampling, several will be topically adjacent to the query. The model must learn
finer-grained distinctions to correctly identify the positive. Each training step
is far more informative.

The intuition: batch size controls the discriminativeness of the training task.
Larger batches make the task harder and more informative, producing better-calibrated
embedding spaces. This is why the progression from DPR (batch size 64) to E5
(batch size 16,384+) corresponds to consistent quality improvements.

### Temperature intuition

Temperature controls how confident the model needs to be to produce low loss.

With high temperature (τ = 0.5), the softmax is very soft. A model that puts the
positive at rank 1 with similarity 0.85 and has a hard negative at similarity 0.80
produces only a small loss - the soft distribution does not strongly penalize the
nearby negative. The model learns to make rough distinctions but not fine-grained
ones.

With low temperature (τ = 0.02), the softmax is very sharp. The same model would
produce very high loss if any negative is at similarity 0.80 when the positive
is at 0.85. The model is forced to push the positive far above all negatives -
producing tightly clustered positive pairs and well-separated negatives.

The intuition: temperature controls the precision requirement of the embedding
space. Low temperature demands geometric precision - positives must be very close
to their queries and negatives must be far. High temperature tolerates approximate
proximity.

## Batch Size: The Deep Analysis

### Why batch size is fundamentally linked to learning quality

The InfoNCE loss for query i in a batch of size B:

```
L_i = -log(
    exp(sim(qᵢ, dᵢ⁺) / τ) /
    Σⱼ₌₁ᴮ exp(sim(qᵢ, dⱼ) / τ)
)
```

The denominator contains B terms - the positive and B-1 negatives. The loss
measures how hard it is to identify the positive among B choices. As B increases:

```
B = 8:    loss ≤ log(8) = 2.08     (model identifies positive among 8)
B = 64:   loss ≤ log(64) = 4.16    (harder task - more informative gradient)
B = 512:  loss ≤ log(512) = 6.24   (much harder - much more informative)
B = 4096: loss ≤ log(4096) = 8.32  (very challenging - very informative)
```

The upper bound of the loss scales logarithmically with batch size. A random
model achieves loss ≈ log(B). As training progresses, loss decreases from
this maximum - the rate and extent of decrease determines model quality.

### Batch size and the false negative effect

With very large batches, the probability of including false negatives (documents
that are actually relevant to the query but treated as negatives) increases:

```
For MS MARCO (8.8M passages, ~1 relevant per query):
  P(false negative in batch of B) ≈ B/8.8M ≈ 0.015% at B=1,000
                                   ≈ 0.19%  at B=16,384

For a dense corpus (100K documents, ~10 relevant per query):
  P(false negative in batch of B) ≈ B×10/100K = B/10,000
                                   ≈ 10% at B=1,000
                                   ← significant false negative contamination
```

For sparse relevance corpora (few relevant documents per query, large corpus),
very large batches are safe. For dense corpora or datasets with many positives,
large batches require false negative filtering at scale.

### The saturation effect

Beyond a certain batch size, marginal improvement from additional negatives
diminishes:

```
Quality vs batch size (approximate, BEIR NDCG@10):
  B = 32:    0.42
  B = 128:   0.46  (+0.04 over B=32)
  B = 512:   0.49  (+0.03 over B=128)
  B = 2048:  0.51  (+0.02 over B=512)
  B = 8192:  0.52  (+0.01 over B=2048)
  B = 32768: 0.53  (+0.01 over B=8192)
```

The relationship is logarithmic rather than linear - doubling batch size
produces diminishing returns at large scales. This explains why the gap
between E5 (16K batch) and a model with 512 batch size is real but not
enormous - you get most of the benefit well before the largest batch sizes.

### Gradient accumulation: a partial substitute

True large batch training requires that all B examples are present simultaneously
when computing the loss - all B document embeddings contribute to each denominator.
Gradient accumulation across micro-batches does not achieve this:

```
True B=512 training:
  Forward pass: compute 512 query embs and 512 doc embs simultaneously
  Loss: InfoNCE over all 512 pairs (512 × 511 negative pairs)
  Backward: gradient through all 512 × 512 similarities

Gradient accumulation with micro_batch=32, accumulate=16:
  Forward pass 1: compute 32 query embs and 32 doc embs
  Loss: InfoNCE over 32 pairs (32 × 31 = 992 negative pairs per micro-batch)
  ... repeat 16 times, accumulate gradients
  Backward: average gradient over 16 micro-batches

Key difference: cross-micro-batch negatives are absent
  Query from micro-batch 1 never sees documents from micro-batches 2-16 as negatives
  Effective negatives: 31 per query (not 511)
  Gradient accumulation achieves the gradient averaging benefit but NOT the
  additional in-batch negatives benefit
```

True large batch training requires distributed training where GPUs each compute
a subset of the batch and cross-GPU negatives are shared. Libraries like
GradCache or multi-GPU InfoNCE implementations enable this.

### GradCache: memory-efficient large batch training

GradCache decouples embedding computation from loss computation:

```
Step 1: Compute embeddings for all B queries and documents
        (using multiple micro-batches, no gradient retained)
Step 2: Cache all B × 2 embeddings
Step 3: Compute full B × B similarity matrix and InfoNCE loss
        (uses cached embeddings, no backward through encoder yet)
Step 4: Propagate gradients back through cached embeddings to encoder
        (using gradient checkpointing)
```

GradCache achieves true large-batch InfoNCE loss with the memory footprint of
small batches. It is the standard approach for training state-of-the-art dense
retrievers on single-node hardware.

## Temperature: The Precision Parameter

### The temperature-quality landscape

Temperature affects three distinct aspects of embedding space geometry:

**Alignment** (positive pairs close together):
Low temperature → high gradient on misaligned positives → strong alignment.

**Uniformity** (embeddings spread uniformly on hypersphere):
Low temperature → sharp softmax → model focuses only on hardest negatives →
poor coverage of the full negative space → less uniformity.

**Separation** (negatives far from positives):
Low temperature → model must push positives well above negatives → good separation.

The optimal temperature balances all three:

```
Too high (τ ≥ 0.2):
  Alignment:   moderate (soft gradients, slow to align)
  Uniformity:  good (gradient spread across many negatives)
  Separation:  poor (easy to satisfy soft softmax)
  Result:      embeddings are spread but not precisely organized

Optimal (τ ≈ 0.02-0.07):
  Alignment:   strong
  Uniformity:  balanced
  Separation:  strong
  Result:      well-organized embedding space with precise retrieval

Too low (τ ≤ 0.01):
  Alignment:   very strong but potentially brittle
  Uniformity:  poor (all gradient concentrated on single hardest negative)
  Separation:  extreme but potentially collapsed
  Result:      embeddings may collapse - all positives cluster in a small region
```

### Temperature calibration empirically

Published optimal temperatures for different settings:

```
Task                             Optimal τ    Source
──────────────────────────────────────────────────────────────────────
MS MARCO dense retrieval         0.02-0.05    DPR, ANCE, E5 papers
Sentence similarity (STS)        0.02-0.05    SimCSE
Image-text retrieval (CLIP)      0.07         CLIP original paper
Code search                      0.05-0.10    CodeBERT papers
Cross-lingual retrieval          0.07-0.10    mE5, LaBSE
Bi-encoder with distillation     0.01-0.03    TAS-B, marginMSE papers
```

The pattern: lower temperatures are better when negatives are already hard
(hard negative mining reduces the need for temperature to create difficulty).
Higher temperatures are better when negatives are random (temperature must
substitute for negative hardness).

### The relationship between temperature and negative hardness

A critical insight: temperature and negative hardness are partially substitutable:

```
Hard negatives + high temperature ≈ Easy negatives + low temperature

Both combinations force the model to make precise distinctions:
  Hard negs + high temp:  negatives are intrinsically close to the positive,
                          temperature doesn't need to create artificial difficulty
  Easy negs + low temp:   negatives are far from the positive, but low temperature
                          makes the model work hard to separate even easy negatives
```

This means:

- If you have strong hard negative mining, temperature can be moderate (0.05-0.07)
- If you rely entirely on in-batch random negatives, lower temperature (0.01-0.03)
  compensates for the easy negatives

The best configurations use both: hard negatives AND calibrated temperature.

### Temperature and the false negative problem revisited

Low temperature amplifies the false negative problem. When τ is very small:

```
False negative d_false at similarity 0.82 (close to positive at 0.85):
  At τ = 0.1: softmax weight ratio = exp(0.82/0.1) / exp(0.85/0.1)
            = exp(-0.3) ≈ 0.74 (moderate impact on gradient)

  At τ = 0.01: softmax weight ratio = exp(0.82/0.01) / exp(0.85/0.01)
             = exp(-3) ≈ 0.05 (large impact - false negative dominates gradient)
```

At very low temperature, false negatives close to the positive produce
extremely large gradients that strongly push the positive away from the
correct region. This is why aggressive temperature reduction requires
simultaneous aggressive false negative filtering.

## Batch Size and Temperature Interaction

The two hyperparameters interact in a specific way that has important practical
implications:

### The difficulty-calibration interaction

Batch size controls the nominal difficulty of the training task (how many
candidates to distinguish). Temperature controls how precisely the model must
complete that task to produce low loss.

```
Jointly considering batch size and temperature:

Small B, high τ:  easy task, softly evaluated → very easy training → poor quality
Small B, low τ:   easy task, precisely evaluated → model learns to be confident
                  on easy questions → good at easy cases, poor on hard ones
Large B, high τ:  hard task, softly evaluated → model learns rough distinctions
                  → moderate quality
Large B, low τ:   hard task, precisely evaluated → model must precisely identify
                  positive among many challenging candidates → highest quality
```

The optimal configuration is large B with appropriately low τ - but the specific
optimal τ for a given B is not independently tunable; it depends on the distribution
of similarities in the batch.

### The effective temperature

A useful concept: the effective temperature is the scaled temperature that produces
the same gradient distribution for different batch sizes:

```
Effective temperature ≈ τ × log(B) / log(B_reference)

At B = 64 and τ = 0.05:   effective_τ = 0.05 × log(64)/log(256) = 0.038
At B = 256 and τ = 0.05:  effective_τ = 0.05 (reference)
At B = 1024 and τ = 0.05: effective_τ = 0.05 × log(1024)/log(256) = 0.063
```

This implies: as batch size increases, slightly increasing temperature produces
comparable learning dynamics. Very large batch sizes with very low temperature
create extremely concentrated gradients that may destabilize training.

### Rule of thumb: batch size and temperature co-tuning

```
B = 32:    τ = 0.01-0.02  (low temp compensates for small batch)
B = 64:    τ = 0.02-0.03
B = 128:   τ = 0.03-0.05
B = 256:   τ = 0.05-0.07  ← DPR configuration
B = 512:   τ = 0.05-0.08
B = 2048:  τ = 0.07-0.10
B = 16384: τ = 0.07-0.15  ← E5/GTE configuration
```

The pattern: as batch size increases, optimal temperature rises. Very large
batches already create hard training tasks; low temperature on top of this
creates instability.

## The Learnable Temperature Approach

Rather than treating temperature as a fixed hyperparameter, some models learn it:

### CLIP's learnable temperature

OpenAI's CLIP model initializes temperature at τ = 0.07 and learns it
as a scalar parameter during training:

```
τ ← exp(log_τ)   (parameterized as log to ensure positivity)
∂L/∂log_τ        (temperature receives gradient from InfoNCE loss)
```

The temperature is clipped to be ≥ 0.01 to prevent collapse.

During CLIP training, the learned temperature converges to approximately 0.07

- close to the hand-tuned optimal. This suggests learnable temperature
  is a useful robustness mechanism but does not dramatically change the
  optimal value for well-calibrated initial settings.

### When to use learnable temperature

```
Use learnable temperature when:
  Training data is heterogeneous (mix of domains, lengths, difficulties)
  → Temperature needs to adapt to varying difficulty across batches

Use fixed temperature when:
  Training data is homogeneous (single domain, similar lengths)
  Computational budget is tight (learnable temp adds minimal cost but requires tuning)
  Reproducing results exactly from prior work
```

## Practical Training Decisions

### Starting configuration for a new domain

Given limited GPU resources and no domain-specific priors:

```
GPU:      Single A100 (80GB)
Batch:    Start at B = 128 per GPU (or maximum that fits in memory)
Temp:     τ = 0.05
Negatives: 7 BM25 hard negatives per query
Training: 3 epochs on domain-labeled pairs

Expected outcome: reasonable quality baseline

Tuning:
  If Recall@100 < 0.75: increase hard negative count to 15
  If training loss spiky: increase τ to 0.07
  If training loss decreases slowly: decrease τ to 0.03
  If GPU memory allows: increase B to 256
```

### Scaling to large batch with multi-GPU

For multi-GPU training with distributed InfoNCE:

```
Setup: 8 × A100 80GB GPUs
Per-GPU batch: 256
Effective batch: 256 × 8 = 2048 (with cross-GPU negative sharing)
Temperature: τ = 0.07-0.10 (increase with larger batch)
Negatives: 7 BM25 + 3 dense hard negatives per query

Cross-GPU negative sharing implementation:
  Each GPU holds 256 query and document embeddings
  AllGather: collect all 2048 document embeddings on each GPU
  Loss: InfoNCE with all 2048 documents as negatives per query
```

### Diagnosing batch size issues

**Symptom: loss decreasing but Recall@100 stagnating**
Cause: batch size is too small - model memorizes batch pattern without
learning generalizable representations.
Fix: increase batch size or gradient accumulation.

**Symptom: loss unstable, spiky despite reasonable LR**
Cause: temperature too low for the batch size and negative difficulty.
Fix: increase temperature by 0.02-0.05.

**Symptom: loss decreases rapidly to very low values but NDCG plateaus**
Cause: temperature too high - model finds a trivial solution.
Fix: decrease temperature.

**Symptom: model performs well on easy queries but fails on hard ones**
Cause: batch is too homogeneous (all easy in-batch negatives, high temperature).
Fix: add hard negatives, decrease temperature.

## Learning Rate and Batch Size

The learning rate must be scaled with batch size to maintain stable training:

### Linear scaling rule

The empirically effective rule for contrastive training:

```
lr(B) = lr_base × (B / B_base)

Example:
  Base config: B=256, lr=2e-5
  Scaled:      B=1024, lr = 2e-5 × (1024/256) = 8e-5
```

The linear scaling rule works for contrastive training up to moderate batch sizes.
For very large batches (B > 4096), square root scaling is sometimes more stable:

```
lr(B) = lr_base × √(B / B_base)

B=1024: lr = 2e-5 × √4 = 4e-5
B=4096: lr = 2e-5 × √16 = 8e-5  (same as linear in this case)
B=16384: lr = 2e-5 × √64 = 1.6e-4  (different from linear's 12.8e-4)
```

For very large batches, linear scaling often leads to instability. Square root
scaling is more conservative and typically more stable.

### Warmup for large batch training

Large batch training is more sensitive to the initial learning rate because
the initial gradients are larger (more negatives → larger gradient magnitude):

```
Standard warmup: 5-10% of total training steps at lr = 0
Large batch warmup rule:
  Warmup steps = max(standard_warmup, log₂(B) × 100)

For B = 2048: warmup = max(standard, 11 × 100) = max(standard, 1100 steps)
```

Warmup prevents the large initial gradients from destabilizing the model
before the optimizer has accumulated enough momentum.

## Beyond Binary Temperature: Multiple Temperatures

Some advanced training configurations use different temperatures for different
components of the loss:

### Per-dimension temperature (anisotropic contrastive learning)

Instead of a single scalar τ, use a different temperature for each embedding
dimension:

```
sim_scaled(q, d) = Σᵢ qᵢ × dᵢ / τᵢ

Low τᵢ for dimensions that should discriminate precisely
High τᵢ for dimensions that carry coarse semantic information
```

This is rarely used in practice but theoretically allows finer control over
which dimensions carry coarse vs fine-grained information.

### Dual temperature (positive and negative scaling)

Scale the positive pair and negative pairs differently:

```
L = -log(
    exp(sim(q, d⁺) / τ_pos) /
    [exp(sim(q, d⁺) / τ_pos) + Σⱼ exp(sim(q, dⱼ⁻) / τ_neg)]
)

τ_pos < τ_neg: requires very tight clustering of positives, relaxed for negatives
```

This asymmetric temperature allows the model to learn tight positive clusters
without over-separating negatives - particularly useful when positive pairs
have legitimate diversity (multiple valid answers for one query).

## Empirical Checklist for Batch Size and Temperature

Before finalizing a training configuration, verify:

```
Checkpoint 1: Is the loss in the expected range?
  Initial loss ≈ log(B) (random model)
  After 1 epoch: loss should drop by 30-60% from initial
  If not dropping: temperature may be too high or batch too small

Checkpoint 2: Is training stable?
  Loss should decrease monotonically with small fluctuations
  Large spikes (> 3× moving average): temperature too low or batch too small
  Sudden plateau: learning rate too low or temperature too high

Checkpoint 3: Are embeddings well-calibrated?
  Pairwise similarities for random document pairs: mean ≈ 0, std ≈ 0.1-0.2
  Positive pair similarities: mean ≈ 0.7-0.9
  Hard negative similarities: mean ≈ 0.6-0.8 (close to but below positives)

Checkpoint 4: Does validation recall improve?
  Recall@100 on held-out queries should improve across epochs
  If recall improves on train set but not validation: overfitting
    → Reduce epochs, add regularization, check for false negatives
```

## My Summary

Batch size and temperature are the two hyperparameters with the largest systematic
effect on dense retrieval model quality. Batch size determines the number of
negatives each query faces simultaneously - larger batches create harder, more
informative training tasks, explaining the consistent quality improvement from
DPR (batch 64) to E5 (batch 16,384+). The relationship is logarithmic: quality
improves significantly up to B ≈ 2048 then exhibits diminishing returns. Temperature
τ controls the precision requirement of the embedding space through the InfoNCE
softmax sharpness: low temperatures demand tight clustering of positive pairs and
strong separation of negatives, while high temperatures tolerate approximate proximity.
The optimal temperature depends on both batch size and negative hardness - temperature
and negative hardness are partially substitutable, with hard negatives allowing
higher temperature and random in-batch negatives requiring lower temperature to
compensate. Gradient accumulation achieves gradient averaging across micro-batches
but not true large batch InfoNCE because cross-micro-batch negatives are absent;
GradCache and distributed InfoNCE implementations are needed for true large batch
contrastive training. Learning rate must be scaled with batch size using the linear
or square root scaling rule. The practical starting point is B = 128-256 with τ =
0.05, increasing both as compute budget allows, and diagnosing training quality
through loss trajectory, embedding calibration, and validation Recall@100.
