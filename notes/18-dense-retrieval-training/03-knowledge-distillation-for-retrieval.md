# Knowledge Distillation for Retrieval

Knowledge distillation for retrieval is the training paradigm in which a smaller,
faster student retrieval model is trained to approximate the behavior of a larger,
more accurate teacher model - most commonly a cross-encoder or ensemble of models

- by learning from the teacher's relevance scores rather than from binary
  positive/negative labels alone. The teacher provides rich, continuous supervision
  signal that captures fine-grained relevance distinctions the student would struggle
  to learn from binary labels with limited training data. The result is a bi-encoder
  student that retrieves at inference speed with quality substantially closer to the
  cross-encoder teacher than supervised fine-tuning alone can achieve. Knowledge
  distillation is the technique behind the most widely deployed production bi-encoders -
  TAS-B, ColBERT v2's distilled variants, and the full E5 and GTE model families all
  rely on distillation as a central component of their training.

## Intuition

Binary labels in retrieval training are informationally poor. A (query, document)
pair is labeled 1 if the document is relevant, 0 if not. But relevance is not
binary in reality:

```
Query: "treatment for type 2 diabetes"

Document A: "Metformin is the first-line treatment for type 2 diabetes.
             It reduces hepatic glucose production and improves insulin sensitivity."
Binary label: 1 (relevant)
True relevance: very high - comprehensive, directly answers the query

Document B: "GLP-1 receptor agonists are second-line agents for type 2 diabetes
             when metformin is insufficient."
Binary label: 1 (relevant)
True relevance: high - relevant but more specific, partial answer

Document C: "Lifestyle modification including diet and exercise is recommended
             for type 2 diabetes management."
Binary label: 1 (relevant)
True relevance: moderate - relevant but about prevention/management, less specific

Document D: "Type 1 diabetes requires insulin therapy, unlike type 2."
Binary label: 0 (not relevant)
True relevance: near zero - about type 1, not type 2
```

Training a bi-encoder with binary labels treats Documents A, B, and C identically
(all labeled 1) and treats their differences as noise. A cross-encoder, by contrast,
correctly scores A > B > C >> D because it can read the full (query, document) pair
with cross-attention.

Distillation transfers the cross-encoder's nuanced judgment to the bi-encoder:
instead of binary 0/1, the bi-encoder learns to reproduce the cross-encoder's
score distribution. The student learns that A should score higher than B, B higher
than C, and all three much higher than D - producing representations that encode
these fine-grained distinctions.

## The Teacher-Student Framework in Dense Retrieval

### Standard teacher types

**Cross-encoder teacher (most common)**

A cross-encoder jointly encodes query and document and produces a precise
relevance score through full self-attention. It is the natural choice as teacher
because:

- It has access to all query-document interaction information
- It achieves near-human relevance judgment accuracy
- It generalizes well to new domains with domain-specific fine-tuning

Cross-encoder teachers are almost always slower than the bi-encoder student
by a factor of 100-1000x, which is precisely why distillation is needed -
you want the teacher's quality in the student's speed.

**Ensemble teacher**

Train multiple bi-encoders or cross-encoders with different initialization seeds
or architectures and average their scores. Ensemble scores are smoother and more
reliable than single-model scores:

```
teacher_score(q, d) = (1/K) × Σₖ score_k(q, d)
```

Ensemble teachers reduce noise from individual model variance. TAS-B uses an
ensemble of multiple cross-encoder variants as its teacher.

**LLM teacher**

Use GPT-4 or Claude to produce relevance judgments for training pairs. The LLM
can provide graded relevance labels (0-3) or pairwise preferences. This is the
highest-quality teacher available but also the most expensive. Practical for
domain adaptation where a small, high-quality training set is needed.

**Cascade teacher**

A two-stage teacher: bi-encoder retrieves top-k, cross-encoder reranks, final
scores from cross-encoder are used as distillation targets. This produces a student
that approximates the full cascade quality while running at bi-encoder speed.

### Student model design

The student is typically a bi-encoder that will be deployed for inference:

```
Student architecture:
  - Smaller or same-size transformer as teacher
  - Separate or shared encoder for query and document
  - Mean or CLS pooling
  - L2 normalization for cosine similarity

Common student models:
  - MiniLM-L-6 (22M params) → fast CPU/GPU inference
  - BERT-base (110M params) → standard production quality
  - DistilBERT (66M params) → good speed/quality tradeoff
  - RoBERTa-base (125M params) → better generalization
```

The student does not need to be smaller than the teacher. Distillation from
a cross-encoder into a same-size bi-encoder is still highly valuable because
the student gains access to cross-encoder supervision without the inference
cost of the cross-encoder.

## Distillation Loss Functions

### Margin MSE loss (TAS-B)

The most widely used distillation loss for retrieval. Rather than matching
absolute scores (which have different scales for bi-encoder and cross-encoder),
match the margins between positive and negative scores:

```
For each training triple (query q, positive d⁺, negative d⁻):

Teacher margin:
  M_teacher = score_teacher(q, d⁺) - score_teacher(q, d⁻)

Student margin:
  M_student = sim(f_q(q), f_d(d⁺)) - sim(f_q(q), f_d(d⁻))

MarginMSE loss:
  L = MSE(M_student, M_teacher)
    = (M_student - M_teacher)²
```

Why margins instead of absolute scores? Cross-encoder scores and cosine
similarities have different ranges and calibrations. A cross-encoder might
output logits in [-5, 5] while cosine similarities are in [-1, 1]. Margins
cancel out these scale differences:

```
Cross-encoder:       d⁺ score = 0.89, d⁻ score = 0.23, margin = 0.66
Cosine similarity:   d⁺ sim   = 0.87, d⁻ sim   = 0.31, margin = 0.56

Student learns to match the ranking order (large margin for clearly better docs)
not the absolute values
```

The student learns that whenever the teacher strongly prefers d⁺ over d⁻
(large positive margin), the student's similarity scores should also strongly
prefer d⁺ over d⁻.

### KL divergence distillation

Rather than matching individual margins, match the full score distribution
over a list of candidates:

```
For query q and candidates [d₁, d₂, ..., dₖ]:

Teacher distribution:
  p_teacher = softmax([score_teacher(q, dᵢ) / τ for dᵢ in candidates])

Student distribution:
  p_student = softmax([sim(f_q(q), f_d(dᵢ)) / τ for dᵢ in candidates])

KL divergence loss:
  L = KL(p_teacher || p_student)
    = Σᵢ p_teacher(i) × log(p_teacher(i) / p_student(i))
```

KL divergence distillation trains the student to match the full ranked
distribution of the teacher, not just pairwise margins. This is more
information-rich but requires the teacher to score all candidates for
each query (expensive).

### Direct score regression

Simply regress the student score to match the teacher score:

```
L = MSE(sim(f_q(q), f_d(d)), σ(score_teacher(q, d)))
```

Where σ is a normalization function to bring teacher and student scores
to the same range. This is the simplest distillation loss but performs
poorly because absolute score calibration is difficult.

### Combined loss: supervised + distillation

The strongest approach combines hard-label ranking loss with soft distillation:

```
L_total = L_InfoNCE(hard labels) + λ × L_MarginMSE(teacher scores)

Where:
  L_InfoNCE: standard contrastive loss on positive/negative labels
  L_MarginMSE: distillation loss matching teacher margins
  λ: distillation weight (typically 0.5-2.0)
```

The hard-label component ensures the student learns basic retrieval signal
from the known (query, positive) pairs. The distillation component provides
fine-grained calibration from the teacher's graded judgments. Together they
consistently outperform either component alone.

## TAS-B - The Canonical Distillation Framework

TAS-B (Topic-Aware Sampling for BERT) from Hofstätter et al. (2021) is the most
influential dense retrieval distillation paper. It introduced the key ideas that
most subsequent work builds on.

### TAS-B training setup

**Teacher:** Ensemble of multiple cross-encoder variants
(trained on MS MARCO with different seeds and architectures).

**Student:** BERT-base bi-encoder.

**Loss:** Balanced pairwise MarginMSE with in-batch negatives.

**Key innovation - Balanced topic-aware sampling:**

Standard random batching for contrastive training mixes queries from completely
different topics in each batch. This means in-batch negatives are easy (different
topics) and provide little learning signal.

TAS-B samples batches so that queries within a batch come from the same
or related topics:

```
Standard random batch:
  q₁: "cardiac arrest treatment"
  q₂: "python list sorting"
  q₃: "coffee brewing methods"
  q₄: "quantum entanglement"
  → In-batch negatives are trivially different topics

TAS-B topic-aware batch:
  q₁: "type 2 diabetes treatment"
  q₂: "metformin side effects"
  q₃: "insulin resistance management"
  q₄: "GLP-1 agonist benefits"
  → In-batch negatives are all diabetes-related (harder)
```

Topic-aware sampling ensures in-batch negatives are topically related,
creating harder negatives that force the model to learn within-topic
discrimination - exactly the scenario where cross-encoder knowledge
is most valuable.

**Performance:**

```
Training strategy                    MS MARCO MRR@10
────────────────────────────────────────────────────
BM25                                  0.187
DPR (random batches, BM25 negs)       0.340
TAS-B (topic-aware, MarginMSE)        0.393    ← large improvement
```

TAS-B's improvement over DPR is almost entirely attributable to (1) distillation
and (2) better batch construction - same architecture, same base model, different
training procedure.

## ColBERT v2 - Distillation for Late Interaction

ColBERT v2 applies distillation to the late interaction architecture:

### ColBERT v2 distillation setup

**Teacher:** ColBERT v1 (already strong) + cross-encoder ensemble.

**Student:** ColBERT with smaller representations (compressed from 128 to 32 dims).

**Residual compression:** ColBERT v2 introduces a key efficiency improvement -
document token vectors are compressed using a learned codebook and the
distillation loss ensures the compressed representations preserve the
teacher's relevance scores.

```
ColBERT v1 document representation:
  Each document stores n token vectors × 128 dims
  Total: ~100 tokens × 128 dims = 12,800 floats per document

ColBERT v2 document representation:
  Each document stores n compressed codes × 32 dims
  Total: ~100 tokens × 32 dims = 3,200 floats per document
  4x compression with distillation to maintain quality
```

Distillation is what enables this compression - without the teacher's signal,
compressing from 128 to 32 dims would cause severe quality degradation. With
distillation, the compressed student learns to preserve the ranking information
that the teacher found important.

## SPLADE Distillation

SPLADE++ (2023) applies distillation from an ensemble teacher to the learned
sparse retrieval architecture:

### SPLADE distillation challenge

SPLADE produces sparse vectors over the vocabulary rather than dense embeddings.
Standard bi-encoder distillation losses apply to cosine similarities, but SPLADE
uses dot products over sparse vectors. The adaptation is straightforward:

```
SPLADE teacher-student:
  Teacher: ensemble of SPLADE-v2-max models (high quality, slow)
  Student: SPLADE with stronger FLOPS regularization (faster, sparser)

Loss: MarginMSE over sparse dot products
  M_teacher = dot(q_teacher_sparse, d⁺_teacher_sparse) - dot(q_teacher_sparse, d⁻_teacher_sparse)
  M_student = dot(q_student_sparse, d⁺_student_sparse) - dot(q_student_sparse, d⁻_student_sparse)
  L = MSE(M_student, M_teacher)
```

SPLADE++ achieves near-max SPLADE quality with half the active terms through
this ensemble distillation approach.

## Cascade Distillation

Train a chain of progressively smaller models, each learning from the one
above it:

```
Level 1: Cross-encoder ensemble (teacher, never deployed)
         → Produces high-quality scores for training data

Level 2: Large bi-encoder (student₁)
         Trained with MarginMSE from Level 1
         → Deployed as premium reranker

Level 3: Medium bi-encoder (student₂)
         Trained with MarginMSE from Level 2
         → Deployed as standard retriever

Level 4: Small bi-encoder (student₃)
         Trained with MarginMSE from Level 3
         → Deployed for latency-sensitive applications
```

Each level benefits from the teacher-student relationship. The smallest model
at level 4 has received a highly compressed version of the cross-encoder's
knowledge through three layers of distillation - substantially better than
direct training from binary labels.

## Offline vs Online Distillation

### Offline distillation (score caching)

Pre-compute teacher scores for all training examples and store them:

```
Step 1 (offline, once): score all (query, candidate) pairs with teacher
  → store teacher_scores.pkl

Step 2 (training): load cached scores, train student
  → student sees same teacher signals every epoch
  → fast training (no teacher inference during training)
  → requires storage for cached scores

Cost: O(n × k) teacher forward passes (n queries, k candidates each)
Storage: 8 bytes × n × k scores

For MS MARCO: 500K queries × 100 candidates × 8 bytes = 400MB
```

Offline distillation is the practical choice for large training sets -
it pays the teacher inference cost once and reuses it.

### Online distillation (dynamic scoring)

Compute teacher scores on-the-fly during training:

```
During each training step:
  1. Sample a mini-batch
  2. Score each (query, candidate) pair with teacher
  3. Compute distillation loss
  4. Update student

Cost: teacher inference runs simultaneously with student training
      → training speed limited by teacher inference speed
      → teacher and student usually run on different GPUs
```

Online distillation enables:

- Fresh teacher scores on dynamically mined hard negatives
- Dynamic curriculum (teacher scores harder candidates as model improves)
- No pre-computation storage requirement

For research experiments with small training sets, online distillation is
cleaner. For production-scale training, offline distillation is standard.

## Domain Adaptation via Distillation

Distillation is particularly powerful for domain adaptation:

### The domain adaptation problem

A general MS MARCO cross-encoder produces excellent scores on web search queries
but mediocre scores on biomedical, legal, or code queries because the pre-training
distribution does not match the target domain.

### Distillation-based domain adaptation pipeline

```
Step 1: Collect unlabeled in-domain corpus
Step 2: Generate synthetic queries for domain documents (GPL approach)
Step 3: Mine candidate documents with BM25 or domain-adapted retriever
Step 4: Score (synthetic query, candidate) pairs with cross-encoder
        → Even a general cross-encoder provides useful signal on domain data
Step 5: Distill into domain-specific bi-encoder using MarginMSE

Result: bi-encoder that:
  - Inherits general retrieval knowledge from bi-encoder initialization
  - Learns domain-specific relevance from synthetic data + cross-encoder scoring
  - No human annotation required
```

GPL (Generative Pseudo-Labeling, Thakur et al. 2021) formalizes this pipeline:

```
GPL = T5 query generation + BM25 negative mining + cross-encoder scoring + MarginMSE training

For any target domain corpus:
  1. T5 generates synthetic queries for each document chunk
  2. BM25 mines hard negatives for each synthetic query
  3. Cross-encoder scores (synthetic query, positive, negative) triples
  4. MarginMSE distillation trains domain bi-encoder

GPL achieves competitive performance with only synthetic data on BEIR domains
```

## Distillation Quality vs Training Budget

The quality of the distillation teacher determines the ceiling quality
of the student, but practical constraints limit teacher quality:

```
Teacher option          Student MRR@10    Teacher inference cost
─────────────────────────────────────────────────────────────────────
No distillation          ~0.340           Free
Single cross-encoder     ~0.380           ~100x bi-encoder cost
Ensemble of 3 CEs        ~0.393           ~300x bi-encoder cost
Ensemble of 10 CEs       ~0.398           ~1000x bi-encoder cost
LLM teacher (GPT-4)      ~0.410+          ~10000x bi-encoder cost
```

The diminishing returns beyond an ensemble of 3-5 cross-encoders means practical
training uses 3-5 cross-encoders as teacher rather than larger ensembles. The
jump from "no distillation" to "single cross-encoder teacher" is by far the
most cost-effective improvement.

## Key Design Decisions

### How many negatives to score with teacher?

More candidates per query give the teacher more fine-grained ranking signal:

```
1 negative per query:
  Teacher provides 1 margin signal → coarse ranking information
  Minimal teacher inference cost

7 negatives per query:
  Teacher provides 7 margins → much richer ranking information
  7x teacher inference cost

50 negatives per query:
  Teacher provides C(50,2)=1225 pairwise relations → rich ranking
  50x teacher inference cost
```

Empirically, 5-15 negatives per query captures most of the distillation benefit.
Beyond 15, returns diminish while cost grows linearly.

### BM25 negatives vs dense negatives for teacher scoring

Both work well. The key is that the teacher should score negatives that are
actually challenging - random negatives receive near-zero teacher scores for
all of them, providing no learning signal.

Practical recommendation: mine BM25 hard negatives (fast), score with cross-encoder
teacher (provides fine-grained signal for lexically similar but non-relevant docs).

### Temperature in distillation loss

Temperature in KL divergence distillation controls how smoothly the teacher's
distribution is transferred:

```
Low temperature (τ = 1.0):
  Hard distribution → student learns sharp preference for top-ranked document
  Risk: student overfits to teacher's exact ordering rather than generalizing

High temperature (τ = 4.0):
  Soft distribution → student learns more uniform relevance gradient
  Benefit: student generalizes better to test distribution
```

Temperature annealing - starting high and decreasing over training - often
produces the best results.

## Evaluation of Distillation Quality

Measuring how well distillation transferred knowledge:

### Teacher-student correlation

How well do student scores correlate with teacher scores on held-out data?

```
Spearman ρ between teacher and student rankings:
  ρ > 0.85: excellent distillation - student closely matches teacher
  ρ = 0.70-0.85: good distillation
  ρ < 0.70: poor distillation - may need more training or better negatives
```

### Compression ratio analysis

Plot quality vs model size across teacher and student:

```
Model size → NDCG@10 tradeoff:
  Cross-encoder (370M):  NDCG@10 = 0.480 (teacher, not deployed)
  BERT-base bi-enc (110M): NDCG@10 = 0.460 (distilled, 3.4x smaller)
  MiniLM-L-6 (22M):      NDCG@10 = 0.443 (distilled, 17x smaller)
  TinyBERT (15M):         NDCG@10 = 0.421 (distilled, 25x smaller)

Without distillation:
  BERT-base bi-enc (110M): NDCG@10 = 0.430
  MiniLM-L-6 (22M):       NDCG@10 = 0.395
  → Distillation adds 0.020-0.030 NDCG across all sizes
```

### Rank correlation at critical positions

Standard Spearman ρ weights all ranks equally. For IR, top-rank correlation
matters most:

```
Normalized Discounted Rank Correlation (NDRC):
  Compute rank correlation but discount positions below top-10
  → Measures how well teacher ranking at top ranks is preserved by student
```

## My Summary

Knowledge distillation trains a fast bi-encoder student to approximate the quality
of an accurate cross-encoder teacher by learning from the teacher's continuous
relevance scores rather than binary positive/negative labels. The teacher's scores
encode fine-grained relevance distinctions - that Document A is much more relevant
than Document B, which is only slightly more relevant than Document C - that binary
labels treat as identical. Margin MSE loss is the standard distillation objective:
the student learns to reproduce the teacher's score margin between positive and
negative documents for each training query, bypassing the scale mismatch between
cross-encoder logits and cosine similarities. TAS-B demonstrated that combining
MarginMSE distillation with topic-aware batch construction improves MS MARCO
MRR@10 from 0.340 (DPR with binary labels) to 0.393 (TAS-B with distillation).
The ensemble teacher - averaging multiple cross-encoder scores - reduces label
noise and consistently outperforms single-teacher distillation. GPL extends
distillation to domain adaptation without human annotation: T5-generated synthetic
queries are scored by a cross-encoder teacher and used to train a domain bi-encoder
via MarginMSE. Offline distillation (pre-computing teacher scores) is standard for
large training sets, while online distillation enables dynamic negative mining but
requires running teacher inference during training. The single highest-leverage
distillation decision is adding any cross-encoder teacher at all - the jump from
no distillation to a single cross-encoder teacher is consistently the largest
quality improvement available in the dense retrieval training pipeline.
