# Knowledge Distillation

Knowledge distillation is a model compression technique where a smaller, faster
student model is trained to mimic the behavior of a larger, more accurate teacher
model. Instead of training the student on hard labels (correct/incorrect), the
student learns from the teacher's soft output distributions, the full probability scores across all classes: which contain richer information about what the teacher has learned. In IR, distillation is the primary technique for producing small fast models that approach the quality of large slow ones.

## Intuition

Training a model from scratch on hard labels is inefficient. The label "relevant"
tells the student nothing about how relevant, or how the document compares to other documents. A cross-encoder teacher that scores document D1 at 0.92 and D2 at 0.87 is communicating that both are relevant but D1 is more so - nuance that a binary label completely loses.

Soft labels from the teacher capture this nuance. The student learns not just what the answer is but how confident the teacher is and how documents relate to each other in score space. This is why distillation consistently outperforms training the same small model from scratch on the same data, the teacher provides richer supervision.

A useful analogy: hard label training is like a student reading a textbook alone.
Distillation is like a student being tutored by an expert who explains not just
the correct answer but their reasoning and degree of certainty.

## The Distillation Framework

```bash
Teacher model (large, accurate, slow):
  -> cross-encoder BERT-large fine-tuned on MS MARCO
  -> produces relevance scores for (query, document) pairs

Student model (small, approximate, fast):
  -> bi-encoder MiniLM-L6
  -> trained to reproduce teacher scores on same pairs

Distillation loss:
  -> minimize distance between teacher scores and student scores
  -> student inherits teacher's ranking judgment without teacher's cost
```

## Types of Knowledge in IR Distillation

### Score distillation (response-based)

The most common type. Train the student to match the teacher's output scores:

```bash
teacher_score(q, d) = cross_encoder(q, d)   → scalar, e.g. 0.87
student_score(q, d) = biencoder(q) · biencoder(d)   → scalar, e.g. 0.71

loss = MSE(student_score, teacher_score)
    or KL_divergence(softmax(student_scores), softmax(teacher_scores))
```

### Ranking distillation

Instead of matching absolute scores, match the ranking order:

```bash
For a query q with documents [d1, d2, d3, d4]:
  Teacher ranking: d1 > d3 > d2 > d4
  Student learns to produce the same ordering

Loss: pairwise ranking loss over teacher-ranked pairs
      L = Σ max(0, margin - score(q,dᵢ) + score(q,dⱼ)) for all i>j in teacher ranking
```

### Feature distillation (intermediate layers)

Match not just outputs but intermediate representations:

```bash
Teacher hidden state at layer 6:  h_T ∈ ℝ^{768}
Student hidden state at layer 3:  h_S ∈ ℝ^{384}

loss = MSE(W × h_S, h_T)   where W is a learned projection
```

Requires aligning teacher and student layers. More complex but produces better
students when done carefully. Used in DistilBERT.

## Key Distillation Papers in IR

### DistilBERT (Sanh et al., 2019)

Distilled BERT-base into a 6-layer, 66M parameter student using:

- Soft label distillation from BERT MLM outputs
- Cosine embedding loss on hidden states
- Hard label MLM loss

Result: 40% smaller, 60% faster, 97% of BERT-base performance on GLUE.
DistilBERT is the standard lightweight base model for many IR applications.

### TAS-B (Hofstätter et al., 2021)

Topic-Aware Sampling with BERT. Distills a cross-encoder teacher into a bi-encoder
student using:

- Margin MSE loss: minimize the difference in score margins between teacher and student
- Balanced topic sampling: ensure training queries cover diverse topics

```bash
Margin MSE loss:
  teacher_margin = teacher_score(q, d+) - teacher_score(q, d-)
  student_margin = student_score(q, d+) - student_score(q, d-)
  loss = MSE(student_margin, teacher_margin)
```

Result: bi-encoder that approaches cross-encoder quality on MS MARCO.

### Sentence-Transformers Distillation (Reimers & Gurevych, 2020)

Distills sentence embeddings from a large teacher to a smaller student using
mean squared error on the embedding space:

```bash
teacher_embedding = teacher_model.encode(text)    → 768-dim
student_embedding = student_model.encode(text)    → 384-dim

loss = MSE(W × student_embedding, teacher_embedding)
```

This is how the all-MiniLM family was created — distilled from larger
sentence-transformer models into fast, compact 384-dim embedding models.

### ColBERT Distillation (ColBERT v2)

ColBERT v2 uses a cross-encoder teacher to generate soft relevance scores for
training triples, then trains the student ColBERT model to minimize KL divergence
between its MaxSim scores and the teacher's relevance scores:

```bash
teacher_score(q, d) = cross_encoder(q, d)
student_score(q, d) = colbert_maxsim(q, d)

loss = KL(softmax(teacher_scores_per_query),
          softmax(student_scores_per_query))
```

## Distillation Training Recipe for IR

### Standard setup

```bash
1. Train teacher cross-encoder on MS MARCO with hard labels
   → BERT-large or RoBERTa-large fine-tuned on passage ranking

2. Generate soft labels
   → for each (query, passage) pair in training set,
     compute teacher_score = cross_encoder(query, passage)

3. Train student bi-encoder on soft labels
   → minimize margin MSE or KL divergence between
     student scores and teacher scores

4. Evaluate student on MS MARCO dev + BEIR
```

### Hard negative mining for distillation

Soft labels from the teacher are most informative when the negatives are hard —
documents that the teacher gives a non-trivial relevance score.

```bash
Easy negative: d- has teacher_score = 0.02   → too easy, little signal
Hard negative: d- has teacher_score = 0.45   → confusing, high signal
```

Mine hard negatives by retrieving top-k candidates with an early student model
or BM25, then filtering to those with non-trivial teacher scores.

## Temperature Scaling

Temperature T controls how soft the teacher's probability distribution is:

```bash
softmax_T(scores) = exp(scores / T) / Σ exp(scores / T)
```

```bash
T = 1  → standard softmax, peaks are sharp
T > 1  → softer distribution, more information transferred
T < 1  → sharper distribution, more like hard labels
```

Higher temperature reveals more of the teacher's uncertainty and inter-document
relationships. T = 2-4 is typical for distillation in IR.

## Distillation vs Training from Scratch

For the same student architecture, distillation consistently outperforms training
from scratch on hard labels:

| Model                             | MS MARCO MRR@10 | BEIR avg NDCG@10 |
| --------------------------------- | --------------- | ---------------- |
| BERT-base cross-encoder (teacher) | 0.358           | 0.411            |
| MiniLM-L6 from scratch            | 0.310           | 0.380            |
| MiniLM-L6 distilled from teacher  | 0.334           | 0.403            |

The distilled MiniLM-L6 is 5x faster than BERT-base and achieves 93% of its
quality, making it the standard choice for production cross-encoder reranking.

## The MiniLM Family - Distillation in Practice

The all-MiniLM models from sentence-transformers are the most widely used
distilled IR models. Understanding their lineage explains why distillation works:

```bash
Teacher: all-mpnet-base-v2 (420M params, 768-dim embeddings)
  ↓ embedding distillation (MSE on embedding space)
Student: all-MiniLM-L12-v2 (33M params, 384-dim embeddings)
  ↓ further distillation
Student: all-MiniLM-L6-v2 (22M params, 384-dim embeddings)
```

Performance comparison:

| Model             | SBERT Benchmark | Params | Encode speed  |
| ----------------- | --------------- | ------ | ------------- |
| all-mpnet-base-v2 | 0.6916          | 420M   | ~2800 sent/s  |
| all-MiniLM-L12-v2 | 0.6734          | 33M    | ~7500 sent/s  |
| all-MiniLM-L6-v2  | 0.6667          | 22M    | ~14200 sent/s |

all-MiniLM-L6-v2 achieves 96% of the teacher's quality at 5x the speed. This is
the standard trade-off distillation enables and why it is the default choice
for most production IR systems.

## Distillation vs Quantization — When to Use Which

| Concern                 | Use distillation | Use quantization |
| ----------------------- | ---------------- | ---------------- |
| Primary goal            | Smaller model    | Faster inference |
| Training data available | Yes              | No               |
| Memory constraint       | Yes              | Yes              |
| Latency constraint      | Yes              | Yes              |
| Quality budget          | Flexible         | Tight            |
| One-time cost           | High (training)  | Low (conversion) |
| Ongoing cost            | Low              | Low              |

In practice, use both together: distill first to get a smaller model, then
quantize the distilled model for additional inference speedup. MiniLM-L6 (distilled)

- int8 quantization + ONNX export is the standard production recipe for fast,
  high-quality IR.

## Where This Fits in the Progression

```bash
01-why-efficiency-matters  → understanding the constraints
02-quantization            → reduce precision, reduce cost
03-knowledge-distillation  → smaller model, same capability  ← you are here
04-model-pruning           → remove redundant parameters
05-onnx-runtime            → optimized execution engine
06-caching-and-batching    → system-level optimizations
07-ann-search              → efficient vector search
```

## My Summary

Knowledge distillation trains a small fast student model to mimic a large accurate teacher by learning from the teacher's soft output scores rather than hard labels.
Soft labels transfer richer supervision — they encode relative confidence and
inter-document relationships that binary labels lose. In IR, the dominant pattern
is cross-encoder teacher → bi-encoder student using margin MSE loss, which trains
the student to match the teacher's score margins across positive and negative
document pairs. The MiniLM family demonstrates the practical result: 5x speedup
with 96% quality retention. Distillation and quantization are complementary — distill first for a smaller model, then quantize for faster inference. Together they make BERT-quality retrieval practical within production latency and cost budgets.
