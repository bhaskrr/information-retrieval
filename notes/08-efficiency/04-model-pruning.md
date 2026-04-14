# Model Pruning

Model pruning is a compression technique that removes parameters, neurons, or
entire structural components from a neural network that contribute little to its
output quality. The goal is the same as distillation - a smaller, faster model -
but the mechanism is different: instead of training a new smaller model from
scratch, pruning starts from a trained model and removes parts of it. The result
is a sparser or structurally smaller model that runs faster with minimal quality
loss.

## Intuition

Not all parts of a neural network are equally important. In a 12-layer BERT model, some attention heads consistently attend to uninformative positions. Some neurons in the feed-forward layers fire rarely and weakly. Some layers contribute almost nothing to the final output.

Pruning identifies and removes these redundant components. The key insight comes
from the lottery ticket hypothesis - large networks contain smaller subnetworks
(winning tickets) that can match the performance of the full network when trained
correctly. Pruning is one way to find and extract these subnetworks.

For IR specifically, pruning is less commonly used than distillation because
distillation produces more predictable and controllable results. But pruning is
worth understanding because it operates on already-trained models without requiring a teacher, making it applicable when you cannot retrain from scratch.

## Types of Pruning

### Unstructured pruning

Remove individual weights - set specific weight values to zero based on their
magnitude. The model becomes sparse but retains its original shape.

```bash
Dense weight matrix:
[0.82, -0.03, 0.71, 0.01, -0.65]
[0.12,  0.44, 0.02, 0.89, -0.11]

After 60% unstructured pruning (remove smallest magnitude weights):
[0.82,  0.00, 0.71, 0.00, -0.65]
[0.00,  0.44, 0.00, 0.89,  0.00]
```

**Advantage**: can achieve very high sparsity (90%+) with low quality loss.

**Disadvantage**: sparse matrices are not efficiently executed on standard GPU/CPU hardware - you need specialized sparse kernels to actually see speedup. Without hardware support, a 90% sparse model runs at the same speed as the dense model.

### Structured pruning

Remove entire structural units - attention heads, neurons, layers. The model
becomes smaller in a hardware-friendly way.

```bash
BERT-base: 12 layers × 12 heads = 144 attention heads
After structured pruning: 12 layers × 8 heads = 96 attention heads
→ actual speedup on standard hardware
```

Structured pruning is the practical choice for IR because it produces models
that run faster on standard GPUs without specialized sparse kernels.

### Magnitude pruning

The simplest criterion: remove weights with the smallest absolute values.
The assumption is that small weights contribute little to the output.

```bash
Importance(w) = |w|
Prune: weights with |w| < threshold
```

Works surprisingly well for unstructured pruning. Less effective for structured
pruning where entire heads or layers must be removed.

### Gradient-based pruning

Use gradient information to estimate how much each parameter affects the loss:

```bash
Importance(w) = |w × ∂L/∂w|
```

Parameters with large weight but small gradient are used rarely - good pruning
candidates. Parameters with large gradient but small weight are actively being
learned - keep them.

### Taylor expansion pruning

A principled approximation of the change in loss when a parameter is removed:

```bash
ΔL ≈ |∂L/∂w × w|   (first-order Taylor approximation)
```

Remove parameters where removing them changes the loss least. Requires a
calibration forward pass to compute gradients.

## Attention Head Pruning

Attention head pruning is the most practically useful form of structured pruning
for transformer-based IR models.

### Finding important heads

Not all attention heads are equally important. Some consistently attend to
meaningful token relationships; others produce nearly uniform attention patterns
(attending equally to all tokens — not useful).

Methods to identify important heads:

**Attention entropy**
High entropy = uniform attention = uninformative head. Low entropy = focused
attention = likely important.

```bash
entropy(head) = -Σ attn_weight × log(attn_weight)
→ prune heads with highest entropy
```

**Gradient sensitivity**
Add a learnable gate gₕ ∈ {0,1} for each head. During fine-tuning, learn which
gates should be 0 (prune) vs 1 (keep):

```bash
output = Σₕ gₕ × head_h(x)
→ after training: heads with gₕ ≈ 0 are pruned
```

**Michel et al. (2019) — "Are Sixteen Heads Really Better than One?"**
Found that many BERT attention heads can be removed at test time with minimal
quality loss. On some tasks, pruning to a single head per layer loses less than
1% accuracy. This result established that BERT is significantly over-parameterized
for many tasks.

### Head pruning results on IR

| Model                        | Heads | MRR@10 | Speedup |
| ---------------------------- | ----- | ------ | ------- |
| BERT-base (full)             | 144   | 0.358  | 1.0x    |
| BERT-base (75% heads pruned) | 36    | 0.341  | 1.4x    |
| BERT-base (50% heads pruned) | 72    | 0.351  | 1.2x    |

Head pruning alone gives modest speedup (1.2-1.5x). Combined with quantization
and ONNX export, the total speedup is more significant.

## Layer Pruning

Removing entire transformer layers is more aggressive than head pruning but
produces larger speedups.

```bash
BERT-base: 12 layers
After removing layers 7-12: 6 layers → ~2x speedup
```

Not all layers are equally important. Early layers capture syntactic patterns;
middle layers capture semantic relationships; later layers are task-specific.
For many IR tasks, the later layers contribute disproportionately — removing
them hurts quality more than removing early layers.

### Layer dropping heuristic

Remove layers whose output is most similar to their input (low residual norm):

```bash
importance(layer_i) = ||output_i - input_i||₂
→ prune layers with smallest residual norm
```

Layers where the residual connection dominates (output ≈ input) are doing little
work and are safe candidates for removal.

## Iterative Pruning

Pruning too aggressively in one step degrades quality sharply. Iterative pruning
achieves higher final sparsity with better quality retention:

```bash
1. Train full model to convergence
2. Prune 10-20% of least important parameters
3. Fine-tune the pruned model to recover quality
4. Repeat steps 2-3 until target sparsity is reached
```

Each fine-tuning step allows the remaining parameters to compensate for the
removed ones. Iterative pruning consistently outperforms one-shot pruning at
the same final sparsity.

## Pruning vs Distillation for IR

| Property                  | Pruning              | Distillation          |
| ------------------------- | -------------------- | --------------------- |
| Starting point            | Trained model        | Pretrained base model |
| Teacher required          | No                   | Yes                   |
| Training data required    | Calibration set only | Full training set     |
| Quality at same size      | Lower                | Higher                |
| Implementation complexity | Medium               | Medium                |
| Best for                  | Quick compression of | Best quality at       |
|                           | existing model       | target size           |
| Speedup type              | Structural or sparse | Structural (new arch) |

For most IR applications, distillation produces better models at the same size.
Pruning is most useful when you have a fine-tuned model you cannot retrain from
scratch — for example a domain-specific cross-encoder where labeled training data
is scarce.

## Pruning in the Context of IR Models

### When pruning makes sense for IR

| Scenario                             | Recommended approach       |
| ------------------------------------ | -------------------------- |
| Fine-tuned domain model, need faster | Prune + quantize + ONNX    |
| No labeled data to retrain           | Prune existing model       |
| Have labeled data, can retrain       | Distill instead            |
| Need maximum compression             | Prune + distill + quantize |
| Quick experiment, minimal effort     | Quantize only              |

### Practical limits for IR quality

From empirical results on cross-encoders and bi-encoders:

| Pruning level     | Quality impact (NDCG@10) | Speedup (structured)  |
| ----------------- | ------------------------ | --------------------- |
| 10% unstructured  | < 0.1%                   | ~0 (hardware limited) |
| 30% unstructured  | ~0.5%                    | ~0 (hardware limited) |
| 30% heads pruned  | ~1-2%                    | ~1.2x                 |
| 50% heads pruned  | ~3-5%                    | ~1.4x                 |
| 50% layers pruned | ~8-15%                   | ~2x                   |

Head pruning gives modest speedups with acceptable quality. Layer pruning gives
larger speedups but significant quality loss - only viable with subsequent
fine-tuning to recover quality.

## Combining Pruning with Other Techniques

The full compression pipeline for an IR model:

```bash
Step 1: Start with full BERT-base cross-encoder (110M params, fp32)
Step 2: Iterative head pruning (30% heads removed, fine-tune after each round)
        → 77M effective params, ~1.2x speedup, ~1% quality loss
Step 3: fp16 quantization
        → 55MB model, 1.5x additional speedup, ~0% quality loss
Step 4: ONNX export + optimization
        → 1.3x additional speedup, ~0% quality loss
Step 5: Dynamic batching at inference
        → 2-4x throughput improvement

Total: ~5x speedup, ~1% quality loss vs original
```

Compare this to pure distillation (MiniLM-L6): 5x speedup, ~7% quality loss.
Pruning + quantization achieves the same speedup with better quality retention
but requires more engineering effort.

## My Summary

Model pruning removes redundant parameters, attention heads, or entire layers from a trained model to reduce its size and inference cost. Unstructured pruning removes individual weights (producing sparsity) but requires specialized hardware to translate to actual speedup. Structured pruning removes entire heads or layers and gives real speedup on standard hardware. For IR models, attention head pruning is the most practical form - 30% of heads can typically be removed with ~1-2% quality loss. Pruning is most useful when you have a fine-tuned domain-specific model that cannot be retrained from scratch - otherwise distillation produces better models at the same size. The full production recipe combines iterative pruning with quantization and ONNX export for maximum efficiency gains.
