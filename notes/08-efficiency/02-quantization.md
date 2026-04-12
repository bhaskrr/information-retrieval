# Quantization

Quantization is the process of reducing the numerical precision of a model's weights
and activations from the default 32-bit floating point (fp32) to lower precision
formats such as 16-bit float (fp16), bfloat16 (bf16), or 8-bit integer (int8).
Reducing precision shrinks model memory footprint and speeds up matrix multiplications - the
dominant operation in transformer inference — with minimal impact on output quality.

## Intuition

A neural network stores its weights as 32-bit floats by default. Each weight takes
4 bytes. BERT-base with 110M parameters requires 440MB just for weights in fp32.

Most of that precision is wasted. A weight with value 0.7342819... does not need
23 bits of mantissa precision to be useful. Rounding it to 0.734 (fp16) or even
0.73 (int8) changes the model's output by a negligible amount in practice — the
difference is smaller than the noise introduced by mini-batch training.

Quantization exploits this. It is a controlled approximation that trades a tiny
amount of mathematical precision for significant reductions in memory usage and
inference time.

## Number Formats

### fp32 (full precision)

- 1 sign bit, 8 exponent bits, 23 mantissa bits
- Range: ±3.4 × 10³⁸
- Memory: 4 bytes per value
- Default training and inference format

### fp16 (half precision)

- 1 sign bit, 5 exponent bits, 10 mantissa bits
- Range: ±65,504
- Memory: 2 bytes per value
- 2x memory reduction over fp32
- Supported natively on modern GPUs (V100, A100, RTX series)
- Risk: smaller range can cause overflow for large activations

### bf16 (bfloat16)

- 1 sign bit, 8 exponent bits, 7 mantissa bits
- Same range as fp32, less mantissa precision
- Memory: 2 bytes per value
- Preferred over fp16 for training — same range as fp32 reduces overflow risk
- Supported on A100, TPUs, newer Intel CPUs

### int8 (8-bit integer)

- Integers in range [-128, 127] or [0, 255]
- Memory: 1 byte per value
- 4x memory reduction over fp32
- Requires calibration — mapping float range to int range
- Matrix multiplications in int8 are 2-4x faster on supported hardware
- Some quality loss, usually 0.5-1% on IR benchmarks

### int4 (4-bit integer)

- Integers in range [-8, 7] or [0, 15]
- Memory: 0.5 bytes per value
- 8x memory reduction over fp32
- Higher quality loss — mainly used for LLM weight compression
- Not commonly used for encoder models in IR

## Quantization Types

### Post-Training Quantization (PTQ)

Quantize a pretrained model without any additional training. The simplest approach.

**Dynamic quantization**
Weights are quantized to int8 ahead of time. Activations are quantized dynamically
at inference time — the quantization scale is computed per-batch.

```bash
Weights: fp32 → int8 (offline, one-time)
Activations: fp32 → int8 → fp32 (per forward pass)
```

Simplest to apply. Works well for CPU inference. Less effective on GPU because
the dynamic dequantization adds overhead.

**Static quantization**
Both weights and activations are quantized. Requires a calibration step — run
a representative dataset through the model to determine the activation ranges,
then fix the quantization scale.

```bash
Calibration: run N samples → measure activation ranges
Production:  weights and activations both int8, scales fixed
```

More complex than dynamic quantization but faster at inference time. Best for
GPU deployment.

**fp16 / bf16 quantization**
Simply cast model weights and computations to fp16 or bf16. No calibration needed.
The most practical starting point — free speedup with essentially zero quality loss.

```python
model = model.half()   # fp32 → fp16
```

### Quantization-Aware Training (QAT)

Simulate quantization during training by inserting fake quantization operations
into the forward pass. The model learns to be robust to quantization noise.

```bash
Training: fp32 weights + fake quantization → gradients compensate
Deployment: actual int8 quantization
```

Recovers most quality lost by PTQ. Requires access to training data and a full
training run — more expensive but produces better models. Used when PTQ quality
loss is unacceptable.

## Quantization for IR Models

### Bi-encoders

Quantization is most impactful here because bi-encoders are used at query time
for every query. The query encoding step is on the critical path.

fp16 quantization: essentially lossless, 2x memory reduction, 1.5-2x speedup.
int8 quantization: 4x memory reduction, 2-3x speedup, ~0.5% quality loss on BEIR.

Document encoding is done offline — quality matters more than speed here.
Use fp32 for offline document encoding if quality is the priority.

### Cross-encoders

Quantization is critical here because cross-encoders are the latency bottleneck.
int8 quantization on cross-encoders typically achieves 2-4x speedup with 0.5-1%
quality loss on MS MARCO MRR@10 — a very favorable tradeoff.

### Embedding index

Separately from model quantization, the stored document embeddings can be
quantized. Storing embeddings in fp16 instead of fp32 halves index memory with
negligible retrieval quality loss. Product quantization (PQ) compresses further
but with larger quality loss — covered in 07-approximate-nearest-neighbour.md.

## Practical Quantization - The Quick Wins

### fp16 - always do this first

```python
import torch

model = model.half()          # cast weights to fp16
model = model.cuda()          # ensure on GPU

# or at load time:
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    torch_dtype=torch.float16
)
```

Zero configuration, zero quality loss, 2x memory reduction. Always apply this
before anything else.

### Using bitsandbytes for int8

```python
# pip install bitsandbytes transformers accelerate

from transformers import AutoModel, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModel.from_pretrained(
    "bert-base-uncased",
    quantization_config=quantization_config,
    device_map="auto"
)
```

One line change. Works with any HuggingFace model. Recommended for GPU inference.

## Quantization Decision Guide

```bash
Starting point: fp32 model, too slow or too large

Step 1 — Apply fp16
  Cost:     one line of code
  Gain:     2x memory, 1.5x speedup
  Quality:  essentially zero loss
  Decision: always do this first

Step 2 — Export to ONNX (covered in 05-onnx-and-runtime-optimization.md)
  Cost:     one-time export
  Gain:     additional 1.3-1.5x speedup from optimized runtime
  Quality:  zero loss
  Decision: always do this too

Step 3 — Apply int8 quantization
  Cost:     calibration step, testing required
  Gain:     2x memory over fp16, 2-3x speedup over fp16
  Quality:  0.5-1% degradation on typical IR metrics
  Decision: apply if fp16 + ONNX is still too slow

Step 4 — Quantization-aware training
  Cost:     full retraining required
  Gain:     recovers most QAT quality loss
  Quality:  near fp32 quality at int8 speed
  Decision: only if PTQ quality loss is unacceptable
```

## What Quantization Cannot Fix

Quantization reduces the cost of each forward pass but does not reduce the number
of forward passes. If your bottleneck is that you are running 1000 cross-encoder
passes per query, int8 quantization makes each pass faster but you still run 1000.

The architectural fixes — knowledge distillation (smaller model), caching (fewer
passes), batching (parallel passes) — are complementary to quantization and often
more impactful. Apply them together.

## Where This Fits in the Progression

```bash
01-why-efficiency-matters  → understanding the constraints
02-quantization            → reduce precision, reduce cost  ← you are here
03-knowledge-distillation  → smaller model, same capability
04-model-pruning           → remove redundant parameters
05-onnx-runtime            → optimized execution engine
06-caching-and-batching    → system-level optimizations
07-ann-search              → efficient vector search
```

## My Summary

Quantization reduces model weight and activation precision from fp32 to fp16 or int8, trading negligible quality for significant reductions in memory and inference time. fp16 quantization is a free win - always apply it first with one line of code,
gaining 2x memory reduction and 1.5x speedup with essentially zero quality loss. int8 quantization requires calibration but delivers 4x memory reduction and 2-4x speedup at the cost of 0.5-1% quality degradation on typical IR benchmarks. For cross-encoder reranking - the dominant latency bottleneck in IR pipelines - int8
quantization combined with ONNX export can bring latency from 500ms to under 150ms per 100 candidates, making production deployment feasible on modest GPU hardware.
