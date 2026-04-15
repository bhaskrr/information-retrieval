# ONNX and Runtime Optimization

ONNX (Open Neural Network Exchange) is an open format for representing machine
learning models as a standardized computational graph. Runtime optimization refers
to the set of techniques applied to this graph - operator fusion, constant folding,
memory planning, hardware-specific kernel selection - that make model inference
faster without changing the model's weights or architecture. Exporting a model to
ONNX and running it through an optimized runtime like ONNX Runtime is one of the
highest-leverage efficiency techniques in IR because it requires no retraining and
produces zero quality loss.

## Intuition

A PyTorch model is a Python program. At inference time, Python overhead, dynamic
dispatch, and unoptimized operator sequencing all add latency on top of the actual
computation. The model spends time on bookkeeping, not just math.

ONNX freezes the model into a static computational graph - a precise description
of every operation in order, with no Python overhead. An optimized runtime then
analyzes this graph and applies a series of mechanical transformations:

- Adjacent operations that can be fused into one kernel (LayerNorm + GELU → single
  kernel call instead of two)
- Constants that are computed at graph compile time rather than every forward pass
- Memory layouts that minimize data movement between GPU cores
- Hardware-specific kernels that are faster than generic implementations

None of these change what the model computes - only how it computes it. The result
is typically a 1.3-2x speedup over PyTorch inference with zero code changes to your
retrieval pipeline.

## The ONNX Ecosystem

```bash
Training framework          ONNX format         Inference runtime
(PyTorch, TensorFlow)  →   (.onnx file)    →   (ONNX Runtime, TensorRT,
                                                 OpenVINO, CoreML, etc.)
```

ONNX acts as the interchange format. You export once from your training framework
and deploy on any supported runtime - CPU, GPU, mobile, edge device. This
decoupling is the secondary value of ONNX beyond raw performance.

## Key Optimization Passes in ONNX Runtime

### Operator fusion

Combines multiple sequential operations into a single kernel call:

```bash
Before fusion:
  MatMul → Add (bias) → LayerNorm → GELU
  = 4 kernel launches, 4 memory read/write cycles

After fusion:
  MatMulAddLayerNormGelu
  = 1 kernel launch, 1 memory read/write cycle
```

Transformer models are especially amenable to fusion - attention blocks contain
many sequential operations that can be merged. ONNX Runtime's transformer-specific optimization pass fuses entire attention blocks into a single kernel.

### Constant folding

Operations whose inputs are all constants are computed once at graph compilation
time and replaced with their output:

```bash
Before: position_embedding = embedding_table[arange(512)]
        (computed every forward pass)

After:  position_embedding = [precomputed constant tensor]
        (computed once at export time)
```

### Graph simplification

Remove redundant operations - unnecessary casts, no-op reshapes, identity
operations - that accumulate in models exported from high-level frameworks.

### Memory planning

Analyze the lifetime of every intermediate tensor and reuse memory buffers
wherever possible, minimizing peak memory usage and reducing memory allocation
overhead.

### Kernel selection

For each operation, select the fastest available kernel for the target hardware.
On NVIDIA GPUs this means choosing between cuBLAS, cuDNN, and custom CUDA kernels
based on tensor shapes and batch sizes.

## ONNX Runtime for Transformers

The `optimum` library from HuggingFace provides a high-level interface for exporting
and optimizing transformer models to ONNX with transformer-specific optimizations:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
```

Three optimization levels:

```bash
Level 1 (O1) — basic graph optimizations
  Constant folding, redundant node elimination, operator fusion
  Speedup: ~1.2x, zero quality loss

Level 2 (O2) — extended optimizations
  All of O1 + transformer-specific attention fusion
  Speedup: ~1.5x, zero quality loss

Level 3 (O3) — layout optimizations
  All of O2 + memory layout transformations
  Speedup: ~1.7x, zero quality loss

Level 4 (O4) — maximum performance
  All of O3 + mixed precision (fp16)
  Speedup: ~2x, negligible quality loss
```

## TensorRT

NVIDIA TensorRT is a higher-performance alternative to ONNX Runtime for NVIDIA
GPU deployment. It performs more aggressive optimization including:

- Layer fusion at the CUDA kernel level
- Precision calibration (automatic fp16/int8 selection per layer)
- Kernel autotuning — profiles multiple kernel implementations and selects fastest
- Engine caching — compiled engine reused across restarts

TensorRT typically gives 2-4x speedup over PyTorch on NVIDIA hardware, compared
to 1.5-2x for ONNX Runtime. The tradeoff is higher setup complexity and NVIDIA-only
support.

## Flash Attention

Flash Attention (Dao et al., 2022) is an algorithmic optimization of the attention
mechanism itself — not a graph-level optimization but a mathematically equivalent
reformulation of the attention computation that is dramatically more memory-efficient
and faster on modern hardware.

Standard attention:

```bash
scores = Q × Kᵀ   → (seq_len, seq_len) matrix - O(n²) memory
weights = softmax(scores / √d)
output = weights × V
```

The (seq_len, seq_len) scores matrix is the bottleneck - for seq_len=512, this is
262,144 elements per head, all of which must be written to and read from GPU VRAM.

Flash Attention computes the same output without materializing the full scores
matrix by computing attention in tiles that fit in GPU SRAM (fast on-chip memory):

```bash
Memory: O(n) instead of O(n²)
Speed:  2-4x faster for long sequences
```

For IR models processing 512-token documents, Flash Attention gives significant
speedups especially on A100/H100 GPUs where SRAM bandwidth is abundant.

## The Full Optimization Stack

Combining all runtime optimizations for a production cross-encoder:

```bash
Starting point: PyTorch BERT-base cross-encoder
  → fp32, dynamic computation graph, Python overhead
  → latency: ~10ms per passage on GPU

Step 1: fp16 quantization
  → model.half()
  → latency: ~6ms  (1.6x speedup)

Step 2: ONNX export + O2 optimization
  → operator fusion, constant folding, memory planning
  → latency: ~4ms  (1.5x additional speedup)

Step 3: ONNX O4 with fp16
  → fp16 kernels in optimized runtime
  → latency: ~3ms  (1.3x additional speedup)

Total: ~3ms per passage vs ~10ms original = 3.3x speedup
For 100 candidates: 300ms vs 1000ms — now within latency budget
```

## When ONNX Gives the Most Benefit

ONNX optimization is most impactful when:

- Running on CPU (operator fusion reduces Python overhead significantly)
- Running many small batches (fixed overhead amortized differently)
- Using transformers with long sequences (Flash Attention integration)
- Deploying on non-NVIDIA hardware (ONNX Runtime supports CPU, Apple Silicon, etc.)

ONNX gives less benefit when:

- Running very large batches on GPU (GPU utilization already high)
- Using TensorRT directly (more aggressive optimization)
- Model is already distilled to tiny size (overhead is small anyway)

## ONNX vs TensorRT Decision Guide

| Scenario                           | Recommendation                |
| ---------------------------------- | ----------------------------- |
| NVIDIA GPU, maximum performance    | TensorRT                      |
| NVIDIA GPU, reasonable performance | ONNX Runtime (simpler setup)  |
| CPU deployment                     | ONNX Runtime                  |
| Apple Silicon                      | ONNX Runtime + CoreML         |
| Multi-hardware deployment          | ONNX Runtime (most portable)  |
| Production with NVIDIA hardware    | TensorRT for inference server |

## My Summary

ONNX exports a PyTorch model to a static computational graph that optimized runtimes like ONNX Runtime can transform through operator fusion, constant folding, and hardware-specific kernel selection. The result is typically a 1.5-2x speedup over PyTorch inference with zero code changes and zero quality loss - making it one of the highest-leverage efficiency techniques in IR. The HuggingFace optimum library makes export and optimization straightforward with a few lines of code. For NVIDIA GPUs, TensorRT provides even more aggressive optimization at the cost of higher setup complexity. ONNX Runtime O2 optimization is the practical default - always apply it alongside fp16 quantization as the baseline efficiency stack before considering more invasive techniques like distillation or pruning.
