# Why Efficiency Matters

Efficiency in IR refers to the set of constraints, tradeoffs, and techniques
governing how fast a retrieval system can respond, how much memory it consumes,
and how much it costs to run at scale. Efficiency is not an optional concern;
in production IR systems, a model that is too slow or too large to deploy is
useless regardless of how well it performs on benchmarks.

## Intuition

Academic IR research optimizes for quality: NDCG@10, MAP, MRR. Production IR
systems optimize for quality under constraints: latency, memory, throughput,
cost. A model that achieves NDCG@10 = 0.55 but takes 5 seconds per query will
never be deployed. A model that achieves NDCG@10 = 0.48 but responds in 50ms
will ship.

The efficiency module is about understanding these constraints and the techniques
that navigate them without sacrificing too much quality.

## The Latency Budget

A typical production search system has a total latency budget of 100-200ms
from query received to results returned. This budget must cover:

| Component                         | Typical Latency |
| --------------------------------- | --------------- |
| Network (client -> server)        | 10-20ms         |
| Query preprocessing               | 1-2ms           |
| First-stage retrieval (BM25)      | 5-25ms          |
| First-stage retrieval (dense ANN) | 10-30ms         |
| Cross-encoder reranking           | 100-500ms       |
| LLM generation (RAG)              | 500-3000ms      |
| Network (server -> client)        | 10-20ms         |

The cross-encoder is almost always the bottleneck. A full BERT-base forward pass
takes ~5ms on GPU. Reranking 100 candidates sequentially = 500ms — already over
budget before network latency is added.

This is why the entire field of efficiency techniques exists: to make the accurate
models fast enough to deploy within real latency budgets.

## The Quality-Efficiency Tradeoff

Every efficiency technique trades some quality for some speed or memory reduction.
The tradeoff is not always linear — some techniques lose very little quality for
large efficiency gains:

| Technique              | Speedup       | Quality loss | Recommended? |
| ---------------------- | ------------- | ------------ | ------------ |
| fp16 quantization      | 1.5-2x        | < 0.1%       | Always       |
| int8 quantization      | 2-4x          | 0.5-1%       | Usually      |
| Knowledge distillation | 3-10x         | 2-5%         | Usually      |
| Model pruning          | 1.5-3x        | 1-3%         | Situational  |
| ONNX export            | 1.5-2x        | 0%           | Always       |
| ANN vs exact search    | 2-10x         | < 1% recall  | Always       |
| Result caching         | ∞ (cache hit) | 0%           | Always       |

fp16 quantization and ONNX export are essentially free, always apply them.
Knowledge distillation is the most impactful technique for large speedups with
manageable quality loss.

## The Memory Constraint

Beyond latency, memory is the second hard constraint. In production:

| Resource                       | Typical limit |
| ------------------------------ | ------------- |
| GPU VRAM (inference server)    | 8-24GB        |
| RAM (document embedding store) | 32-128GB      |
| Disk (index storage)           | 500GB-10TB    |

BERT-base has 110M parameters at fp32 = 440MB just for weights. Plus activations,
KV cache, and batch processing overhead — a single BERT-base instance easily
consumes 2-4GB of VRAM. Running multiple concurrent queries requires multiple
instances or careful batching.

For the embedding index: 1M documents × 768 dimensions × 4 bytes (fp32) = 3GB.
At fp16 = 1.5GB. With product quantization = 150MB. Memory reduction techniques
determine whether a corpus fits in RAM or requires disk-based retrieval.

## The Throughput Constraint

Throughput = queries processed per second. For a search engine serving many users:

```bash
Small application:    10 QPS    (queries per second)
Medium application:   100 QPS
Large application:    1000+ QPS
Web search scale:     100,000+ QPS
```

A single GPU running BERT-base cross-encoder at 100 docs per query:

- ~100ms per query → 10 QPS single-threaded
- With batching and optimization: 50-100 QPS
- Meeting 1000 QPS requires ~10-20 GPU instances → significant cost

Every 2x speedup from efficiency techniques halves the number of required GPU
instances and halves the infrastructure cost.

## Why BERT is Too Slow as-is

A vanilla BERT-base encoder running on GPU:

| Operation                  | Time     |
| -------------------------- | -------- |
| Single BERT forward pass   | 5-10ms   |
| Encode 100 doc candidates  | 500ms-1s |
| Encode 1000 doc candidates | 5-10ms   |

For a system serving 100 QPS with 100 candidates per query:

- 100 queries × 500ms reranking = 50 GPU-seconds per second of wall time
- Requires 50+ GPUs just for reranking — impractical for most applications

The solution is not to avoid neural models but to make them faster:

- Replace BERT-base (110M params) with MiniLM-L6 (22M params) via distillation
- Quantize to int8 for 2-4x additional speedup
- Export to ONNX for optimized runtime execution
- Cache frequent query results

Combined: 10-20x speedup over vanilla BERT, making 100 QPS achievable on 2-3 GPUs.

## The Full Efficiency Stack

The efficiency techniques in this module form a stack — each layer addresses a
different bottleneck:

```bash
Layer 1 — Model architecture
  Knowledge distillation → smaller student model
  Model pruning          → fewer attention heads and parameters

Layer 2 — Numerical precision
  fp16 / bf16 quantization → half memory, faster matrix operations
  int8 quantization        → quarter memory, even faster on supported hardware

Layer 3 — Runtime optimization
  ONNX export              → platform-optimized execution
  TensorRT                 → GPU-specific optimization
  Flash attention          → memory-efficient attention computation

Layer 4 — Index efficiency
  ANN search               → approximate but fast nearest neighbour
  Product quantization     → compressed embedding vectors
  HNSW / IVF indices       → structured search over vectors

Layer 5 — System optimization
  Result caching           → skip recomputation for repeated queries
  Dynamic batching         → process multiple queries together
  Async retrieval          → parallelize retrieval and reranking
```

Understanding which layer to optimize for your specific bottleneck is more
important than knowing every technique.

## The Efficiency Mindset

Three principles to carry through the rest of this module:

**1. Measure before optimizing**
Profile your system. The bottleneck is almost always the cross-encoder, almost
never the inverted index. Do not optimize what is not slow.

**2. Apply cheap wins first**
fp16 quantization, ONNX export, and result caching cost almost nothing to implement
and give free speedups. Apply these before attempting distillation or pruning.

**3. Distillation over architecture search**
For most IR applications, distilling a large cross-encoder into a smaller one
is more effective than manually designing a custom architecture. MiniLM-L6 distilled
from BERT-large outperforms custom small models designed from scratch.

## Where This Fits in the Learning Path

```bash
Phase 1-7:  learn what models exist and how they work
Phase 8:    learn how to make them fast enough to deploy  ← you are here
```

Everything in Phases 1-7 is about maximizing quality. Phase 8 is about delivering
that quality within the real-world constraints of latency, memory, and cost.
A retrieval system that works in a Colab notebook is a prototype. A retrieval system
that serves 100 QPS under 100ms is a product. This module is the bridge between
the two.

## My Summary

Efficiency in IR is about delivering retrieval quality within hard constraints on
latency, memory, and cost. The cross-encoder reranker is almost always the system
bottleneck — a single BERT-base forward pass takes 5-10ms, making 100-candidate
reranking infeasible within typical 100-200ms latency budgets. The efficiency stack
addresses this across five layers: model architecture (distillation, pruning),
numerical precision (fp16, int8), runtime optimization (ONNX, TensorRT), index
efficiency (ANN, PQ), and system optimization (caching, batching). Always profile
before optimizing, apply free wins first (fp16, ONNX, caching), and use knowledge
distillation as the primary technique for large speedups with manageable quality
loss.
