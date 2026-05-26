# Reranking Latency Optimization

Reranking latency optimization is the set of engineering and modeling techniques
that reduce the time cost of second-stage retrieval scoring to fit within production
latency budgets without unacceptable quality degradation. A cross-encoder that
achieves NDCG@10 = 0.50 is worthless if it takes 2 seconds per query in an
application that requires 200ms end-to-end response. Latency optimization for
reranking spans model compression (quantization, distillation, pruning), inference
infrastructure (batching, caching, hardware acceleration), algorithmic changes
(early exit, adaptive k₁, cascade scoring), and architectural choices that trade
precision for speed. The goal is to find the point on the latency-quality Pareto
frontier that satisfies the application's requirements - not to minimize latency
at any quality cost, but to identify the maximum quality achievable within the
latency budget.

## Intuition

A cross-encoder reranking 100 candidates performs 100 separate transformer forward
passes - each encoding the concatenated (query, document) sequence. If each forward
pass takes 7ms on a GPU (typical for MiniLM-L-6), 100 passes cost 700ms. That is
far outside the 100-200ms reranking budget for interactive search.

Multiple strategies can bring this to 70ms or less without major quality loss:

```
Starting point:  100 candidates × 7ms each = 700ms

Reduce k₁:       50 candidates × 7ms each = 350ms   (-50%)
Quantize int8:   100 candidates × 3ms each = 300ms   (-57%)
ONNX export:     100 candidates × 2ms each = 200ms   (-71%)
Dynamic batching: 100 candidates in parallel ≈ 15ms  (-98%)
Smaller model:   100 candidates × 1ms each = 100ms   (-86%)
Combined (↓k₁ + quantize + ONNX):
                 50 × 1.5ms = 75ms           (-89%)
```

No single technique is a silver bullet, but combining multiple optimizations
compounds their effect. The engineering challenge is understanding which combination
preserves quality and which crosses the quality floor for your application.

## The Latency Budget Framework

Before optimizing, establish the latency budget for each pipeline component:

```
Total end-to-end budget: 300ms (interactive search typical)

Component allocation:
  Network/API overhead:     20ms
  Query preprocessing:       5ms
  First-stage retrieval:    30ms (BM25 or ANN)
  Reranker inference:       70ms  ← our optimization target
  Response assembly:        10ms
  Network/API return:       20ms
  ──────────────────────────────
  Total:                   155ms  ← margin within 300ms budget
```

The reranker budget is whatever remains after first-stage and overhead costs.
A tight 50ms budget requires aggressive optimization. A generous 200ms budget
allows for larger models and more candidates.

### Latency vs quality at different budgets

```
Budget      Configuration                    Approx NDCG@10
────────────────────────────────────────────────────────────────────
< 20ms      No reranking (bi-encoder only)   0.48
20-50ms     TinyBERT cross-encoder, k₁=25   0.51
50-100ms    MiniLM-L-6, k₁=50               0.53
100-150ms   MiniLM-L-12, k₁=100             0.55
150-250ms   RoBERTa cross-encoder, k₁=100   0.57
250-500ms   monoT5-base, k₁=100             0.59
> 500ms     LLM reranker or monoT5-3b       0.62+
```

## Model-Level Optimizations

### Quantization for cross-encoders

Quantization reduces the numerical precision of model weights from float32 to
int8 or even int4, trading a small amount of accuracy for significant speed
and memory gains:

**Dynamic int8 quantization:**

```
float32 model: weight ∈ [-3.2, 3.2] → 32-bit float per value
int8 model:    weight ∈ [-128, 127]  → 8-bit integer per value

Conversion: scale = max(|weights|) / 127
            int_weight = round(weight / scale)
```

Practical gains for MiniLM cross-encoder:

```
MiniLM-L-6 float32:  ~7ms per pair, 84MB model
MiniLM-L-6 int8:     ~3ms per pair, 22MB model
Quality change:       NDCG@10 -0.002 (negligible)
```

The 2-3x speedup from int8 quantization with negligible quality loss is almost
always worth applying.

**Static int8 quantization:**
More aggressive than dynamic - calibrate on representative data to set optimal
quantization scales:

```
Step 1: collect activation statistics on calibration set (1000 example pairs)
Step 2: compute optimal per-tensor quantization scales
Step 3: quantize weights and activations statically
Gain: slightly faster than dynamic quantization
Cost: requires calibration step, marginal quality degradation
```

**FP16 (half precision):**
On modern GPUs with tensor cores (A100, H100, RTX 30xx), fp16 runs faster than
fp32 while maintaining near-identical accuracy:

```
MiniLM-L-6 float32:  7ms on A100
MiniLM-L-6 float16:  3ms on A100 (tensor cores accelerate fp16)
Quality change:       essentially identical
```

FP16 is the minimum optimization to apply on any GPU - it costs nothing in
quality and provides consistent 2x speedup on modern hardware.

### Knowledge distillation for faster rerankers

The MiniLM and TinyBERT families are explicitly trained to distill larger models:

**MiniLM-L-6-v2 (22M params) vs BERT-base (110M):**

```
BERT-base cross-encoder:    ~35ms per pair on GPU, NDCG@10 = 0.42
MiniLM-L-6 cross-encoder:   ~7ms per pair on GPU,  NDCG@10 = 0.39
```

5x speedup for 7% quality reduction - often an excellent tradeoff.

**Task-specific distillation:**
Beyond general model distillation, distilling from a large teacher to a small
student specifically for your domain:

```
Teacher: monoT5-3b or RoBERTa cross-encoder fine-tuned on domain data
Student: MiniLM-L-6 initialized from pretrained model
Training: MSE loss on teacher scores for (query, document) pairs in your domain
Result:  Student matches teacher quality on domain despite 14x fewer parameters
```

This is the highest-leverage optimization available when domain-specific quality
is critical - you get a fast model tailored to your domain rather than a generic
fast model.

### ONNX Runtime export

Exporting cross-encoders to ONNX (Open Neural Network Exchange) format and
running them through the ONNX Runtime inference engine produces significant
speedups over native PyTorch inference:

```
Why ONNX is faster than PyTorch:
  PyTorch: dynamic computation graph, Python overhead per operation
  ONNX:    static computation graph, compiled kernels, no Python overhead
  ONNX Runtime: applies graph-level optimizations (operator fusion, memory planning)

Typical gains:
  MiniLM-L-6 PyTorch:      7ms per pair
  MiniLM-L-6 ONNX Runtime: 2-3ms per pair
  Combined int8 + ONNX:    1-2ms per pair
```

### Sparse attention for long documents

Cross-encoders process the concatenated (query, document) sequence. For long
documents (512+ tokens), attention computation scales quadratically with sequence
length. Sparse attention patterns reduce this:

```
Standard attention: O(n²) where n = query_length + doc_length
Sparse attention:   O(n × k) where k = local window size

For 512-token sequences:
  Standard: 262,144 attention operations
  Sparse (window=64): 32,768 attention operations (8x reduction)
```

Flash Attention (covered in 08-efficiency/08-flash-attention.md) applies to
cross-encoders and reduces memory usage proportionally to sequence length,
enabling longer document processing within the same latency budget.

## Algorithmic Optimizations

### Adaptive k₁ - dynamic candidate set size

Not all queries need 100 candidates to find their top-10 results. Easy queries
(with clearly dominant relevant documents) can use k₁ = 25. Hard queries need
k₁ = 200. Adaptive k₁ uses the first-stage scores to decide how many candidates
to rerank:

```
First-stage top-k scores: [0.95, 0.47, 0.46, 0.45, ...]
Observation: large gap between rank 1 (0.95) and rank 2 (0.47)
→ The top document is clearly dominant
→ Safe to use k₁ = 25 (smaller candidate set, faster reranking)

First-stage top-k scores: [0.82, 0.81, 0.80, 0.79, 0.78, ...]
Observation: no clear gap, many documents with similar scores
→ Many candidates are potentially relevant
→ Use k₁ = 200 (larger candidate set to not miss relevant docs)
```

Adaptive k₁ reduces average reranking cost by 20-40% with minimal quality
impact by applying expensive reranking only when needed:

```
Threshold rule:
  if score_gap(rank_1, rank_10) > δ:    use k₁ = 25
  elif score_gap(rank_1, rank_50) > δ:  use k₁ = 50
  else:                                  use k₁ = 100
```

### Early exit - stop reranking when confident

If the reranker has already found several high-confidence relevant documents,
it may not need to score all remaining candidates:

```
Process candidates in order of first-stage score:
  After scoring top-20: if top-10 scores >> bottom-10 scores
    → high confidence in current ranking, exit early

Early exit condition:
  score(rank_10) - score(rank_11) > confidence_threshold
  → Significant gap means rank-11 unlikely to enter top-10
  → Stop reranking, return current top-10
```

Early exit requires careful calibration - the cross-encoder score gap is not
always a reliable confidence indicator. But when tuned on validation data, it
can reduce average reranking cost by 30-50% for easy query sets.

### Cascade scoring with multiple models

Instead of using one expensive model for all 100 candidates, use a two-model
cascade:

```
Stage 2a: fast model (TinyBERT, k₁=100) → top-20 candidates
Stage 2b: quality model (RoBERTa, k₁=20) → top-10 final results

Total cost:
  TinyBERT × 100 = 100 × 1ms  = 100ms
  RoBERTa × 20   =  20 × 15ms = 300ms
  Total: 400ms
  vs direct RoBERTa × 100 = 100 × 15ms = 1500ms

Savings: 3.75x faster at modest quality loss
```

The key insight: use an expensive quality model only on the small set of
candidates that a cheap fast model already identified as promising. The fast
model handles 80% of the filtering cost.

### Prefiltering with lightweight signals

Before the cross-encoder, apply fast lightweight relevance filters to reduce
the candidate set:

**Exact match prefilter:**

```
Filter out documents with zero BM25 score for any query term
→ Removes documents that share no vocabulary with the query
→ Fast (inverted index operation), removes 30-50% of candidates
→ Low risk of removing relevant documents (vocabulary overlap usually indicates relevance)
```

**Length filter:**

```
For queries about specific facts: very long documents may contain the answer
but rank poorly for cross-encoders (relevant sentence buried in noise)
→ Consider chunking long documents before reranking
→ Alternatively: prefer documents of similar length to query
```

**Semantic threshold filter:**

```
Filter out candidates with bi-encoder score < threshold
→ Documents far from query in embedding space unlikely to be relevant
→ Threshold calibrated to maintain >95% recall on validation set
```

## Infrastructure Optimizations

### Dynamic batching

Cross-encoders process (query, document) pairs one at a time or in small batches.
Dynamic batching groups multiple query requests together for parallel processing:

```
Without batching:
  Request 1: score 100 pairs → 700ms
  Request 2: score 100 pairs → 700ms (waits for Request 1)
  Total for 2 requests: 1400ms

With dynamic batching (batch_size=2):
  Requests 1+2 grouped: score 200 pairs in parallel → 850ms
  Total for 2 requests: 850ms each (40% reduction)
```

Dynamic batching provides near-linear throughput improvement with batch size,
at the cost of latency variance (requests must wait to form a batch):

```
Batch size   Throughput      Latency
1            1x              T
2            1.8x            T + wait_time
4            3.2x            T + 2×wait_time
8            5.5x            T + 4×wait_time
```

For high-QPS applications where multiple requests arrive concurrently, dynamic
batching is the highest-leverage infrastructure optimization.

### Result caching

Many production systems see the same or similar queries repeatedly. Caching
reranking results eliminates latency entirely for cached queries:

```
Cache key:  hash(query_text + sorted(candidate_doc_ids))
Cache value: ranked list of doc_ids + reranker scores
TTL:        depends on corpus update frequency (hours to days)
```

**Cache hit rates in production:**

```
Enterprise search:       40-60% cache hit rate (repetitive queries)
E-commerce:              20-40% (seasonal/popular products)
General web search:      5-15% (diverse query distribution)
News search:             < 5% (time-sensitive, unique queries)
```

For enterprise and e-commerce applications, caching alone can reduce average
reranking latency by 40-50%.

### GPU memory bandwidth optimization

Cross-encoder inference is often memory-bandwidth-limited rather than
compute-limited - the bottleneck is loading model weights from GPU memory,
not performing matrix multiplications. Optimizations:

**Persistent model residency:**
Keep the cross-encoder model loaded in GPU memory between requests. Loading
a MiniLM model from CPU memory to GPU memory takes 200-500ms - avoidable if
the model is always resident.

**KV cache for repeated queries:**
For the query portion of the (query, document) input, the key-value attention
states can be cached and reused across all document scoring operations:

```
Without KV cache:
  Encode [query + doc₁] → full attention computation
  Encode [query + doc₂] → full attention computation (query recomputed)

With KV cache:
  Encode [query] → cache query KV states
  Encode [doc₁] with cached query KV → partial attention computation
  Encode [doc₂] with cached query KV → partial attention computation

Savings: query portion of computation eliminated for k₁-1 documents
```

For a 512-token sequence with 20-token query and 492-token document:

```
Without KV cache: compute 512 tokens × 100 times = 51,200 token computations
With KV cache:    compute 20 tokens once + 492 tokens × 100 times = 49,220
Savings: modest (4%) for long documents, significant (30%+) for short documents
```

### Horizontal scaling

When a single GPU cannot meet latency requirements, distribute inference:

```
Request routing:
  Load balancer → GPU server 1 (handles 50% of queries)
                → GPU server 2 (handles 50% of queries)

Each server runs its own cross-encoder instance
Latency: unchanged (single request still hits one server)
Throughput: 2x (two servers process requests in parallel)
```

Horizontal scaling improves throughput, not latency. It is the right solution
when the system can meet latency for a single request but cannot handle high QPS.

## The Optimization Decision Tree

```
Is latency currently acceptable?
  Yes → No optimization needed, monitor and revisit if load increases
  No  →
        Is GPU being used?
          No → Enable GPU inference first (10-20x speedup immediately)
          Yes →
                Is fp16 enabled?
                  No → Enable fp16 (2x speedup, free on modern GPUs)
                  Yes →
                        Is ONNX Runtime in use?
                          No → Export to ONNX (1.5-2x additional speedup)
                          Yes →
                                Is k₁ > 50?
                                  Yes → Reduce k₁ or use adaptive k₁
                                  No →
                                        Is model > MiniLM-L-6?
                                          Yes → Distill to smaller model
                                          No  → Consider hardware upgrade
                                                or horizontal scaling
```

## Quality Impact of Optimizations

The critical question for each optimization: how much quality does it trade for
speed? Approximate quality impact on MS MARCO MRR@10:

```
Optimization               Speed gain    Quality loss
──────────────────────────────────────────────────────────────────────
FP16 quantization          2x            ~0.001 MRR (negligible)
Int8 dynamic quantization  2-3x          ~0.003 MRR (negligible)
ONNX Runtime               1.5-2x        0 (exact same computation)
Reduce k₁: 100→50          2x            ~0.008 MRR (small)
Reduce k₁: 100→25          4x            ~0.018 MRR (moderate)
Distill L-12 to L-6        5x            ~0.005 MRR (small)
Distill BERT to TinyBERT   12x           ~0.018 MRR (moderate)
Early exit (aggressive)    2-3x          ~0.010 MRR (context-dependent)
Cascade (TinyBERT→RoBERTa) 3.75x         ~0.004 MRR (small)
```

The safe optimizations (FP16, ONNX, int8) combine to give ~5x speedup with
essentially zero quality loss. Aggressive optimizations (small k₁, tiny model)
reach 10-20x speedup at moderate quality cost.

## Measuring and Monitoring Reranker Latency

In production, latency is not a single number - it is a distribution:

```
Metrics to track:
  p50 (median):  typical user experience
  p95:           slow but not extreme cases
  p99:           near-worst case (used for SLA definition)
  p99.9:         extreme outliers (document length outliers, cold starts)

Typical production targets for interactive search:
  p50 < 50ms
  p95 < 150ms
  p99 < 300ms
```

Alerting rules:

```
Alert if p99 latency > 500ms for 5+ consecutive minutes
Alert if p50 latency increases > 50% from rolling 1-hour baseline
Alert if error rate > 0.1% (inference failures, OOM)
```

The most common production latency problems:

**Document length variance:**
Unusually long documents (OCR output, full-text articles) can cause latency
spikes. Mitigations: truncation to max_length, length-based k₁ reduction.

**GPU memory fragmentation:**
Long-running GPU inference processes accumulate memory fragmentation over time,
degrading performance. Mitigation: periodic process restart or memory defragmentation.

**Cold start:**
The first query after a model deployment loads the model from disk, causing a
one-time latency spike. Mitigation: warm-up requests after deployment.

## Combining Optimizations - A Production Example

A typical production optimization journey for a cross-encoder reranker:

```
Starting point:
  Model:     bert-base cross-encoder (110M params)
  Precision: float32
  Runtime:   PyTorch
  k₁:        100
  Measured:  950ms p50, 1.2s p99

Step 1 - Enable fp16:
  Measured:  480ms p50, 600ms p99   (2x improvement)

Step 2 - Export to ONNX Runtime:
  Measured:  240ms p50, 320ms p99   (2x additional improvement)

Step 3 - Apply int8 quantization:
  Measured:  120ms p50, 160ms p99   (2x additional improvement)

Step 4 - Distill to MiniLM-L-6:
  Measured:  35ms p50, 50ms p99     (3.5x additional improvement)

Step 5 - Reduce k₁ to 50:
  Measured:  18ms p50, 25ms p99     (2x additional improvement)

Final: 18ms p50 from 950ms starting point (53x total improvement)
NDCG@10 change: -0.028 (from 0.420 to 0.392)
```

Whether this tradeoff is acceptable depends entirely on the application's
latency requirements and quality floor. For many interactive applications,
NDCG = 0.392 at 18ms is far more valuable than NDCG = 0.420 at 950ms.

This note closes the reranking module. The full reranking deep-dive covers the
complete lifecycle of production reranking: designing the pipeline (note 01),
training the model (note 02), choosing the ranking objective (note 03), scaling
to LLM quality (note 04), and making it fast enough to deploy (this note).

## My Summary

Reranking latency optimization combines model-level and infrastructure-level
techniques to fit cross-encoder or LLM reranking within production latency
budgets. The highest-leverage model optimizations are: FP16 quantization (2x
speedup, zero quality loss on GPU), ONNX Runtime export (1.5-2x speedup, zero
quality cost), int8 dynamic quantization (2-3x speedup, negligible quality loss),
and knowledge distillation to MiniLM-L-6 (5x speedup, ~0.5% quality reduction).
Algorithmic optimizations include adaptive k₁ (dynamically choosing how many
candidates to rerank based on first-stage score distribution), early exit (stopping
when confident about the top-k ranking), and cascade scoring (cheap fast model
filters to top-20, expensive quality model scores final 20). Infrastructure
optimizations include dynamic batching for high-QPS systems, result caching for
repetitive query distributions, query KV cache reuse across documents, and
horizontal GPU scaling for throughput-limited systems. The combination of FP16,
ONNX, and int8 quantization typically yields 5x speedup with negligible quality
loss - the safe baseline optimization every production reranker should apply.
Further speedup requires quality-speed tradeoffs that must be calibrated against
the application's quality floor rather than applied universally.
