# Caching and Batching

Caching and batching are system-level optimization techniques that reduce the
total computational work a retrieval system performs without changing any model
architecture or precision. Caching stores previously computed results and reuses
them when the same computation is requested again. Batching groups multiple
independent computations together so they can be processed in parallel on GPU
hardware. Both techniques are orthogonal to model-level optimizations like
quantization and distillation - they address different bottlenecks and their
benefits compound when applied together.

## Intuition

### Caching

The best way to make a computation faster is to not do it at all. Search systems
have highly predictable query patterns - a small fraction of queries account for
a large fraction of traffic. The top 1000 queries on a search engine may represent
30-50% of all daily traffic. If you cache the results for these queries, half your
traffic requires zero computation.

This principle applies at multiple levels in the retrieval pipeline:

- Query result caching: cache the final ranked list for frequent queries
- Embedding caching: cache encoded document vectors (already done in dense
  retrieval via the offline index, but also applicable to query vectors for
  repeated queries)
- Reranker score caching: cache cross-encoder scores for frequently seen
  (query, document) pairs

### Batching

A GPU is designed for massive parallelism. Running a single BERT forward pass
uses a small fraction of the GPU's available compute - most cores sit idle.
Running 32 forward passes simultaneously uses the GPU at near-full capacity and
takes only slightly longer than running one.

Batching groups multiple requests together into a single forward pass. For a
system serving many concurrent users, this turns many small underutilized GPU
operations into fewer large fully-utilized ones - dramatically improving throughput
(queries per second) with minimal impact on individual query latency.

## Caching Strategies

### Level 1 - Query result cache

Cache the complete ranked result list for a query:

```bash
key:   hash(query_text)
value: [(doc_id, score), ...] top-k results
TTL:   minutes to hours depending on content freshness requirements
```

Highest impact, simplest implementation. Works best for head queries (popular
searches) that repeat frequently. Provides zero benefit for tail queries (rare
unique searches) which are often the majority by count but minority by traffic.

### Level 2 - Query embedding cache

Cache the encoded query vector:

```bash
key:   hash(query_text)
value: query_embedding (768-dim float array)
TTL:   hours to days (embeddings do not change unless model is retrained)
```

Eliminates the query encoding step for repeated queries. Useful when:

- The same query is submitted frequently (autocomplete, saved searches)
- Query encoding is on the critical path (no GPU available, using BERT not MiniLM)

### Level 3 - Document embedding cache

Cache encoded document representations. This is essentially what the dense retrieval
index already does - document vectors are precomputed offline. The cache is the
index itself.

For dynamic corpora where documents are updated frequently:

- Incremental indexing: only re-encode changed documents
- Delta index: maintain a small cache of recently changed document vectors,
  merge periodically with the main index

### Level 4 - Reranker score cache

Cache cross-encoder scores for (query, document) pairs:

```bash
key:   hash(query_text + doc_id)
value: relevance_score (float)
TTL:   hours (scores do not change unless model is retrained)
```

Useful when the same document repeatedly appears in the candidate set for popular
queries - its reranker score can be reused without re-running the cross-encoder.

### Cache eviction policies

```
Policy      Description                         Best for
──────────────────────────────────────────────────────────────
LRU         Evict least recently used           General use
LFU         Evict least frequently used         Skewed traffic
TTL         Evict after fixed time              Time-sensitive content
Size-based  Evict largest items first           Memory-constrained
```

LRU is the default for most IR caching scenarios. TTL is essential when the
underlying content changes (news articles, product listings, live data).

## Dynamic Batching

### Static batching

Process a fixed batch of N requests together. Simple but wasteful - if requests
arrive unevenly, you either wait to fill the batch (adding latency) or process
partial batches (wasting GPU capacity).

### Dynamic batching

Accumulate incoming requests for a short window (1-5ms) then process together.
Balances latency and throughput:

```bash
Window: 2ms
Requests arriving: [q1 at t=0ms, q2 at t=0.5ms, q3 at t=1.2ms, q4 at t=2.3ms]
Batch 1: [q1, q2, q3] processed at t=2ms
Batch 2: [q4, ...] processed at t=4ms
```

Queries q1-q3 experience 0-2ms additional wait for batching but the batch
processes 3x as fast as three individual forward passes.

### Continuous batching

The modern approach for LLM-based systems. Instead of fixed windows, maintain
a queue and continuously form batches from available requests. Requests join
in-flight batches where possible.

## GPU Utilization and Batch Size

Understanding GPU utilization explains why batching matters:

```bash
Single BERT-base forward pass on A100:
  Matrix multiplications use: ~15% of GPU compute
  Memory bandwidth use:       ~20% of GPU memory bandwidth
  Effective GPU utilization:  ~15%

Batch of 32 on A100:
  Matrix multiplications use: ~90% of GPU compute
  Memory bandwidth use:       ~85% of GPU memory bandwidth
  Effective GPU utilization:  ~85%

Speedup: ~6x throughput for ~6x more requests, near-constant per-batch latency
```

The practical implication: a single GPU running single requests processes ~20 QPS.
The same GPU with batch size 32 processes ~120 QPS - 6x more throughput for zero
additional hardware cost.

## Padding and Sequence Length Optimization

Transformer models process all sequences in a batch to the same length (the
longest sequence). Short sequences are padded with [PAD] tokens - wasted
computation.

### Sorting by length before batching

Group requests of similar length together:

```bash
Unsorted batch: [512 tokens, 45 tokens, 378 tokens, 67 tokens]
→ padded to 512 tokens for all → 4 × 512 = 2048 tokens processed

Sorted batches: [45, 67] and [378, 512]
→ batch 1: 2 × 67 = 134 tokens
→ batch 2: 2 × 512 = 1024 tokens
→ total: 1158 tokens processed (44% less computation)
```

### Dynamic padding

Pad each batch only to the length of its longest sequence, not the global maximum:

```python
def collate_with_dynamic_padding(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    # pad all items to max_len, not to model's max_length
```

### Truncation strategy

For long documents in cross-encoder reranking, truncate intelligently:

- Keep the beginning and end of the document (often most informative)
- Use a sliding window approach for very long documents

## Cache Sizing Guidelines

Sizing your cache correctly matters - too small and hit rates are poor, too large
and memory is wasted:

```
Cache type          Recommended size        Memory estimate
──────────────────────────────────────────────────────────────────
Query results       Top 10K queries         ~50MB (10K × 5KB avg)
Query embeddings    Top 100K queries        ~300MB (100K × 3KB)
Reranker scores     Top 1M query-doc pairs  ~8MB (1M × 8 bytes)
Document embeddings Entire corpus (index)   See dense-retrieval.md
```

For most applications, caching the top 10K query results covers 40-60% of traffic.
The Zipf distribution of query frequency means the gains are heavily front-loaded -
the first 1K cached queries often cover 30-40% of traffic alone.

## Redis for Production Caching

For production systems, replace the in-memory LRU cache with Redis:

Redis advantages over in-memory cache:

- Survives process restarts
- Shared across multiple server instances
- Built-in TTL management
- Configurable eviction policies (LRU, LFU)
- Can store values up to 512MB
- Cluster mode for horizontal scaling

## My Summary

Caching and batching are system-level optimizations that reduce computation without
touching model weights. Caching stores previously computed results at multiple
levels - query results, query embeddings, reranker scores - and reuses them for
repeated requests. Given the Zipf distribution of query frequency, caching the top
10K queries typically covers 40-60% of all traffic with zero computation. Batching
groups concurrent requests into single GPU forward passes, dramatically improving
throughput - a GPU at batch size 32 processes ~6x more queries per second than at
batch size 1. Dynamic batching with a short accumulation window (2-5ms) balances
the throughput benefit against added latency. Length sorting before batching reduces
padding waste by 20-40% for corpora with variable-length documents. These techniques
are orthogonal to model-level optimizations - apply them together for maximum
efficiency gains.
