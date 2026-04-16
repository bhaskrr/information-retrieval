# Approximate Nearest Neighbour Search

Approximate Nearest Neighbour (ANN) search is a family of algorithms that find
vectors close to a query vector in high-dimensional space without exhaustively
comparing against every vector in the index. By trading a small amount of recall
accuracy for dramatically faster search, ANN makes dense retrieval practical at
scale - enabling millisecond query times over indices of millions or billions of
document vectors.

## Intuition

Exact nearest neighbour search compares a query vector against every document
vector and returns the closest ones. For 1 million documents with 768-dimensional
vectors, that is 1 million dot products per query - feasible but slow (~50ms on
GPU). For 1 billion documents it becomes ~50,000ms - completely impractical.

ANN accepts a small accuracy compromise: instead of guaranteeing the exact nearest
neighbours, it returns a set of vectors that are very likely to include the true
nearest neighbours with high probability. In practice, a well-tuned ANN index
returns 95-99% of the true nearest neighbours in 1-5ms - a 10-50x speedup for
a 1-5% recall cost. For IR this tradeoff is almost always acceptable because
the retrieval model itself introduces far more noise than the 1-5% ANN miss rate.

## Exact vs Approximate Search

| Method       | Recall | Latency (1M docs, 768-dim) | Memory |
| ------------ | ------ | -------------------------- | ------ |
| Exact        | 100%   | ~50ms                      | 3GB    |
| IVF (ANN)    | ~97%   | ~3ms                       | 3GB    |
| HNSW (ANN)   | ~99%   | ~1ms                       | 6GB    |
| IVF-PQ (ANN) | ~92%   | ~1ms                       | 200MB  |

The right choice depends on your corpus size, memory budget, and acceptable
recall loss.

## Core ANN Algorithms

### IVF - Inverted File Index

The simplest ANN approach. Clusters document vectors into nlist groups at index
build time using k-means. At query time, searches only the nprobe nearest clusters
instead of all vectors.

```bash
Build time:
  1. Run k-means on all document vectors → nlist cluster centroids
  2. Assign each document vector to its nearest centroid
  3. Store vectors grouped by cluster

Query time:
  1. Compute distance from query to all nlist centroids → O(nlist × d)
  2. Select nprobe nearest centroids
  3. Search vectors within those clusters → O(nprobe × cluster_size × d)
  4. Return top-k results
```

Parameters:

- nlist: number of clusters (typically sqrt(N) to 4×sqrt(N) where N = corpus size)
- nprobe: clusters to search at query time (tradeoff between speed and recall)

```bash
nlist = 1000, nprobe = 10:
  Search 1% of corpus → ~100x speedup vs exact
  Recall@10: ~92%

nlist = 1000, nprobe = 50:
  Search 5% of corpus → ~20x speedup vs exact
  Recall@10: ~98%
```

IVF is the standard first choice for large corpora. Simple to understand, easy
to tune, good recall-speed tradeoff.

### HNSW - Hierarchical Navigable Small World

A graph-based ANN algorithm. Builds a multi-layer graph where nodes are document
vectors and edges connect nearby vectors. Query time traverses the graph from
the top (coarse) layer to the bottom (fine) layer, narrowing down nearest
neighbours at each step.

```bash
Layer 2 (sparse):  few nodes, long-range connections → coarse navigation
Layer 1 (medium):  more nodes, medium connections    → medium resolution
Layer 0 (dense):   all nodes, short connections      → fine-grained search
```

Query traversal:

```bash
1. Start at entry point in top layer
2. Greedily move to node closest to query
3. Move down to next layer at current position
4. Repeat until reaching layer 0
5. Explore neighbors of current node → return top-k nearest
```

Parameters:

- M: number of connections per node (higher = better recall, more memory)
- ef_construction: beam size during index build (higher = better quality, slower build)
- ef_search: beam size during query (tradeoff: higher = better recall, slower search)

```bash
M=16, ef_search=64:   Recall@10 ~99%, latency ~1ms
M=32, ef_search=128:  Recall@10 ~99.5%, latency ~2ms
```

HNSW consistently achieves the best recall-speed tradeoff among ANN algorithms
and is the default choice for most production IR systems when memory is available.

### Product Quantization (PQ)

PQ is a compression technique that dramatically reduces the memory footprint of
an ANN index by compressing document vectors into compact codes.

```bash
Original vector: 768 dimensions × 4 bytes = 3072 bytes per document

PQ compression:
  Split 768-dim vector into m subspaces of (768/m) dims each
  Quantize each subspace to one of k centroids (using k-means)
  Store only centroid indices: m × log₂(k) bits

Example: m=8, k=256 (8 bits per subspace)
  Storage: 8 bytes per document (384x compression!)
  Recall@10: ~90-95% (some loss vs exact)
```

PQ enables billion-scale indices in memory that would otherwise require terabytes.

### IVF-PQ

Combines IVF (fast coarse search) with PQ (compressed vectors):

```bash
Build: cluster vectors (IVF) + compress residuals within clusters (PQ)
Query: find nearest clusters (IVF) + approximate distances within clusters (PQ)
```

The standard choice for billion-scale retrieval systems. Used in Facebook's
production search, Google's nearest-neighbour service, and most large-scale
recommender systems.

### ScaNN (Scalable Approximate Nearest Neighbours)

Google's ANN library. Uses anisotropic quantization - quantizes vectors to
minimize the error on the inner product specifically (most important for IR) rather
than minimizing Euclidean reconstruction error (what PQ does).

Typically 2-3x faster than FAISS at the same recall level for inner product
similarity. Available as a Python library and used in Google's production retrieval
systems.

## FAISS - The Standard ANN Library

Facebook AI Similarity Search (FAISS) is the most widely used ANN library in IR.
Supports CPU and GPU, implements IVF, HNSW, PQ, and combinations thereof.

### Index factory strings

FAISS uses a string syntax to specify index types:

```python
import faiss

# Exact search
index = faiss.IndexFlatIP(d)

# IVF with exact vectors
index = faiss.index_factory(d, "IVF1000,Flat")

# HNSW
index = faiss.index_factory(d, "HNSW32")

# IVF with PQ compression
index = faiss.index_factory(d, "IVF1000,PQ16")

# OPQ preprocessing + IVF-PQ (better quality)
index = faiss.index_factory(d, "OPQ16_64,IVF1000,PQ16")
```

### GPU acceleration

FAISS supports moving indices to GPU for ~10x additional speedup:

```python
res   = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

## Choosing the Right Index

| Corpus size | Memory budget | Recommendation                             |
| ----------- | ------------- | ------------------------------------------ |
| < 100K      | Any           | Flat (exact) - no need for ANN             |
| 100K - 1M   | Ample         | HNSW - best recall-speed tradeoff          |
| 100K - 1M   | Tight         | IVF - good tradeoff, less memory than HNSW |
| 1M - 100M   | Ample         | HNSW or IVF                                |
| 1M - 100M   | Tight         | IVF-PQ - compressed vectors                |
| > 100M      | Any           | IVF-PQ or ScaNN - only practical options   |

Rule of thumb: start with HNSW for most IR applications. Switch to IVF-PQ when
memory becomes the constraint.

## ANN in the Full IR Pipeline

```bash
Query
  ↓
Encode query → 768-dim vector   (bi-encoder, ~5ms)
  ↓
ANN search in FAISS HNSW        (~1ms for 1M docs)
  → top-100 candidate doc IDs
  ↓
Fetch candidate texts from store (~1ms)
  ↓
Cross-encoder reranking          (~50ms for top-50)
  ↓
Final top-10 results
```

Total: ~57ms. Within a 100ms latency budget.

ANN search is the final piece of the efficiency stack. With quantized models,
distilled encoders, ONNX runtimes, result caching, dynamic batching, and HNSW
indices, a complete production IR system can serve hundreds of QPS within tight
latency budgets on modest hardware.

## My Summary

ANN search finds approximate nearest neighbours in high-dimensional vector spaces
without exhaustive comparison, trading 1-5% recall for 10-100x speedup. IVF
clusters vectors and searches only nearby clusters at query time - simple, tunable, good tradeoff. HNSW builds a multi-layer navigable graph - fastest query time, best recall, higher memory cost. Product quantization compresses 768-dim float vectors into byte codes achieving 100-400x memory compression at 5-10% recall cost, enabling billion-scale indices. FAISS is the standard library implementing all these approaches with CPU and GPU support. For most IR applications, HNSW is the right default choice; IVF-PQ is the right choice when memory is the binding constraint at large scale.
