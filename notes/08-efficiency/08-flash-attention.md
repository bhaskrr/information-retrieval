# Flash Attention

Flash Attention is an exact, IO-aware algorithm for computing self-attention that
dramatically reduces memory usage and increases speed by restructuring the attention
computation to minimize data movement between GPU high-bandwidth memory (HBM) and
GPU on-chip SRAM. Introduced by Dao et al. at Stanford in 2022 and extended in
Flash Attention 2 (2023) and Flash Attention 3 (2024), it has become the default
attention implementation in virtually every modern transformer - from BERT to
GPT-4 - because it makes the same mathematical computation faster and more
memory-efficient without any approximation or quality loss.

## Intuition

Standard attention has a memory problem that is not obvious from the algorithm's
mathematical definition. The computation requires materializing the full attention
matrix of shape (seq_len × seq_len) - for a sequence of 512 tokens that is 262,144
floating point numbers per attention head. For a 12-head model with fp32 precision,
the attention matrices alone consume:

```
512 × 512 × 12 heads × 4 bytes = 12.6 MB per layer
× 12 layers = 151 MB just for attention matrices
```

This seems manageable. The real problem is not total size but data movement. GPU
computation is fast. Reading and writing to GPU memory (HBM) is comparatively slow.
Standard attention writes the full attention matrix to HBM, then reads it back to
compute the weighted sum of values - two full HBM read/write passes over the
attention matrix for every forward pass.

Flash Attention restructures the computation into tiles that fit entirely in SRAM
(the fast on-chip memory), performing the full attention computation for each tile
without ever writing the full attention matrix to HBM. The mathematical result is
identical - Flash Attention is exact, not approximate - but the data movement is
reduced from O(n²) to O(n), producing 2-4x speedup and allowing much longer
sequences on the same hardware.

## The Memory Hierarchy on Modern GPUs

Understanding Flash Attention requires understanding the GPU memory hierarchy:

```
SRAM (on-chip):
  Size:       ~20MB on A100
  Bandwidth:  ~19 TB/s
  Latency:    ~10 cycles

HBM (off-chip, "GPU memory"):
  Size:       40-80GB on A100
  Bandwidth:  ~2 TB/s
  Latency:    ~hundreds of cycles
```

SRAM is 10x faster than HBM but 4000x smaller. The key insight in Flash Attention:
if you can keep your working data in SRAM during computation, you avoid expensive
HBM reads and writes. Standard attention cannot do this because the full attention
matrix is larger than SRAM. Flash Attention restructures the algorithm to work in
tiles that fit in SRAM.

## Standard Attention - The Baseline

The standard scaled dot-product attention formula:

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V
```

Where Q, K, V are (seq_len × head_dim) matrices.

Standard implementation (three separate HBM read/write passes):

```
Step 1: S = Q × Kᵀ                     → write S (seq_len × seq_len) to HBM
Step 2: P = softmax(S / √d)             → read S from HBM, write P to HBM
Step 3: O = P × V                       → read P from HBM, compute output
```

Each arrow is an expensive HBM access. For long sequences:

- Writing S: O(n²) elements written to HBM
- Reading S: O(n²) elements read from HBM
- Writing P: O(n²) elements written to HBM
- Reading P: O(n²) elements read from HBM

Total HBM data movement: O(n²) - quadratic in sequence length.

## Flash Attention - The Algorithmic Innovation

Flash Attention fuses the three steps into a single pass using tiling and the
online softmax algorithm.

### Key insight 1 - Tiling

Divide Q, K, V into blocks that fit in SRAM. Process blocks of Q against all blocks
of K and V, computing partial attention outputs that are accumulated across blocks:

```
For each block of Q (size: BLOCK_Q × head_dim):
    For each block of K, V (size: BLOCK_KV × head_dim):
        Load Q_block, K_block, V_block into SRAM
        Compute partial scores: S_block = Q_block × K_block.T
        Compute partial attention: O_partial = softmax(S_block) × V_block
        Accumulate: O += O_partial (with rescaling)
    Write final O_block to HBM
```

No full n×n attention matrix is ever written to HBM - only the output O is written,
which is O(n) in size.

### Key insight 2 - Online softmax

The naive approach to tiling fails because softmax requires seeing all values to
compute the normalization denominator. Flash Attention uses an online softmax
algorithm that computes a running maximum and running sum, updating them as new
blocks are processed:

```
For each new block of scores S_block:
    m_new = max(m_old, max(S_block))
    l_new = exp(m_old - m_new) × l_old + sum(exp(S_block - m_new))
    O     = (exp(m_old - m_new) × O_old + exp(S_block - m_new) × V_block) / l_new
```

Where m is the running maximum and l is the running normalization sum. This
allows exact softmax computation without materializing the full attention matrix.

### Memory complexity

```
Standard attention:  O(n²) memory (stores full n×n attention matrix)
Flash Attention:     O(n)  memory (only stores output O and running statistics)
```

For n=4096 (long context):

```
Standard: 4096 × 4096 × 4 bytes × 12 heads × 12 layers = ~9.7 GB
Flash:    4096 × 768 × 4 bytes × 12 layers = ~230 MB
```

42x memory reduction - enabling context lengths that were impossible with standard attention.

## Flash Attention 2

Flash Attention 2 (Dao, 2023) improves over Flash Attention 1 in three ways:

### 1. Better parallelism

Flash Attention 1 parallelizes over batch size and heads. Flash Attention 2
additionally parallelizes over the sequence length dimension of Q, enabling
better GPU utilization for long sequences where the first two dimensions are small.

### 2. Reduced non-matmul operations

Modern GPUs have specialized hardware (tensor cores) for matrix multiplications
(matmul) that is much faster than general floating-point arithmetic. Flash
Attention 2 reduces non-matmul operations (rescaling in the online softmax) to
maximize time spent on matmul:

```
Flash Attention 1: 4 non-matmul FLOPs per attention FLOP
Flash Attention 2: 2 non-matmul FLOPs per attention FLOP
```

### 3. Causal masking optimization

For autoregressive (causal) attention where tokens only attend to previous tokens,
Flash Attention 2 skips computation for masked positions, roughly halving the work
for causal attention:

```
Standard causal: compute full n×n matrix, zero out upper triangle
FA2 causal:      only compute lower triangle blocks, skip upper triangle entirely
```

### Performance comparison

```
Method               Memory (seq=2048)   Speed vs standard
──────────────────────────────────────────────────────────
Standard attention   ~1.2 GB             1.0x (baseline)
Flash Attention 1    ~150 MB             2-4x faster
Flash Attention 2    ~150 MB             4-8x faster
```

## Flash Attention 3

Flash Attention 3 (Shah et al., 2024) targets Hopper architecture (H100) GPUs
specifically:

- **Warp specialization** - different groups of GPU threads handle different
  parts of the computation simultaneously (producer/consumer overlap)
- **Pingpong scheduling** - overlap attention computation with data loading
- **Low-precision FP8 support** - enable int8/fp8 precision with hardware support
- **Speedup** - 1.5-2x over Flash Attention 2 on H100

Flash Attention 3 is most relevant for inference at extreme scale. For most
IR practitioners, Flash Attention 2 is the practical target.

## Flash Attention in IR

Flash Attention is relevant to IR practitioners primarily through three pathways:

### 1. Faster bi-encoder encoding

Bi-encoders (sentence-transformers) use BERT-style transformers. Flash Attention
accelerates every forward pass:

```
Standard BERT-base encoding 512 tokens:   ~8ms per sample
Flash Attention BERT-base encoding 512 tokens: ~3ms per sample
```

For offline document encoding at index time, this translates directly to faster
indexing. For online query encoding, it contributes to lower latency.

### 2. Longer context for late chunking

Flash Attention makes long-context encoding practical. BERT is limited to 512
tokens with standard attention due to memory constraints. With Flash Attention,
Jina Embeddings v2 achieves 8192-token context by replacing standard attention
with Flash Attention - the same computation, 16x longer sequences.

This is the enabling technology behind late chunking (07-advanced/08-late-chunking.md)

- without Flash Attention, encoding full documents before chunking would be
  infeasible at the token lengths needed.

### 3. Faster cross-encoder reranking

Cross-encoders are the latency bottleneck in IR pipelines. Each (query, document)
pair is encoded as a concatenated sequence. Flash Attention reduces this forward
pass time:

```
Standard cross-encoder (512 tokens):    ~5ms per pair on A100
Flash Attention cross-encoder (512 tokens): ~2ms per pair
```

For reranking 100 candidates: 500ms → 200ms. A significant improvement within a
200ms latency budget.

### 4. RAG with long context

RAG pipelines increasingly use long-context LLMs for generation - including large
context windows of retrieved passages. Flash Attention is what makes these long-
context LLMs feasible:

```
GPT-4 (32K context):  only possible with Flash Attention
Claude (200K context): only possible with Flash Attention
Llama 2 (4K context): 4x speedup with Flash Attention
```

Without Flash Attention, 100K-token context windows would require hundreds of
gigabytes of HBM just for attention matrices.

## Flash Attention Requirements and Compatibility

```
Requirement              Details
────────────────────────────────────────────────────────────────────
GPU architecture         Ampere (A100, RTX 30xx) or newer for FA2
                         Hopper (H100) for FA3
                         Turing (T4, RTX 20xx) for FA1 only
CUDA version             >= 11.6 for FA2
PyTorch version          >= 2.0 for built-in SDPA
Dtype                    fp16 or bf16 required (not fp32)
Sequence length          Any - benefits increase with length
Installation (FA2)       pip install flash-attn (builds from source)
Installation (PyTorch)   F.scaled_dot_product_attention (no install)
```

## Flash Attention for the IR Practitioner

As an IR engineer, you interact with Flash Attention primarily through three
decisions:

**Decision 1 - Which embedding model to use**
Models trained or fine-tuned with Flash Attention can support longer contexts.
When choosing between models, check if they use Flash Attention for encoding -
it determines maximum sequence length and encoding speed.

```
Standard BERT-base:      512 tokens max, ~8ms per sample
Jina Embeddings v2:      8192 tokens max, ~30ms per sample (via FA)
```

**Decision 2 - How to load HuggingFace models**
Always specify `attn_implementation="sdpa"` when loading transformers for
retrieval. It is free performance:

```python
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="sdpa",   # free speedup
    torch_dtype=torch.float16     # also recommended
)
```

**Decision 3 - Chunking strategy for RAG**
Flash Attention enables late chunking over longer documents. If your RAG pipeline
uses standard chunking because documents were too long, revisit with a Flash
Attention-enabled long-context encoder.

Flash Attention is the only efficiency technique in this module that operates
at the algorithmic level - it changes how attention is computed, not how models
are compressed or deployed. It produces the same mathematical result as standard
attention but with O(n) memory instead of O(n²), enabling longer context windows
and faster throughput that all downstream IR applications benefit from.

## My Summary

Flash Attention restructures the scaled dot-product attention computation into
tiles that fit in GPU SRAM, avoiding expensive HBM reads and writes of the full
n×n attention matrix. The result is mathematically identical to standard attention
but uses O(n) memory instead of O(n²) and runs 2-8x faster, primarily by reducing
data movement between GPU memory and compute units. Flash Attention 2 adds sequence-
parallel execution and reduces non-matmul operations. In IR, Flash Attention matters
because it enables long-context encoding for late chunking, accelerates bi-encoder
document encoding at index time, speeds up cross-encoder reranking, and makes long-
context RAG pipelines feasible. For practitioners, the actionable steps are: use
`attn_implementation="sdpa"` when loading HuggingFace models, prefer float16 over
float32, and select long-context embedding models (Jina v2, E5-Mistral) for
applications requiring context beyond 512 tokens.
