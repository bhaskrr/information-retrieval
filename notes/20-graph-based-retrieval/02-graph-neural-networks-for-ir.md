# Graph Neural Networks for IR

Graph Neural Networks (GNNs) for information retrieval are neural architectures
that compute document, query, or entity representations by aggregating information
from a node's neighbors in a graph structure through iterative message passing,
producing embeddings that encode both the node's own content and the relational
context of its position in the graph. Where standard bi-encoders compute a
document's embedding from its text alone, GNN-based retrieval models compute a
document's embedding from its text combined with the text and structure of
related documents - citing papers, linked pages, co-purchased items, or
neighboring entities in a knowledge graph. This produces representations that
are structurally aware: two documents with similar text but very different
graph neighborhoods receive different embeddings, and two documents with
different text but similar graph neighborhoods (e.g., both heavily cited by
the same influential papers) receive embeddings pulled closer together. This
note covers the core GNN architectures - GCN, GraphSAGE, GAT, and their
retrieval-specific adaptations - and how message passing is integrated into
retrieval training and inference pipelines.

## Intuition

Imagine trying to understand a person's professional expertise. You could read
their resume alone (text-only approach) - degrees, job titles, listed skills.
Or you could also look at who they collaborate with, who cites their work, what
projects they have contributed to, and what skills their close collaborators
have (graph-aware approach). The second approach often reveals expertise that
the resume alone does not capture - someone whose resume says "software engineer"
but who collaborates exclusively with machine learning researchers and contributes
to ML infrastructure projects is likely to have ML expertise that a text-only
read of their resume would miss.

GNNs formalize this intuition for documents, entities, and any other graph-
structured data. A node's representation is updated by aggregating information
from its neighbors - and crucially, this is done iteratively across multiple
layers, so a node's final representation incorporates not just its immediate
neighbors but neighbors-of-neighbors, capturing increasingly distant relational
context with each additional layer.

The key engineering challenge GNNs solve: how do you aggregate a variable number
of neighbors (a paper might have 5 citations or 500) into a fixed-size update
to a node's representation, in a way that is permutation-invariant (the order
in which you consider the neighbors should not matter) and trainable end-to-end
alongside the text encoder?

## The Message Passing Framework

All major GNN architectures follow the same general computational pattern,
called message passing, applied iteratively across multiple layers:

```
For each node v at layer l:
  1. Collect messages from neighbors:
     messages = {m(h_u^(l-1), h_v^(l-1)) : u ∈ neighbors(v)}

  2. Aggregate messages into a single vector:
     aggregated = AGGREGATE(messages)

  3. Update the node's representation:
     h_v^(l) = UPDATE(h_v^(l-1), aggregated)

Where:
  h_v^(l) is node v's representation after l layers of message passing
  h_v^(0) is the initial representation (typically from a text encoder)
```

Different GNN architectures are distinguished primarily by how they define
the message function m(), the aggregation function AGGREGATE(), and the
update function UPDATE().

### Why multiple layers matter - the receptive field

Each layer of message passing expands a node's "receptive field" - the set
of nodes whose information can influence its final representation:

```
Layer 0 (initial):     node sees only itself (text-only representation)
Layer 1:                node sees itself + direct neighbors (1-hop)
Layer 2:                node sees itself + neighbors + neighbors-of-neighbors (2-hop)
Layer 3:                node sees up to 3-hop neighborhood

Example for citation graph:
  1-hop: papers this paper directly cites or is cited by
  2-hop: papers that cite (or are cited by) papers in the 1-hop neighborhood
         - captures "intellectual lineage" two steps removed
  3-hop: rapidly expands to a large fraction of the connected graph component
         - often too diffuse to provide meaningful signal, risk of oversmoothing
```

Most practical GNN-based retrieval systems use 2-3 layers - enough to capture
meaningful multi-hop relational context without the oversmoothing problem
where node representations become indistinguishable after too many rounds
of neighbor averaging.

## Graph Convolutional Networks (GCN)

The foundational GNN architecture, adapting the convolution operation from
CNNs (which aggregate local spatial neighborhoods in images) to graph-structured
data (which aggregates local graph neighborhoods).

### The GCN layer

```
h_v^(l) = σ( Σ_{u ∈ N(v) ∪ {v}}  (1 / √(d_u × d_v)) × W^(l) × h_u^(l-1) )

Where:
  N(v) = neighbors of node v
  d_u, d_v = degree of nodes u and v (number of edges)
  W^(l) = learnable weight matrix at layer l (shared across all nodes)
  σ = nonlinear activation function (ReLU, typically)
```

The normalization term 1/√(d_u × d_v) is critical - it prevents high-degree
nodes (e.g., a heavily-cited foundational paper with thousands of citing
papers) from disproportionately dominating the aggregation, and prevents
low-degree nodes from receiving unstably scaled updates.

### Strengths and limitations for retrieval

```
Strengths:
  Simple, well-understood, computationally efficient
  Works well for homogeneous graphs (single node type, single edge type)
  Effective baseline for citation-graph-based document ranking

Limitations:
  Treats all neighbors with equal importance after degree normalization
    - cannot learn that some citations matter more than others
  Requires the full graph structure at training and inference time
    (transductive - does not naturally generalize to unseen nodes)
  Struggles with heterogeneous graphs (multiple node/edge types) without
    significant architectural modification
```

GCN's transductive limitation is particularly important for retrieval systems:
a standard GCN is trained on a fixed graph and cannot directly produce
embeddings for new documents added after training without retraining or
specialized inductive extensions. This motivated GraphSAGE.

## GraphSAGE - Inductive Representation Learning

GraphSAGE (SAmple and aggreGatE) was specifically designed to address GCN's
inductive limitation - the ability to generate embeddings for nodes never seen
during training, which is essential for retrieval systems where new documents
are continuously added to the corpus.

### The GraphSAGE approach

```
For each node v:
  1. Sample a fixed-size set of neighbors (rather than using all neighbors)
     sampled_neighbors = sample(N(v), sample_size=k)

  2. Aggregate sampled neighbor representations using a learned aggregator
     h_N(v) = AGGREGATE({h_u^(l-1) : u ∈ sampled_neighbors})

  3. Concatenate with the node's own representation and transform
     h_v^(l) = σ( W^(l) × CONCAT(h_v^(l-1), h_N(v)) )

  4. Normalize (L2 normalization, common for retrieval embeddings)
     h_v^(l) = h_v^(l) / ||h_v^(l)||₂
```

The key innovations distinguishing GraphSAGE from GCN:

**Neighbor sampling:** Rather than aggregating over all neighbors (which is
expensive and unstable for high-degree nodes), GraphSAGE samples a fixed
number of neighbors at each layer, making computation cost predictable
regardless of node degree.

**Learned aggregation functions:** GraphSAGE supports multiple aggregation
function choices:

```
Mean aggregator:
  h_N(v) = mean({h_u : u ∈ sampled_neighbors})
  Simple, fast, but loses information about neighbor diversity

LSTM aggregator:
  h_N(v) = LSTM({h_u : u ∈ sampled_neighbors})
  More expressive (can weight neighbors differently based on order),
  but requires an arbitrary ordering of unordered neighbors (a conceptual
  mismatch, though it works reasonably in practice)

Pooling aggregator:
  h_N(v) = max({σ(W_pool × h_u + b) : u ∈ sampled_neighbors})
  Element-wise max pooling after a learned transformation,
  captures the most salient signal from any neighbor per dimension
```

**Inductive capability:** Because GraphSAGE learns aggregation functions
(not node-specific parameters), it can compute embeddings for nodes that
were not present during training, as long as their neighborhood structure
is available at inference time. A newly published paper, once its citations
are known, can be embedded using the same trained aggregation functions
without retraining.

### Why inductive capability matters for retrieval

```
Scenario: A new corpus document arrives daily (news articles, new papers,
          new products)

Transductive GNN (standard GCN):
  Requires retraining or expensive graph-wide recomputation to incorporate
  new nodes - impractical for daily updates

Inductive GNN (GraphSAGE):
  New node's embedding computed by sampling its neighbors and applying
  the already-trained aggregation function - no retraining needed
  Practical for continuous, incremental corpus updates
```

This inductive property is the primary reason GraphSAGE and its successors
are preferred over plain GCN for production retrieval systems with dynamic,
growing corpora.

## Graph Attention Networks (GAT)

Graph Attention Networks address GCN's limitation of treating all neighbors
with equal (degree-normalized) importance by learning to weight neighbors
differently based on their relevance to the central node - analogous to
how self-attention in transformers learns to weight different tokens.

### The GAT attention mechanism

```
For node v with neighbor u:
  Attention coefficient:
    e_vu = LeakyReLU( a^T × [W×h_v || W×h_u] )

  Normalized attention weight (softmax over all neighbors):
    α_vu = exp(e_vu) / Σ_{k ∈ N(v)} exp(e_vk)

  Weighted aggregation:
    h_v^(l) = σ( Σ_{u ∈ N(v)} α_vu × W × h_u^(l-1) )

Where:
  a = learnable attention vector
  W = learnable weight matrix
  || denotes concatenation
```

The attention weights α_vu are learned and node-pair-specific - unlike GCN's
fixed degree-based normalization, GAT learns which neighbors matter more for
a given node's representation, conditioned on both nodes' content.

### Multi-head attention for GNNs

Following the transformer architecture's multi-head attention pattern, GAT
typically computes multiple independent attention mechanisms in parallel and
concatenates (or averages) their outputs:

```
h_v^(l) = CONCAT(head_1, head_2, ..., head_K)

Where each head_k computes an independent attention-weighted aggregation
with its own learned parameters
```

Multiple attention heads allow the model to capture different types of
relational importance simultaneously - one head might learn to emphasize
recent citations, another might emphasize citations from highly-cited papers,
another might emphasize topical similarity among neighbors.

### When GAT outperforms GCN and GraphSAGE for retrieval

```
GAT advantages:
  Citation/link importance is genuinely heterogeneous:
    a single highly relevant citation should matter more than ten
    tangentially related ones
  Interpretability: attention weights provide a (partial) explanation
    for which relational connections drove a given relevance decision
  Better empirical performance on graphs where neighbor importance
    varies substantially (most real-world citation and entity graphs)

GAT costs:
  More parameters than GCN (additional attention parameters per layer)
  Higher computational cost (attention computation over all neighbor pairs)
  More hyperparameters to tune (number of attention heads, attention
    dropout rate)
```

## Applying GNNs to Retrieval: Architecture Patterns

### Pattern 1 - GNN as a document encoder enhancement

The most direct application: use a GNN to refine bi-encoder document embeddings
by incorporating graph context, while queries are encoded with a standard
text-only encoder:

```
Document encoding:
  text_emb(d) = text_encoder(d)             ← standard bi-encoder text encoding
  graph_emb(d) = GNN(text_emb(d), neighbors(d))  ← GNN refines with graph context
  final_emb(d) = combine(text_emb(d), graph_emb(d))   ← e.g., concatenation or
                                                          weighted sum

Query encoding:
  query_emb(q) = text_encoder(q)            ← standard, no graph context
                                               (queries typically have no
                                               natural graph position)

Retrieval:
  score(q, d) = cosine_similarity(query_emb(q), final_emb(d))
```

This pattern is the most common in production graph-augmented retrieval
systems because it requires minimal architectural change to the query side
(which usually lacks meaningful graph structure of its own) while enriching
document representations with relational context.

### Pattern 2 - Joint query-document graph

For domains where queries can also be situated in the graph (e.g., a query
that mentions a specific entity present in a knowledge graph), both query
and document representations can be computed through the same GNN:

```
Construct a unified graph including:
  Document nodes (with text content)
  Entity nodes (extracted from documents and queries)
  Query nodes (linked to entities mentioned in the query)

Apply GNN message passing across this unified heterogeneous graph
Both query and document final representations incorporate shared
  entity context, enabling matching through entity-mediated paths
  even when direct text similarity is low
```

This pattern requires entity linking infrastructure (covered in note 06 of
this module) and is more complex to implement but can capture relevance
relationships that pure text-side GNN refinement misses.

### Pattern 3 - GNN-based reranking

Rather than modifying the embeddings used for first-stage retrieval, apply
GNN-based scoring as a second-stage reranker over the candidates retrieved
by a standard first-stage system:

```
Stage 1: Standard BM25 or bi-encoder retrieval → top-100 candidates
Stage 2: Construct a local subgraph from the top-100 candidates and their
         immediate graph neighbors (citations, links, related entities)
Stage 3: Apply GNN message passing within this local subgraph
Stage 4: Re-score and re-rank the top-100 candidates using GNN-refined
         representations
```

This pattern limits the computational cost of GNN inference to a small
local subgraph rather than requiring graph-aware embeddings for the entire
corpus, making it more practical for latency-sensitive applications while
still capturing relational signal among the most promising candidates.

## Training GNN-Based Retrieval Models

GNN-based retrieval models are trained using the same contrastive learning
framework covered in the dense retrieval training module (18-dense-retrieval-training/),
with the GNN integrated into the document (or document and query) encoder:

```
Training objective: standard InfoNCE contrastive loss
  L = -log( exp(sim(q, d⁺)/τ) / Σⱼ exp(sim(q, dⱼ)/τ) )

Where sim(q, d) uses GNN-refined document embeddings:
  sim(q, d) = cosine_similarity(text_encoder(q), GNN(text_encoder(d), graph(d)))

Training requires:
  1. The graph structure available at training time (citation edges,
     hyperlinks, entity relations)
  2. Batched subgraph sampling (since the full graph is typically too
     large to process in a single forward pass)
  3. Backpropagation through both the text encoder and the GNN layers
     (end-to-end joint training)
```

### Subgraph sampling for mini-batch training

Because production-scale graphs (millions of documents, billions of edges)
cannot fit in GPU memory for a single forward pass, GNN training requires
sampling smaller subgraphs for each training batch:

```
For each training batch:
  1. Sample a set of "seed" nodes (the documents in this batch's
     query-document pairs)
  2. For each seed node, sample its k-hop neighborhood
     (using GraphSAGE-style fixed-size neighbor sampling)
  3. Construct the resulting subgraph (seed nodes + sampled neighbors)
  4. Run GNN message passing within this subgraph
  5. Compute embeddings for the seed nodes (used in the contrastive loss)
  6. Backpropagate through the subgraph
```

The neighbor sampling strategy (how many neighbors to sample at each hop,
uniform vs importance-weighted sampling) significantly affects both training
efficiency and final model quality - this is one of the most important
practical hyperparameters in GNN-based retrieval training.

## Computational Cost Considerations

GNN-based retrieval introduces meaningful computational overhead compared
to text-only retrieval, at both training and inference time:

```
Training cost:
  Standard bi-encoder: O(batch_size) forward passes through text encoder
  GNN-augmented: O(batch_size × average_neighbors_sampled) additional
                 forward passes through GNN layers, plus subgraph
                 construction overhead

Inference cost (indexing time):
  Standard bi-encoder: O(corpus_size) text encoder forward passes
  GNN-augmented: O(corpus_size × neighbors_per_node) additional GNN
                 computation - but this is typically a one-time indexing
                 cost (graph-refined embeddings precomputed and stored,
                 same as standard dense retrieval embeddings)

Inference cost (query time):
  Standard bi-encoder: O(1) query encoding + ANN search
  GNN-augmented (Pattern 1, document-side only):
    O(1) query encoding + ANN search - IDENTICAL to standard bi-encoder,
    because document embeddings were precomputed offline with GNN refinement
  GNN-augmented (Pattern 2, joint graph) or (Pattern 3, GNN reranking):
    Additional query-time graph construction and GNN inference required
```

The critical practical insight: Pattern 1 (GNN as document encoder enhancement,
with precomputed document embeddings) introduces zero query-time latency
overhead compared to standard dense retrieval, because all graph-aware
computation happens once during indexing. This makes Pattern 1 the most
practical choice for latency-sensitive production retrieval systems, while
Patterns 2 and 3 trade additional query-time cost for richer, query-aware
graph reasoning.

## Oversmoothing and Depth Limitations

A well-documented failure mode in GNN architectures, important to understand
before applying GNNs to retrieval: as the number of message-passing layers
increases, node representations tend to converge toward indistinguishable
values - a phenomenon called oversmoothing.

```
Intuition: each layer averages a node's representation with its neighbors.
After many layers, a node's representation has effectively averaged over
an exponentially growing neighborhood (the k-hop neighborhood grows
exponentially with graph connectivity), eventually approaching the average
representation of the entire connected graph component.

Practical consequence:
  2-3 layers: typically beneficial, captures meaningful multi-hop context
  4-6 layers: diminishing returns, beginning to oversmooth
  7+ layers: representations often become nearly indistinguishable across
             nodes, destroying the discriminative power needed for retrieval
```

This is a critical difference from text transformers, where adding more
layers (within reason) generally continues to improve representation quality.
For GNNs, the optimal depth is typically much shallower - most production
GNN-based retrieval systems use 2-3 layers, occasionally up to 4, rarely more.

### Mitigations for oversmoothing

```
Residual connections:
  h_v^(l) = h_v^(l-1) + GNN_layer(h_v^(l-1), neighbors)
  Allows the model to preserve earlier-layer (less-smoothed) information
  alongside deeper relational context

Jumping knowledge networks:
  Concatenate or aggregate representations from ALL layers (not just the
  final layer) when producing the final node embedding:
    h_v^final = AGGREGATE(h_v^(1), h_v^(2), ..., h_v^(L))
  Allows the model to adaptively use shallower or deeper representations
  depending on what is most useful for each node

PairNorm and other normalization techniques:
  Explicitly normalize node representations at each layer to maintain
  a minimum distance between distinct nodes' representations, directly
  counteracting the smoothing tendency
```

## My Summary

Graph Neural Networks compute document or entity representations by aggregating
information from graph neighbors through iterative message passing, producing
embeddings that incorporate both textual content and relational context. The
three foundational architectures are GCN (degree-normalized neighbor averaging,
simple but transductive and treats all neighbors equally), GraphSAGE (sampled
neighbor aggregation with learned aggregation functions, critically inductive -
can embed new nodes without retraining, making it the practical choice for
dynamic corpora), and GAT (learned attention weights over neighbors, captures
heterogeneous neighbor importance at the cost of additional computation). For
retrieval applications, the most practical architecture pattern uses the GNN
to refine document embeddings with graph context while queries remain encoded
by a standard text-only encoder - this pattern allows all GNN computation to
happen offline during indexing, introducing zero additional query-time latency
compared to standard dense retrieval. Training follows the same contrastive
InfoNCE framework as standard dense retrieval, with subgraph sampling required
to make mini-batch training tractable on production-scale graphs that cannot
fit in GPU memory. A critical architectural constraint is oversmoothing - GNN
representations become increasingly indistinguishable as layer depth increases,
because each layer averages over an exponentially growing neighborhood, making
2-3 layers the practical depth ceiling for most retrieval applications, in
contrast to text transformers where deeper is generally better. Residual
connections and jumping knowledge networks partially mitigate oversmoothing
by preserving shallower-layer representations alongside deeper relational context.
