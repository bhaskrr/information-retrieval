# Vector Databases

A vector database is a purpose-built data store designed to index, store, and
query high-dimensional vectors efficiently. Unlike general-purpose databases that
store structured rows or document databases that store JSON, vector databases are
optimized for the specific operations that dense retrieval requires: inserting
vectors, building approximate nearest neighbour (ANN) indexes, performing similarity
search at query time, and filtering results by metadata alongside vector similarity.
Vector databases are the infrastructure layer that makes dense retrieval practical
at scale without the operational complexity of running and tuning FAISS directly.

## Intuition

FAISS is an excellent ANN library but it is a library - not a service. It stores
vectors in memory on a single machine, has no network API, no persistence beyond
manual save/load, no filtering by metadata, no access control, no monitoring, and
no horizontal scaling. Building a production dense retrieval system on raw FAISS
requires engineering all of these around it.

Vector databases provide these capabilities out of the box. They are to FAISS what
Elasticsearch is to Lucene - they wrap the core algorithm in production-ready
infrastructure that handles the concerns beyond the algorithm itself:

```bash
FAISS (library):
  + Fast ANN search
  - In-memory only (no persistence)
  - Single machine
  - No metadata filtering
  - No REST API
  - No access control
  - No monitoring

Vector database (managed service):
  + Fast ANN search (same HNSW/IVF algorithms)
  + Persistent storage
  + Horizontal scaling
  + Metadata filtering alongside vectors
  + REST/gRPC API
  + Access control
  + Built-in monitoring
  + Managed updates and deletes
```

The tradeoff is control vs convenience. FAISS gives you full control over every
parameter. A vector database abstracts these away in exchange for operational
simplicity.

## Key Vector Database Concepts

### Collections and namespaces

The equivalent of a database table. Each collection stores vectors of the same
dimension with associated metadata:

```bash
Collection: "research_papers"
  Vectors: 384-dimensional embeddings (one per document chunk)
  Metadata: {title, authors, year, domain, chunk_id}
```

### Points and payloads

Individual records in a vector database:

```bash
Point:
  id:      "D001_chunk_3"
  vector:  [0.23, -0.41, 0.87, ...]   (384 dimensions)
  payload: {
    title:    "Attention Is All You Need",
    authors:  ["Vaswani et al."],
    year:     2017,
    domain:   "deep learning",
    chunk_id: 3,
    text:     "The architecture follows an encoder-decoder structure..."
  }
```

### Similarity metrics

Vector databases support multiple similarity functions:

```bash
Cosine:      angle between vectors (most common for text embeddings)
Dot product: inner product (use when vectors are L2-normalized - same as cosine)
Euclidean:   L2 distance (less common for text, used in image search)
```

### Filters

Metadata filters applied alongside vector similarity search:

```bash
Query: "transformer architecture" (vector)
Filter: year >= 2020 AND domain == "information retrieval"
→ returns semantically similar results from the filtered subset
```

Two filtering strategies:

- **Pre-filter**: apply metadata filter first, then ANN search within filtered subset
- **Post-filter**: ANN search first, then apply metadata filter to results

Pre-filter is more accurate but slower when the filter is very selective. Post-filter
is faster but may return fewer results than requested if many candidates fail the filter.

### Namespaces and tenants

Partition a single collection for multi-tenant applications:

```bash
namespace: "user_alice"   → Alice's private documents
namespace: "user_bob"     → Bob's private documents
namespace: "shared"       → Shared public documents
```

Query within a namespace - users only see their own data.

## Major Vector Databases

### Pinecone

Fully managed, serverless vector database. No infrastructure to manage.

```bash
Key properties:
  Fully managed SaaS      → no operational overhead
  Serverless option        → pay per query, no idle cost
  Namespaces               → multi-tenancy built in
  Sparse + dense           → hybrid search support
  Metadata filtering       → rich filter expressions
  Scale                    → billions of vectors
  Latency                  → ~10-50ms p99

Best for:
  Teams that want zero infrastructure management
  Applications with variable or unpredictable query volume
  Quick prototyping → production without DevOps
```

### Weaviate

Open-source vector database with built-in ML capabilities. Can be self-hosted
or used as a managed cloud service.

```bash
Key properties:
  Open source              → full control, no vendor lock-in
  GraphQL + REST API       → flexible querying
  BM25 built-in           → hybrid search without external BM25 system
  Modules system           → integrate embedding models directly
  Multi-modal              → text, images, audio vectors
  Schema-based             → typed collections with validation

Best for:
  Teams that want hybrid BM25 + dense in a single system
  Applications requiring complex semantic search with structured data
  Organizations that need self-hosted for data privacy
```

### Qdrant

Open-source, written in Rust for high performance. Strong filtering capabilities.

```bash
Key properties:
  Written in Rust          → very high performance, low memory
  Payload indexing         → fast metadata filtering (pre-filter)
  Named vectors            → multiple vector spaces per point
  Sparse vectors           → SPLADE/BM25 vectors alongside dense
  Quantization             → int8, binary quantization built-in
  Snapshots                → easy backup/restore

Best for:
  High-performance requirements on limited hardware
  Applications needing strong metadata filtering alongside vectors
  SPLADE or hybrid sparse+dense in a single system
```

### Chroma

Open-source, Python-native, designed for simplicity. Popular for RAG prototyping.

```bash
Key properties:
  Python-first API         → extremely simple setup
  Persistent + in-memory   → flexible deployment
  LangChain/LlamaIndex     → deep integration with RAG frameworks
  Embeddings included      → can compute embeddings internally
  Lightweight              → runs on laptop without Docker

Best for:
  Prototyping RAG pipelines quickly
  Learning and experimentation
  Small to medium corpora (< 1M vectors)
  Not for large-scale production (limited scaling)
```

### Milvus

Open-source, cloud-native, designed for massive scale.

```bash
Key properties:
  GPU acceleration          → fast indexing and search on GPU
  Multiple index types      → HNSW, IVF, DiskANN, ANNOY
  Streaming + batch         → handle real-time vector inserts
  Kubernetes native         → designed for cloud deployment
  Attu GUI                  → visual management interface
  Scale                     → tens of billions of vectors

Best for:
  Very large scale (> 100M vectors)
  Teams with Kubernetes infrastructure
  GPU-accelerated search
  High-throughput applications
```

### pgvector

PostgreSQL extension that adds vector storage and ANN search to Postgres.

```bash
Key properties:
  Inside PostgreSQL         → single database for all data
  SQL interface             → familiar query language
  ACID transactions         → strong consistency guarantees
  Joins with regular data   → combine vector search with relational queries
  Limited scale             → tens of millions of vectors max

Best for:
  Applications already using PostgreSQL
  Need for transactional consistency alongside vector search
  Simple retrieval without extreme scale requirements
  Teams that do not want to operate a separate vector store
```

## Comparison Table

```
Database    Self-hosted   Scale         Hybrid    Filtering   Best for
────────────────────────────────────────────────────────────────────────
Pinecone    No (SaaS)     Billions      Yes       Yes         Managed, scalable
Weaviate    Yes/Cloud     100M+         Yes (BM25)Yes         All-in-one search
Qdrant      Yes/Cloud     100M+         Yes(sparse)Yes(fast)  Performance
Chroma      Yes           < 1M          No        Basic       Prototyping
Milvus      Yes (k8s)     Billions      Limited   Yes         GPU, massive scale
pgvector    Yes (Postgres)Tens of M     No        Yes (SQL)   Postgres users
```

## Choosing a Vector Database

```
Situation                                   Recommendation
────────────────────────────────────────────────────────────────────────
Prototyping a RAG pipeline                  Chroma - simplest setup
Production RAG, no DevOps team              Pinecone - fully managed
Hybrid BM25 + dense, single system          Weaviate
High performance, self-hosted               Qdrant
Already using PostgreSQL                    pgvector (if < 10M vectors)
> 100M vectors, GPU available               Milvus
Enterprise, data privacy required           Weaviate or Qdrant self-hosted
Multi-tenant SaaS application               Pinecone (namespaces) or Qdrant
```

## Vector Database vs FAISS vs Elasticsearch

```
Property                FAISS            Vector DB        Elasticsearch
────────────────────────────────────────────────────────────────────────
Deployment              Library          Service          Service
Persistence             Manual           Built-in         Built-in
Horizontal scaling      Manual           Built-in         Built-in
Metadata filtering      No               Yes              Yes (full SQL-like)
BM25 hybrid             No               Some (Weaviate)  Yes (native)
REST API                No               Yes              Yes
Access control          No               Yes              Yes (paid features)
Monitoring              No               Yes              Yes
Update/delete vectors   Manual rebuild   Yes              Yes
Managed cloud option    No               Yes (all)        Yes (Elastic Cloud)
Cost                    Free             Free + Cloud     Free + Elastic Cloud
Best for                Research         Production dense Production search
                        prototypes       retrieval        with BM25 + kNN
```

## Production Considerations

### Embedding model versioning

When you update the embedding model, all existing vectors must be recomputed.
Plan for this:

```bash
1. Create new collection with new model name in metadata
2. Recompute and index all vectors using new model
3. Run AB test: old collection vs new collection on sample queries
4. Switch application traffic to new collection
5. Delete old collection after validation period
```

### Indexing pipeline

For large corpora, use async batch indexing:

```bash
Documents → chunking → embedding (batch, GPU) → vector DB bulk insert
                                              → metadata validation
                                              → index refresh confirmation
```

### Backup and disaster recovery

Most managed vector databases handle backups automatically. For self-hosted:

```bash
Qdrant:   snapshots API → point-in-time backup of collection
Weaviate: backup module → S3/GCS/Azure integration
Chroma:   copy persistent directory
Milvus:   MinIO backup integration
```

### Cold start problem

New collections have empty indexes - no vectors, no meaningful ANN structure.
Bulk load all existing documents before opening to traffic.

## My Summary

Vector databases are purpose-built services for storing and querying dense vectors
at scale. They provide what FAISS lacks: persistence, horizontal scaling, metadata
filtering, REST APIs, access control, and managed infrastructure. The major options
serve different needs: Chroma for prototyping (simplest setup, no server required),
Qdrant for performance-critical self-hosted deployments (Rust, fast filtering),
Pinecone for zero-infrastructure managed deployments, Weaviate for all-in-one search
with BM25 + dense hybrid, Milvus for massive scale with GPU acceleration, and
pgvector for teams already using PostgreSQL. Choosing between a vector database and
Elasticsearch depends on whether BM25 is a first-class requirement - if it is,
Elasticsearch with kNN is often simpler than running a separate vector database.
If pure dense retrieval is the primary need and operational simplicity matters,
a managed vector database like Pinecone or cloud-hosted Qdrant eliminates
significant infrastructure overhead.
