# Elasticsearch for IR

Elasticsearch is a distributed search engine built on Apache Lucene that provides
BM25-based full-text search, vector search (kNN), and hybrid search capabilities
through a REST API. It is the most widely deployed search engine in production,
used by companies ranging from startups to large enterprises for application search,
log analysis, e-commerce, and increasingly for RAG pipelines. For IR practitioners,
Elasticsearch is the bridge between the algorithms covered in this repo and a
production-grade system that handles indexing, query processing, scaling, and
infrastructure concerns out of the box.

## Intuition

Everything covered in this repo so far has been at the algorithmic level - how
BM25 scoring works, how dense retrieval encodes documents, how reranking improves
ranking quality. Elasticsearch is where these algorithms meet production engineering
concerns: how do you index ten million documents without downtime? How do you serve
a thousand queries per second with sub-100ms latency? How do you update documents
without rebuilding the entire index? How do you monitor retrieval quality over time?

Elasticsearch handles the distributed systems complexity so you can focus on
retrieval quality. Its core abstractions map directly to IR concepts:

| IR concept      | Elasticsearch abstraction         |
| --------------- | --------------------------------- |
| Document corpus | Index                             |
| Document        | Document (JSON)                   |
| Inverted index  | Lucene index segments             |
| BM25 scoring    | Default similarity function       |
| Dense vectors   | kNN field (dense_vector)          |
| Hybrid search   | bool query + knn query combined   |
| Reranking       | Script score or custom reranker   |
| Query pipeline  | Ingest pipeline + search pipeline |

## Core Architecture

### Nodes and clusters

Elasticsearch runs as a distributed cluster of nodes:

```
Cluster
├── Master node          - manages cluster state, index routing
├── Data nodes (N)       - store shards, execute queries
└── Coordinating nodes   - route requests, aggregate results
```

For IR applications at startup scale, a single node handles millions of documents.
For larger corpora or high query volume, scale horizontally by adding data nodes.

### Indexes and shards

An Elasticsearch index is divided into shards - each shard is a complete Lucene
index:

```
Index: "documents" (5 shards, 1 replica each)
  Shard 0 primary + replica
  Shard 1 primary + replica
  Shard 2 primary + replica
  Shard 3 primary + replica
  Shard 4 primary + replica
```

Documents are routed to shards by document ID hash. Queries execute in parallel
across all shards, results are merged and returned. For most IR applications,
1-3 shards with 1 replica is sufficient. Over-sharding (many small shards) is
a common performance mistake.

### Mappings

The mapping defines how document fields are indexed:

```json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "english"
      },
      "content": {
        "type": "text",
        "analyzer": "english"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 384,
        "index": true,
        "similarity": "cosine"
      },
      "doc_id": {
        "type": "keyword"
      },
      "timestamp": {
        "type": "date"
      }
    }
  }
}
```

Field types that matter for IR:

- `text` - analyzed, tokenized, inverted indexed → BM25 scoring
- `keyword` - exact match, not analyzed → filtering, faceting
- `dense_vector` - vector field for kNN search
- `date`, `integer`, `float` - structured data for filtering

### Analyzers

Analyzers define the text processing pipeline applied at index and query time:

```
Input text → Character filters → Tokenizer → Token filters → Tokens → Index
```

Built-in analyzers:

```
standard:   lowercase + basic punctuation removal + english stopwords
english:    standard + stemming (Porter) + english stopwords (best for IR)
simple:     split on non-letters, lowercase
whitespace: split on whitespace only, no normalization
```

Custom analyzer for IR:

```json
{
  "analysis": {
    "analyzer": {
      "ir_analyzer": {
        "type": "custom",
        "tokenizer": "standard",
        "filter": [
          "lowercase",
          "english_stop",
          "english_stemmer",
          "asciifolding"
        ]
      }
    },
    "filter": {
      "english_stop": {
        "type": "stop",
        "stopwords": "_english_"
      },
      "english_stemmer": {
        "type": "stemmer",
        "language": "english"
      }
    }
  }
}
```

The `asciifolding` filter normalizes accented characters - "résumé" → "resume".
Critical to apply the same analyzer at index and query time (Elasticsearch does
this automatically for `text` fields).

## BM25 Retrieval in Elasticsearch

Elasticsearch uses BM25 as its default similarity function since version 5.0.
The default parameters (k1=1.2, b=0.75) are reasonable starting points:

```json
{
  "settings": {
    "similarity": {
      "custom_bm25": {
        "type": "BM25",
        "k1": 1.5,
        "b": 0.75
      }
    }
  }
}
```

### Basic search query

```json
{
  "query": {
    "match": {
      "content": {
        "query": "information retrieval evaluation metrics",
        "analyzer": "english"
      }
    }
  },
  "size": 10
}
```

### Multi-field search with field boosting

```json
{
  "query": {
    "multi_match": {
      "query": "transformer architecture attention",
      "fields": ["title^3", "abstract^2", "content"],
      "type": "best_fields"
    }
  }
}
```

The `^3` and `^2` boost factors increase the weight of title and abstract matches
relative to content matches. Title matches are typically more relevant than body
matches for most queries.

### Boolean query for filtering + ranking

```json
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "dense retrieval"
          }
        }
      ],
      "filter": [
        {
          "range": {
            "year": { "gte": 2020 }
          }
        },
        {
          "term": {
            "domain": "information retrieval"
          }
        }
      ]
    }
  }
}
```

The `must` clause contributes to BM25 scoring. The `filter` clause is a hard
constraint - documents not matching are excluded but filter clauses do not
affect score. Use `filter` for structural constraints (date ranges, categories)
and `must` for content matching.

## Vector Search (kNN) in Elasticsearch

Elasticsearch 8.x added native kNN search using HNSW indexes:

```json
{
  "knn": {
    "field": "embedding",
    "query_vector": [0.23, -0.41, 0.87, ...],
    "k": 10,
    "num_candidates": 100
  }
}
```

`num_candidates` controls the HNSW search width - higher values improve recall
at the cost of latency. Rule of thumb: `num_candidates` = 5-10 × `k`.

### Indexing documents with embeddings

Documents must include precomputed embeddings:

```json
{
  "doc_id": "D001",
  "title": "Attention Is All You Need",
  "content": "We propose a new simple network architecture...",
  "embedding": [0.23, -0.41, 0.87, 0.12, ...]
}
```

Embeddings are computed offline (using sentence-transformers) and included
in the document payload at index time.

## Hybrid Search in Elasticsearch

Combine BM25 and kNN in a single query using the `hybrid` retrieval approach
or the `bool` + `knn` combination:

### Approach 1 - Bool + kNN (pre-8.8)

```json
{
  "query": {
    "bool": {
      "should": [
        {
          "match": {
            "content": "transformer attention mechanism"
          }
        }
      ]
    }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [0.23, -0.41, 0.87, ...],
    "k": 10,
    "num_candidates": 100,
    "boost": 0.5
  },
  "size": 10
}
```

Elasticsearch combines the BM25 score from `bool.should` with the kNN score
using linear combination. The `boost` on kNN controls relative weighting.

### Approach 2 - Reciprocal Rank Fusion (Elasticsearch 8.8+)

Elasticsearch 8.8 introduced native RRF support:

```json
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "match": {
                "content": "transformer attention"
              }
            }
          }
        },
        {
          "knn": {
            "field": "embedding",
            "query_vector": [0.23, -0.41, 0.87, ...],
            "num_candidates": 100
          }
        }
      ],
      "rank_constant": 60,
      "rank_window_size": 100
    }
  },
  "size": 10
}
```

RRF in Elasticsearch natively combines rankings from multiple retrieval methods -
identical to the RRF implementation covered in 07-advanced/02-hybrid-search.md.

## Index Lifecycle Management

### Bulk indexing

For large corpora, use the bulk API rather than indexing one document at a time:

```python
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch("http://localhost:9200")

def generate_docs(documents: dict[str, dict]):
    for doc_id, doc in documents.items():
        yield {
            "_index": "ir_documents",
            "_id":    doc_id,
            "_source": doc
        }

success, errors = bulk(
    es,
    generate_docs(documents),
    chunk_size=500,
    max_retries=3
)
```

Bulk indexing is 10-50x faster than individual document indexing. Use
`chunk_size=500` as a starting point - larger chunks reduce overhead but
increase memory usage.

### Index aliases for zero-downtime reindexing

When you need to rebuild an index (new mapping, new embeddings), use aliases
to avoid downtime:

```python
# Create new index
es.indices.create(index="documents_v2", body=new_mapping)

# Index all documents into documents_v2
# ...

# Atomically swap alias from v1 to v2
es.indices.update_aliases(body={
    "actions": [
        {"remove": {"index": "documents_v1", "alias": "documents"}},
        {"add":    {"index": "documents_v2", "alias": "documents"}}
    ]
})
```

All queries against the alias "documents" automatically route to the new index
without any application code changes.

### Update strategies

For dynamic corpora with frequent updates:

**Full reindex** - rebuild the entire index periodically (nightly, weekly).
Simplest. Appropriate when updates are batched and staleness is acceptable.

**Partial update** - update individual documents using the Update API.
Appropriate for real-time changes to specific documents.

**Delta index** - maintain a small separate index for recent changes, merge
periodically. Appropriate for high-velocity update streams.

## Performance Tuning

### Indexing performance

| Setting                   | Default | Recommended for bulk indexing             |
| ------------------------- | ------- | ----------------------------------------- |
| refresh_interval          | 1s      | -1 (disable during bulk, re-enable after) |
| number_of_replicas        | 1       | 0 (disable during bulk, re-enable after)  |
| index.translog.durability | request | async (risk: lose data on crash)          |
| bulk chunk_size           | -       | 500 documents                             |

### Query performance

| Setting             | Guidance                                  |
| ------------------- | ----------------------------------------- |
| knn.num_candidates  | 5-10 × k (higher = better recall, slower) |
| result_window (max) | 10,000 documents default                  |
| search.max_buckets  | Increase for aggregation-heavy queries    |
| query_cache         | Enable for repeated filter queries        |

### Memory sizing

| Component          | Rule of thumb                                                              |
| ------------------ | -------------------------------------------------------------------------- |
| JVM heap           | 50% of available RAM, max 32GB                                             |
| Filesystem cache   | Remaining 50% - critical for performance                                   |
| Dense vector index | ~4 bytes × dims × doc_count for HNSW, Example: 384 dims × 1M docs = ~1.5GB |

## Common Pitfalls

| Pitfall                                  | Symptom                                     | Fix                                                     |
| ---------------------------------------- | ------------------------------------------- | ------------------------------------------------------- |
| Over-sharding                            | Slow queries on small index, high cpu       | 1-3 shards for < 50GB index                             |
| Wrong analyzer at query time             | Stemmed index, unstemmed query -> misses    | Use same analyzer in mapping + query                    |
| Mapping explosion                        | Dynamic mapping creates thousands of fields | Use explicit mapping, disable dynamic on unknown fields |
| No refresh before searching (bulk index) | Documents not visible after indexing        | Add refresh=True after bulk or wait 1s                  |
| kNN with heavy filters                   | Filter applied after kNN -> poor results    | Use pre-filter for restrictive filters                  |
| Dense vectors not normalized             | kNN closine similarity incorrect            | Normalize embeddings before indexing                    |

## Where This Fits in the Progression

```
Choosing a retrieval stack  → decide which components to use
Elasticsearch for IR        → implement the chosen stack  ← you are here
Vector databases            → alternative managed infrastructure
Monitoring quality          → measure and improve post-launch
```

## My Summary

Elasticsearch is the most widely deployed production search engine - it wraps
Lucene's BM25 and HNSW vector search in a distributed, REST-accessible system
that handles indexing, sharding, query execution, and scaling. For IR practitioners,
its key abstractions map directly to IR concepts: text fields with analyzers for
BM25, dense_vector fields with HNSW for kNN, and bool + knn queries for hybrid
search. The RRF retriever added in version 8.8 provides native hybrid search with
the same reciprocal rank fusion algorithm covered in the hybrid search notes.
Production-grade Elasticsearch IR requires attention to: index mapping design
(field types, analyzers, similarity settings), bulk indexing performance (disable
refresh and replicas during initial load), query performance (num_candidates tuning
for kNN, field boosting for BM25), and memory sizing (heap at 50% RAM, rest for
filesystem cache). The most common mistakes are over-sharding small indexes,
analyzer mismatch between indexing and querying, and not normalizing dense vectors
before indexing.
