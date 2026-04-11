# Evaluation at Scale — BEIR and Zero-shot Benchmarking

Evaluation at scale in modern IR refers to systematic benchmarking of retrieval
systems across many domains, tasks, and query types to measure generalization rather
than just in-domain performance. BEIR (Benchmarking IR) is the standard framework
for this — a heterogeneous benchmark containing 18 datasets across diverse domains
that tests whether retrieval models trained on one distribution perform well on
completely different ones without any additional fine-tuning. This zero-shot
evaluation reveals which models genuinely understand retrieval versus which have
simply memorized patterns from their training data.

## Intuition

A model that achieves MAP 0.40 on MS MARCO is impressive. But MS MARCO is web
search queries from Bing — a very specific distribution of queries and documents.
Does that model also work on biomedical literature? Legal documents? Financial
reports? Twitter data? News articles? Scientific papers?

Training and testing on the same distribution inflates perceived performance.
A model can learn surface-level patterns specific to MS MARCO without learning
anything generalizable about relevance. When deployed on a different domain in
production, it fails.

BEIR exposes this by evaluating models trained on MS MARCO across 18 completely
different datasets — no fine-tuning allowed. A model that scores well across all
18 domains has learned something genuinely useful about retrieval. A model that
scores well only on MS MARCO has overfit to a specific distribution.

## The Generalization Problem in IR

### In-domain vs out-of-domain performance

Consider the vocabulary gap between domains:

```bash
MS MARCO (web queries):
  "what is the capital of france"
  "how to fix a leaky faucet"
  "best python libraries for machine learning"

TREC-COVID (biomedical):
  "coronavirus remdesivir treatment efficacy"
  "ACE2 receptor SARS-CoV-2 binding mechanism"
  "COVID-19 cytokine storm immunopathology"
```

A bi-encoder fine-tuned on MS MARCO has never seen "ACE2", "remdesivir", or
"cytokine storm" in its training data. Its representations for these terms are
essentially random — inherited from BERT pretraining but not refined for retrieval.
BM25, on the other hand, handles these terms naturally via exact matching and IDF.

This is why BM25 outperforms many neural models on BEIR despite being conceptually
simpler — it generalizes perfectly by design.

### The training data dependency of dense retrievers

Dense retrievers require large amounts of labeled query-document pairs to train
effectively. MS MARCO provides ~500K training triples for English web queries.
For specialized domains:

```bash
Biomedical retrieval:    TREC-COVID, NFCorpus (~3K examples)
Legal retrieval:         FIQA, small datasets
Financial QA:            Very limited labeled data
Code search:             CodeSearchNet, different format
```

The scarcity of labeled data in specialized domains means models trained on MS MARCO
must transfer without fine-tuning — which is exactly what BEIR tests.

## BEIR — Benchmarking Information Retrieval

### Overview

Introduced by Thakur et al. (2021). A collection of 18 datasets spanning 9 IR tasks
across diverse domains. All datasets use the same evaluation protocol — NDCG@10 as
the primary metric — making results directly comparable across systems.

### The 18 datasets

```bash
Dataset          Domain              Task              Size (corpus)
────────────────────────────────────────────────────────────────────
MSMARCO          Web                 Passage retrieval  8.8M passages
TREC-COVID       Biomedical          Ad-hoc retrieval   171K papers
NFCorpus         Medical             Ad-hoc retrieval   3.6K documents
NQ               Wikipedia           Open-domain QA     2.7M passages
HotpotQA         Wikipedia           Multi-hop QA       5.2M passages
FiQA-2018        Financial           QA                 57K passages
Arguana          Counterargument     Retrieval          8.7K documents
Touché-2020      Controversial topics Argument retrieval 382K documents
DBPedia          Entity              Entity retrieval   4.6M entities
SCIDOCS          Scientific          Citation retrieval 25K papers
FEVER            Fact verification   Retrieval          5.4M passages
Climate-FEVER    Climate             Fact verification  5.4M passages
SciFact          Scientific claims   Fact verification  5K papers
Quora            Duplicate questions Duplicate det.     523K questions
CQADupStack      Community QA        Duplicate det.     457K questions
TREC-NEWS        News                Background linking 595K articles
Robust04         News                Ad-hoc retrieval   528K articles
BioASQ           Biomedical          QA                 14.9M passages
```

### The zero-shot protocol

1. Train your retrieval model on MS MARCO (or use a pretrained model)
2. Evaluate directly on each BEIR dataset — no fine-tuning on the target domain
3. Report NDCG@10 for each dataset
4. Average across all datasets for a single aggregate score

No hyperparameter tuning per dataset. No data from the target domain. Pure
generalization.

## Key Findings from BEIR

### Finding 1 — BM25 is a strong zero-shot baseline

Despite being a hand-crafted function with no learned parameters, BM25 outperforms
most neural retrievers on BEIR. Its exact matching strength generalizes perfectly
because it makes no domain-specific assumptions.

```bash
BM25 average NDCG@10 on BEIR:          0.428
Dense (msmarco-distilbert):             0.396
Dense (msmarco-bert-base):              0.411
```

Neural models that dramatically outperform BM25 on MS MARCO actually underperform
BM25 on BEIR. This is the generalization gap.

### Finding 2 — Larger models generalize better

Larger pretrained models (more parameters, more pretraining data) generalize better
out-of-domain, even without task-specific fine-tuning.

```bash
all-MiniLM-L6-v2 (22M params):   BEIR avg ≈ 0.41
all-mpnet-base-v2 (109M params):  BEIR avg ≈ 0.43
E5-large (335M params):           BEIR avg ≈ 0.48
```

### Finding 3 — SPLADE generalizes better than dense

SPLADE's learned sparse representations generalize better than dense bi-encoders
because they maintain term-level matching similar to BM25 while adding semantic
expansion.

```bash
SPLADE++ on BEIR average:         0.453
Dense (bi-encoder) on BEIR:       0.411
Hybrid (BM25 + dense) on BEIR:    0.461
SPLADE + Dense on BEIR:           0.482
```

### Finding 4 — Domain matters enormously

Performance varies wildly by domain. A model ranking 1st on biomedical datasets
may rank 5th on financial datasets. Aggregate scores hide domain-specific strengths
and weaknesses.

```bash
Model performance variance across BEIR datasets:
  Best dataset NDCG@10:   ~0.80+ (Quora duplicate detection)
  Worst dataset NDCG@10:  ~0.10  (some argument retrieval tasks)
```

## Beyond BEIR — Other Large-Scale Evaluation Frameworks

### MTEB (Massive Text Embedding Benchmark)

56 datasets across 8 task types in 112 languages. The most comprehensive embedding
benchmark:

```bash
Tasks covered:
  Retrieval              → 15 datasets (includes BEIR subset)
  Clustering             → 11 datasets
  Classification         → 12 datasets
  Semantic similarity    → 10 datasets
  Reranking              → 4 datasets
  Pair classification    → 3 datasets
  Summarization          → 1 dataset
  Bitext mining          → 4 datasets
```

MTEB leaderboard is the standard reference for choosing embedding models.
A model with high MTEB retrieval score and high MTEB clustering score has learned
genuinely general representations.

### TREC Deep Learning Track (2019-present)

Reusable test collections with deep (pooled) relevance judgments for MS MARCO
passages and documents. Better judgment depth than standard MS MARCO eval — more
reliable for comparing state-of-the-art systems.

### LoTTE (Long-Tail Topic-stratified Evaluation)

Tests retrieval on niche, long-tail topics not well covered by MS MARCO. Queries
from StackExchange forums across 5 domains: writing, recreation, science, technology,
lifestyle.

### BRIGHT (Reasoning-Intensive Generalization of Information Retrieval Tasks)

Evaluates retrieval on queries requiring multi-step reasoning — the hardest
generalization challenge for current IR systems.

## Evaluation Protocol at Scale

### Standard evaluation pipeline

```python
# The standard way to evaluate any retrieval system on BEIR

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Load dataset
dataset = "trec-covid"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# corpus:  {doc_id: {"title": str, "text": str}}
# queries: {query_id: str}
# qrels:   {query_id: {doc_id: relevance_score}}
```

### Metrics computed by BEIR

BEIR reports a standard set of metrics for each dataset:

```bash
NDCG@10       → primary metric, used for leaderboard ranking
Recall@100    → how many relevant docs in top-100 (retrieval coverage)
Precision@1   → top-1 precision
MAP@100       → mean average precision at cutoff 100
MRR@10        → mean reciprocal rank at cutoff 10
```

### Running a complete BEIR evaluation

```python
# pip install beir sentence-transformers

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
import json
import os


def evaluate_on_beir(model_name: str,
                     datasets: list[str],
                     batch_size: int = 128) -> dict[str, dict]:
    """
    Evaluate a sentence-transformers bi-encoder on multiple BEIR datasets.

    Args:
        model_name: huggingface model name or path
        datasets:   list of BEIR dataset names
        batch_size: encoding batch size

    Returns:
        {dataset_name: {metric: score}}
    """
    results = {}

    for dataset in datasets:
        print(f"\nEvaluating on {dataset}...")

        # Download and load dataset
        url = (
            f"https://public.ukp.informatik.tu-darmstadt.de/"
            f"thakur/BEIR/datasets/{dataset}.zip"
        )
        data_path = util.download_and_unzip(url, f"datasets/{dataset}")
        corpus, queries, qrels = GenericDataLoader(
            data_folder=data_path
        ).load(split="test")

        # Initialize retrieval model
        beir_model = DRES(
            models.SentenceBERT(model_name),
            batch_size=batch_size
        )
        retriever = EvaluateRetrieval(beir_model, score_function="dot")

        # Retrieve
        retrieved = retriever.retrieve(corpus, queries)

        # Evaluate
        ndcg, map_score, recall, precision = retriever.evaluate(
            qrels, retrieved, retriever.k_values
        )

        results[dataset] = {
            "NDCG@10":    ndcg["NDCG@10"],
            "MAP@100":    map_score["MAP@100"],
            "Recall@100": recall["Recall@100"],
            "MRR@10":     retriever.evaluate_custom(
                qrels, retrieved, [10], metric="mrr"
            )["MRR@10"]
        }

        print(f"  NDCG@10: {results[dataset]['NDCG@10']:.4f}")
        print(f"  Recall@100: {results[dataset]['Recall@100']:.4f}")

    return results


def compute_beir_average(results: dict[str, dict],
                          metric: str = "NDCG@10") -> float:
    """Compute average metric across all evaluated datasets."""
    scores = [v[metric] for v in results.values() if metric in v]
    return sum(scores) / len(scores) if scores else 0.0


def beir_summary_table(results: dict[str, dict]) -> str:
    """Format results as a markdown table."""
    header = "| Dataset | NDCG@10 | MAP@100 | Recall@100 |"
    sep    = "|---------|---------|---------|------------|"
    rows   = []
    for dataset, metrics in results.items():
        row = (
            f"| {dataset:<20} "
            f"| {metrics.get('NDCG@10', 0):.4f}  "
            f"| {metrics.get('MAP@100', 0):.4f}  "
            f"| {metrics.get('Recall@100', 0):.4f}     |"
        )
        rows.append(row)

    avg_ndcg = compute_beir_average(results, "NDCG@10")
    avg_row  = (
        f"| **Average**          "
        f"| **{avg_ndcg:.4f}** "
        f"|         "
        f"|            |"
    )

    return "\n".join([header, sep] + rows + [avg_row])


# Example — evaluate on a small subset of BEIR
small_beir = [
    "scifact",     # scientific fact verification, small corpus
    "fiqa",        # financial QA
    "arguana",     # argument retrieval
]

# Uncomment to run actual evaluation:
# results = evaluate_on_beir(
#     model_name="msmarco-distilbert-base-v4",
#     datasets=small_beir
# )
# print(beir_summary_table(results))

# Mock results for illustration
mock_results = {
    "scifact": {"NDCG@10": 0.671, "MAP@100": 0.634, "Recall@100": 0.912},
    "fiqa":    {"NDCG@10": 0.296, "MAP@100": 0.243, "Recall@100": 0.534},
    "arguana": {"NDCG@10": 0.415, "MAP@100": 0.389, "Recall@100": 0.968},
}

print("BEIR Evaluation Results:")
print(beir_summary_table(mock_results))
print(f"\nAverage NDCG@10: {compute_beir_average(mock_results):.4f}")


# BM25 baseline comparison
from rank_bm25 import BM25Okapi
import numpy as np

def evaluate_bm25_on_corpus(corpus: dict,
                              queries: dict,
                              qrels: dict,
                              k: int = 10) -> float:
    """
    Simple NDCG@K evaluation of BM25 on a corpus.
    Demonstrates how to run evaluation from scratch.
    """
    import math

    doc_ids   = list(corpus.keys())
    doc_texts = [
        f"{v.get('title', '')} {v.get('text', '')}".strip()
        for v in corpus.values()
    ]

    tokenized = [text.lower().split() for text in doc_texts]
    bm25 = BM25Okapi(tokenized)

    ndcg_scores = []

    for query_id, query_text in queries.items():
        if query_id not in qrels:
            continue

        # Retrieve
        scores      = bm25.get_scores(query_text.lower().split())
        top_indices = np.argsort(scores)[::-1][:k]
        top_doc_ids = [doc_ids[i] for i in top_indices]

        # NDCG@K
        relevances = [
            qrels[query_id].get(doc_id, 0)
            for doc_id in top_doc_ids
        ]

        dcg  = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(relevances)
        )
        ideal_rels = sorted(qrels[query_id].values(), reverse=True)[:k]
        idcg = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(ideal_rels)
        )

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
```

## What Good Scores Look Like at Scale

Ballpark NDCG@10 on BEIR across model families (as of 2025):

```bash
Model family                        BEIR avg NDCG@10
────────────────────────────────────────────────────
BM25                                0.428
Bi-encoder (small, e.g. MiniLM)     0.41 – 0.43
Bi-encoder (large, e.g. E5-large)   0.47 – 0.50
SPLADE variants                     0.45 – 0.49
Hybrid (BM25 + dense)               0.46 – 0.50
Hybrid (SPLADE + dense)             0.48 – 0.52
+ Cross-encoder reranking           0.50 – 0.56
LLM-based reranking (large models)  0.54 – 0.60+
```

Progress on BEIR has been slower than on MS MARCO because generalization is
genuinely harder. A 2-point improvement on BEIR is significant; a 2-point
improvement on MS MARCO alone may just be overfitting.

## Practical Lessons for System Building

### Lesson 1 — Always evaluate out-of-domain

If you are building an IR system for a specific domain, do not just measure
in-domain performance. Evaluate on related out-of-domain datasets to understand
where the system will fail when deployed on slightly different queries.

### Lesson 2 — BM25 is your first baseline

Before building any neural retrieval system, implement BM25 and measure its
performance on your task. If your neural system cannot beat BM25 after fine-tuning,
something is wrong.

### Lesson 3 — Recall@K matters as much as NDCG@K

NDCG@10 measures ranking quality. Recall@100 measures retrieval coverage — are
the relevant documents even in the top 100? A reranker cannot recover documents
not in the candidate set. Monitor both.

### Lesson 4 — Domain-specific fine-tuning dramatically helps

Even a small number of labeled examples from the target domain (100-1000) can
close most of the generalization gap. If labeled data is available for your domain,
fine-tune rather than relying on zero-shot transfer.

### Lesson 5 — Hybrid search is the safe default

Across all BEIR datasets, hybrid search (BM25 + dense) consistently outperforms
either alone. When in doubt, use hybrid. The improvement from adding a sparse
component is reliable across domains.

## Where This Fits in the Progression

```bash
Dense Retrieval         → first-stage neural retrieval
Bi-encoders             → efficient first-stage retrieval
Cross-encoders          → accurate reranking
SPLADE                  → learned sparse retrieval
Reranking               → two-stage pipeline
Hybrid Search           → combining multiple signals
ColBERT                 → late interaction retrieval
RAG                     → retrieval powering generation
Evaluation at Scale     → measuring generalization  ← you are here
```

Evaluation at scale is the lens through which everything else in this module is
validated. Without BEIR-style evaluation, you cannot know whether a model that
performs well in development will hold up in production across different query
types and domains.

## My Summary

Evaluation at scale tests whether retrieval systems generalize beyond their training
distribution. BEIR provides 18 heterogeneous datasets across diverse domains for
zero-shot evaluation — no fine-tuning on the target domain allowed. Its key finding
is that BM25 outperforms many neural retrievers out-of-domain despite being
conceptually simpler, because it makes no domain-specific assumptions. SPLADE and
hybrid search generalize better than pure dense retrieval. MTEB extends this
evaluation to 56 datasets across 8 task types. The practical lessons are clear:
always run out-of-domain evaluation, use BM25 as your first baseline, monitor
recall not just NDCG, and use hybrid search as the safe default when domain-specific
labeled data is unavailable.
