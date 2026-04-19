# Learning to Rank

Learning to Rank (LTR) is a family of supervised machine learning approaches that train a model to produce optimal document rankings given a query. Instead of using a hand-crafted scoring function like BM25 or a single relevance score like a cross-encoder, LTR systems learn a ranking function from labeled training data
(query, document, relevance) triples that combines many features to produce a final ranked list. LTR is the bridge between classical retrieval models and neural ranking, and forms the backbone of ranking in production search engines at Google, Bing, and Baidu.

## Intuition

BM25 uses two signals term frequency and inverse document frequency combined by a fixed formula. It does not learn from data. A cross-encoder uses a neural network but produces only a single relevance score per (query, document) pair.

LTR asks: what if we could use hundreds of signals simultaneously and learn how to combine them from examples of what users considered relevant? Signals like:

```bash
BM25 score                    ← term matching
TF-IDF score                  ← term weighting
PageRank / authority          ← document importance
Query term density            ← how concentrated are query terms
Document freshness            ← how recently was it updated
Click-through rate            ← what users actually clicked
Title match score             ← query terms in title
URL match score               ← query terms in URL
Dense retrieval score         ← semantic similarity
Cross-encoder score           ← fine-grained relevance
```

LTR takes all of these as input features and learns a function that combines them to produce rankings that match human relevance judgments as closely as possible.

## The Three Paradigms

LTR approaches are classified by what their loss function operates over.
This classification is fundamental it determines what the model learns to optimize and how well that objective aligns with the actual ranking task.

### Pointwise

Treats ranking as regression or classification on individual documents.
Predicts a relevance score for each (query, document) pair independently, then ranks by predicted score.

```bash
Input:  features(query, document)
Output: predicted relevance score (regression) or
        P(relevant | query, document) (classification)

Loss:   MSE between predicted and true relevance score
        or cross-entropy on binary relevant/not-relevant label
```

Training examples are individual (query, document, label) triples:

```bash
(q1, d1, 3)   ← highly relevant
(q1, d2, 1)   ← marginally relevant
(q1, d3, 0)   ← not relevant
```

Advantage: simple, maps to standard regression/classification.
Disadvantage: does not directly optimize ranking two documents with scores 0.9 and 0.8 are ranked correctly, but so are 0.9 and 0.1. The absolute score values matter to the loss but not to the ranking.

### Pairwise

Treats ranking as binary classification on document pairs. Learns to predict which of two documents should be ranked higher for a given query.

Input: features(query, $doc_i$), features(query, $doc_j$)

Output: P($doc_i$ should above $doc_j$)

Loss: binary cross-entropy on (query, $doc_i$, $doc_j$) triples
where label = 1 if $doc_i$ is more relevant than $doc_j$

Training examples are (query, relevant_doc, less_relevant_doc) triples:

```bash
(q1, d1, d2)   ← d1 is more relevant than d2
(q1, d1, d3)   ← d1 is more relevant than d3
(q1, d2, d3)   ← d2 is more relevant than d3
```

More directly aligned with ranking than pointwise the model explicitly learns relative ordering. Still not a perfect proxy for list-level metrics like NDCG because the loss treats all pairs equally regardless of their position in the ranking.

Key pairwise algorithms: RankNet, RankSVM.

### Listwise

Treats the entire ranked list as the training unit. Optimizes a list-level objective that directly corresponds to a ranking metric like NDCG or MAP.

```bash
Input:  all (query, document) feature vectors for a query
Output: permutation of documents (ranked list)

Loss:   approximation of NDCG or MAP at the list level
```

Training examples are entire query result lists:

```bash
q1 → [(d1, rel=3), (d2, rel=1), (d3, rel=0), (d4, rel=2)]
```

Most directly aligned with what we actually want to optimize. Harder to implement and train than pointwise or pairwise approaches.

Key listwise algorithms: ListNet, LambdaMART (listwise interpretation).

## Key Algorithms

### RankNet (Burges et al., 2005)

The first major pairwise neural ranking approach. A feedforward neural network trained to predict the probability that document i should rank above document j:

$s_i = network(features_i)$ → score for document i

$s_j = network(features_j)$ → score for document j

$P(i > j) = sigmoid(s_i - s_j)$

Loss = $-y_{ij} × log(P(i > j)) - (1 - y_{ij}) × log(1 - P(i > j))$

Where $y_{ij} = 1$ if document i is more relevant than j, 0 otherwise.

Training with stochastic gradient descent on all pairs. Introduced the key insight that the neural network only needs to output a single score per document pairs are formed during loss computation, not model architecture.

### LambdaRank (Burges et al., 2006)

Extends RankNet by weighting pair gradients by the change in NDCG that would result from swapping the two documents. Pairs whose swap would change NDCG the most get larger gradient weight the model focuses learning effort on the pairs that matter most for the ranking metric.

$λ_{ij} = |ΔNDCG_{ij}| × ∂C/∂(s_i - s_j)$

Where $ΔNDCG_{ij}$ is the absolute change in NDCG from swapping ranks of i and j.

LambdaRank does not directly optimize NDCG but its gradient approximation works extremely well in practice. It was state-of-the-art for several years on the LETOR benchmark.

### LambdaMART (Wu et al., 2010)

Combines LambdaRank gradients with MART (Multiple Additive Regression Trees), also known as gradient boosted trees. The most widely used LTR algorithm in production for over a decade.

```bash
LambdaMART = LambdaRank gradient + MART (gradient boosted trees)
```

Instead of training a neural network with LambdaRank gradients, LambdaMART fits gradient boosted decision trees iteratively each tree corrects the errors of the previous ensemble.

Why LambdaMART became dominant:

- Gradient boosted trees handle heterogeneous feature types naturally
- No need to normalize features
- Built-in feature importance
- Fast training and inference
- Strong performance on tabular features (BM25 scores, PageRank, click data)

LambdaMART is implemented in LightGBM, XGBoost, and the dedicated RankLib and XGBoost libraries. It is still used in production ranking at major search engines.

### ListNet (Cao et al., 2007)

A listwise approach that minimizes the KL divergence between the predicted probability distribution over permutations and the ground truth distribution:

```bash
P_pred(π) = softmax of predicted scores over all documents
P_true(π) = softmax of true relevance scores over all documents

Loss = KL(P_true || P_pred)
```

Directly optimizes the probability of the correct ranking. More principled than LambdaRank but in practice LambdaMART usually outperforms it.

### Neural LTR From LambdaMART to BERT

The transition from LambdaMART to neural LTR mirrors the broader transition in IR from classical to neural methods:

```bash
LambdaMART (2010):
  Input: hand-crafted features (BM25, PageRank, TF-IDF, ...)
  Model: gradient boosted trees
  Loss:  LambdaRank gradient

Neural LambdaRank / LambdaLoss (2018-2019):
  Input: same hand-crafted features
  Model: deep neural network instead of trees
  Loss:  LambdaRank or direct NDCG approximation

BERT-based LTR (2019-present):
  Input: raw query + document text
  Model: BERT fine-tuned with pairwise or listwise loss
  Loss:  pairwise margin or listwise NDCG approximation
```

MonoBERT and MonoT5 (covered in 06-neural-ir/05-cross-encoders.md) are essentially BERT-based LTR models using pairwise training on MS MARCO.

## Feature Engineering for LTR

Classical LTR requires careful feature engineering. Features fall into three groups:

### Query-independent features (document features)

Properties of the document that do not depend on the query:

```bash
PageRank score             ← link graph importance
Document length            ← number of tokens
URL length and depth       ← structural signal
Document freshness         ← days since last update
Click-through rate (CTR)   ← aggregate user preference
```

### Query-dependent features (query-document features)

Properties of the (query, document) pair:

```bash
BM25 score                 ← term-based relevance
TF-IDF cosine similarity   ← weighted term matching
Dense retrieval score      ← semantic similarity
Query term coverage        ← fraction of query terms in document
Min/Max/Avg IDF of query   ← query term rarity
Query term density         ← query terms / document length
Title field match          ← query terms in title specifically
```

### Query features

Properties of the query itself:

```bash
Query length               ← number of tokens
Query clarity score        ← pre-retrieval difficulty estimate
Query type (navigational/informational) ← from intent classifier
```

A typical industrial LTR system uses 200-500 features. The feature set is often more important than the choice of LTR algorithm LambdaMART with good features consistently outperforms neural LTR with poor features.

## LETOR The Standard LTR Benchmark

LETOR (Learning to Rank for Information Retrieval) is the standard benchmark dataset for LTR research, provided by Microsoft Research.

```bash
LETOR 4.0:
  MQ2007: 1,692 queries, 69,623 documents, 46 features
  MQ2008: 784 queries, 15,211 documents, 46 features
  Relevance: 3-point graded (0, 1, 2)
  Split: 5-fold cross-validation

Features include: BM25 variants, TF-IDF variants, language model scores,
                  document structure signals, all computed for body, title,
                  anchor, URL fields
```

Any new LTR algorithm is expected to report results on LETOR for comparison.

## LTR in the Modern IR Pipeline

LTR did not disappear with the rise of neural IR it evolved. Modern production ranking systems use LTR at multiple stages:

```bash
Stage 1 First-stage retrieval
  BM25 or dense retrieval → top-1000 candidates
  No LTR here speed is the constraint

Stage 2 Pre-ranking (lightweight LTR)
  LambdaMART on 50-100 fast features → top-100
  Eliminates obviously irrelevant candidates cheaply
  Latency: ~5ms

Stage 3 Reranking (heavy LTR / neural)
  Cross-encoder or BERT-based LTR on rich features → top-10
  May include dense score, cross-encoder score, BM25 as features
  Latency: ~100-300ms

Stage 4 Post-ranking (business rules)
  Diversity, freshness, personalization adjustments
  Not ML but affects final order
```

LambdaMART is still used at Stage 2 in many production systems because it is fast, interpretable, and robust. Neural LTR (cross-encoders) dominates Stage 3.

## The Connection to Everything Else in This Repo

LTR is the unifying framework that connects every scoring approach covered so far:

```bash
BM25 score          → a feature for LTR
TF-IDF score        → a feature for LTR
Dense cosine sim    → a feature for LTR
Cross-encoder score → a feature for LTR (or LTR IS the cross-encoder)
NDCG, MAP           → the metrics LTR optimizes
Evaluation          → how we measure LTR quality
```

A cross-encoder fine-tuned on MS MARCO with pairwise loss is LTR specifically a pairwise neural LTR model. The distinction between "LTR" and "neural reranking" is largely historical. They are the same paradigm at different points on the
complexity spectrum.

LTR is the conceptual hinge between classical retrieval (hand-crafted scores) and neural IR (end-to-end learned representations). It introduced the key idea that ranking is a supervised learning problem with its own loss functions,
evaluation metrics, and training data requirements an idea that every neural IR model inherits directly.

## My Summary

Learning to Rank trains models to combine multiple retrieval signals BM25 scores, dense similarity, document authority, freshness, click data into an optimal ranking using supervised learning from human relevance judgments. Three paradigms
exist: pointwise (predict relevance score per document), pairwise (predict which of two documents ranks higher), and listwise (optimize the full ranked list directly).
LambdaMART, combining gradient boosted trees with LambdaRank gradients, dominated production ranking for over a decade and remains widely deployed. Modern neural LTR
(BERT cross-encoders with pairwise or listwise loss) inherits the same framework with raw text as input instead of hand-crafted features. LTR is the conceptual bridge between classical retrieval functions and neural reranking understanding it makes every neural IR model feel like a natural evolution rather than a disconnected technique.
