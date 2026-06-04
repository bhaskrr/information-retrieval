# Hard Negative Mining Strategies

Hard negative mining is the process of selecting training negatives that are
difficult for a retrieval model to correctly distinguish from positive examples -
documents that appear relevant to a query but are not actually relevant. Standard
in-batch negatives in contrastive training are random documents from other queries
in the batch, which are typically easy to distinguish because they belong to
completely different topics. Hard negatives belong to the same topic or share
vocabulary with the query, forcing the model to learn fine-grained semantic
distinctions rather than coarse topical separation. The quality of hard negatives
is one of the two most important factors determining dense retrieval model quality
(the other being training data scale). Models trained on random negatives plateau
quickly; models trained on hard negatives continue improving because each training
step forces progressively more precise semantic discrimination.

## Intuition

Consider training a model to distinguish relevant from irrelevant documents for
the query "treatment for type 2 diabetes":

**Easy negative (random in-batch):**
"The French Revolution began in 1789 with the storming of the Bastille."
→ Trivially irrelevant. The model learns nothing useful from this pair -
it already knows French history documents are unrelated to diabetes treatment.

**Medium negative (same topic area):**
"Risk factors for developing type 2 diabetes include obesity and sedentary lifestyle."
→ Topically related but about prevention, not treatment. The model learns that
diabetes-adjacent documents are not sufficient - it must distinguish treatment
content specifically.

**Hard negative (same specific topic):**
"Metformin is the first-line medication for type 2 diabetes, improving insulin
sensitivity and reducing hepatic glucose production."
→ Directly about type 2 diabetes treatment - this is a true positive. If this
appeared as a negative, it would be a false negative that corrupts training.

**Hard negative (similar but non-relevant):**
"Insulin therapy is the primary treatment for type 1 diabetes in patients who
cannot produce their own insulin."
→ About diabetes treatment but for type 1, not type 2. The model must learn
that type 1 and type 2 diabetes have different treatments. This is the ideal
hard negative: related enough to be challenging, different enough to be
genuinely not relevant.

The progression from easy to hard negatives directly mirrors the progression of
what the model needs to learn - from "distinguish completely different topics"
to "distinguish subtle semantic differences within the same specific domain."

## Why Hard Negatives Matter: Empirical Evidence

The impact of hard negative quality is one of the clearest empirical findings
in dense retrieval research:

```
Training strategy                    MS MARCO Dev MRR@10
─────────────────────────────────────────────────────────────
BM25 first stage (no dense model)    0.187
In-batch negatives only              0.256
+ BM25 hard negatives (1 per q)      0.318
+ BM25 hard negatives (7 per q)      0.340  ← DPR original paper
+ ANCE dynamic negatives             0.355
+ Distillation + hard negs           0.393  ← TAS-B
+ Large batch + hard negs + distill  0.420  ← E5-base range
```

Each step from random negatives to BM25 hard negatives to dynamic mining to
distillation represents a consistent, large improvement. The transition from
in-batch only to BM25 hard negatives alone is a 24% relative improvement
on MRR@10.

## Strategy 1 - BM25 Hard Negatives (Static Mining)

The simplest and most widely used hard negative strategy. Use BM25 to retrieve
the top-k documents for each training query, exclude known positives, and use
the retrieved documents as negatives.

### Mining procedure

```
For each query q in training set:
  1. Retrieve BM25 top-k documents (k = 100 is typical)
  2. Identify positive documents: P_q ⊆ {BM25 top-k}
  3. Hard negatives: H_q = {BM25 top-k} - P_q
  4. Sample n negatives from H_q (typically n = 7-15)

Storage: pre-compute and save (query, positive, hard_negatives_list) triples
Training: load triplets and train with in-batch + hard negatives
```

### Why BM25 hard negatives are valuable

BM25 hard negatives share vocabulary with the query - they contain the exact
terms the user searched for but are not actually relevant. This forces the model
to learn:

- Term weighting: some terms are more informative than others for relevance
- Context sensitivity: the same term can indicate relevance or irrelevance
  depending on surrounding context
- Negation and qualifiers: "type 1 diabetes" is not relevant for "type 2 diabetes"
  despite containing "diabetes"

### BM25 negative scoring

Rather than sampling uniformly from BM25 top-k, score-weighted sampling
preferentially selects harder negatives:

```
Weight each BM25 negative by its BM25 score normalized by the positive's score:
  w(d⁻) = BM25(q, d⁻) / BM25(q, d⁺)

Higher w → closer to positive in BM25 score → harder negative
Sample proportional to w: harder negatives selected more often
```

This ensures training time is spent on the most informative negatives rather
than on randomly-scored retrievals.

### Limitations of BM25 hard negatives

BM25 hard negatives are challenging only on the lexical dimension - they share
terms with the query but are not relevant. A dense model trained on BM25 hard
negatives learns to distinguish lexical similarity from semantic relevance, but
it does not face the complementary challenge: documents that are semantically
similar to the query (same concepts in different vocabulary) but not relevant.

Example:

```
Query: "cardiac arrest resuscitation techniques"

BM25 hard negative: "cardiac arrest epidemiology in urban populations"
  (shares "cardiac arrest" but is about epidemiology, not resuscitation)
  ← BM25 retrieves this, it is a genuine hard negative for BM25 vocabulary reasons

Semantic hard negative: "CPR performance in patients with respiratory failure"
  (same conceptual territory - resuscitation - but for a different condition)
  ← BM25 may not retrieve this (different vocabulary), but dense model might
```

Dense hard negatives address the gaps left by BM25 hard negatives.

## Strategy 2 - Dense Hard Negatives (Periodic Mining)

Use the current state of the dense model itself to mine hard negatives. Documents
that the current model retrieves highly but are not relevant are exactly the
negatives the model needs to learn to reject.

### Mining procedure

```
After each k training steps (or epochs):
  1. Encode all corpus documents with current model
  2. Build FAISS index
  3. For each query in training set:
     a. Retrieve top-k with current model
     b. Exclude known positives
     c. Use retrieved non-positives as hard negatives
  4. Update training dataset with new hard negatives
  5. Continue training with updated negatives
```

This is called ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation),
introduced by Xiong et al. at Microsoft in 2020.

### Why dense negatives are uniquely valuable

Dense hard negatives are chosen specifically because the current model fails on
them - they expose the model's specific blind spots:

```
Training step 1000:
  Model can correctly distinguish topically different queries
  Dense negatives at this point: topically similar documents in same domain

Training step 5000:
  Model can correctly distinguish topically similar documents
  Dense negatives at this point: semantically similar documents with different implications

Training step 10000:
  Model distinguishes most semantic nuances
  Dense negatives at this point: near-duplicate documents with critical distinguishing detail
```

The adaptive nature of dense mining means the difficulty automatically scales
with the model's current ability - the Goldilocks property that static BM25
negatives lack.

### Asynchronous dense negative mining

Running full corpus encoding for negative mining is expensive. Asynchronous
mining schedules mining to minimize training interruption:

```
Main process:     Training with current hard negatives (continuous)
Background process: Encoding corpus, building new FAISS index, mining negatives
                    (runs while main training continues)

Handoff: when background process completes, swap in new hard negatives
         main training continues without interruption

Frequency: every 10,000-50,000 training steps depending on corpus size
```

For MS MARCO (8.8M passages) with a GPU, full corpus encoding takes
~2-4 hours. Asynchronous mining means this cost is paid in wall-clock time
but not in training step time.

## Strategy 3 - Mixed Strategy Negatives

Combining BM25 and dense negatives provides complementary challenges:

```
For each query, use:
  - In-batch random negatives: easy, efficient, many per query
  - BM25 hard negatives: lexical discrimination challenge
  - Dense hard negatives: semantic discrimination challenge

Training loss:
  L = L_InfoNCE(query, positive, {in-batch ∪ BM25 ∪ dense negatives})
```

In practice, DPR uses 1 BM25 hard negative per query. ANCE replaces BM25
with dense negatives. TAS-B uses a mix of BM25 and dense with distillation.

### Negative type mixing ratios

Empirical findings on effective mixing ratios:

```
DPR configuration:
  In-batch: 63 per query (batch size 64)
  BM25 hard: 7 per query
  Dense hard: 0
  MRR@10: ~0.340

ANCE configuration:
  In-batch: 127 per query (batch size 128)
  BM25 hard: 0
  Dense hard: 1 (updated periodically)
  MRR@10: ~0.355

Mixed (reported best):
  In-batch: 63 per query
  BM25 hard: 3 per query
  Dense hard: 3 per query
  MRR@10: ~0.368
```

The consensus: a mix of BM25 and dense hard negatives consistently outperforms
either alone, though the exact ratio matters less than whether both types are
present.

## Strategy 4 - Cross-Encoder Scored Negatives

Use a cross-encoder to assign precise relevance scores to the hard negatives,
then select negatives with the highest cross-encoder scores:

```
Step 1: Mine top-50 candidates per query with BM25 or dense retrieval
Step 2: Cross-encoder scores each candidate: score(q, d) → float
Step 3: Sort candidates by cross-encoder score
Step 4: Use top-scored non-relevant candidates as hard negatives
        (highest score = most confusable for the cross-encoder = hardest)
```

This produces the hardest possible negatives - documents that even the most
accurate relevance model (cross-encoder) nearly ranks as positive. Training
against these forces the bi-encoder to approach cross-encoder-level discrimination.

### Cross-encoder negative quality vs cost

```
Negative type            Informativeness    Mining cost per 1M queries
─────────────────────────────────────────────────────────────────────────
Random in-batch          Very low           Free
BM25 top-k               Low-medium         Minutes (BM25 fast)
Dense top-k              Medium-high        Hours (full corpus encoding)
Cross-encoder scored     Very high          Days (cross-encoder for each pair)
```

Cross-encoder negatives produce the best models but are impractical for
large-scale training. TAS-B addresses this by distilling cross-encoder scores
into bi-encoder training rather than explicit negative selection.

## Strategy 5 - Curriculum Hard Negative Mining

Stage negative difficulty to prevent training instability:

### Stage 1 - Warm-up with easy negatives (epochs 1-2)

Train only with in-batch random negatives. Allow the model to learn basic
semantic structure before facing hard negatives. A freshly initialized model
cannot learn from hard negatives effectively - it has no basis to distinguish
them from positives.

Evidence: ANCE reports that starting with hard negatives from epoch 1 with
random initialization leads to training collapse in some configurations.

### Stage 2 - Introduce BM25 hard negatives (epochs 2-3)

Add lexical hard negatives. The model now has basic semantic understanding
and can begin learning lexical-semantic distinctions.

### Stage 3 - Switch to dense hard negatives (epoch 4+)

Replace or supplement with dense model-mined negatives. The model's current
representations define what constitutes a challenging negative for it.

### Stage 4 - Continue with updated dense negatives (epoch 5+)

Periodically re-mine negatives as the model improves. Each mining cycle
provides harder negatives that push the model further.

```
Curriculum schedule:
  Epoch 1-2:  in-batch only
  Epoch 3:    in-batch + BM25 (7 per query)
  Epoch 4:    in-batch + BM25 (3) + initial dense (3)
  Epoch 5+:   in-batch + updated dense (5-7), re-mine every 10K steps
```

## Strategy 6 - Synthetic Hard Negatives

Generate hard negatives using LLMs rather than retrieving them:

### Perturbation-based negatives

Generate negatives by slightly modifying relevant documents to make them
non-relevant while preserving surface similarity:

```
Positive: "Aspirin inhibits COX-1 and COX-2 enzymes, reducing prostaglandin synthesis"
           and thereby reducing inflammation and pain.

Synthetic negative (entity swap):
  "Ibuprofen inhibits COX-1 and COX-2 enzymes, reducing prostaglandin synthesis"
  ← Correct drug class but wrong specific drug for aspirin-specific queries

Synthetic negative (negation):
  "Aspirin does not inhibit COX-2 enzymes in therapeutic doses"
  ← Factually incorrect but superficially similar
```

Perturbation-based negatives force fine-grained entity discrimination and
factual accuracy checking.

### LLM-generated distractors

Prompt an LLM to generate plausible but incorrect answers to queries:

```
Prompt:
  Query: {query}
  Relevant document: {positive_document}

  Generate a plausible-sounding but incorrect document that would appear
  relevant to this query but actually does not answer it correctly.
  The distractor should:
  1. Use similar vocabulary to the relevant document
  2. Be on the same topic
  3. Contain a key factual error or address a subtly different question

  Distractor:
```

LLM-generated distractors are highly effective hard negatives but expensive
to generate at scale. Practical approach: generate for 10-20% of training
queries and mix with BM25 negatives for the remainder.

## False Negative Filtering

Hard negatives selected by BM25 or dense retrieval will inevitably include
some true positives - documents that genuinely answer the query but were not
annotated as such in the training data. Training against false negatives
corrupts the model by penalizing it for correctly identifying relevant documents.

### Detection approaches

**Known positive filtering:**
The most basic approach. Filter out any document that appears in the known
positive set for the query.

**BM25 score threshold:**
Documents with very high BM25 scores for the query are likely relevant even
if not annotated. Use a threshold:

```
If BM25(q, d⁻) > α × BM25(q, d⁺):
  likely false negative → exclude from training
```

Where α ∈ [0.5, 0.9]. Conservative (α = 0.9) misses fewer false negatives;
aggressive (α = 0.5) may exclude some true negatives.

**Cross-encoder verification:**
Score candidate negatives with a cross-encoder and exclude high-scoring ones:

```
If cross_encoder(q, d⁻) > threshold (e.g., 0.7):
  likely relevant → exclude from negative set
```

More precise than BM25 filtering but requires running cross-encoder inference
on all candidate negatives.

**Retrieval-based annotation:**
For high-value training data, manually annotate retrieved candidates to
identify false negatives. Expensive but produces the cleanest negative sets.

### Impact of false negatives on training

```
False negative rate    MRR@10 degradation
────────────────────────────────────────────────
0%                     baseline
5%                     ~0.003 (slight)
10%                    ~0.010 (noticeable)
20%                    ~0.025 (significant)
30%+                   ~0.050+ (severe degradation)
```

False negatives above 10% of the hard negative pool typically cause significant
degradation. For densely labeled datasets (many positive documents per query),
false negative filtering is essential.

## Hard Negative Quality Evaluation

Before training, evaluate whether your hard negatives are actually hard:

### Difficulty metrics

**Hard negative accuracy:**
What fraction of hard negatives does the current model correctly rank below
the positive? If accuracy is very high (> 0.95), negatives are too easy.
If accuracy is very low (< 0.60), negatives may include false negatives.

```
Target range: 0.70-0.90 hard negative accuracy
Too easy (> 0.90): negatives provide little learning signal
Too hard (< 0.60): possible false negative contamination
```

**Similarity distribution:**
Plot the distribution of cosine similarities between queries and hard negatives:

```
Ideal hard negative distribution:
  Mean similarity:    0.7-0.85 (close to positive similarity of ~0.85-0.95)
  Std similarity:     0.05-0.15 (spread across the difficulty range)

Easy negative distribution:
  Mean similarity:    0.3-0.5 (clearly dissimilar from query)
  → Negatives are too easy; training on them wastes compute
```

**NDCG@10 on hard negative set:**
Evaluate the model's ability to rank the positive above the hard negatives:

```
If NDCG@10(positive + hard_negatives) ≈ 0.90:
  Negatives are too easy - model already succeeds with little effort

If NDCG@10(positive + hard_negatives) ≈ 0.50:
  Good difficulty - model struggles but can learn

If NDCG@10(positive + hard_negatives) ≈ 0.20:
  Negatives may be too hard or include false negatives
  → Investigate false negative rate
```

## Cross-Domain Hard Negatives

For domain adaptation, use cross-domain negatives to teach the model domain
boundary awareness:

```
Target domain: biomedical retrieval
Cross-domain negatives: medical documents that are highly relevant in general
                        but not in the specific biomedical context

Example:
  Query: "metformin mechanism of action insulin resistance"
  Cross-domain negative: "insulin resistance lifestyle interventions review"
    (relevant in general health context, but not the pharmacology answer)
  Target domain positive: "Metformin activates AMP-activated protein kinase,
                           reducing hepatic glucose production and improving
                           peripheral insulin sensitivity"

Training on this teaches: general health relevance ≠ pharmacological relevance
```

Cross-domain negatives are particularly valuable for domain-adapted models
that must distinguish domain-specific precision from topically-related generality.

## Hard Negative Mining Decision Guide

```
Situation                              Recommended strategy
────────────────────────────────────────────────────────────────────────
Starting from scratch, no GPU           BM25 hard negatives (7 per query)
                                        Fast, no GPU needed, large improvement

Have GPU, want best quality             Mixed BM25 (3) + dense (3) + in-batch
                                        Complementary challenges

Large corpus (>10M docs), GPU cluster   ANCE dynamic mining
                                        Self-improving negatives, highest quality

Few labeled pairs (<1000)               BM25 hard negatives + cross-encoder
                                        verification to filter false negatives

Domain adaptation, small corpus         BM25 + LLM-generated distractors
                                        Address vocabulary gap

Maximizing quality, have teacher model  Cross-encoder verified negatives
                                        Most expensive but highest quality
```

## My Summary

Hard negative mining selects training negatives that are genuinely challenging for
the retrieval model - documents that appear relevant to a query but are not -
forcing the model to learn fine-grained semantic discrimination rather than coarse
topical separation. The four main strategies are: BM25 hard negatives (retrieve
documents with high lexical overlap but non-relevant content - the simplest and
most impactful strategy, moving MRR@10 from 0.256 to 0.340 over random negatives),
dense hard negatives (ANCE-style periodic re-mining using the current model's
representations to identify its specific failure cases), mixed strategies combining
both for complementary lexical and semantic challenges, and cross-encoder scored
negatives using a cross-encoder teacher to identify the most confusable non-relevant
documents. False negative filtering is critical - documents incorrectly labeled as
negatives that are actually relevant corrupt training and must be removed through
BM25 score thresholding, cross-encoder verification, or manual annotation. Curriculum
learning (easy negatives first, then progressively harder) prevents training
instability from hard negatives applied to randomly initialized models. Negative
quality evaluation - measuring the similarity gap between positives and negatives,
and the model's discrimination accuracy - confirms that mined negatives are in the
effective difficulty range for the current model.
