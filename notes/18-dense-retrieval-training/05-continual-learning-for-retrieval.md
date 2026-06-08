# Continual Learning for Retrieval

Continual learning for retrieval is the practice of updating dense retrieval models
and their associated indexes over time as the world changes - new documents are
added to the corpus, existing documents become outdated, user query distributions
shift, and domain knowledge evolves - without completely retraining from scratch
on every update cycle. A retrieval model trained once on static data gradually
becomes less effective as the distribution of content it serves diverges from its
training distribution. Continual learning encompasses the techniques for refreshing
model knowledge, managing index staleness, incorporating new training signal from
production interactions, and controlling the rate of forgetting previously learned
representations during incremental updates. It is the operational dimension of
dense retrieval that makes a system remain effective for months and years rather
than just at the moment of initial deployment.

## Intuition

A dense retrieval model trained on a corpus of web pages from 2022 faces a
fundamentally different challenge in 2025. New entities, products, technologies,
and events have emerged. The vocabulary of relevant queries has evolved. Documents
that were highly relevant are now outdated. New documents cover topics that did not
exist during training.

If you retrain from scratch every time the world changes, you incur the full cost
of pretraining and fine-tuning on each update cycle - potentially weeks of GPU time
and significant engineering effort. If you never update, the model's performance
gradually degrades as the distribution drift between training data and production
data widens.

Continual learning navigates this tradeoff. The goal is to incorporate new signal -
new documents, new queries, new relevance judgments - while retaining the useful
representations already learned. The challenge is catastrophic forgetting: neural
networks have a strong tendency to overwrite previously learned representations
when trained on new data, because the gradient updates that help with new examples
often undo the weight configurations that helped with old ones.

The analogy: a new employee learns company-specific knowledge quickly, but if they
are then assigned to an entirely different project with different procedures, they
may forget the original company processes. Effective continual learning is like
periodic refreshers that keep prior knowledge alive while adding new information.

## When Retrieval Models Need Updating

Several distinct mechanisms cause retrieval quality to degrade over time:

### Corpus distribution shift

New documents enter the corpus with vocabulary, topics, and writing styles not
well represented in the training data:

```
Training data (2022): tech corpus dominated by Python, JavaScript, cloud services
Corpus in 2025:        now includes Rust, TypeScript, LLM tooling, AI agents

Impact:
  Queries about LLM tooling → model may not embed these well
  Documents about AI agents → may not cluster with related content
  New vocabulary → subword fragments with poor learned representations
```

The embedding model has never seen "LangGraph" or "llama.cpp" during training -
these terms get fragmented into subwords with no semantic weight behind them.

### Query distribution shift

The queries users submit evolve as their needs change:

```
2022 queries: "how to use Docker containers", "React state management"
2024 queries: "LLM inference optimization", "RAG pipeline evaluation"

Impact:
  New query patterns the model has not been trained to handle
  Changed user expectations about what constitutes a relevant result
  Emerging jargon and terminology the model does not recognize
```

### Relevance distribution shift

What users consider relevant changes as their sophistication, context, and
needs evolve. A document that was highly relevant for a query two years ago
may now be outdated, superseded, or simply no longer the preferred answer style.

### Model drift from production feedback

If implicit feedback signals (clicks, dwell time) are used to generate new
training pairs, the distribution of what gets selected as training data can
gradually drift from the original training distribution, creating feedback loops
that amplify certain result patterns over others.

## The Catastrophic Forgetting Problem

The central challenge in continual learning: when a neural network is trained
on new data D_new, gradient updates that minimize the loss on D_new tend to
increase the loss on previously learned data D_old.

### Why forgetting happens

Dense retrieval models are over-parameterized - they have far more parameters
than the training data explicitly constrains. When trained on D_old, many parameter
configurations could equally fit the training objective, and the specific
configuration found depends on random initialization and optimization trajectory.

When training on D_new, the optimizer moves parameters to minimize the new loss.
This movement is unconstrained with respect to D_old - the optimizer has no
mechanism to avoid overwriting the specific configuration that worked for D_old.

```
Loss landscape:
  θ*_old:  parameters minimizing L(D_old)
  θ*_new:  parameters minimizing L(D_new)
  θ*_both: parameters minimizing L(D_old ∪ D_new)

Naive fine-tuning on D_new:
  Starts at θ*_old
  Gradient descent toward θ*_new
  May land far from θ*_both if the loss landscapes conflict
```

### Forgetting severity in retrieval models

Catastrophic forgetting in dense retrieval is particularly severe for
domain-specific vocabulary. General semantic relationships (synonymy,
hyponymy, basic factual associations) are encoded redundantly across
many parameters and are relatively resistant to forgetting. Domain-specific
terminology and specialized semantic relationships are more sparsely encoded
and are easily overwritten.

Empirical observation: fine-tuning a model on a new medical domain for 2 epochs
can degrade general retrieval quality (BEIR average) by 5-15% while improving
domain quality by 15-25%. The tradeoff is not symmetric - forgetting is faster
than learning.

## Continual Learning Strategies

### Strategy 1 - Experience Replay

Maintain a buffer of training examples from previous data and mix them with
new training examples during updates:

```
Buffer: reservoir of (query, positive, negative) triples from previous training
New data: fresh labeled pairs from current corpus/queries

Mixed training:
  Each batch: fraction p from buffer, fraction (1-p) from new data
  p = 0.3: 30% replay, 70% new data

Buffer management:
  Fixed-size buffer: randomly sample from all historical data
  Recency-weighted: recent examples more likely to be selected
  Importance-weighted: examples where loss is high (hard examples) prioritized
```

Experience replay is the most practical continual learning approach for retrieval
because it requires only storing a representative sample of historical training
data - no architectural changes needed.

**Replay fraction selection:**

```
p = 0.5:  equal weight to old and new → slow adaptation, minimal forgetting
p = 0.2:  20% old, 80% new → faster adaptation, some forgetting acceptable
p = 0.0:  no replay (fine-tuning) → fastest adaptation, severe forgetting

Rule of thumb:
  High-value existing representations: p = 0.3-0.5
  Acceptable performance tradeoff: p = 0.1-0.2
  Domain pivot (previous data irrelevant): p = 0.0
```

### Strategy 2 - Elastic Weight Consolidation (EWC)

Constrain weight updates to avoid overwriting parameters important for previous
tasks:

```
Augmented loss for new task:
  L_EWC = L_new(θ) + λ × Σᵢ Fᵢ × (θᵢ - θ*ᵢ)²

Where:
  θ*ᵢ:  optimal parameter values from previous training
  Fᵢ:   Fisher information - measures how important θᵢ is for old tasks
  λ:    regularization strength

Fisher information approximation:
  Fᵢ = E[(∂ log p(y|x, θ*) / ∂θᵢ)²]
  Estimated from the diagonal of the Fisher information matrix on old data
```

EWC identifies which parameters matter most for old performance (high Fisher
information) and penalizes changing them during new training. Parameters with
low Fisher information are free to change to accommodate new data.

**EWC in practice for retrieval:**

The Fisher information matrix must be computed on the old training data before
updating. For large models (BERT-base with 110M parameters), storing the full
Fisher information is feasible, but computing it requires a forward pass over
the full old training set - significant compute.

A simpler approximation: penalize all weight changes equally (L2 regularization
toward the previous model):

```
L_simple = L_new(θ) + λ × ||θ - θ*_old||²

This is equivalent to EWC with uniform Fisher information.
Less precise but much cheaper to compute.
```

### Strategy 3 - Progressive Neural Networks

Freeze previously trained model parameters entirely. Add new network modules
for new tasks. Connect new modules to old modules through learned adapter
connections:

```
Architecture:
  Frozen column (old model):     [layer 1] → [layer 2] → ... → [layer 12]
  New column (new task):         [layer 1'] → [layer 2'] → ... → [layer 12']
  Lateral connections:           [layer k] → adapter → [layer k']

Training:
  Frozen column: no gradient
  New column + adapters: full gradient

Output: combine frozen and new column representations
```

Progressive networks guarantee zero catastrophic forgetting (frozen parameters
cannot change) at the cost of growing model size with each update. For retrieval,
this means the document index must also be updated to use the combined representation.

### Strategy 4 - Adapter-Based Updates

Insert small trainable modules (adapters) into a frozen pretrained model.
Only adapter parameters are updated - the base model is frozen:

```
Standard BERT layer:
  [attention] → [FFN] → output

With adapter:
  [attention] → [FFN] → [down_project: 768→64] → [up_project: 64→768] → output
                                                  ↑ adapter (tiny, trainable)
```

Adapter parameters are typically 1-5% of the base model parameters. Only adapters
are trained during updates - the base model is frozen. This provides:

- Zero forgetting of base model representations
- Fast training (few parameters to update)
- Easy version management (swap adapters for different domains/time periods)
- Efficient multi-domain serving (shared base + domain-specific adapters)

**Adapter approaches for retrieval:**

```
LoRA (Low-Rank Adaptation):
  Modify attention weights with low-rank updates:
  W' = W + BA  (B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d}, r << d)

Prefix tuning:
  Add trainable prefix tokens to attention:
  [prefix₁, prefix₂, ..., prefixₖ, query_tokens...]

Bottleneck adapters:
  Small FFN inserted between each transformer layer
```

LoRA is currently the most popular approach because it modifies the existing
attention computation rather than adding new modules, keeping the inference-time
architecture identical to the base model.

### Strategy 5 - ANCE with Periodic Refresh

ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation) naturally
supports continual learning through its periodic hard negative re-mining:

```
Initial training: train on static labeled data with BM25 hard negatives
Deployment:       index corpus, serve queries, log interactions

Period 1 update (every month):
  1. Encode updated corpus with current model
  2. Mine new hard negatives reflecting current model capabilities
  3. Collect implicit feedback pairs from production logs
  4. Fine-tune with: new feedback + refreshed hard negs + 20% experience replay
  5. Re-encode updated corpus, rebuild index

Period 2 update:
  Repeat with new corpus additions and fresh feedback
```

This cycle continuously adapts the model to corpus changes and user preferences
while the hard negative refresh ensures training stays challenging.

## Index Management for Continual Learning

Beyond the model, the vector index must be managed:

### Three index management strategies

**Full rebuild:**
Re-encode all documents with the updated model and rebuild the entire FAISS
or vector database index from scratch.

```
Cost:     high (encode all documents, rebuild index structure)
Benefit:  all documents encoded with the latest model
Use when: major model update, significant corpus change, ≥ monthly cadence
```

**Incremental update:**
Encode and index only new/changed documents. Keep existing document embeddings
from the previous model.

```
Cost:     low (only process changed documents)
Limitation: mismatch between old documents (old model embeddings) and new
            documents (new model embeddings) in the same index
Severity: depends on how much the model changed
          small adapter update → small embedding shift → acceptable mismatch
          full fine-tune → large embedding shift → significant quality loss
Use when: model change is small (adapter-only), frequent corpus updates
```

**Hybrid (dual index):**
Maintain two indexes - old model index and new model index - and serve
results from both, fusing with RRF:

```
Architecture:
  Old index: all documents encoded with model v1
  New index: all documents encoded with model v2 (new docs first, full rebuild ongoing)
  Query time: retrieve from both indexes, fuse results

Gradual transition:
  Week 1: 80% old index, 20% new index (new docs only)
  Week 2: 50/50 (old + growing new)
  Week 3: 20% old, 80% new (most docs re-encoded)
  Week 4: 100% new (full rebuild complete)
```

The dual index approach enables zero-downtime model updates with gradual
quality improvement as the new index grows.

### Incremental indexing with embedding compatibility

If the adapter-based update changes only the upper layers of the model, embeddings
from the lower layers change less. This can be exploited:

```
BERT layers 1-8:  frozen (embeddings change minimally after adapter update)
BERT layers 9-12: updated via adapters (embeddings change more)

Strategy:
  Documents embedded before update: re-encode with updated model
  Priority: documents that changed most (use embedding shift as proxy)
    change_score = ||emb_new(doc) - emb_old(doc)||
    re-encode documents with highest change_score first
```

Prioritized re-encoding focuses rebuild effort on documents where the old
embedding is most stale.

## Temporal Relevance and Document Freshness

For time-sensitive retrieval (news, financial, medical), content freshness
is a relevance dimension independent of semantic match:

### Freshness scoring

Add a recency component to the retrieval score:

```
final_score(q, d, t) = α × semantic_similarity(q, d)
                     + β × freshness_score(d, t)

freshness_score(d, t) = exp(-λ × (t_now - t_published(d)))

Where:
  λ controls the freshness decay rate
  t_published(d) is the document publication time
  t_now is the current time

Application:
  News retrieval:     λ = 0.5 (strong freshness preference, half-life ~2 days)
  Scientific papers:  λ = 0.001 (weak freshness preference, half-life ~700 days)
  Legal documents:    λ = 0.0 (no freshness penalty - historical precedent relevant)
```

### Temporal embedding augmentation

Encode time as a learnable embedding added to document representations:

```
doc_embedding = f_encoder(doc_text) + temporal_embedding(publication_date)

Where temporal_embedding maps dates to learned vectors that encode
temporal relationships (recent documents cluster with recent documents)
```

This allows the model to learn domain-specific recency preferences from
training data rather than hardcoding a decay function.

## Monitoring for Drift Detection

Continual learning requires detecting when a model update is needed before
quality degradation becomes user-visible:

### Embedding distribution monitoring

Track statistics of the query and document embedding distributions over time:

```
Daily metrics:
  Mean pairwise cosine similarity between random query embeddings
  Mean cosine similarity between queries and their top-retrieved documents
  Fraction of queries with very low maximum similarity to any document

Alert thresholds:
  Mean query similarity drops > 10% from 30-day rolling baseline
  Zero-result rate increases > 5% week-over-week
```

These signals detect corpus distribution shift - when new documents arrive
that the current model cannot match well to incoming queries.

### Implicit feedback degradation signals

Monitor online quality metrics for drift:

```
Weekly:
  Click-through rate on top-3 results
  Query reformulation rate
  Session abandonment rate

Alert: any metric degrades > 5% from 4-week rolling average
       → trigger investigation, possible model refresh
```

### Proactive drift detection with query log analysis

Periodically analyze incoming query logs for new vocabulary and topics
not represented in the model's training data:

```
Process:
  1. Collect 10K recent queries
  2. Tokenize with the model's tokenizer
  3. Flag queries with > 30% OOV (out-of-vocabulary) tokens
  4. Flag queries in topics not covered by training corpus

Threshold:
  OOV rate > 15% of queries → significant vocabulary drift → schedule update
  New topic cluster > 10% of queries → corpus gap → add new documents + fine-tune
```

## Update Scheduling Strategies

### Event-driven updates

Trigger model updates in response to specific events rather than on a fixed schedule:

```
Events that trigger updates:
  Corpus size increases > 20% since last training
  New product launch adds > 1000 new documents in specialized domain
  Query distribution shift detected (OOV rate or topic drift threshold exceeded)
  Significant user satisfaction drop (CTR or reformulation rate degradation)

Advantages: updates are timely and necessary
            no unnecessary updates when the model is still effective
Disadvantages: unpredictable resource scheduling
```

### Scheduled updates

Retrain or fine-tune on a fixed schedule regardless of detected drift:

```
Daily:   Lightweight adapter update with previous day's implicit feedback
Weekly:  ANCE hard negative refresh with new index
Monthly: Full fine-tune with accumulated new labeled pairs + experience replay
Yearly:  Full retraining from scratch on expanded labeled dataset
```

Scheduled updates are predictable and easy to operate at the cost of potentially
unnecessary updates (when the model is still performing well) or delayed updates
(when significant drift occurs between scheduled updates).

### Adaptive scheduling

Combine event detection with scheduled baseline updates:

```
Baseline: monthly scheduled update (ensures regular refresh even without detected drift)
Events:   immediate update if:
  - CTR drops > 10% week-over-week
  - OOV rate exceeds 20%
  - Corpus grows > 30%
```

## Evaluation for Continual Learning

Standard offline evaluation does not adequately assess continual learning quality.
Additional evaluation dimensions are needed:

### Backward transfer

How much does an update on new data degrade performance on old data?

```
Backward transfer = NDCG@10(old queries, after update) - NDCG@10(old queries, before update)

Target: backward transfer > -0.02 (acceptable degradation)
Alert:  backward transfer < -0.05 (significant forgetting)
```

### Forward transfer

Does training on historical data improve performance on new data without
explicit training on it?

```
Forward transfer = NDCG@10(new queries, after old training) - random baseline

High forward transfer: old training generalizes to new queries/documents
Low forward transfer:  new queries/documents require explicit training
```

### Online quality trajectory

Track NDCG@10 (from implicit feedback) over time on a rolling basis:

```
Quality trajectory for healthy continual learning:
  - Stable or improving NDCG during stable periods
  - Temporary dip after corpus change (before update)
  - Quick recovery after each update
  - Long-term trend: stable or slowly improving

Quality trajectory for poor continual learning:
  - Gradual downward trend between updates
  - Large drops after corpus changes
  - Slow recovery after updates
  - Long-term trend: declining
```

## The Practical Continual Learning Playbook

For most production retrieval teams, the following playbook captures the
essential practices without requiring sophisticated continual learning research:

### Tier 1 - Minimal viable continual learning

```
Who:      Small team, limited ML infrastructure
What:     Monthly full fine-tune from scratch on accumulated labeled pairs
          + experience replay of historical training data
When:     Schedule monthly, trigger immediately if quality drops > 10%
Cost:     ~8-24 GPU hours per month for BERT-base class models
```

### Tier 2 - Standard production practice

```
Who:      Medium team with ML infrastructure
What:     ANCE-style periodic update:
          1. Monthly ANCE refresh with hard negative re-mining
          2. Weekly adapter update with implicit feedback
          3. Dual index for zero-downtime updates
When:     Scheduled + event-triggered
Cost:     ~2-4 GPU hours/week, ~16-32 GPU hours/month
```

### Tier 3 - Advanced continual learning

```
Who:      Large team with dedicated ML platform
What:     Streaming updates with online ANCE
          Per-domain adapter management
          Continuous A/B testing of model versions
          Automated drift detection and update scheduling
When:     Near-continuous (daily or faster)
Cost:     Significant ongoing GPU resource allocation
```

## Where This Fits in the Progression

```
Contrastive learning          → core training paradigm
Hard negative mining          → improving negative quality
Knowledge distillation        → teacher-student training
Batch size and temperature    → hyperparameter deep dive
Continual learning            → keeping models fresh  ← you are here
```

This note closes the dense retrieval training module. The module covers the
complete lifecycle of a dense retrieval model: the core training paradigm
(contrastive learning), improving training quality (hard negatives, distillation,
hyperparameters), and keeping the model effective over its operational lifetime
(continual learning).

## My Summary

Continual learning for retrieval addresses the gradual degradation of dense models
as the corpus, queries, and relevance criteria evolve over time. The central
challenge is catastrophic forgetting - gradient updates for new data tend to
overwrite representations learned from old data. The main mitigation strategies
are experience replay (mix historical training examples into each update batch
at 20-50% ratio), elastic weight consolidation (penalize changes to parameters
important for old tasks using Fisher information), adapter-based updates (freeze
the base model, train only small LoRA or bottleneck adapter modules), and ANCE
with periodic hard negative refresh (continuously re-mine hard negatives as the
corpus and model evolve). Index management requires choosing between full rebuild
(expensive, all documents current), incremental update (cheap, but embedding
mismatch between old and new documents), or dual index with gradual transition
(zero-downtime, gradual quality improvement). Drift detection monitors embedding
distribution statistics, implicit feedback signals, and incoming query vocabulary
for OOV rates that indicate when a model update is needed. The practical playbook
scales from monthly full fine-tune with experience replay for small teams to
near-continuous streaming updates with automated drift detection for large teams.
