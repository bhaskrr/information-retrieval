# Human Evaluation Protocols

Human evaluation protocols are the structured procedures, guidelines, and quality
control mechanisms used to collect reliable relevance judgments from human
annotators for IR system evaluation. They cover every aspect of the annotation
process: how relevance is defined for a specific domain and task, how annotators
are recruited and trained, how annotation tasks are presented to minimize cognitive
load and bias, how disagreements between annotators are measured and resolved, and
how the final relevance judgments are validated for quality. Getting human evaluation
protocols right is what separates a reliable, reproducible evaluation benchmark from
a noisy dataset whose results cannot be trusted or reproduced. Every major IR
benchmark - TREC, BEIR, MS MARCO, MIRACL - is built on carefully designed human
evaluation protocols, and the quality of those protocols directly determines the
quality of the benchmarks built on them.

## Intuition

Two annotators are given the same query and the same document and asked to judge
relevance. The first annotator says "relevant" - the document addresses the topic.
The second says "not relevant" - the document mentions the topic but does not answer
the specific question. Who is right?

Both annotators are right according to their interpretation of "relevance." The
problem is not that annotators disagree on the facts - both read the same document.
The problem is that the annotation task was underspecified. What counts as relevant?
Does the document need to directly answer the question or is topical overlap
sufficient? How specific must the match be? Can a document be relevant if it
provides background but not the specific information requested?

Human evaluation protocols are the answer to these underspecification problems.
Clear guidelines, graded relevance scales, worked examples, and inter-annotator
agreement measurement turn the inherently subjective concept of relevance into a
reproducible, measurable operationalization. A well-designed protocol produces
annotation datasets where different annotators applying the same guidelines reach
the same judgments at a high rate - and where that rate can be measured and
reported so evaluation consumers understand how reliable the data is.

## Relevance Definitions

The first and most important design decision in any human evaluation protocol:
how is relevance defined?

### Binary relevance

Documents are relevant (1) or not relevant (0):

```
Definition: A document is relevant if it contains information that
            directly addresses the user's information need as expressed
            by the query.

Not relevant: document is off-topic or tangential
Relevant:     document directly addresses the query
```

Binary relevance is simple to annotate and produces clean training and evaluation
data. It loses precision - a comprehensive perfect answer and a partial vague
answer both get label=1.

Used in: MS MARCO, most early TREC tracks, simple classification training.

### Graded relevance (TREC scale)

Four levels of relevance commonly used in TREC tracks:

```
0 - Not relevant:     The document does not address the topic.
                      It may share surface vocabulary but provides
                      no useful information for the information need.

1 - Marginally relevant: The document mentions the topic but does not
                         provide substantive information about it.
                         Tangentially related.

2 - Relevant:         The document provides useful, substantive information
                      about the topic. Addresses part of the information need
                      but may be incomplete or less direct.

3 - Highly relevant:  The document directly and comprehensively addresses
                      the information need. An ideal answer.
```

Graded relevance enables NDCG computation, which rewards placing highly relevant
documents at the top. It requires more annotator judgment but produces richer
evaluation data.

Used in: TREC-DL, TREC-COVID, most modern benchmark construction.

### Domain-specific relevance definitions

Generic relevance definitions are insufficient for specialized domains. Domain
knowledge must be encoded in the guidelines:

**Legal IR:**

```
0 - Not on point: Different legal issue, jurisdiction, or time period
1 - Background:   Related legal principle but different holding
2 - Related:      Similar issue, potentially persuasive authority
3 - On point:     Same legal issue, directly applicable holding
```

**Medical IR:**

```
0 - Not relevant: Different condition, population, or treatment
1 - Background:   General medical information, not condition-specific
2 - Related:      Addresses the condition but not the specific question
3 - Directly relevant: Clinical evidence directly addressing the question
    with appropriate study design
```

**Code search:**

```
0 - Wrong language or task: Different programming language or task
1 - Approach relevant: Shows relevant approach but incomplete
2 - Functionally relevant: Correct approach, may need adaptation
3 - Directly usable: Can be directly used or minimally adapted
```

Domain-specific definitions require annotators with domain expertise and produce
higher-quality data than generic definitions applied by non-expert annotators.

## Annotation Guidelines

Annotation guidelines are the reference document annotators consult during the
annotation process. Well-designed guidelines are the single biggest factor in
annotation quality.

### Structure of effective guidelines

**1. Task definition**

Clearly describe what the annotator is being asked to do and why. Include context
about the end use of the data:

```
Task: You are evaluating the quality of search results for an information
retrieval system. For each query-document pair, you will judge how well the
document answers the query using the relevance scale described below.

Context: These judgments will be used to train and evaluate search systems.
Your goal is to assess relevance from the perspective of someone genuinely
seeking the information expressed by the query.
```

**2. Relevance scale with worked examples**

Each relevance level must be defined with multiple positive and negative examples.
Worked examples are far more valuable than abstract definitions:

```
Relevance level 0 - Not relevant

Definition: The document does not provide any useful information for the query.

Example:
  Query: "symptoms of type 2 diabetes"
  Document: "Insulin was first isolated in 1921 by Banting and Best at the
            University of Toronto. The discovery revolutionized treatment of
            diabetes mellitus."
  Judgment: 0 - This is historical background about insulin discovery, not
            information about symptoms. A person looking for symptoms would
            not find this useful.

Not-example (would be 2, not 0):
  Document: "Type 2 diabetes often presents with increased thirst, frequent
            urination, and fatigue. Blurred vision and slow-healing sores
            are also common."
  Judgment: 2 - This directly describes symptoms.
```

**3. Decision rules for difficult cases**

Every annotation task has edge cases. Anticipate them and provide explicit rules:

```
Decision rule: Relevance vs. Tangential relevance

If the document is about the general topic area but does not address the
specific question, assign grade 1 (marginally relevant), not grade 2.

Example:
  Query: "What is the half-life of aspirin in the bloodstream?"
  Document: "Aspirin is widely used as an analgesic and anti-inflammatory.
            Common side effects include gastrointestinal irritation."
  Judgment: 1 - The document is about aspirin but does not mention pharmacokinetics
            or half-life. It is topically related but does not address the query.

Decision rule: Outdated information

If the document was accurate when written but is now outdated:
  - If the query asks for current information: grade 1 or 0 depending on extent
  - If the query has no time sensitivity: judge based on content, not date
```

**4. What NOT to consider**

Explicitly tell annotators what factors to exclude from their judgment to prevent
irrelevant quality signals from contaminating relevance labels:

```
Do NOT consider:
  - Document length (short documents can be highly relevant)
  - Writing quality or formatting
  - Source credibility (unless the query is about credible sources)
  - Whether you personally agree with the document's claims
  - Whether the document is easy or hard to read

DO consider:
  - Whether the document contains information that answers the query
  - How directly and specifically it addresses the information need
  - Whether someone asking this query would find this document useful
```

**5. Edge cases catalog**

A growing list of edge cases encountered during annotation with their canonical
resolution:

```
Edge case: Query has multiple interpretations
  Resolution: Judge relevance for the most likely interpretation unless the
  query specifies context. If truly ambiguous, default to the most common
  interpretation and note in the comments field.

Edge case: Document answers a related but different question
  Resolution: Grade 1 if the information provides useful context, grade 0
  if the document is on a different topic entirely.

Edge case: Document is behind a paywall or requires login
  Resolution: Judge based on the visible excerpt and metadata. Note that
  full content is unavailable.
```

## Inter-Annotator Agreement (IAA)

Inter-annotator agreement measures how consistently different annotators apply
the same guidelines to the same data. It is both a quality control metric (low
IAA indicates problems with the guidelines or training) and a benchmark
characteristic (reported in papers so readers understand annotation reliability).

### Cohen's Kappa (κ)

The standard IAA metric. Measures observed agreement corrected for chance agreement:

```
κ = (P_observed - P_chance) / (1 - P_chance)

Where:
  P_observed = fraction of items where annotators agree
  P_chance   = fraction expected to agree by chance (given marginal distributions)

Interpretation:
  κ < 0.20:  slight agreement (essentially random)
  κ = 0.20-0.40: fair agreement
  κ = 0.40-0.60: moderate agreement
  κ = 0.60-0.80: substantial agreement  ← target for IR annotation
  κ > 0.80:  near-perfect agreement
```

For binary relevance annotation, κ > 0.60 is the standard target. For graded
relevance, κ > 0.50 is typically acceptable because adjacent-grade disagreements
(e.g., 2 vs 3) are less consequential than non-adjacent ones (e.g., 0 vs 3).

### Fleiss' Kappa

Extension of Cohen's kappa to more than two annotators:

```
Fleiss' κ = (P̄ - P_e) / (1 - P_e)

Where P̄ = mean pairwise agreement, P_e = expected chance agreement

Used when: 3+ annotators judge each item
Common in: large-scale annotation projects with multiple annotator pools
```

### Krippendorff's Alpha

A more flexible IAA metric that handles:

- More than two annotators
- Missing data (not all annotators judge all items)
- Ordinal scales (neighboring grades partially agree in graded relevance)

```
α = 1 - D_o / D_e

Where D_o = observed disagreement, D_e = expected chance disagreement

For ordinal relevance scales (0, 1, 2, 3), Krippendorff's alpha with
ordinal metric is more appropriate than Cohen's kappa because it treats
a 0 vs 3 disagreement as more severe than a 1 vs 2 disagreement.
```

### Adjacent agreement rate

A practical complement to kappa that measures how often annotators are within
one grade of each other on a graded scale:

```
Adjacent agreement = fraction of pairs where |grade_A - grade_B| ≤ 1

For 4-level relevance scale (0-3):
  Exact agreement:    grade_A = grade_B
  Adjacent agreement: |grade_A - grade_B| ≤ 1

Target: adjacent agreement > 0.85 even when exact agreement is ~0.70
```

Adjacent agreement captures that a 2 vs 3 disagreement is much less
consequential for evaluation than a 0 vs 3 disagreement.

## Annotator Recruitment and Training

The quality of annotation depends heavily on who is doing the annotation and
how well they have been trained.

### Annotator requirements by domain

**General web search:**
Native speakers of the query language. Basic computer literacy. No domain
expertise required. Crowdworkers (Mechanical Turk, Scale AI, Appen) are
appropriate.

**Scientific and medical:**
At minimum a relevant bachelor's degree. Reading comprehension of academic
text. For clinical questions: medical training required. Graduate students
in the relevant field are often appropriate.

**Legal:**
Legal background required for case relevance. Law students or paralegals
at minimum. For jurisdiction-specific relevance: licensed attorneys.

**Code:**
Ability to read and understand code in the target language. Software engineers
with at least 2 years of experience.

### Training pipeline

A standard annotator training process for a new IR evaluation task:

**Step 1 - Onboarding (1-2 hours)**
Annotators read the guidelines in full. They complete a brief quiz on
the relevance scale definitions and decision rules to confirm understanding.

**Step 2 - Practice set (30-60 minutes)**
A set of 20-30 items with known gold-standard labels. Annotators complete
these independently and receive feedback on disagreements with the gold
standard. The practice set highlights common difficult cases.

**Step 3 - Calibration batch (2-4 hours)**
Annotators each label the same 50-100 items. IAA is computed. Annotators
whose kappa is below the threshold (typically 0.55) receive additional
feedback and repeat the calibration batch.

**Step 4 - Production annotation**
Annotators work on disjoint sets of items with a small overlap (10-15%)
maintained for continuous IAA monitoring. Items falling below the IAA
threshold are flagged for adjudication.

### Continuous quality monitoring

During production annotation, monitor annotator quality:

```
Metrics tracked per annotator:
  - IAA with other annotators on overlap set
  - Agreement with gold-standard quality control items
  - Distribution of relevance labels (flag if distribution is extreme)
  - Annotation speed (too fast = rushing, too slow = confusion)

Responses to quality issues:
  - Low IAA: additional training and feedback, reduce workload
  - Drift in distribution: retrain on guidelines, fresh calibration batch
  - Failed quality control: review items, possible replacement
```

## Adjudication of Disagreements

When annotators disagree, a systematic resolution process is needed:

### Decision hierarchy for adjudication

**Level 1 - Majority vote:** If 3+ annotators judge the same item,
majority vote resolves disagreements without additional work.

**Level 2 - Expert annotator review:** Disagreements between 2 annotators
go to a designated expert annotator (senior annotator, domain expert)
who reviews both annotations and the document, makes a final judgment,
and provides a written rationale.

**Level 3 - Consensus discussion:** For difficult or ambiguous cases where
expert review does not clearly resolve the disagreement, a group discussion
among 2-3 expert annotators reaches consensus. The reasoning is documented
and added to the edge cases catalog.

**Level 4 - Guideline clarification:** If a category of items consistently
produces disagreements, the annotation guidelines are updated to address
the ambiguity. All previously annotated items in that category are reviewed
against the updated guidelines.

### Adjudication documentation

Every adjudicated item should be documented with:

- Original annotations from each annotator
- Expert/consensus final judgment
- Written rationale for the final judgment
- Whether this case is now added to the edge cases catalog

This documentation serves two purposes: it builds the edge cases catalog
for future annotation tasks, and it creates an audit trail that allows
evaluation benchmark consumers to understand how difficult cases were handled.

## Annotation Task Design

Beyond guidelines, the design of the annotation interface and task presentation
significantly affects annotation quality.

### Presenting the annotation task

**Context provision:**
Annotators should see the query in the context it was submitted - not as an
isolated string. For conversational queries, show the conversation history.
For navigational queries, show the user's apparent goal.

**Document truncation:**
Documents longer than 500-1000 words should be truncated to the most relevant
portion. Presenting very long documents increases annotator fatigue without
improving judgment quality. When truncating, preserve the beginning and sections
most likely to contain relevant content.

**Highlight query terms:**
Highlighting query terms in the document helps annotators quickly locate
potentially relevant passages. This reduces cognitive load without biasing the
relevance judgment (the relevance decision still requires reading).

**Order and presentation:**
Avoid presenting documents in ranking order (annotators may be biased by
knowing a document was ranked first). Randomize document presentation order
within each annotation task.

### Task batching strategy

Annotating random samples is less efficient than intelligent batching:

**Same query, multiple documents:**
Group documents for the same query together. Annotators develop context
about the query and can make consistent comparative judgments. Recommended
batch size: 5-20 documents per query.

**Across-query batching:**
For large annotation projects, annotators may specialize by topic area rather
than labeling all query types. Annotators build expertise in the topic area
and produce higher quality labels than generalists.

**Quota sampling:**
Ensure the annotation set includes a representative distribution of:

- Relevant and non-relevant documents (roughly 20-30% relevant after pooling)
- Simple and complex queries
- Short and long documents
- High-confidence easy cases and low-confidence borderline cases

Without deliberate quota sampling, annotation sets can become dominated by
easy cases that do not adequately test system discrimination.

## Pooling for Large-Scale Evaluation

Annotation at scale requires pooling - only a subset of all possible query-document
pairs can be annotated, so which documents get annotated must be carefully chosen.

### TREC-style pooling

The standard protocol for creating large IR test collections:

```
Step 1: Define evaluation queries (50-250 topics)
Step 2: Submit runs from many systems (participants) for each query
Step 3: Pool: collect the top-K documents from each run for each query
         (K = 100 is typical, creating pools of 200-1000 unique documents per query)
Step 4: Annotate only the pooled documents (not the full corpus)
Step 5: Documents not in any run are assumed non-relevant (a simplifying assumption)
```

Pooling enables large-scale annotation without annotating every document.
The key risk is pool bias: systems that retrieve documents not included in
the pool cannot get credit for them, disadvantaging systems that are very
different from the pool contributors.

### Minimum pool depth

The pool must be deep enough to ensure high recall:

```
Pool depth K:  How many results per system to include in the pool
              Deeper pools = better coverage = more annotation cost

Finding optimal K:
  Sample 20 queries with deep pools (K=100)
  For each, compute how many additional relevant documents appear at depths 21-100
  If the "unjudged relevant" rate is low at K=20: pooling at K=20 is sufficient
  If many new relevants appear at depths 21-100: increase K
```

Modern practice often uses K=50 to K=100 for TREC-style evaluation, with
importance sampling to prioritize documents from diverse systems.

## Evaluating Annotation Quality

Before using annotation data to evaluate retrieval systems, validate the
annotation data itself:

### Internal consistency checks

**Gold standard items:** Include previously annotated "gold" items in every
annotator's queue and measure agreement with the gold standard. Annotators
falling below 75% agreement with gold standard are flagged for review.

**Repeated items:** Include a small fraction (5-10%) of items annotated twice
by the same annotator, separated by time. Consistency within an annotator
(intra-annotator agreement) should be ≥ 0.80 kappa.

**Implausible distribution check:** If an annotator assigns 90%+ of documents
as "highly relevant" or 90%+ as "not relevant," their labels require review -
real document sets should produce moderate distributions.

### Label distribution validation

The distribution of relevance labels in the final dataset should be examined:

```
Expected distribution for pooled annotation:
  Grade 0 (not relevant):        50-60%
  Grade 1 (marginally relevant): 15-25%
  Grade 2 (relevant):            15-25%
  Grade 3 (highly relevant):      5-15%

If distribution is extreme (e.g., 90% grade 0): pooling may have failed
If distribution is flat (25% at each level): very unusual, check guidelines
```

Validate distribution separately per annotator - systematic outliers in an
individual annotator's distribution indicate calibration problems.

## Reporting Standards for Evaluation Papers

When publishing evaluation benchmarks or system comparisons that rely on
human evaluation, the following should be reported:

### Required reporting

```
1. Relevance scale used (binary, 4-level, domain-specific)
2. Number of annotators per item
3. Annotator background (crowdworker, domain expert, etc.)
4. Inter-annotator agreement metric and value (Cohen's κ, Fleiss' κ, α)
5. Adjudication process for disagreements
6. Total number of annotated items
7. Query and document sources
```

### Best practice reporting

```
8. Annotation guidelines (available as supplementary material or appendix)
9. Annotator training procedure
10. Quality control measures
11. Label distribution
12. Difficult case categories and how they were resolved
13. Limitations (pool bias, domain coverage, annotator expertise level)
```

## Human Evaluation for Generative Retrieval

As RAG systems become more common, human evaluation protocols must extend
beyond document relevance to assess generated answer quality:

### Additional dimensions for RAG evaluation

**Factual accuracy:** Is the information in the generated answer correct?
Requires domain expert annotators who can verify claims.

**Attribution accuracy:** When the answer cites specific documents, are
the citations accurate? Does the cited document actually support the claim?

**Completeness:** Does the answer address all aspects of the question, or
does it miss important aspects?

**Conciseness:** Is the answer appropriately concise without omitting
important information?

### Side-by-side (SxS) evaluation for generative systems

For comparing two RAG systems, side-by-side evaluation is more reliable than
independent scoring:

```
Annotator sees:
  Query: {user question}
  System A response: {answer from System A with citations}
  System B response: {answer from System B with citations}

Judgment: Which system response is better?
  A is much better | A is somewhat better | Tie | B is somewhat better | B is much better
```

Side-by-side evaluation with 5-point preference scales produces cleaner signals
than independent 4-point relevance labels for comparing generative systems.

Human evaluation protocols close the evaluation advanced module by addressing
the human foundation that all other evaluation approaches rest on. LLM-as-judge
calibration requires human labels to measure against. A/B tests must be validated
against human preference studies. Statistical significance is only meaningful if
the underlying metric was measured reliably. Human evaluation protocols are what
make these foundations trustworthy.

## My Summary

Human evaluation protocols are structured procedures that turn the subjective
concept of relevance into a reproducible, measurable annotation process. The core
components are: a clearly defined relevance scale (binary for simplicity, 4-level
graded for NDCG computation, domain-specific for specialized domains), annotation
guidelines with worked examples and explicit decision rules for edge cases,
inter-annotator agreement measurement (Cohen's kappa targeting ≥ 0.60 for binary,
≥ 0.50 for graded), and adjudication procedures for resolving disagreements.
Annotator recruitment must match domain requirements - crowdworkers for general
web search, domain experts for medical or legal annotation. Training follows a
standard pipeline of guidelines reading, practice set with feedback, calibration
batch with IAA measurement, and continuous quality monitoring in production.
TREC-style pooling makes large-scale annotation tractable by annotating only
documents retrieved by submitted systems rather than the full corpus, at the cost
of pool bias against novel systems. Reporting standards require documenting the
relevance scale, annotator background, IAA values, adjudication process, and
sample counts so evaluation consumers can assess annotation reliability. For
generative RAG systems, protocols extend to factual accuracy, attribution quality,
and side-by-side preference evaluation in addition to traditional document relevance.
