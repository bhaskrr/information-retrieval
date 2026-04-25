# Online Evaluation

Online evaluation measures retrieval system quality using real users interacting
with a live system - observing actual behavior rather than comparing against
pre-collected relevance judgments. Where offline evaluation (precision, recall,
NDCG) measures how well a system matches human annotations on a fixed test
collection, online evaluation measures whether real users find the system
useful, complete their tasks, and return. It is the complement to offline
evaluation: offline evaluation is reproducible and cheap; online evaluation
is noisy and expensive but measures what actually matters - user satisfaction
in the real world.

## Intuition

Offline evaluation has a fundamental gap: it measures agreement with a fixed
set of relevance judgments, not actual user satisfaction. A system can score
well on NDCG@10 while being frustrating to use, and a system can feel excellent
to users while scoring lower than a competitor on a benchmark.

Consider: a retrieval system returns the same ten relevant documents as the
annotated ground truth but displays them in a confusing layout, with slow load
times, and no snippet preview. NDCG@10 = 1.0. User satisfaction = low.

Or: a system returns nine of ten relevant documents plus one borderline case
that happens to be the exact format the user needs. NDCG@10 drops slightly.
User clicks immediately and task completes. User satisfaction = high.

Online evaluation captures this gap by observing real behavior. When users
click, dwell, return, and complete tasks - these are signals that offline
metrics cannot provide. Online evaluation is how production search teams at
Google, Bing, Amazon, and every major search system actually decide which
system is better.

## The Two Core Methods

### A/B Testing

Split traffic randomly between two systems - Control (A) and Treatment (B).
Measure behavioral metrics for each group. Determine whether the difference
is statistically significant.

```
Traffic split:
  50% of users → System A (control, current system)
  50% of users → System B (treatment, new system)

Measure per group:
  Click-through rate, session length, task completion, return visits

Determine winner:
  Conduct statistical significance test
  If B significantly better → ship B
  If no significant difference → ship A (conservative) or B (if cheaper)
  If B significantly worse → reject B
```

### Interleaving

Present results from both systems in a single ranked list and observe which
system's results users prefer through their clicking behavior. More statistically
efficient than A/B testing - detects differences with fewer users.

```
System A ranking: [D1, D2, D3, D4, D5]
System B ranking: [D2, D4, D1, D5, D3]

Interleaved list: [D1(A), D2(B), D3(A), D4(B), D5(A)]

User clicks D2 and D4 → B wins for this query
User clicks D1 and D3 → A wins for this query

Aggregate win rates across many queries → determine winner
```

## Behavioral Metrics - What to Measure

### Click-Through Rate (CTR)

The fraction of queries that result in at least one click:

```
CTR = clicks / impressions
```

High CTR indicates results look relevant from the snippet preview. Low CTR
suggests poor result quality or poor presentation (title, snippet, URL).

Limitations: CTR measures intent to engage, not actual satisfaction. Users
click on results that look relevant but may abandon after reading. CTR is
easily gamed - sensationalist titles increase CTR without improving satisfaction.

### Click Position Distribution

Which rank positions users click on most often reveals both user behavior
patterns and system quality:

```
Healthy distribution: most clicks at rank 1-3, declining at lower ranks
Quality signal: if many clicks at rank 5+ → relevant results not surfacing early
```

### Time to First Click (T2FC)

How long after seeing results does the user click? Short T2FC suggests the
first result is immediately obviously relevant:

```
T2FC < 1s   → user found what they needed immediately
T2FC 1-5s   → user scanned results, found relevant one
T2FC > 10s  → user struggled to find a relevant result
```

### Dwell Time

How long does the user spend on the clicked page before returning to results?
Long dwell time suggests the clicked document was genuinely useful:

```
Dwell time < 5s  → likely a bad click (document was not what they expected)
Dwell time > 30s → user engaged with content, likely satisfied
```

Bounce rate (immediate return with no dwell) is a strong negative signal.

### Session Abandonment

Did the user abandon the session without clicking anything? High abandonment
suggests results were completely irrelevant or the query was not understood:

```
Abandonment = sessions with 0 clicks / total sessions
```

A sudden increase in abandonment rate is a strong quality regression signal.

### Query Reformulation Rate

How often do users immediately rephrase their query after seeing results?
High reformulation rate suggests the system did not understand or satisfy
the original query:

```
Reformulation = queries immediately followed by another query on same topic
              / total queries
```

### Long Clicks

A "long click" is a click followed by a long dwell time (e.g. > 30s) without
returning to results. Long clicks are the strongest positive signal available
from click data - they indicate the user found what they were looking for and
stayed on the result page.

### Return Rate

Do users return to the system within a short window (1 hour, 1 day)?
High return rate indicates user trust and satisfaction:

```
Return rate = users who return within 24h / total users
```

### Task Completion Rate

For systems with identifiable task completion signals (purchase, sign-up,
download), the fraction of sessions that complete the task is the most
direct satisfaction metric.

## A/B Testing - Implementation Details

### Traffic assignment

Random assignment based on a stable hash of user identifier:

```python
import hashlib

def assign_variant(user_id: str, experiment_id: str) -> str:
    hash_val = int(
        hashlib.md5(f"{user_id}:{experiment_id}".encode()).hexdigest(),
        16
    )
    return "control" if hash_val % 2 == 0 else "treatment"
```

Stable hashing ensures the same user always sees the same variant - preventing
users from experiencing both systems and getting confused.

### Sample size calculation

Before running an experiment, determine how many users are needed to detect
a meaningful difference with sufficient statistical power:

```
n = (z_α/2 + z_β)² × 2σ² / δ²
```

Where:

- z_α/2 = critical value for significance level α (1.96 for α=0.05)
- z_β = critical value for power (0.84 for 80% power)
- σ² = variance of the metric
- δ = minimum detectable effect (the smallest improvement worth detecting)

For CTR experiments, δ is typically 0.5-2% (0.005-0.02 absolute change).

### Running duration

Run experiments long enough to capture:

- Day-of-week effects (behavior varies Monday vs Saturday)
- Novelty effects (users explore a new system initially then settle)
- Minimum: 1-2 weeks for general search
- Longer: 4 weeks for low-traffic systems or small expected effects

### Novelty effect

Users often click more on a new system simply because it looks different,
not because it is better. The novelty effect fades after a few days. Always
look at metric trends over time - a genuine improvement maintains its
advantage as novelty fades.

## Interleaving - Implementation Details

### Team Draft Interleaving (TDI)

The most widely used interleaving method. Documents are alternately picked
from each system's ranked list in a "draft" fashion:

```
System A ranking: [D1, D2, D3, D4, D5]
System B ranking: [D2, D1, D4, D3, D5]

Round 1: A picks D1 (not yet in list), B picks D2 (not yet in list)
         Interleaved list: [D1(A), D2(B)]

Round 2: A picks D3 (D1 already taken), B picks D4 (D2 already taken)
         Interleaved list: [D1(A), D2(B), D3(A), D4(B)]

Round 3: A picks D5, B picks D5 (same doc, assign to first system = A)
         Interleaved list: [D1(A), D2(B), D3(A), D4(B), D5(A)]

User clicks D2 and D4 → B credit: 2, A credit: 0 → B wins
```

Winning criterion: the system whose documents accumulate more clicks wins
the query. Aggregate wins across all queries to determine the experiment winner.

### Balanced Interleaving (BI)

A variant that randomizes which system picks first to avoid position bias:

```
50% of queries: A picks first
50% of queries: B picks first
```

Prevents position bias from systematically favoring one system.

### Probabilistic Interleaving (PI)

Assigns documents probabilistically using a softmax of scores rather than
deterministic rank order. More robust to near-tied documents in rankings.

### Advantages of interleaving over A/B testing

```
Property               A/B Testing          Interleaving
────────────────────────────────────────────────────────────────
Users needed           Large (thousands)    Small (hundreds)
Detection speed        Slow (weeks)         Fast (days)
Position bias          Absent               Present (managed)
System interaction     None                 Users see both
Metric coverage        Any metric           Click-based only
Implementation         Simple               Moderate
```

Interleaving is 10-100x more statistically efficient than A/B testing for
detecting ranking quality differences. Netflix, Bing, and Yandex have published
results showing interleaving detects changes that would require months of A/B
testing in days of interleaving.

## Click Models - Understanding Biased Clicks

Raw click data is biased. Users are more likely to click on higher-ranked
results simply because they are higher - not because they are more relevant.
Click models account for this position bias to extract unbiased relevance signals.

### Position bias

The probability of clicking decreases with rank position regardless of relevance:

```
P(click at rank 1) = 0.35
P(click at rank 3) = 0.12
P(click at rank 5) = 0.06
P(click at rank 10) = 0.02
```

A document at rank 1 receives 17x more clicks than the same document at rank 10.

### The Examination Hypothesis

A user clicks a document if and only if they examine it AND find it relevant:

```
P(click | document d at rank r) = P(examine rank r) × P(relevant | d)
```

This factorization is the foundation of most click models.

### Cascade Model

The simplest click model. Users scan results top-to-bottom and click the first
relevant document:

```
P(examine rank r) = Π P(not click at rank i)  for i < r
                  = Π (1 - P(click at rank i))
```

After a click, the user stops. Fails to model skip behavior (examining but not
clicking a relevant document to look for a better one).

### DBN - Dynamic Bayesian Network Model

Models examination, click, and satisfaction as latent variables:

```
P(click at rank r) = P(examine r) × P(attracted by snippet)
P(continue after click) = P(not satisfied with clicked document)
```

Learns attraction (probability of clicking given examination) and satisfaction
(probability of stopping after clicking) separately. More realistic than cascade.

### Position-Based Model (PBM)

Simple factorization that learns position-specific examination probabilities:

```
P(click | d, r) = P(examine | r) × P(relevant | d)
```

The examination probability γ_r is learned from click data and applied to
de-bias relevance estimates. Widely used in production because it is simple
and effective.

## Implicit Feedback for Training

Beyond evaluation, click signals are used to train retrieval models:

### Click-through data as weak supervision

A query-document pair that received a click is a weak positive signal:

```
(query, clicked_document)  → positive training example
(query, shown_but_not_clicked_document) → negative training example
```

Used to train LambdaMART and neural retrieval models when human relevance
judgments are scarce or expensive. MS MARCO was itself constructed from Bing
click logs.

### Pairwise click preferences

A clicked document is preferred over a document shown above it that was not
clicked (skip signal):

```
User sees: [D1, D2, D3]
User clicks: D3, not D1 or D2
→ D3 > D1 and D3 > D2 as pairwise training signal
```

Skipped over documents provide a stronger negative signal than unexamined ones.

### Session-level signals

Multi-query sessions where a user reformulates after clicking provide rich signal:

```
Query 1: "dense retrieval"
→ clicks D3, dwell 8 seconds, returns
Query 2: "dense retrieval vs BM25"   ← reformulation suggests D3 did not satisfy
→ clicks D7, dwell 47 seconds, leaves session
→ D7 on query 2 = satisfied result
```

## Online vs Offline Evaluation - When to Use Which

```
Situation                          Use offline evaluation
────────────────────────────────────────────────────────────────────────
Comparing retrieval models         Yes - reproducible, fast, cheap
Debugging quality regressions      Yes - trace back to specific queries
Academic research                  Yes - standard benchmarks, comparable
Early-stage development            Yes - fast iteration without live traffic
Measuring model improvements       Yes - NDCG, MAP on labeled test set

Situation                          Use online evaluation
────────────────────────────────────────────────────────────────────────
Shipping a new system to users     Yes - must validate real user behavior
Measuring business impact          Yes - clicks, revenue, retention
Detecting UX/presentation effects  Yes - offline cannot capture these
Final decision before deployment   Yes - the ground truth for production
Long-term user satisfaction        Yes - return rate, trust signals
```

The standard production workflow: iterate using offline evaluation, validate
with online evaluation before each major release, monitor behavioral metrics
continuously in production.

## The Offline-Online Correlation Problem

A persistent challenge: offline metrics (NDCG, MAP) correlate imperfectly
with online metrics (CTR, dwell time). A system that improves NDCG@10 by
3 points may or may not improve CTR significantly.

Known sources of correlation failure:

- Relevance annotations are static; user needs evolve
- Annotation granularity does not match click behavior
- Offline metrics ignore UX factors (page load, snippet quality)
- Position bias affects clicks but not offline evaluation
- Diversity matters online but standard offline metrics ignore it

**Practical advice**: use offline metrics to rank candidate systems quickly, but
always A/B test promising candidates before shipping. Treat offline evaluation
as necessary but not sufficient evidence of quality improvement.

Online evaluation is the final step in the evaluation module - the point where
everything measured offline is validated against actual user behavior. It is
also the bridge from evaluation to production: a system that passes both offline
and online evaluation is ready to deploy.

## My Summary

Online evaluation measures retrieval quality through real user behavior rather
than pre-collected relevance judgments. A/B testing splits traffic randomly
between systems and compares behavioral metrics - CTR, dwell time, abandonment
rate, long click rate - using statistical significance tests to determine
winners. Interleaving presents results from both systems in a single ranked list
and observes click preferences, detecting differences with 10-100x fewer users
than A/B testing. Behavioral metrics are biased by position - the position-based
click model and other click models de-bias raw click signals to produce cleaner
relevance estimates. The offline-online correlation problem means that offline
NDCG improvements do not always translate to online improvements: always A/B
test promising candidates before deployment. In production, offline evaluation
guides iteration speed and online evaluation validates before shipping - both
are necessary, neither is sufficient alone.
