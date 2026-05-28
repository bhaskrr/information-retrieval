# Online Evaluation and A/B Testing

Online evaluation is the practice of measuring retrieval system quality using
signals from real user interactions with a live system - clicks, dwell time,
reformulations, and other behavioral indicators - rather than static relevance
judgments assigned by human annotators on fixed query sets. A/B testing is the
primary framework for online evaluation: it splits live traffic between two
system variants, collects user interaction signals for both, and applies
statistical hypothesis testing to determine whether one variant genuinely
produces better user outcomes. Together, online evaluation and A/B testing
provide the ground truth that offline benchmarks can only approximate - actual
user satisfaction with actual queries on an actual production system. They are
the final validation step before any major retrieval system change is deployed
at full scale.

## Intuition

Offline evaluation measures what expert annotators think is relevant. Online
evaluation measures what users actually do.

These are meaningfully different. An annotator judges a document as highly
relevant to "python tutorial" because it is a well-written, accurate Python
beginner guide. But if users consistently skip it and click the third result
instead - a more interactive tutorial with runnable code examples - then the
annotator's judgment and the user's actual preference diverge. Online evaluation
captures the user preference. Offline evaluation captures the annotator's
judgment.

Neither is strictly better. Offline evaluation with expert annotators is the
standard for systematic benchmarking - it is reproducible, shareable, and
enables controlled comparisons. Online evaluation reveals preferences that
annotations miss - interactivity, page layout, loading speed, recency, trust
signals - that affect real user satisfaction but are not captured in text
relevance labels.

A/B testing is how you bridge the gap. You have measured a 2% NDCG@10
improvement offline. Is this real? Does it translate to better user outcomes?
Run an A/B test on production traffic, measure the behavioral signals, apply
significance testing, and find out. The offline gain is a hypothesis. The A/B
test result is the answer.

## Why Offline Evaluation Alone Is Insufficient

### The annotation gap

Relevance judgments are created by human annotators who read documents and
assess their relevance to queries. This process has systematic biases:

**Annotator expertise mismatch:** A non-expert annotator may judge a general
overview as highly relevant when an expert user wants a primary research paper.
The annotator's notion of relevance may not match the user's.

**Static judgments for dynamic needs:** Relevance is not static. "Best Python
framework" has different answers in 2019 vs 2024. Offline test collections age -
a document that was highly relevant when annotated may be outdated by the time
a new system is evaluated.

**No behavioral signals:** Annotators cannot assess how engaging, trustworthy,
or actionable a document actually is for users. They see only text.

### Goodhart's Law in IR evaluation

Once a team knows what metric they are being evaluated on, they optimize for
that metric, which may diverge from the true goal. A retrieval system optimized
for NDCG@10 on a fixed test collection can learn to do well on that specific
collection without improving actual search quality.

Online evaluation provides a moving target that is harder to overfit - users'
actual needs are diverse and unpredictable in ways that test collections are not.

### The reproducibility-realism tradeoff

```
Offline evaluation:    Reproducible, controlled, shareable, standardized
                       → Ideal for research comparison and development
                       → May not reflect production query distribution

Online evaluation:     Realistic, captures true user preferences
                       → Ideal for deployment validation
                       → Harder to reproduce or share (data privacy)
                       → Requires live traffic (not available during development)
```

A mature IR team uses both: offline evaluation to guide development decisions
and compare systems systematically, online evaluation to validate before
deployment and continuously monitor after.

## Implicit Feedback Signals

Online evaluation relies on implicit feedback - signals inferred from user
behavior rather than explicitly provided by users. Users rarely rate search
results directly, but their behavioral signals carry relevance information.

### Click-through rate (CTR)

Fraction of searches where a specific document was clicked:

```
CTR(document, query) = clicks(document, query) / impressions(document, query)
```

CTR is the most abundant implicit signal and correlates with relevance - users
click results that appear relevant. But CTR is heavily confounded by position
bias: users disproportionately click results at rank 1-3 regardless of quality.
Raw CTR is a noisy relevance signal without position bias correction.

### Dwell time

How long a user spends on a clicked result before returning to search:

```
Short dwell time (< 5 seconds):  Pogo-sticking - document was not useful
Long dwell time (> 30 seconds):  Document was engaged with - likely useful
```

Dwell time is more reliable than CTR as a quality signal because it measures
engagement rather than just initial attraction. A misleading title may attract
a click (high CTR) but a bad document quickly returns users to search (low dwell).

### Session abandonment and abandonment type

```
Successful abandonment:  User queries, clicks, reads, then leaves
                         (task completed - good retrieval outcome)

Frustrated abandonment:  User queries, gets poor results, reformulates multiple
                         times, then leaves without reading anything
                         (retrieval failed to serve the need)
```

Distinguishing between successful and frustrated abandonment requires modeling
session context, not just individual clicks.

### Query reformulation rate

Fraction of searches followed by a modified query within a short time window
(typically 60 seconds):

```
High reformulation rate → users are not satisfied with results → bad retrieval
Low reformulation rate  → users found what they needed → good retrieval
```

Query reformulations are strong negative signals - users explicitly indicate
that the initial results were insufficient by trying again.

### SERP abandonment (zero-click rate)

Fraction of searches where the user views results but does not click anything
and does not reformulate - they simply leave. This is one of the strongest
signals of complete retrieval failure.

## Position Bias and Debiasing

The central challenge in online evaluation: users click results that appear
at higher positions regardless of their relevance. A document moved from rank
5 to rank 1 gets more clicks even if nothing else changes. This makes raw
click signals systematically misleading as relevance indicators.

### The position bias mechanism

```
User behavior model (simplified):
  Probability of examining position k:  P(exam | k) = 1 / k^η
  Probability of clicking if examined:  P(click | exam, relevant) = P(relevance)

  Observable:  CTR(k) = P(exam | k) × P(relevance at k)
  Want:        P(relevance at k)  ← true relevance signal
  Have:        P(exam | k)        ← confound we need to remove
```

Raw CTR conflates examination probability (position effect) with relevance.
Debiasing separates them.

### Inverse Propensity Weighting (IPW)

The standard approach for click debiasing. Weight each click by the inverse
probability of being examined at that position:

```
Unbiased relevance estimate:
  relevance(doc, position) = click(doc, position) / P(exam | position)

Where P(exam | position) is estimated from:
  Random result shuffling experiments
  Position-based click models (PBM)
  Intervention harvesting
```

IPW corrects for position bias but introduces variance - rare positions have
high inverse weights, amplifying noise.

### Result Randomization

A direct approach: randomly shuffle the order of top results for a small
fraction of users and observe how clicks change with position:

```
For 1% of users: serve randomly ordered results
For 99% of users: serve standard ranked results

From the randomized subset:
  Estimate click probability at each position for each document
  → This gives P(click | position, document) under random ordering
  → Remove position effect by normalizing
```

Result randomization is expensive (1% of users get intentionally bad rankings)
but provides clean click propensity estimates.

### Interleaving

A highly efficient alternative to A/B testing for comparing two retrieval
systems using implicit feedback. Instead of splitting traffic, interleaving
serves both systems simultaneously to every user:

```
Standard A/B:
  User group A: sees results from System A
  User group B: sees results from System B

Interleaving (Team Draft):
  For each user: combine results from System A and System B
  Randomly assign System A to odd positions, System B to even (or vice versa)
  Observe which system's documents get more clicks
  The system whose results get more clicks wins

Advantages over A/B:
  Requires 10-100x fewer users for same statistical power
  No between-user variance (same users see both systems)
  Faster to reach significance
```

### Team Draft Interleaving

The standard interleaving algorithm:

```
Create interleaved list from System A (ranked list a₁, a₂, ...) and System B (b₁, b₂, ...):

Step 1: Randomly choose which system starts (A or B)
Step 2: Alternate: add top-ranked document from chosen system not yet in list
Step 3: Continue until list is full

Result: interleaved list where both systems have representatives
        at approximately equal positions on average

Click attribution:
  Each click attributed to the system that ranked that document higher
  If both ranked it equally: split credit

Winner: system whose attributed clicks > other system's attributed clicks
```

Team Draft Interleaving is unbiased with respect to position because both
systems contribute documents to approximately the same rank positions on average.

## A/B Testing Framework for IR

The full A/B testing pipeline for deploying a retrieval system change:

### Step 1 - Define the experiment

Before running:

```
Hypothesis:    New bi-encoder + reranker pipeline improves user satisfaction
               over current BM25-only system

Primary metric:    CTR (or dwell-time-weighted CTR, reformulation rate)
Secondary metrics: Session length, page views, conversion rate
Guardrail metrics: Query error rate, p99 latency (must not degrade)

Traffic split:  50/50 control/treatment
Minimum duration: 2 weeks (capture weekly patterns)
Minimum sample:   Computed from power analysis for expected effect size
```

### Step 2 - Assignment mechanism

User or session-level assignment, not query-level:

```
Query-level assignment:     Same user sees different systems for different queries
                            → Cross-contamination: user learns system A is better
                            → Session context breaks

User-level assignment:      Each user consistently sees one system
                            → Clean comparison
                            → Stable user experience

Session-level assignment:   Compromise: one system per session
                            → Acceptable for most search systems
                            → Avoids within-session confusion
```

Assignment is hash-based for reproducibility and even splitting:

```
user_hash = MD5(experiment_name + user_id)
variant   = "treatment" if user_hash % 100 < 50 else "control"
```

### Step 3 - Collect and aggregate metrics

Per-query metrics collected in production:

```
For each search session (user, query, result_list, clicks, times):
  ndcg_implicit = estimate_ndcg_from_clicks(result_list, clicks)
  ctr           = len(clicks) / len(result_list)
  reformulated  = 1 if next_query_within_60s else 0
  dwell_time    = sum(time_on_page for clicked_result)
  abandoned     = 1 if len(clicks) == 0 and no_reformulation else 0
```

### Step 4 - Statistical testing

Apply the significance tests from the previous note (01-statistical-significance):

```
For each metric M:
  control_scores   = [M(q) for q in control_queries]
  treatment_scores = [M(q) for q in treatment_queries]
  result           = randomization_test(control_scores, treatment_scores)

Report:
  Primary metric: p-value, effect size, 95% confidence interval
  Secondary metrics: p-values with multiple testing correction
  Guardrails: confirm no significant degradation
```

### Step 5 - Decision

```
Deploy treatment if:
  Primary metric significantly improved (p < 0.05)
  AND effect size ≥ minimum meaningful threshold (defined by product team)
  AND no guardrail metrics significantly degraded
  AND treatment has been running for minimum duration (capture weekly cycles)

Hold / iterate if:
  Primary metric not significant
  OR guardrail degraded
  OR insufficient data collected

Rollback if:
  Guardrail metrics significantly worsened (error rate, latency)
```

## Metrics for IR A/B Testing

### Direct relevance proxies

**CTR@1 (first-click rate):** Did the user click the first result? Most
sensitive metric for ranking quality - good ranking puts the best document first.

**DCG from clicks:** Approximate DCG using click positions as relevance signals:

```
DCG_clicks = Σ click(position) / log₂(position + 1)
```

**Dwell-time weighted CTR:**

```
satisfaction_score = Σ (click × min(dwell_time, 30) / 30)
                       ↑ weight clicks by reading engagement
```

### Session-level satisfaction

**SERP CTR:** At least one click in the session (task completion proxy)

**Last query CTR:** User clicked on the last query in a session before
ending (strong task completion signal - they found what they needed)

**Session reformulation rate:** Number of query reformulations per session
(lower is better - fewer reformulations means first results satisfied the need)

**Session length:** For informational queries, longer sessions indicate
more engagement with results (positive). For navigational queries, shorter
sessions indicate faster task completion (positive).

### Counter metrics

**Pogo-sticking rate:** Click followed immediately by return to SERP
(dwell time < 5 seconds). Strong negative quality signal.

**Rage clicks:** Multiple rapid clicks on the same position (frustration signal).

**No-result click rate:** Fraction of queries where zero results are clicked.
High no-result rate indicates the retrieval system fails to surface anything
worth clicking.

## Sample Size and Duration

### Minimum sample size

Use power analysis (from the previous note) with estimated effect size for
the online metric being measured:

```
Expected online effect: CTR improvement from 0.42 to 0.43 (+2.4% relative)
Standard deviation of per-query CTR: ~0.20

Required n (80% power, α=0.05):
  Cohen's d = (0.43 - 0.42) / 0.20 = 0.05
  n ≈ (1.96 + 0.84)² / 0.05² ≈ 3,136 queries per variant

Conservative recommendation: 2x required n = ~6,272 queries per variant
```

### Minimum duration

Even with sufficient sample size, A/B tests should run for a minimum duration
to capture temporal patterns:

```
Minimum: 1 week (capture weekly query distribution shift)
Standard: 2 weeks (two full weekly cycles)
Recommended: Until both time and sample requirements are met

Why duration matters:
  Day 1-2: novelty effect - users may behave differently with a new system
  Days 3-7: novelty wears off, behavior stabilizes
  Week 2:  weekly pattern (weekday vs weekend query mix) captured
```

### Common A/B testing mistakes in IR

**Peeking problem:** Checking results before the predetermined sample size is
reached and stopping early if results look significant. This inflates the
false positive rate dramatically.

```
Example: You peek at results every day. On day 4, p = 0.04. You stop.
Reality: If you ran 10 days, average p = 0.22 (not significant).
         You stopped at a false positive - the improvement was noise.
Fix:     Use sequential testing (O'Brien-Fleming) if early stopping is needed.
         Otherwise, commit to the predetermined sample size before starting.
```

**Network effects / contamination:** Users in control group interact with
users in treatment group, spreading the treatment effect.

```
Example: Treatment system improves result diversity. Users in treatment group
         share better search results on social media. Control group users
         see these shared results and benefit indirectly.
Fix:     For social or recommendation systems, use cluster-level assignment
         (assign groups of socially connected users to the same variant).
```

**Novelty / learning effects:** Users behave differently with a new system
initially, then adapt.

```
Example: New system uses a different UI pattern. Week 1 CTR drops (unfamiliar).
         Week 3 CTR improves (users adapt). A 1-week A/B test would show
         the new system as worse when it is actually better long-term.
Fix:     Run tests for at least 2 weeks. Analyze CTR over time within the test.
```

**Multiple comparison inflation:** Running many A/B tests simultaneously
and picking the best-performing one.

```
Example: Run 20 A/B tests of different system variants. The one with the
         best CTR improvement wins and gets deployed.
Reality: With α=0.05 and 20 tests, you expect 1 false positive by chance.
         You just deployed the one that got lucky.
Fix:     Apply Bonferroni correction to the significance threshold.
         Or accept a stricter per-test α (e.g., 0.005 with 10 simultaneous tests).
```

## Connecting Online and Offline Evaluation

The relationship between offline NDCG and online behavioral metrics is
empirically measurable and critically important for calibrating offline evaluation:

### NDCG-to-CTR correlation

Measure how well offline NDCG predicts online CTR improvement across many
historical A/B tests:

```
For each past A/B test:
  offline_improvement = NDCG_treatment - NDCG_control
  online_improvement  = CTR_treatment - CTR_control

Correlation:
  High correlation (r > 0.7): offline NDCG is a good proxy for user satisfaction
  Low correlation (r < 0.4):  offline NDCG does not reflect actual user needs
                               → need to revisit offline evaluation setup
```

A team that has run many A/B tests can plot offline NDCG improvements against
online CTR improvements and fit a regression line. This calibrates how much
offline NDCG improvement is "worth" in terms of expected online improvement -
and whether offline evaluation is even tracking the right signal.

### Surrogate metric validation

The goal of surrogate metric validation is to determine which offline metrics
most reliably predict online success:

```
Experiment: run 50 A/B tests, record offline and online metrics for each
Analysis:   which offline metric (NDCG@1, NDCG@5, NDCG@10, MRR, MAP, ...)
            best predicts online CTR improvement?
Result:     use that metric as the primary offline development signal
```

Different teams find different answers depending on their query distribution -
NDCG@1 may be most predictive for navigational search, NDCG@10 for informational.

## Continuous Online Evaluation

Beyond episodic A/B tests, production IR systems should have continuous
online quality monitoring:

### Rolling metric tracking

```
Daily:  CTR, reformulation rate, zero-click rate
        (detect sudden quality regressions from index or model changes)

Weekly: Distribution of session lengths, dwell times
        (detect gradual quality drift)

Monthly: Click position distribution
         (detect index freshness issues - do clicked documents appear higher?)
```

### Implicit NDCG from production logs

Estimate NDCG continuously from click logs without explicit relevance judgments:

```
For each query:
  Retrieved list: [D₁, D₂, D₃, D₄, D₅, ...]
  Clicks observed: user clicked D₃ (rank 3), D₁ (rank 1)

Implicit NDCG approximation:
  Treat clicks as binary relevance signals
  NDCG_implicit = DCG(click_positions) / ideal_DCG

Average over rolling 7-day window → online quality estimate
```

This is noisy (position bias, incomplete signals) but provides a continuous
quality signal that can detect regressions quickly.

Online evaluation and A/B testing close the loop that offline evaluation opens.
Offline evaluation says "System B appears to be 2% better on NDCG@10." Online
evaluation says "Yes, users prefer System B in production - CTR increased 1.8%
with p = 0.003." The combination of both gives the complete picture needed for
confident deployment decisions.

## My Summary

Online evaluation measures retrieval quality using real user behavioral signals -
clicks, dwell time, reformulations - rather than static relevance annotations.
A/B testing is the primary framework: split production traffic between control
and treatment variants, collect behavioral metrics for both, and apply significance
testing to determine whether the treatment genuinely improves user outcomes.
Implicit feedback signals include CTR (confounded by position bias), dwell time
(more reliable engagement signal), reformulation rate (strong negative quality
signal), and session abandonment type. Position bias - users' tendency to click
higher-ranked results regardless of relevance - must be corrected through inverse
propensity weighting, result randomization, or interleaving. Interleaving (Team
Draft) is 10-100x more sample-efficient than A/B testing for pairwise system
comparison because it eliminates between-user variance. Common A/B testing
mistakes include early stopping (peeking), contamination between user groups,
novelty effects that inflate or suppress short-term metrics, and multiple
comparison inflation from simultaneous tests. The relationship between offline
NDCG and online CTR should be empirically measured across historical A/B tests to
calibrate whether the offline metric actually predicts online user satisfaction -
this calibration is one of the most valuable investments an IR team can make in
their evaluation infrastructure.
