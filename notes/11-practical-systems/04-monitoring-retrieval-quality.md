# Monitoring Retrieval Quality

Monitoring retrieval quality is the practice of continuously measuring how well a
deployed retrieval system is serving users after launch. Unlike offline evaluation
on fixed test collections, production monitoring must detect quality degradation,
distribution shift, and silent failures using a combination of automatic metrics,
user signals, and periodic human evaluation - all without ground truth relevance
labels for the vast majority of queries. It is the difference between a system that
was good at launch and a system that stays good over time.

## Intuition

A retrieval system that scored NDCG@10 = 0.48 on your offline benchmark may be
performing very differently in production. The benchmark queries were collected
months ago. User behavior has changed. New documents were added that have different
characteristics. The query distribution shifted. The embedding model was updated.
The index grew to ten times its original size. Any of these can silently degrade
quality without triggering any error or alert.

Without monitoring, you discover quality problems through user complaints, support
tickets, or declining engagement metrics - all lagging indicators that arrive weeks
after the problem began. With monitoring, you detect quality degradation within hours
of its onset and can respond before users notice.

The challenge is that unlike model inference monitoring (where you can detect error
spikes or latency increases instantly), retrieval quality monitoring requires
measuring something inherently subjective - relevance - at scale, without asking a
human to judge every single query.

## What to Monitor

### 1. System health metrics (operational)

These are standard infrastructure metrics that detect technical failures:

| Metric                         | Alert threshold         |
| ------------------------------ | ----------------------- |
| Query latency (p50, p95, p99)  | > 2x baseline           |
| Query error rate               | > 1% of queries         |
| Index size                     | > 90% of capacity       |
| Embedding service availability | < 99.9% uptime          |
| Memory usage                   | > 80% of limit          |
| Throughput (QPS)               | > 90% of capacity limit |
| Cache hit rate                 | < expected baseline     |

These are the easiest metrics to collect and the most urgent to alert on. A
retrieval system that crashes or times out is worse than one with degraded quality.

### 2. Retrieval quality metrics (semantic)

These require more sophisticated measurement but directly reflect user experience:

**Zero-result rate**
Fraction of queries that return zero results. Should be near zero for most systems.

```
zero_result_rate = count(queries with 0 results) / total_queries
Alert if: zero_result_rate > 0.5%
```

**Low-confidence rate**
Fraction of queries where the top result score falls below a threshold:

```
low_confidence_rate = count(queries where top_score < threshold) / total_queries
Alert if: low_confidence_rate increases by > 20% week-over-week
```

**Score distribution shift**
Track the distribution of retrieval scores over time. A significant shift indicates
the index or model behavior has changed:

```
Monitor: mean, p25, p75, p95 of top-1 retrieval scores
Alert if: mean drops by > 2 standard deviations from rolling baseline
```

**Result diversity**
Measure the average pairwise similarity among top-k results. If all returned results
are nearly identical, the index may have a coverage problem:

```
diversity = 1 - mean(pairwise_cosine_similarity(top_k_results))
Alert if: diversity drops below historical baseline
```

### 3. User behavior signals (implicit feedback)

User actions are noisy but abundant proxies for relevance:

**Click-through rate (CTR)**
Fraction of searches where the user clicks at least one result:

```
CTR = count(searches with at least one click) / total_searches
Alert if: CTR drops > 10% week-over-week
```

**Position of first click (PFC)**
Average rank position of the first clicked result. Lower is better:

```
PFC = mean(rank of first click across all clicked searches)
Alert if: PFC increases significantly (users are scrolling further to find relevant results)
```

**Query reformulation rate**
Fraction of queries followed by a similar modified query within 60 seconds.
High reformulation indicates the initial results were unsatisfying:

```
reformulation_rate = count(queries followed by reformulation) / total_queries
Alert if: reformulation_rate > 20% or increases significantly
```

**Dwell time**
Time spent on a clicked result before returning to search. Long dwell time
(> 30 seconds) suggests the result was useful. Short dwell time (< 5 seconds)
suggests a bad result (pogo-sticking):

```
pogo_stick_rate = count(clicks with dwell_time < 5s) / total_clicks
Alert if: pogo_stick_rate > 30% or increases significantly
```

**Zero-click rate**
Fraction of searches where the user does not click anything and abandons:

```
zero_click_rate = count(searches with no click and no reformulation) / total_searches
Alert if: zero_click_rate > 40% or increases significantly
```

### 4. Sampled human evaluation (ground truth)

All implicit signals are noisy. Periodic human evaluation on a sample of real
production queries provides ground truth:

```
Cadence:    Weekly or monthly sample evaluation
Sample size: 50-200 queries per evaluation round
Process:
  1. Sample queries from production logs
  2. Have annotators rate top-5 results for each query
  3. Compute NDCG@5 on sampled queries
  4. Compare to previous evaluation round
```

This is expensive but essential - implicit signals can be misleading. CTR can stay
high even when quality degrades if users click out of desperation rather than
satisfaction.

## Detecting Distribution Shift

Query distribution shift is one of the most common causes of silent quality
degradation. The system was evaluated on one distribution but now receives a
different one:

### Query distribution shift

Monitor the distribution of query characteristics over time:

```
Features to track:
  - Average query length
  - Fraction of queries with named entities
  - Fraction of queries with code/technical terms
  - Fraction of queries by language
  - Top-N most frequent queries (changes indicate new topics)
```

Alert when these distributions shift beyond historical variance. A sudden increase
in average query length or a new dominant query pattern may indicate a new user
segment or use case that the current index does not serve well.

### Document distribution shift

Monitor characteristics of the document corpus as it changes:

```
Features to track:
  - Total document count
  - Document length distribution
  - Average embedding norm (indicates potential model mismatch)
  - Topic distribution of new vs existing documents
```

Alert when new documents have significantly different characteristics from the
training distribution of the embedding model.

### Embedding drift

If the embedding model is updated, the query vectors and document vectors may
no longer be comparable:

```
Symptom: sudden drop in retrieval scores across all queries
Cause:   query encoder updated but document index not rebuilt
Fix:     rebuild entire index with new embedding model
```

Monitor for mismatched model versions between query encoder and document encoder.

## Retrieval Quality Dashboards

### Key metrics to display

```
Panel 1 - System Health (real-time)
  Query latency p50, p95, p99 (time series)
  Error rate (time series)
  Throughput QPS (time series)
  Cache hit rate (gauge)

Panel 2 - Retrieval Quality (hourly/daily)
  Zero-result rate (time series)
  Low-confidence rate (time series)
  Score distribution (histogram, rolling)
  Result diversity score (time series)

Panel 3 - User Signals (daily/weekly)
  CTR by position (position bias curve)
  Query reformulation rate (time series)
  Dwell time distribution (histogram)
  Zero-click rate (time series)

Panel 4 - Distribution Health (weekly)
  Query length distribution (histogram)
  Top query terms heatmap
  Document corpus growth rate
  Embedding score drift (rolling mean)
```

## A/B Testing Retrieval Changes

Before deploying changes to the retrieval stack in production, run an A/B test:

```
Traffic split:
  Control:  50% of users → current retrieval system
  Treatment: 50% of users → new retrieval system (e.g. added reranker)

Duration:    1-2 weeks minimum (capture weekly patterns)
Sample size: 1000+ queries per variant for statistical significance

Metrics to compare:
  Primary:   CTR, query reformulation rate, dwell time
  Secondary: p95 latency, system error rate
  Guardrail: zero-click rate must not increase
```

A/B testing catches cases where offline NDCG improvement does not translate to
real user improvement - and vice versa.

## Monitoring Stack Recommendations

| Stack              | Tools                  | Use case                    |
| ------------------ | ---------------------- | --------------------------- |
| Simple             | Python logging + CSV   | Prototype, < 1K QPS         |
| Medium             | Prometheus + Grafana   | Self-hosted production      |
| Managed            | Datadog or New Relic   | Team without DevOps         |
| Full observability | OpenTelemetry + Jaeger | Distributed system tracing  |
| ML-specific        | Evidently AI + MLflow  | ML model monitoring         |
| Enterprise         | Elastic Observability  | If already on Elasticsearch |

## The Monitoring Flywheel

Monitoring is not just about detecting problems - it creates a feedback loop
that continuously improves the system:

```
Deploy retrieval system
    ↓
Monitor quality metrics (automatic, continuous)
    ↓
Detect quality issues (via alerts or scheduled review)
    ↓
Investigate root cause (query analysis, error sampling)
    ↓
Improve system (new model, better chunking, query preprocessing)
    ↓
A/B test improvement
    ↓
Validate with sampled human evaluation
    ↓
Deploy to production
    ↓ (back to top)
```

Teams that close this loop weekly improve retrieval quality faster than
teams that only look at metrics quarterly - even if their retrieval algorithms
are initially identical.

## My Summary

Monitoring retrieval quality requires tracking three layers of signals: operational
health metrics (latency, error rate, throughput), automatic retrieval quality metrics
(zero-result rate, score distributions, result diversity), and user behavior signals
(CTR, reformulation rate, dwell time, position of first click). Each layer has
different collection costs and signal quality - operational metrics are free and
immediate, user signals are abundant but noisy, and human annotation is expensive
but ground truth. Alert on anomalies by comparing current metrics against rolling
baselines rather than fixed thresholds - distributions shift over time and thresholds
go stale. A/B testing is essential for validating retrieval changes before full
deployment since offline NDCG improvement does not always translate to production
user improvement. Sampled human evaluation on production queries at regular intervals
provides the ground truth signal that all other metrics are approximating. The
monitoring flywheel - detect, investigate, improve, validate, deploy - is what
separates retrieval systems that improve over time from those that silently degrade.
