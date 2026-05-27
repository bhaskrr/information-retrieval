# Statistical Significance Testing

Statistical significance testing in IR is the practice of determining whether
an observed difference in retrieval quality between two systems - say, NDCG@10
of 0.452 for System A versus 0.441 for System B - is a genuine performance
difference or could plausibly have arisen by chance due to the particular set
of queries used in the evaluation. Without significance testing, every numerical
difference looks like a real improvement. With significance testing, you can
make principled claims about whether System A actually outperforms System B, or
whether the eleven-thousandth difference in NDCG is measurement noise from
having only evaluated on 50 queries. It is the difference between a result that
can be published or deployed with confidence and one that might reverse with a
different query sample.

## Intuition

Imagine you flip a coin 10 times and get 7 heads. Is the coin biased toward
heads? Maybe - or maybe you just had a lucky streak. With only 10 flips, it is
hard to know. Flip 1000 times and get 700 heads - now you can be confident the
coin is biased.

NDCG@10 evaluation works the same way. You measure System A vs System B on 50
queries. System A wins on 32, System B wins on 18. Is System A genuinely better,
or did it just happen to get lucky with this particular set of 50 queries? With
50 queries, the answer is genuinely uncertain. With 5000 queries, a 32/18 split
becomes overwhelmingly convincing.

Statistical significance testing formalizes this intuition. It asks: if the
two systems were actually equal (the null hypothesis), how likely would we be to
observe a performance difference at least as large as what we measured, purely
by chance? If that probability (the p-value) is very small - say, less than 0.05 -
we reject the hypothesis that the systems are equal and conclude the difference is real.

This matters enormously in IR research and deployment:

**In research:** Papers routinely report improvements of 0.5-2% NDCG. Without
significance testing on an appropriate number of queries, many of these reported
improvements are noise. SIGIR and ECIR require statistical significance testing
for empirical comparisons.

**In production:** You have deployed System A and want to know if System B is
genuinely better before migrating 100 million users. A significance test on your
evaluation set quantifies your confidence before making that call.

## The Null Hypothesis in IR Evaluation

For comparing two retrieval systems:

```
H₀ (null hypothesis):    System A and System B have the same mean performance
                          on the population of queries. Any observed difference
                          is due to random query sampling.

H₁ (alternative):        System A and System B have different mean performance.
                          The observed difference reflects a genuine population-level
                          difference.

Desired outcome:          Reject H₀ with p < 0.05 (or p < 0.01 for stronger claims)
                          → We can conclude System A ≠ System B
                          → Direction of the difference tells us which is better
```

The p-value is the probability of observing a difference at least as large as
the measured one, assuming H₀ is true. Small p-value → unlikely to be chance
→ reject H₀.

## Why Standard Tests Apply to IR

IR evaluation produces one metric value per query per system. For n queries and
two systems A and B:

```
System A scores: [a₁, a₂, ..., aₙ]   (NDCG@10 for each query)
System B scores: [b₁, b₂, ..., bₙ]   (NDCG@10 for each query)
Differences:     [d₁, d₂, ..., dₙ]   where dᵢ = aᵢ - bᵢ
```

We want to test whether the mean of the differences is significantly different
from zero. This is a paired comparison - the same queries are evaluated on both
systems, so per-query differences remove query difficulty as a confounding factor.
Query 47 is hard for both systems; its difficulty cancels out in the difference.

The pairing is what makes this more powerful than an unpaired test - it is the
same reason clinical trials compare treatment vs placebo within the same patient
population rather than separate populations.

## The Paired t-Test

The workhorse of IR significance testing. Assumes the per-query score
differences are approximately normally distributed.

### Formula

```
Null hypothesis: μ_d = 0  (mean difference is zero)

Test statistic:
  t = (d̄ × √n) / s_d

Where:
  d̄   = mean of per-query differences = (1/n) Σ dᵢ
  s_d  = standard deviation of differences = √( (1/(n-1)) Σ (dᵢ - d̄)² )
  n    = number of queries

Degrees of freedom: ν = n - 1
p-value: P(|T| ≥ |t|) where T ~ t-distribution(ν)
```

### Interpretation

```
If p < 0.05: reject H₀ at 5% significance level
             (less than 5% chance of seeing this difference by luck)

If p < 0.01: reject H₀ at 1% significance level
             (stronger evidence against H₀)

If p ≥ 0.05: fail to reject H₀
             (cannot conclude systems are different - NOT the same as "they are equal")
```

### When it is valid

The t-test assumes:

1. Per-query differences are approximately normally distributed
2. Observations (queries) are independent
3. Both systems evaluated on the same queries (paired)

NDCG differences are approximately normal for moderate n (≥ 25 queries) by the
Central Limit Theorem, even if individual NDCG scores are not normally distributed.
For very small query sets (< 25), the assumption is shaky.

### Common mistake - applying it unpaired

```
Wrong:   t_test(system_A_scores, system_B_scores)   ← unpaired
Correct: t_test(system_A_scores - system_B_scores)  ← paired
```

The unpaired version treats the two systems as independent samples, losing
the pairing benefit and dramatically reducing statistical power.

## Wilcoxon Signed-Rank Test

A non-parametric alternative to the paired t-test that does not assume normality.
More appropriate for small query sets or when NDCG distributions are heavily
skewed.

### How it works

Instead of using raw differences, Wilcoxon ranks the absolute differences and
uses the sign to determine direction:

```
Step 1: Compute differences dᵢ = aᵢ - bᵢ for each query
Step 2: Discard queries where dᵢ = 0 (ties)
Step 3: Rank |dᵢ| from smallest (rank 1) to largest (rank n)
Step 4: Assign signs: positive rank if dᵢ > 0, negative if dᵢ < 0
Step 5: Test statistic W = sum of positive ranks (or negative ranks)
Step 6: Compare W to critical values for Wilcoxon distribution
```

### When to prefer Wilcoxon over t-test

```
Use Wilcoxon when:
  → Small query sets (n < 25)
  → NDCG distribution shows heavy skew or outliers
  → You cannot justify the normality assumption
  → Distribution of score differences has heavy tails

Use t-test when:
  → Larger query sets (n ≥ 25)
  → Score differences approximately normal
  → You want a familiar, widely-understood test
  → Computational simplicity matters

In practice: run both and report whichever is appropriate for your
query set size. They usually agree for large n.
```

### Comparison: t-test vs Wilcoxon

```
Property              Paired t-test           Wilcoxon
────────────────────────────────────────────────────────────────
Assumption            Normality of diffs      Distribution-free
Power (n > 25)        Higher                  Slightly lower
Power (n < 25)        May be inflated         More reliable
Outlier sensitivity   High                    Low (uses ranks)
Interpretability      Easy (mean diff)        Moderate
Standard in IR        Yes                     Yes
```

## The Bootstrap Test

The bootstrap test is the most flexible and assumption-free significance test
for IR. It generates an empirical null distribution by resampling:

### How it works

```
Step 1: Compute observed metric difference:
  Δ_obs = mean_NDCG(system_A) - mean_NDCG(system_B)

Step 2: Generate bootstrap null distribution:
  Repeat 10,000 times:
    For each query i:
      With probability 0.5: swap system A and B scores for this query
    Compute Δ_bootstrap = mean(swapped_A) - mean(swapped_B)

Step 3: Compute p-value:
  p = count(|Δ_bootstrap| ≥ |Δ_obs|) / 10,000
```

The bootstrap simulates what the score distribution would look like if the
systems were actually equal (H₀ true) by randomly swapping labels. The p-value
is the fraction of bootstrap samples that produced a difference as extreme as
observed.

### Bootstrap vs t-test for IR

```
Bootstrap advantages:
  → No distributional assumptions
  → Handles any evaluation metric (NDCG, MAP, MRR - doesn't matter)
  → Works for small n
  → Accounts for correlation structure in the data

Bootstrap disadvantages:
  → Computationally expensive (10,000 repetitions)
  → Random seed dependence (results vary slightly between runs)
  → Less familiar to reviewers than t-test or Wilcoxon
```

For IR research, the bootstrap test is considered the gold standard when
normality cannot be assumed or sample sizes are small.

## The Randomization Test (Permutation Test)

Also called the swap randomization test or the Fischer randomization test.
The IR community's most rigorous significance test, advocated by Mark Sanderson
and other IR evaluation researchers.

### How it works

```
Step 1: Compute observed difference:
  Δ_obs = mean_NDCG(system_A) - mean_NDCG(system_B)

Step 2: Generate randomization distribution:
  Repeat B times (B = 10,000 to 100,000):
    For each query i, randomly assign:
      Either: use A score for system A, B score for system B (original)
      Or:     use B score for system A, A score for system B (swapped)
    Compute Δ_permuted = mean(assigned_A) - mean(assigned_B)

Step 3: p-value = count(|Δ_permuted| ≥ |Δ_obs|) / B
```

This is nearly identical to the bootstrap but is theoretically cleaner:
it directly models the hypothesis that assignments are exchangeable under H₀.

### Why IR researchers prefer it

The randomization test makes minimal assumptions about the data generating
process. Under H₀ (systems are equal), the labels "system A" and "system B"
are arbitrary - we could flip them for any query without changing the null
distribution. The test exploits this directly.

For TREC-style evaluation with 50-250 topics, the randomization test is
particularly appropriate because:

1. NDCG distributions are not normal (bounded by 0 and 1, often bimodal)
2. Topic sets are small
3. The test is distribution-free

## Effect Size - The Missing Piece

Significance tells you whether a difference is real. Effect size tells you
whether it matters.

A system improvement from NDCG@10 = 0.440 to NDCG@10 = 0.441 can be
statistically significant with 10,000 queries - but practically meaningless.
A system improvement from 0.440 to 0.460 with only 100 queries might not
reach significance - but if replicated, would be highly valuable.

### Cohen's d for IR

```
d = (mean_A - mean_B) / pooled_standard_deviation

Interpretation:
  d < 0.2:   small effect  (borderline practical significance)
  d = 0.2:   small effect
  d = 0.5:   medium effect
  d ≥ 0.8:   large effect  (clearly practically significant)
```

### IR-specific effect size interpretation

```
NDCG@10 difference    Practical significance
──────────────────────────────────────────────────────
< 0.005              Negligible (measurement noise territory)
0.005 - 0.010        Small (may matter for large-scale deployment)
0.010 - 0.020        Moderate (meaningful improvement)
0.020 - 0.050        Large (substantial improvement)
> 0.050              Very large (major architecture change)
```

Best practice: always report both p-value and effect size. A result can be:

- Significant and large effect → confident, meaningful improvement
- Significant and small effect → real but trivial improvement
- Not significant and large effect → promising but underpowered study
- Not significant and small effect → no meaningful difference

## Multiple Testing Correction

When comparing many systems simultaneously, the probability of false positives
inflates rapidly. If you compare 20 systems pairwise (190 comparisons) and use
p < 0.05 for each, you expect ~9.5 false positives by chance.

### Bonferroni correction

Most conservative. Divide the significance threshold by the number of comparisons:

```
Adjusted α = 0.05 / m   (where m = number of comparisons)

For 10 comparisons: α = 0.05/10 = 0.005
For 20 comparisons: α = 0.05/20 = 0.0025
```

Too conservative for large m - rejects many real effects.

### Holm-Bonferroni (step-down)

Less conservative than Bonferroni while controlling family-wise error rate:

```
Step 1: Sort p-values from smallest to largest: p₁ ≤ p₂ ≤ ... ≤ pₘ
Step 2: Reject null i if pᵢ ≤ α / (m - i + 1)
Step 3: Stop at the first non-rejection
```

### Benjamini-Hochberg (FDR control)

Controls the False Discovery Rate rather than family-wise error - better for
large-scale comparisons in IR research:

```
Step 1: Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
Step 2: Find largest k such that p_k ≤ (k/m) × α
Step 3: Reject nulls 1, 2, ..., k
```

For comparing many IR systems simultaneously (ablation studies, hyperparameter
search), BH-FDR control is the recommended approach.

## How Many Queries Do You Need?

Power analysis answers: given a desired significance level α and effect size δ,
how many queries n are needed to detect that effect with probability β (power)?

### Approximate formula for paired t-test

```
n ≈ (z_{α/2} + z_β)² × σ² / δ²

Where:
  z_{α/2} = 1.96 for α = 0.05 (two-tailed)
  z_β     = 0.84 for 80% power, 1.28 for 90% power
  σ       = standard deviation of per-query differences
  δ       = minimum detectable effect size (NDCG difference)
```

### Practical rules of thumb for IR evaluation

```
Effect size       Queries needed (80% power, α=0.05)
──────────────────────────────────────────────────────
0.020 NDCG        ~200 queries
0.010 NDCG        ~800 queries
0.005 NDCG        ~3,200 queries
0.002 NDCG        ~20,000 queries
```

These numbers explain why:

- Academic IR papers often use 50-250 TREC topics and can only reliably detect
  effects > 0.015 NDCG
- Industry papers with tens of thousands of queries can detect much smaller effects
- Many published IR improvements that appear to be 0.3-0.5% NDCG are likely noise
  on typical evaluation set sizes

### The evaluation set size problem in practice

```
TREC tracks: 50-250 topics → can reliably detect effects ≥ 0.015 NDCG
BEIR:        varies by dataset (6 to 7,437 queries)
Academic labs: typically 50-200 queries → underpowered for small effects
Industry:    1,000-100,000 queries → powered even for small effects
```

This creates an important asymmetry: small effects that are real but below the
detection threshold of academic evaluation sets may be significant and valuable
at industry scale. Academic papers may report "no significant difference" for
what is actually a meaningful improvement in production.

## Reporting Best Practices

What to include when reporting significance test results in IR research or
engineering documents:

### Minimum required information

```
1. Which test was used (t-test, Wilcoxon, bootstrap, randomization)
2. Whether the test was one-tailed or two-tailed (almost always two-tailed)
3. The p-value (not just "p < 0.05" - report the actual value)
4. The effect size (Cohen's d or absolute NDCG difference)
5. Number of queries evaluated on
6. Whether multiple testing correction was applied (if multiple comparisons)
```

### Example of good reporting

```
System A achieved NDCG@10 = 0.452, compared to System B's NDCG@10 = 0.441,
a difference of 0.011. This difference was statistically significant by
Wilcoxon signed-rank test (p = 0.032, two-tailed, n = 85 queries).
Effect size was d = 0.31 (small-to-medium). Note: this evaluation is powered
to detect effects ≥ 0.015 NDCG at 80% power; smaller improvements may
exist but cannot be detected on this query set.
```

### Example of bad reporting

```
System A (NDCG = 0.452) significantly outperforms System B (NDCG = 0.441).
```

Missing: which test, p-value, effect size, n queries, correction for multiple
comparisons.

## Significance Testing in Production IR

The academic significance testing machinery applies directly to production
decisions, with one key difference: you typically have far more queries.

### A/B testing significance for IR

When running an A/B test of a new retrieval system in production:

```
Null hypothesis: System A and System B have equal mean NDCG across user queries

Data collection: log per-query NDCG (from click-based implicit feedback)
                 for 1,000-100,000 queries per variant

Test: paired t-test or Wilcoxon (paired on same query observed by both)
      or bootstrap (if distributions are non-normal)

Decision rule: deploy System B if:
  p < 0.05 (significant improvement)
  AND effect size ≥ minimum meaningful improvement (defined by product team)
  AND no significant regression on important query subgroups
```

### Sequential testing for early stopping

If you want to stop an A/B test early when significance is reached:

Standard significance tests are invalid if you "peek" at results before
the predetermined sample size is reached - this inflates the false positive
rate. Sequential testing frameworks (O'Brien-Fleming, Pocock) allow interim
looks with controlled error rates.

## Quick Reference: Which Test to Use

```
Situation                                   Recommended test
──────────────────────────────────────────────────────────────────────
n ≥ 25, differences approx normal           Paired t-test
n < 25 or non-normal distribution           Wilcoxon signed-rank
Any n, want assumption-free test            Bootstrap or randomization test
IR research paper (gold standard)           Randomization test
Multiple system comparisons                 Any of above + BH-FDR correction
Production A/B test with large n            Paired t-test (CLT applies)
Sequential testing (early stopping)         Sequential t-test (O'Brien-Fleming)
```

## My Summary

Statistical significance testing determines whether an observed NDCG difference
between two retrieval systems is a genuine performance gap or sampling noise from
the particular query set used. The four main tests are: the paired t-test (most
common, requires approximate normality, n ≥ 25), the Wilcoxon signed-rank test
(distribution-free, better for small n or skewed distributions), the bootstrap
test (assumption-free, works for any metric), and the randomization permutation
test (the IR community's gold standard, directly models the exchangeability null
hypothesis). All are paired tests - per-query score differences eliminate query
difficulty as a confounding factor. Effect size (Cohen's d) is equally important
as the p-value: a result can be statistically significant but practically negligible
(d < 0.2, NDCG difference < 0.005). Power analysis shows that detecting small
improvements (0.5-1% NDCG) requires hundreds to thousands of queries - typical
TREC evaluation sets of 50-250 topics are underpowered for small effects. Multiple
testing correction (Holm-Bonferroni or Benjamini-Hochberg FDR) is required when
comparing many systems simultaneously to avoid inflated false positive rates.
