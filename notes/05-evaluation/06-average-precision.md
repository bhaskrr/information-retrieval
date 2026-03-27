# Average Precision

## The Problem With Precision@K Revisited

Recall the blindspot from the previous file:

```bash
Model A: [r, r, r, nr, nr]  →  Precision@5 = 0.6
Model B: [nr, nr, r, r, r]  →  Precision@5 = 0.6
```

Same score. But Model A is clearly better, it surfaces all three relevant items in the first three positions. A user scanning from the top would find something useful almost immediately. A user of Model B has to wade through two irrelevant items first.

Precision@K has no way to distinguish these two models because it only counts relevant items within K, it doesn't care where within K they appear.

**Average Precision (AP) fixes this** by computing precision at every rank where a relevant item appears and averaging those values. Models that rank relevant items higher get rewarded with higher precision values at those early ranks.

---

## Definition

$$AP = \frac{1}{R} \sum_{k=1}^{N} P@k \times \text{rel}(k)$$

Where:

- $R$ = total number of relevant items for this user
- $N$ = total number of items in the ranked list
- $P@k$ = Precision@K computed at position k
- $\text{rel}(k)$ = 1 if the item at position k is relevant, 0 otherwise

In plain terms: **compute Precision@k only at positions where a relevant item appears, then average those precision values.**

The key insight is that $\text{rel}(k)$ acts as a gate, positions with irrelevant items contribute nothing to the sum. You only compute and accumulate precision values at the exact positions where your model got something right.

---

## Why This Rewards Early Relevant Items

When a relevant item appears early in the list, the Precision@k at that position is high because few irrelevant items have accumulated yet. When a relevant item appears late, Precision@k at that position is low because many irrelevant items have already diluted it.

```bash
Relevant item at position 1:  P@1 = 1/1 = 1.0   ← high contribution
Relevant item at position 2:  P@2 = 2/2 = 1.0   ← high contribution
Relevant item at position 8:  P@8 = 3/8 = 0.375 ← lower contribution
Relevant item at position 10: P@10= 4/10= 0.4   ← lower contribution
```

The earlier the relevant item, the higher its Precision@k value, and therefore the higher its contribution to AP. This is exactly the position-sensitive behavior that Precision@K lacked.

---

## A Worked Example

Ranked list of 10 recommendations, 5 relevant items in catalog:

```bash
Position 1:  Movie A  relevant
Position 2:  Movie B  not
Position 3:  Movie C  relevant
Position 4:  Movie D  not
Position 5:  Movie E  relevant
Position 6:  Movie F  not
Position 7:  Movie G  relevant
Position 8:  Movie H  not
Position 9:  Movie I  not
Position 10: Movie J  relevant
```

Relevant items appear at positions: 1, 3, 5, 7, 10.  
R = 5 (total relevant items).

Computing P@k only at relevant positions:

```nash
Position 1:  P@1  = 1/1  = 1.000   rel(1) = 1  →  contributes 1.000
Position 3:  P@3  = 2/3  = 0.667   rel(3) = 1  →  contributes 0.667
Position 5:  P@5  = 3/5  = 0.600   rel(5) = 1  →  contributes 0.600
Position 7:  P@7  = 4/7  = 0.571   rel(7) = 1  →  contributes 0.571
Position 10: P@10 = 5/10 = 0.500   rel(10)= 1  →  contributes 0.500
```

$$AP = \frac{1}{5}(1.000 + 0.667 + 0.600 + 0.571 + 0.500) = \frac{3.338}{5} = 0.668$$

---

## Comparing Two Models With AP

Now let's see AP correctly distinguish Model A and Model B from the opening example.
Assume 3 relevant items total (R = 3), list length = 5:

**Model A:** [r, r, r, nr, nr]

```bash
Position 1: P@1 = 1/1 = 1.000  →  contributes 1.000
Position 2: P@2 = 2/2 = 1.000  →  contributes 1.000
Position 3: P@3 = 3/3 = 1.000  →  contributes 1.000

AP = (1.000 + 1.000 + 1.000) / 3 = 1.000
```

**Model B:** [nr, nr, r, r, r]

```bash
Position 3: P@3 = 1/3 = 0.333  →  contributes 0.333
Position 4: P@4 = 2/4 = 0.500  →  contributes 0.500
Position 5: P@5 = 3/5 = 0.600  →  contributes 0.600

AP = (0.333 + 0.500 + 0.600) / 3 = 0.478
```

AP correctly identifies Model A (AP = 1.0) as far superior to Model B (AP = 0.478), even though both had Precision@5 = 0.6. The metric now captures what Precision@K missed.

---

## AP at K - Truncating the Evaluation

In practice, you often don't want to evaluate the entire ranked list, just the top K positions. **AP@K** computes Average Precision only within the top K:

$$AP@K = \frac{1}{\min(R, K)} \sum_{k=1}^{K} P@k \times \text{rel}(k)$$

The denominator changes to $\min(R, K)$, you can't retrieve more relevant items than either R or K allows. If R = 5 and K = 3, the maximum possible AP@K is 1.0 when all 3 top positions are relevant (you can't retrieve all 5 relevant items in 3 slots, so you normalize by 3).

AP@10 is the most common variant in RecSys papers.

---

## Limitations of AP

**AP assumes binary relevance.** An item is either relevant (1) or not (0). It has no notion of a 5-star rating being more relevant than a 3-star rating. This is fine for many RecSys tasks but inadequate when you have graded relevance signals.

**AP can be noisy with few relevant items.** If a user has only 1–2 relevant items, AP becomes highly sensitive to where exactly those items land. A single position difference can swing AP dramatically.

**AP is a per-user metric.** To evaluate a system across all users, you need to aggregate AP values. That aggregation is called **Mean Average Precision (MAP)**.

---

## Summary

| Aspect        | Detail                                                 |
| ------------- | ------------------------------------------------------ |
| What it fixes | Precision@K's position-blindness within K              |
| How           | Computes P@k at every relevant position, averages      |
| Range         | [0, 1], higher is better                               |
| Denominator   | Total relevant items R (or min(R, K) for AP@K)         |
| Limitation    | Binary relevance only, aggregated via MAP across users |
