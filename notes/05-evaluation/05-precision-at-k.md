# Precision@K

## The Core Insight

Every metric so far has evaluated recommendations as an unordered set. But real recommendation systems don't present items as sets - they present them as **ranked lists**. A user sees position 1 first, then position 2, and so on. Most users never scroll past the first few results.

Precision@K acknowledges this reality with a simple fix:

> **Only evaluate the top K positions. Everything below K doesn't exist.**

This single change transforms Precision from a set metric into a rank-aware metric. Not perfectly rank-aware - we'll see its remaining blindspot - but significantly more realistic than raw Precision.

---

## Definition

$$\text{Precision@K} = \frac{\text{number of relevant items in top K recommendations}}{K}$$

That's it. Count the relevant items in your top K recommendations, divide by K.

**What changes compared to raw Precision:**

- The denominator is always K, not the total number of recommendations
- You're explicitly saying "I only care about the first K positions"
- Items ranked below K have zero effect on the score

---

## A Worked Example

Same setup as before - 5 relevant items in the catalog. Our model's ranked recommendation list:

```bash
Position 1:  Movie A  relevant
Position 2:  Movie B  not relevant
Position 3:  Movie C  relevant
Position 4:  Movie D  not relevant
Position 5:  Movie E  relevant
Position 6:  Movie F  not relevant
Position 7:  Movie G  relevant
Position 8:  Movie H  not relevant
Position 9:  Movie I  not relevant
Position 10: Movie J  relevant
```

Computing Precision@K for different values of K:

```bash
Precision@1  = 1/1  = 1.0    (top 1 item is relevant)
Precision@3  = 2/3  ≈ 0.67   (2 relevant in top 3)
Precision@5  = 3/5  = 0.6    (3 relevant in top 5)
Precision@10 = 5/10 = 0.5    (5 relevant in top 10)
```

Notice how the choice of K changes your evaluation. A model that front-loads relevant items will score high at small K but potentially lower at large K as irrelevant items accumulate.

---

## Choosing K

K is not arbitrary, it should reflect the actual interface the system serves:

| Interface            | Typical K |
| -------------------- | --------- |
| Home screen row      | 5-10      |
| Search results page  | 10-20     |
| Email digest         | 5         |
| Candidate generation | 100-500   |
| Full catalog ranking | 1000+     |

In academic papers, K = 5, 10, and 20 are most common. Always report multiple K values - a model that wins at K=5 may lose at K=20, which tells something meaningful about where relevant items are concentrated in its ranked list.

---

## What Precision@K Still Ignores

Precision@K is better than raw Precision but it still has a significant blindspot:

**It treats all positions within K equally.**

In the example above, a model that ranks relevant items at positions [1, 2, 3] scores the same Precision@5 as one that ranks them at positions [3, 4, 5] - both have 3 relevant items in the top 5.

But these two models are not equal. The first model surfaces relevant items immediately. The second buries them. Users of the first model are much more likely to find something useful before giving up.

```bash
Model A: [r, r, r, nr, nr]  Precision@5 = 3/5 = 0.6
Model B: [nr, nr, r, r, r]  Precision@5 = 3/5 = 0.6

Same score. Very different user experience.
```

This is the motivation for **Average Precision** and **NDCG** - metrics that reward systems for placing relevant items as high as possible, not just within the top K.

---

## Recall@K

Alongside Precision@K, it's common to compute **Recall@K** - how many of the total relevant items appear in the top K:

$$\text{Recall@K} = \frac{\text{number of relevant items in top K}}{|\text{total relevant items}|}$$

Using the same example (5 relevant items total):

```bash
Recall@1  = 1/5 = 0.2
Recall@3  = 2/5 = 0.4
Recall@5  = 3/5 = 0.6
Recall@10 = 5/5 = 1.0
```

Recall@K answers a different question than Precision@K:

- **Precision@K:** Of what I showed, how much was useful?
- **Recall@K:** Of everything useful, how much did I show?

Both matter. In a short list (K=5), Recall@K is often low even for good models, there simply isn't room to retrieve all relevant items. Recall@K becomes more meaningful at larger K values.

---

## Hit Rate@K

In the leave-one-out evaluation setting (one test item per user), Precision@K and Recall@K collapse into a simpler metric: **Hit Rate@K** (also called HR@K or Hits@K).

Since each user has exactly one relevant test item:

$$\text{Hit Rate@K} = \frac{\text{number of users whose test item appears in top K}}{|\text{total users}|}$$

Per user, it's binary - either the test item is in the top K (hit = 1) or it isn't (hit = 0). Averaged across users, it becomes the fraction of users for whom the model successfully retrieved their test item.

Hit Rate@K is extremely common in RecSys papers that use leave-one-out evaluation.

```bash
User 1: test item at position 3  → hit@10 = 1
User 2: test item at position 15 → hit@10 = 0
User 3: test item at position 7  → hit@10 = 1
User 4: test item at position 2  → hit@10 = 1

Hit Rate@10 = 3/4 = 0.75
```

---

## Summary

| Metric      | Formula                                     | Use case                        |
| ----------- | ------------------------------------------- | ------------------------------- |
| Precision@K | relevant in top K / K                       | How useful is the top K list?   |
| Recall@K    | relevant in top K / total relevant          | How complete is the top K list? |
| Hit Rate@K  | users with test item in top K / total users | Leave-one-out evaluation        |

**Remaining blindspot:** Precision@K treats all positions within K equally. A relevant item at rank 1 scores identically to one at rank K.
