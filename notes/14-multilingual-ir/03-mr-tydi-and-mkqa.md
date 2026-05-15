# Mr.TyDi and MKQA

Mr.TyDi and MKQA are the two most important evaluation benchmarks for multilingual
and cross-lingual information retrieval. Mr.TyDi (Multilingual Retrieval TyDi) is a
monolingual retrieval benchmark covering 11 typologically diverse languages, where
queries and documents are in the same language, testing whether a retrieval model
works well natively in each supported language. MKQA (Multilingual Knowledge
Questions and Answers) is a cross-lingual retrieval benchmark where 10,000 questions
from English Natural Questions are translated into 26 languages and used to retrieve
English Wikipedia passages, testing whether a model can find English answers to
non-English questions. Together they provide comprehensive coverage of both
monolingual multilingual retrieval and directed cross-lingual retrieval - the two
central evaluation scenarios in multilingual IR research and deployment.

## Intuition

Evaluating multilingual retrieval requires answering two distinct questions that
demand different benchmarks:

**Question 1 - Does the model work well in each individual language?**
A multilingual model might excel in English and French but be essentially useless
in Bengali or Telugu. Monolingual evaluation within each language reveals these
per-language gaps. Mr.TyDi answers this question by providing retrieval test
collections in 11 languages where everything - queries, documents, relevance
judgments - is in the same language.

**Question 2 - Can the model retrieve across language boundaries?**
A model might work well monolingually in each language but fail to align meaning
across languages, making it useless for cross-lingual retrieval. MKQA answers
this question by fixing the document collection to English Wikipedia and testing
whether queries in 26 different languages can retrieve the right English passages.

Using only BEIR (which is entirely English) to evaluate a multilingual system is
like testing a car only on dry pavement and claiming it handles all conditions.
Mr.TyDi and MKQA are the rain, snow, and off-road tests that reveal whether
multilingual IR actually works across the full range of intended use cases.

## Mr.TyDi in Depth

### Origins and design

Mr.TyDi is built on top of TyDi QA, a question answering dataset collected from
Wikipedia editors in 11 languages who wrote questions they genuinely did not know
the answers to. This "genuinely curious" design produces more natural and diverse
queries than translation-based datasets where questions are artificially created
in one language and then translated.

The retrieval task was derived by pairing each question with the Wikipedia passages
that contain its answer - creating (question, relevant passage) pairs for retrieval
evaluation.

### Languages and typological diversity

The 11 languages were chosen for typological diversity - covering different language
families, scripts, and linguistic features:

| Language   | Family       | Script         | Key feature                           |
| ---------- | ------------ | -------------- | ------------------------------------- |
| Arabic     | Semitic      | Arabic         | RTL, morphologically rich             |
| Bengali    | Indo-Aryan   | Bengali        | SOV word order, complex morphology    |
| English    | Germanic     | Latin          | Baseline, well-resourced              |
| Finnish    | Uralic       | Latin          | Agglutinative, no articles            |
| Indonesian | Austronesian | Latin          | Morphologically simple, no inflection |
| Japanese   | Japonic      | CJK + Hiragana | No word boundaries, honorifics        |
| Korean     | Koreanic     | Hangul         | Agglutinative, SOV                    |
| Russian    | Slavic       | Cyrillic       | Rich case system, verbal aspect       |
| Swahili    | Bantu        | Latin          | Noun class system, agglutinative      |
| Telugu     | Dravidian    | Telugu         | SOV, agglutinative, retroflex         |
| Thai       | Tai-Kadai    | Thai           | No word boundaries, tonal             |

This diversity is intentional - a model that scores well across all 11 languages
has demonstrated genuine multilingual retrieval capability, not just English-biased
performance with some multilingual window dressing.

### Dataset statistics

| Language   | Corpus size | Train queries | Dev queries | Test queries |
| ---------- | ----------- | ------------- | ----------- | ------------ |
| Arabic     | 292,441     | 3,495         | 300         | 300          |
| Bengali    | 304,059     | 1,713         | 190         | 190          |
| English    | 32,907,100  | 3,547         | 308         | 744          |
| Finnish    | 447,815     | 3,021         | 308         | 308          |
| Indonesian | 1,469,399   | 2,817         | 303         | 303          |
| Japanese   | 7,000,027   | 3,697         | 307         | 307          |
| Korean     | 1,496,126   | 1,295         | 303         | 303          |
| Russian    | 9,782,368   | 3,547         | 300         | 300          |
| Swahili    | 136,689     | 2,072         | 300         | 300          |
| Telugu     | 548,224     | 3,880         | 303         | 303          |
| Thai       | 568,855     | 3,319         | 300         | 300          |

Key observation: corpus sizes vary enormously - English Wikipedia has 32 million
passages while Swahili has only 136 thousand. This reflects the real-world imbalance
in multilingual content availability and creates harder retrieval tasks for
smaller corpora (more concentrated topic coverage but also less noise).

### Evaluation metrics

Mr.TyDi uses two primary metrics:

**MRR@10 (Mean Reciprocal Rank at 10)**

```
MRR@10 = (1/|Q|) × Σ_q (1 / rank_q)

Where rank_q = rank of first relevant document in top-10 results
             = 11 if no relevant document in top-10 (treated as 0 contribution)
```

MRR@10 measures how quickly the model surfaces a relevant result. A relevant
document at rank 1 contributes 1.0; at rank 2 contributes 0.5; at rank 5
contributes 0.2; outside top-10 contributes 0.

**Recall@100**

```
Recall@100 = fraction of queries where at least one relevant passage
             appears in the top-100 retrieved passages
```

Recall@100 measures retrieval coverage - does the model even find any relevant
passage if allowed 100 results? This is the first-stage retrieval metric,
critical for RAG pipelines where a reranker then selects from the top-100.

### State-of-the-art performance

Approximate state-of-the-art Mr.TyDi scores (MRR@10) as of 2025:

| Model                       | Average | Best language    | Worst language  |
| --------------------------- | ------- | ---------------- | --------------- |
| BM25 (baseline)             | 0.367   | English (0.631)  | Swahili (0.271) |
| mDPR (zero-shot)            | 0.316   | English (0.611)  | Telugu (0.191)  |
| mE5-base (zero-shot)        | 0.524   | English (0.742)  | Telugu (0.382)  |
| mE5-large (zero-shot)       | 0.561   | English (0.773)  | Swahili (0.415) |
| mGTE-base (zero-shot)       | 0.538   | English (0.751)  | Swahili (0.398) |
| Fine-tuned on Mr.TyDi train | 0.620+  | English (0.800+) | Telugu (0.490+) |

Key observations:

- BM25 outperforms mDPR zero-shot on several languages - domain adaptation matters
- Modern zero-shot models (mE5, mGTE) substantially outperform BM25 on average
- Telugu and Swahili are consistently the hardest languages (low-resource)
- Fine-tuning on Mr.TyDi training data provides substantial improvement

## MKQA in Depth

### Origins and design

MKQA was created by translating 10,000 questions from Natural Questions (a
real Google search QA dataset) into 26 languages using professional translators
for high-resource languages and crowdworkers for lower-resource ones. The document
collection is fixed to English Wikipedia - every question in every language must
retrieve its answer from English text.

This design directly tests cross-lingual retrieval capability: the model must
understand a question in language X and match it against evidence in English.

### Languages

```
Arabic, Chinese (Simplified), Chinese (Traditional), Czech, Danish, Dutch,
English, Finnish, French, German, Hebrew, Hindi, Hungarian, Italian, Japanese,
Korean, Norwegian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish,
Thai, Turkish, Vietnamese

Total: 26 languages
```

### Dataset statistics

```
Total questions:       10,000 per language (same questions across all languages)
Document collection:   English Wikipedia (~6M passages from NQ preprocessing)
Question types:        Single-hop factual, named entity, numerical, yes/no
Answer types:          Short answer spans, entity names, numbers, dates
```

### Evaluation metrics

MKQA evaluation is more complex than Mr.TyDi because it involves both retrieval
and answer extraction. The full pipeline is:

```
Query (any language) → retrieve English passages → extract answer span → evaluate
```

**Retrieval-level metrics:**

```
Recall@K:  fraction of questions where the answer appears in top-K passages
           (K = 5, 20, 100 commonly used)
```

**End-to-end metrics (for RAG evaluation):**

```
EM (Exact Match):  fraction of questions where extracted answer exactly matches
                   gold answer string
F1:                token-level overlap between extracted and gold answer
```

For retrieval evaluation, Recall@100 is the most commonly reported metric -
it measures whether the retrieval component would provide useful context to a
downstream reader.

### MKQA difficulty factors

MKQA is harder than Mr.TyDi for several reasons:

**Translation artifacts** - machine-translated questions may not accurately
capture the original intent, especially for culturally-specific queries.

**Named entity translation variation** - person and place names are translated
differently across languages and may not match the English Wikipedia text:

```
German query: "Barack Obama" → same in English (consistent)
Japanese query: "バラク・オバマ" → transliterated, still matches English
Arabic query: "باراك أوباما" → Arabic script, must be aligned through semantics
```

**Cultural specificity** - some questions make cultural assumptions that do not
translate cleanly:

```
"When did the US Founding Fathers sign the Constitution?"
Japanese: direct translation (culturally foreign concept, harder to answer)
German: direct translation (concept known, easier)
```

**Cross-lingual alignment gap** - the fundamental challenge: the query is in
language X, the answer is in English, and the model must bridge this gap purely
through semantic alignment.

## Comparing Mr.TyDi and MKQA

| Property            | Mr.TyDi                   | MKQA                         |
| ------------------- | ------------------------- | ---------------------------- |
| Primary task        | Monolingual retrieval     | Cross-lingual retrieval      |
| Query language      | Same as documents         | 26 languages                 |
| Document language   | Same as query             | English only                 |
| Languages           | 11                        | 26                           |
| Corpus type         | Wikipedia (per language)  | English Wikipedia            |
| Query source        | Native speakers (organic) | Translated from English NQ   |
| Relevance judgments | Human-labeled passages    | Gold answer spans            |
| Primary metric      | MRR@10, Recall@100        | Recall@K, EM, F1             |
| Tests               | Per-language quality      | Cross-lingual alignment      |
| Use for training    | Yes (training splits)     | Evaluation only              |
| Best reveals        | Low-resource failures     | Cross-lingual alignment gaps |

## Using Mr.TyDi and MKQA for Model Selection

### Decision protocol

When selecting a multilingual retrieval model for production:

```
Step 1: Run zero-shot evaluation on Mr.TyDi
  → Check per-language MRR@10 for languages in your target set
  → If any target language MRR@10 < 0.35: consider fine-tuning

Step 2: Run zero-shot evaluation on MKQA (if cross-lingual needed)
  → Check Recall@100 for source languages your users will query in
  → If Recall@100 < 0.60 for key languages: consider QT fallback

Step 3: Compare models on your specific target languages
  → Do not rely solely on average scores - per-language gaps are large

Step 4: Fine-tune on Mr.TyDi training data if budget allows
  → Training data available for all 11 Mr.TyDi languages
  → Typically 10-20% MRR@10 improvement over zero-shot
```

### Interpreting benchmark results

| Score range (MRR@10) | Interpretation                                     |
| -------------------- | -------------------------------------------------- |
| < 0.30               | Poor - BM25 may outperform; model not transferring |
| 0.30 - 0.45          | Below BM25 on many languages - adaptation needed   |
| 0.45 - 0.55          | Comparable to BM25; neural advantage not clear     |
| 0.55 - 0.65          | Clear neural advantage; good general deployment    |
| 0.65 - 0.75          | Strong; approaching fine-tuned model performance   |
| > 0.75               | Excellent; likely fine-tuned on this language      |

### For training data selection

```
Mr.TyDi training splits (available):
  Arabic, Bengali, English, Finnish, Indonesian, Japanese, Korean,
  Russian, Swahili, Telugu, Thai

  → Use these when fine-tuning for specific languages

MKQA (evaluation only - no training split):
  → Evaluate cross-lingual quality but do not train on it
  → Use mMARCO for cross-lingual training data instead
```

### Connecting benchmarks to deployment quality

```
Mr.TyDi MRR@10 ≈ 0.55  → approximate NDCG@10 ≈ 0.52 in production
                           (similar difficulty level)

MKQA Recall@100 ≈ 0.70  → ~70% of user queries will have the answer
                           in your top-100 retrieved passages

Below these thresholds, consider:
  1. Fine-tuning on Mr.TyDi training data for the weak language
  2. Adding QT fallback for languages with Recall@100 < 0.60
  3. Hybrid search to boost coverage
```

Mr.TyDi and MKQA close the multilingual IR module by providing the measurement
framework that makes everything else accountable. Every multilingual embedding
model, every CLIR technique, every fine-tuning strategy covered in this module
is ultimately evaluated against these benchmarks - they are the ground truth for
whether multilingual retrieval actually works.

## My Summary

Mr.TyDi and MKQA are the two benchmark pillars of multilingual IR evaluation.
Mr.TyDi evaluates monolingual retrieval in 11 typologically diverse languages using
MRR@10 and Recall@100 - revealing per-language performance gaps that average scores
conceal. MKQA evaluates cross-lingual retrieval by testing whether queries in 26
languages can retrieve answers from English Wikipedia - directly measuring cross-
lingual alignment quality. Together they answer the two central questions of
multilingual IR deployment: does the model work well within each target language,
and can it retrieve across language boundaries? Mr.TyDi training data (available
for all 11 languages) provides the fine-tuning signal that typically improves
zero-shot MRR@10 by 10-20 percentage points. MKQA is evaluation-only - for cross-
lingual training, mMARCO (MS MARCO translated into 13 languages) is the standard
resource. Production deployment decisions should be made per-language based on
these benchmarks rather than relying on aggregate scores that mask large variation
between high-resource and low-resource languages.
