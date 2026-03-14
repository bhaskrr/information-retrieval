# Stemming and Lemmatization

Both are vocabulary normalization techniques that reduce inflected or derived forms of a
word to a common base form, so that "running", "runs", and "ran" all map to the same term
in the index.

- **Stemming**: chops word endings using fixed rules to get an approximate root (stem).
- **Lemmatization**: maps a word to its proper dictionary base form (lemma) using
  vocabulary and morphological analysis.

## Why does it matter?

Without this step, "compute", "computing", "computed", and "computation" are four separate terms in the index. A query for "compute" would miss documents that only use "computing". Stemming and lemmatization improve recall by collapsing these variants into one term.

## Stemming

Stemming applies a series of string transformation rules to strip suffixes. No linguistic knowledge involved - purely mechanical.

### Porter Stemmer (most widely used)

Operates in 5 rule-based passes. Examples:

- "caresses" --> "caress"
- "running" --> "run"
- "happiness" --> "happi"   <-- not a real word, but consistent
- "generalization" --> "general"

The stem does not need to be a real word, it just needs to be consistent. As long as "generalize" and "generalization" both reduce to "general", the index works correctly.

### Lancaster Stemmer

More aggressive than Porter. Faster but noisier.

- "eating" --> "eat" (Porter) vs "eat" (Lancaster) - similar here
- "maximum" --> "maxim" (Porter) vs "max" (Lancaster) - Lancaster overshoots

### Snowball Stemmer

An improved version of Porter by the same author. Supports multiple languages.
The go-to choice for most English IR applications today.

## Lemmatization

Lemmatization uses a morphological lexicon (dictionary of word forms) and optionally part-of-speech tagging to find the true base form.

Examples:

- "better" --> "good"        (requires knowing it's an adjective)
- "was" --> "be"
- "running" --> "run"        (verb) vs "running" → "running" (adjective, as in "running water")
- "mice" --> "mouse"

The same word can lemmatize differently depending on its POS tag, this is why lemmatization is more accurate but also slower and more complex.

## Stemming vs. Lemmatization

| Property            | Stemming                        | Lemmatization                      |
|---------------------|---------------------------------|------------------------------------|
| Output              | Approximate root (may not exist)| Real dictionary word               |
| Speed               | Fast                            | Slower                             |
| Accuracy            | Lower                           | Higher                             |
| Linguistic knowledge| None                            | Requires lexicon + POS tagging     |
| Best for            | Large-scale IR, search engines  | NLP tasks, QA, text understanding  |

## Where it fits in the pipeline

```bash
raw text → tokenization → normalization → stopword removal → stemming/lemmatization → index
```

## When neither helps (and can hurt)

- **Overstemming**: two different words collapse to the same stem
  - "universal", "university", "universe" -> "univers", now indistinguishable.
- **Understemming**: variants that should collapse, don't
  - "alumnus" and "alumni" may not stem to the same form.
- **Domain mismatch**: medical or legal text has vocabulary that general stemmers
  handle poorly.

## Which to use in practice?

- Building a search engine or IR pipeline -> **Snowball stemmer**, fast, good enough
- NLP task where word meaning matters (QA, NLI, summarization) -> **lemmatization**
- Modern neural IR -> **neither**, subword tokenizers like BPE (used in BERT) handle
  morphological variation implicitly through learned representations

## My Summary

Stemming and lemmatization both reduce word variants to a common form to improve recall.
Stemming is fast and crude, it chops suffixes by rule and the result may not be a real
word. Lemmatization is slower and accurate, it uses a dictionary and POS context to find
the true base form. For classical IR, Snowball stemming is the practical default; for
neural IR, neither is needed since subword tokenizers handle it implicitly.
