# Normalization

Normalization is the process of converting tokens into a standard, canonical form so that
variations of the same word map to the same token in the index.

## Why does it matter?

Without normalization, "USA", "U.S.A", and "usa" would be treated as three different terms
in the index, even though a user querying any one of them means the same thing. Normalization
is what makes retrieval robust to surface-level variation in text.

## Techniques

### 1. Lowercasing

The most basic and universally applied step.

- "Python" → "python", "NASA" → "nasa"
- Risk: "US" (United States) and "us" (pronoun) become identical. Blind lowercasing
  can hurt precision for named entities.

### 2. Accent / Diacritic Removal

- "naïve" → "naive", "résumé" → "resume"
- Important for multilingual corpora. Less relevant for English-only.

### 3. Punctuation & Special Character Removal

- "state-of-the-art" → "state of the art" or "stateoftheart"
- Hyphens are tricky: "co-operate" and "cooperate" should match, but "F-16" losing its hyphen changes meaning.

### 4. Number Normalization

- Decide: keep numbers, remove them, or replace with a placeholder like `<NUM>`
- "10" and "ten" will still not match, that requires deeper semantic handling

### 5. Equivalence Classing

Group tokens that mean the same thing under one representative term.

- "colour" and "color" → "color"
- Common in cross-lingual IR

## Tradeoffs to keep in mind

- Aggressive normalization improves recall (more matches) but can hurt precision
  (false matches between words that look similar but mean different things)
- The right level of normalization is domain-dependent — a legal IR system treats
  "shall" and "must" very differently; a casual web search engine probably doesn't.

## My Summary

Normalization is the quiet, unglamorous step that makes everything else work reliably.
It standardizes surface variation so the index sees "U.S.A", "u.s.a", and "usa" as one
term. Done too aggressively it conflates things that shouldn't match; done too lightly
it misses obvious equivalences. The pipeline order matters: normalize first, then stem.
