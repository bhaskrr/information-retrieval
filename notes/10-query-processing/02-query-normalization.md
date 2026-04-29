# Query Normalization

Query normalization is the set of transformations applied to tokenized query terms
to convert them into a canonical, standardized form. After tokenization splits the
raw query into units, normalization ensures that surface-level variations of the
same term - different cases, accented characters, punctuation styles, unicode
representations - all map to the same index entry. Without normalization, "Python",
"PYTHON", and "python" are three different terms that retrieve three different
documents from the index.

## Intuition

An index is a lookup table. When you type "BERT" into a search box, the system
looks up "BERT" in the index. If all documents were indexed under "bert" (lowercase),
the lookup returns nothing - not because the documents are irrelevant but because
the surface form does not match.

Normalization prevents this. It answers the question: what is the canonical form
of this term that we will use everywhere - both when indexing documents and when
processing queries? As long as both sides of the pipeline apply the same
transformations, the lookup succeeds.

The key constraint: normalization must be applied identically to both the query
and the document index. A normalization step applied only to queries creates a
guaranteed mismatch. Applied only to documents, the same problem appears in reverse.

## Normalization Operations

### 1. Lowercasing

The most universal normalization step. Convert all characters to lowercase:

```bash
"Python"    --> "python"
"BERT"      --> "bert"
"NLP"       --> "nlp"
"iPhone"    --> "iphone"
```

Ensures case variants of the same word map to the same token.

**Risk - proper noun ambiguity:**
Some terms change meaning with case. "US" (United States) and "us" (pronoun) are
different. "Apple" (company) and "apple" (fruit) are different. For most IR tasks
this ambiguity is acceptable - the context usually resolves it. For high-precision enterprise search it may not be.

**When to skip lowercasing:**

- Case-sensitive code search (Python `True` ≠ `true`)
- Legal documents where acronyms have specific meaning
- Queries where users explicitly capitalize to signal proper nouns

### 2. Unicode Normalization

Unicode has multiple representations for the same visual character. Unicode
normalization collapses these to a single canonical form:

```bash
"naïve"     --> "naive"     (NFC/NFD normalization + accent removal)
"résumé"    --> "resume"
"café"      --> "cafe"
"Ångström"  --> "angstrom"
```

**Four Unicode normalization forms:**

```bash
NFC  - canonical decomposition + canonical composition
       "é" stored as single code point
NFD  - canonical decomposition
       "é" stored as "e" + combining accent
NFKC - compatibility decomposition + canonical composition
       "ﬁ" (ligature) -> "fi", "²" -> "2", "①" -> "1"
NFKD - compatibility decomposition
       Most aggressive - best for IR
```

NFKD followed by stripping combining characters is the standard approach for
IR - it maximizes term matching across spelling variants.

**When to preserve accents:**

- Multilingual IR where accents distinguish meaning ("resume" ≠ "résumé" in French)
- Names and proper nouns where accents are part of identity

### 3. Punctuation Normalization

Decide which punctuation to remove, which to preserve, and which to replace:

```bash
Remove entirely:     ! ? , ; : ( ) [ ] { } " '
Preserve (meaning-bearing): . (decimal), - (hyphen in compounds), _ (code identifiers)
Replace with space:  / \ | @ # $ % ^ & * + = ~
```

The key insight is that punctuation is not uniformly removable. "C++" is not the
same as "C". "file.txt" has different meaning depending on whether the period is
preserved. "state-of-the-art" is a single concept even though it contains hyphens.

**Practical rules for IR:**

```bash
"C++"              -> "c++" (preserve, meaningful)
"http://site.com"  -> "site com" (URL-specific rules)
"file.txt"         -> "file txt" or "file.txt" (depends on use case)
"top-k"            -> "top-k" or "top k" (both valid)
"co-occurrence"    -> "co-occurrence" and "co occurrence" (index both)
"don't"            -> "dont" or expand contraction (application-specific)
"2.5"              -> "2.5" (preserve decimal point)
```

### 4. Whitespace Normalization

Collapse multiple whitespace characters into a single space:

```bash
"dense   retrieval"   -> "dense retrieval"
"query\ttokenization" -> "query tokenization"
"line\nbreak"         -> "line break"
"  leading space"     -> "leading space"
"trailing space  "    -> "trailing space"
```

Simple but essential - double spaces, tabs, and newlines in copy-pasted queries
create invisible tokenization failures.

### 5. Number Normalization

Handle numbers consistently across queries and documents:

**Strategy A - keep as-is:**

```bash
"10"    -> "10"
"ten"   -> "ten"
"10th"  -> "10th"
```

Maximum precision but "10" and "ten" never match.

**Strategy B - normalize to digit form:**

```bash
"ten"    -> "10"
"first"  -> "1st"
"third"  -> "3rd"
```

Requires a word-to-number mapping. Improves matching across representation styles.

**Strategy C - replace all numbers with placeholder:**

```bash
"10"    -> "<NUM>"
"2024"  -> "<NUM>"
"3.14"  -> "<NUM>"
```

Reduces vocabulary size. Useful when exact numbers are not meaningful for retrieval
(rare in IR, more common in NLP classification).

For most IR tasks, Strategy A (keep as-is) is the practical default. Strategy B
is worth adding for conversational and voice query systems where users may say
"top ten" instead of "top 10".

### 6. Special Character Normalization

Handle domain-specific special characters:

```bash
Comparison operators: >=, <=, !=  -> normalize to words or remove
Mathematical:         ²  ->  2, √  ->  sqrt,  π  ->  pi
Currency:             $  ->  dollar or remove
Trademark:            ™  ->  remove,  ®  ->  remove
Bullets/arrows:       →  ·  •  -> remove or replace with space
Ligatures:            ﬁ  ->  fi,  ﬂ  ->  fl,  œ  ->  oe
```

## Normalization Pipeline Order

The order of operations matters because some steps affect subsequent steps:

```bash
1. Unicode normalization (NFKD)         <- before any text operations
2. Accent/diacritic removal             <- after unicode decomposition
3. Contraction expansion                <- before lowercasing (needs capitalization)
4. Lowercasing                          <- after contraction expansion
5. Punctuation normalization            <- after case normalization
6. Whitespace normalization             <- after punctuation (which creates spaces)
7. Number normalization                 <- after all text normalization
8. Final tokenization                   <- produces clean token list
```

Applying lowercasing before contraction expansion creates problems:

```bash
"Don't" -> lowercase -> "don't" -> expand -> "do not"
"Don't" -> expand   -> "Do not" -> lowercase -> "do not"

Both work in this case, but more complex cases can fail when order is wrong.
```

## Normalization Aggressiveness Tradeoff

Every normalization step trades precision for recall:

```bash
More normalization -> higher recall (more documents match)
                   -> lower precision (unrelated documents match)

Less normalization -> lower recall (fewer documents match)
                   -> higher precision (matches are more specific)
```

```bash
Example: query "NLP"

No normalization:   matches only "NLP"             (high precision)
Lowercase:          matches "nlp", "NLP", "Nlp"    (balanced)
Acronym expansion:  matches "natural language processing" too   (high recall)
```

The right level depends on the application:

- Technical documentation search: less normalization (precision matters)
- General web search: more normalization (recall matters)
- Enterprise search: domain-tuned normalization rules

## Normalization vs Stemming vs Lemmatization

These three operations are often confused:

| Operation     | What it does                 | Example            |
| ------------- | ---------------------------- | ------------------ |
| Normalization | Surface-form standardization | "U.S.A" -> "usa"   |
| Stemming      | Approximate root extraction  | "running" -> "run" |
| Lemmatization | Dictionary base form mapping | "better" -> "good" |

They are applied at different pipeline stages:

```bash
Raw text
  -> tokenization
  -> normalization    ← this note
  -> stemming/lemmatization (optional, covered in 02-text-processing/)
  -> index
```

Normalization happens first because stemming and lemmatization assume clean,
normalized input. Stemming "U.S.A." produces "u.s.a." - useless. Normalize
to "usa" first, then stem produces "usa" - correct.

## My Summary

Query normalization converts tokenized query terms into canonical forms by applying
a sequence of standardizing transformations: unicode normalization collapses multiple
representations of the same character, accent removal handles diacritics, lowercasing
eliminates case variation, punctuation normalization removes or preserves symbols
based on whether they carry meaning, and whitespace normalization collapses irregular
spacing. The critical constraint is consistency - identical normalization must be
applied to both the query and the document index. Normalization aggressiveness trades
recall (more variants match) against precision (unrelated variants also match). The
right level is application-specific: general web search benefits from aggressive
normalization, technical documentation search from conservative normalization. Applied
correctly, normalization is invisible - users get relevant results regardless of whether
they typed "BERT", "bert", or "Bert". Applied incorrectly, it silently destroys
retrieval quality through index-query mismatches.
