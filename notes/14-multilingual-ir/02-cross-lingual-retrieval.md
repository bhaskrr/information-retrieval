# Cross-Lingual Retrieval

Cross-lingual retrieval (CLIR) is the task of finding relevant documents in one
language given a query expressed in a different language. A user writes a query
in French and retrieves relevant documents written in English. A researcher asks
a question in Japanese and finds answers in German scientific papers. Unlike
multilingual retrieval where queries and documents can be in any supported language
and the system returns results from a mixed-language corpus, CLIR specifically
addresses the directed case of querying across a language boundary - the query
language and the document collection language are known and different. It is the
foundational capability that makes global information access possible for users
who cannot or prefer not to write queries in the dominant language of the document
collection.

## Intuition

Most of the world's information is written in a small number of languages -
English, Chinese, and Spanish account for a disproportionate share of digital
text. A researcher in Thailand, a student in Morocco, or a journalist in Hungary
who needs access to information predominantly published in English faces a real
barrier: either learn to write effective English queries or miss large fractions
of available information.

Cross-lingual retrieval eliminates this barrier at the retrieval layer. The user
writes their query in their native language. The system retrieves relevant
documents from the English (or any other language) collection. The meaning of
the query - not its surface form - determines what is retrieved.

Two fundamentally different approaches exist for this translation at the meaning
level:

**Translate-then-retrieve** - convert the query to the document language using
machine translation, then use standard monolingual retrieval. Simple but dependent
on translation quality.

**Dense cross-lingual retrieval** - use a multilingual embedding model to
represent both query and documents in a shared semantic space, bypassing explicit
translation. More elegant and increasingly more accurate.

The convergence of high-quality multilingual embeddings and practical machine
translation has made CLIR a solved problem for high-resource language pairs. The
frontier is low-resource languages, specialized domains, and handling queries
whose meaning depends on cultural context that may not transfer across languages.

## CLIR Approaches

### Approach 1 - Query Translation (QT)

Translate the query into the document language, then use standard monolingual
retrieval on the translated query:

```
User query (French): "Quels sont les traitements pour la COVID-19?"
Machine translation: "What are the treatments for COVID-19?"
Monolingual retrieval: standard BM25 or dense retrieval in English
```

**Advantages:**

- Leverages any existing monolingual retrieval system unchanged
- Modern neural MT (DeepL, Google Translate) is very high quality for
  high-resource language pairs
- Translation quality directly maps to retrieval quality
- Easy to implement and debug - translation errors are visible

**Disadvantages:**

- MT errors propagate to retrieval errors
- Domain-specific terminology often mistranslated
- Adds latency (MT API call before retrieval)
- Expensive at scale if using commercial MT APIs
- Low-resource language pairs have poor MT quality

**When to use:** High-resource language pairs, existing monolingual index,
latency is not critical, team has no ML expertise for multilingual embeddings.

### Approach 2 - Document Translation (DT)

Translate all documents into the query language at index time. Users query
in their native language against the translated collection:

```
Index time: translate all English documents to French, German, Spanish, etc.
Query time: French query → French index → retrieve translated documents
```

**Advantages:**

- Query time is identical to monolingual retrieval (no extra latency)
- Translation happens once offline, amortized over many queries
- Can serve many query languages from one translated index

**Disadvantages:**

- Enormously expensive to translate large corpora into many languages
- Storage multiplied by number of target languages
- Translation errors are baked into the index permanently
- Updating the collection requires re-translating new documents

**When to use:** Small, static document collections with known target query
languages. Rarely practical at web scale.

### Approach 3 - Dense Cross-Lingual Retrieval

Use a multilingual embedding model to encode both queries and documents. No
translation required - the embedding model maps both to a shared semantic space:

```
Index time: encode all English documents with multilingual model → FAISS index
Query time: encode French query with same multilingual model → ANN search
```

**Advantages:**

- No explicit translation required - implicit alignment in embedding space
- Single index serves all query languages
- Lower latency than QT (no MT API call)
- Handles terminology that is hard to translate (proper nouns, technical terms)
- Improving as multilingual models improve

**Disadvantages:**

- Requires a capable multilingual embedding model
- Cross-lingual alignment quality varies by language pair
- Low-resource languages still underperform
- Less interpretable than translation-based approaches

**When to use:** Modern production systems with multilingual requirements,
GPU available for encoding, need to serve many query languages from one index.

### Approach 4 - Hybrid (QT + Dense)

Combine query translation with dense retrieval:

```
Approach A - translate query, then encode with multilingual model:
  French query → MT → English query → multilingual encoder → retrieve

Approach B - ensemble QT retrieval and dense retrieval with RRF:
  French query → MT → BM25 English retrieval
  French query → multilingual encoder → dense retrieval
  → RRF fusion of both result sets
```

Consistently stronger than either approach alone. The translated BM25 handles
exact terminology matches; dense retrieval handles semantic similarity that
survives language switching.

## Dense CLIR Architecture

The standard dense CLIR pipeline extends the monolingual bi-encoder architecture:

```
Index time:
  for each document d in target language corpus:
    d_vec = multilingual_encoder(d)    → 768-dim vector
    store in FAISS index

Query time:
  q_vec = multilingual_encoder(query_in_any_language)
  top_k = ANN_search(q_vec, FAISS_index)
  return top_k documents (in target language)
```

The multilingual encoder is the critical component - it must map equivalent
meanings in different languages to nearby regions of the embedding space.

### Cross-lingual alignment quality

Alignment quality is measured by how close translation pairs are in embedding
space versus how far apart unrelated same-language pairs are:

```
Alignment gap = mean_sim(translations) - mean_sim(random_pairs)
High gap → good cross-lingual alignment
Low gap  → poor alignment, queries will miss documents in other language
```

For mE5-base on high-resource languages (EN↔FR, EN↔DE, EN↔ES):

```
Alignment gap ≈ 0.35-0.45   → strong alignment
```

For low-resource languages (EN↔SW, EN↔TE, EN↔BN):

```
Alignment gap ≈ 0.15-0.25   → weaker alignment
```

## CLIR-Specific Challenges

### Terminology and technical vocabulary

Technical terms in specialized domains are often language-specific or have
imprecise translations:

```
English: "common law"
French:  "droit commun" (literal) but conceptually different legal system
German:  "Gewohnheitsrecht" (different nuance)
```

A multilingual model trained on general web text may not capture these
domain-specific distinctions. Domain-specific CLIR systems need additional
fine-tuning on parallel domain text.

### Named entity handling

Named entities (people, organizations, places) often appear in their original
language or have standardized transliterations:

```
Query (French):    "recherches de Yoshua Bengio"
Expected retrieval: English documents about Yoshua Bengio
Problem:           "Bengio" is consistent across languages (no problem)
                   but less common names may be transliterated differently
```

Named entity recognition and standardization improve CLIR for entity-heavy queries.

### Cultural concepts without direct translation

Some concepts have no direct translation and require circumlocution:

```
Japanese "木漏れ日" (komorebi): sunlight through leaves
German "Torschlusspanik": fear of missing opportunities as one ages
Portuguese "Saudade": longing for something absent
```

A CLIR system for queries about these concepts must handle the mismatch between
the culturally-specific source expression and whatever approximation exists in
the target language.

### Script and encoding issues

Non-Latin scripts require correct Unicode handling throughout the pipeline:

```
Arabic:  right-to-left script, morphologically rich, diacritics affect meaning
Chinese: character-based, no word boundaries, Traditional vs Simplified
Thai:    no word boundaries, requires Thai-specific tokenization
Korean:  syllable blocks (Hangul), morphological decomposition needed
```

Each script family requires script-aware preprocessing to avoid breaking
tokenization or embedding quality.

## Fine-Tuning for Cross-Lingual Retrieval

Pretrained multilingual models (mE5, LaBSE) provide strong zero-shot CLIR.
For specific language pairs or domains, fine-tuning improves performance:

### Strategy 1 - mMARCO fine-tuning

MS MARCO translated into 13 languages. Fine-tune a multilingual bi-encoder
on cross-lingual training pairs:

```
Training pairs: (English query, relevant English passage)
                (Spanish query, relevant English passage)
                (French query, relevant English passage)
All from mMARCO.

Loss: MultipleNegativesRankingLoss with in-batch negatives
      (queries in different languages serve as negatives for each other)
```

### Strategy 2 - Translate-train

Use MT to create cross-lingual training pairs from existing monolingual labeled data:

```
Original:   (English query, English relevant doc) from MS MARCO
Augmented:  (French query [MT], English relevant doc)
            (German query [MT], English relevant doc)
            (Spanish query [MT], English relevant doc)

Fine-tune on augmented data:
→ model sees cross-lingual pairs explicitly
→ learns to align query and document languages
```

Simple and effective. Quality depends on MT quality for query translation.

### Strategy 3 - Mr.TyDi / MIRACL fine-tuning

Use language-specific retrieval training data:

```
Mr.TyDi:  Wikipedia-based QA in 11 languages with passage-level relevance
MIRACL:   18-language retrieval training data with human-labeled qrels
```

Fine-tuning on Mr.TyDi/MIRACL training splits produces the strongest
cross-lingual retrieval for languages covered by these datasets.

## Evaluation Benchmarks

### CLEF (Cross-Language Evaluation Forum)

The original CLIR benchmark series, running since 2000. Multilingual newspaper
collections with cross-lingual queries. Historical importance but superseded by
neural-era benchmarks.

### MKQA (Multilingual Knowledge Questions and Answers)

```
Languages:  26 languages
Task:        Open-domain QA, retrieve English Wikipedia passages
             given queries in 26 languages
Queries:    10,000 questions from Natural Questions, translated
Notes:      Tests CLIR where queries are always in non-English language,
            documents always in English
```

### Mr.TyDi (Multilingual Retrieval TyDi)

```
Languages:  11 languages (Arabic, Bengali, English, Finnish, Indonesian,
            Japanese, Korean, Russian, Swahili, Telugu, Thai)
Task:        Monolingual Wikipedia retrieval in each language
Corpus:      Wikipedia in each language
Notes:       Primarily monolingual, used for CLIR evaluation when
             model trained on one language tests on another
```

### MIRACL (Multilingual Information Retrieval Across a Continuum of Languages)

```
Languages:  18 languages
Task:        Ad-hoc Wikipedia retrieval
Primary metric: nDCG@10
Cross-lingual track: query in language X, retrieve documents in language Y
```

### XTREME-R

```
Languages:  40 languages
Tasks:       Multiple cross-lingual understanding tasks including retrieval
Notes:       XCOPA, XQuAD, XNLI, and cross-lingual retrieval tasks
```

| Language pair       | Zero-shot NDCG@10 | Fine-tuned NDCG@10 |
| ------------------- | ----------------- | ------------------ |
| EN→EN (monolingual) | 0.55-0.65         | 0.65-0.75          |
| FR→EN, DE→EN, ES→EN | 0.50-0.60         | 0.60-0.72          |
| ZH→EN, JA→EN, KO→EN | 0.45-0.55         | 0.55-0.65          |
| AR→EN, HI→EN        | 0.40-0.52         | 0.52-0.63          |
| SW→EN, TE→EN, BN→EN | 0.30-0.45         | 0.45-0.58          |

The monolingual upper bound shows how much performance is "lost" to the
cross-lingual gap. High-resource European languages lose 5-10%. Low-resource
languages lose 15-25% even with fine-tuning.

## CLIR vs Translation: When to Use Which

| Scenario                             | Best approach                       |
| ------------------------------------ | ----------------------------------- |
| Real-time, latency < 100ms           | Dense CLIR (no MT latency)          |
| High-resource language pair          | Either - QT+BM25 or Dense both work |
| Low-resource language pair           | QT (dense alignment weaker)         |
| Domain-specific terminology          | QT (MT better for domain terms)     |
| No multilingual GPU available        | QT + BM25 (CPU-friendly)            |
| Static corpus, many query languages  | DT (translate corpus once)          |
| Dynamic corpus, many query languages | Dense CLIR (one index for all)      |
| Maximum retrieval quality            | Hybrid QT + Dense + RRF             |

## My Summary

Cross-lingual retrieval finds relevant documents in one language given queries
in a different language. Three main approaches exist: query translation (translate
the query to the document language, use standard monolingual retrieval -
simple, interpretable, dependent on MT quality), document translation (translate
the collection to the query language at index time - practical only for small
static corpora), and dense cross-lingual retrieval (use multilingual embeddings
to align query and document representations without translation - the modern
production default). Hybrid approaches combining translated BM25 with dense
retrieval via RRF consistently outperform either alone. CLIR quality degrades
predictably from high-resource European languages (5-10% below monolingual)
to low-resource languages (15-25% below monolingual). Fine-tuning on mMARCO
or MIRACL training data significantly closes the gap for covered languages.
The practical decision rule: use dense CLIR for latency-sensitive multilingual
production systems, QT+BM25 for high-resource pairs with tight resource budgets,
and hybrid for maximum quality when latency allows.
