# Multilingual Embeddings

Multilingual embeddings are dense vector representations that map text from
multiple languages into a single shared vector space, where semantically
equivalent sentences in different languages are close together regardless of
the language they are written in. A query in French and a relevant document in
English produce nearby vectors. A question in Spanish and its answer in German
are geometrically close. This cross-lingual alignment enables retrieval across
language boundaries without requiring translation of either queries or documents
at inference time - the embedding model handles the alignment implicitly through
its training on multilingual data.

## Intuition

Monolingual dense retrieval works because semantically similar sentences in the
same language produce similar embeddings. The embedding space is a geometric
representation of meaning - documents and queries cluster by topic and semantic
content. The key insight is that meaning can be represented language-independently.
"A dog is chasing a cat" and "Un perro persigue a un gato" express identical
meaning in different surface forms. If an embedding model learns to map both to
the same region of vector space, cross-lingual retrieval becomes as simple as
monolingual retrieval.

The challenge is learning this alignment without explicit translation signals for
every language pair. A model must learn that the French word "chien", the Spanish
"perro", the German "Hund", and the English "dog" all refer to the same concept
and should produce nearby embeddings. Modern multilingual embedding models achieve
this through a combination of parallel corpora (texts that are translations of each
other), shared subword vocabularies across languages, and massive multilingual
pretraining.

The practical consequence: a single multilingual index can serve users in any
supported language. A user in Japan, Germany, Brazil, and the United States can
all query the same document corpus in their native language and receive relevant
results - even when the relevant documents are written in a language different
from the query.

## The Cross-Lingual Retrieval Problem

Standard monolingual retrieval assumes query and document share the same language.
Cross-lingual IR breaks this assumption in two ways:

### Cross-lingual retrieval (CLIR)

Query and documents are in different languages:

```
Query:    "Welche Behandlungen gibt es für COVID-19?" (German)
Document: "Remdesivir reduces hospitalization in COVID-19 patients." (English)
```

The system must understand that the German query asks about COVID-19 treatments
and that the English document about remdesivir is a relevant answer.

### Multilingual retrieval (MLIR)

Both queries and documents can be in any of multiple languages, with queries
potentially matching documents in a different language:

```
Collection: documents in English, French, Spanish, German, Chinese, Japanese
Query:      any language
Result:     relevant documents regardless of their language
```

MLIR is the harder and more practically important problem - real-world corpora
are increasingly multilingual, and users should find relevant content regardless
of what language it was written in.

## How Multilingual Embeddings Are Trained

### Approach 1 - Multilingual MLM pretraining (mBERT, XLM)

Train a masked language model on text from many languages simultaneously.
The model learns shared subword representations across languages through:

**Shared vocabulary** - a single wordpiece or BPE vocabulary learned across all
languages. Some tokens appear in multiple languages, forcing shared representations.

**Cross-lingual transfer** - if the model learns to predict masked tokens in
French, it implicitly learns that French syntax and vocabulary align with patterns
it has seen in English. The transformer layers develop language-agnostic
representations for common semantic concepts.

```
mBERT training:
  Corpus: Wikipedia in 104 languages
  Vocabulary: 119,500 WordPiece tokens shared across all languages
  Architecture: standard BERT-base
  Objective: MLM on each language's Wikipedia independently
```

mBERT achieves surprisingly strong cross-lingual transfer despite never seeing
explicit parallel text during training - purely from the shared vocabulary and
parameter sharing across languages.

### Approach 2 - Parallel corpus training (XLM, LASER)

Train explicitly on parallel texts (translation pairs) to enforce cross-lingual
alignment:

**Translation Language Modeling (TLM)**
A variant of MLM where the input is a concatenated translation pair. The model
must predict masked tokens in one language using context from both languages:

```
Input: "[CLS] The cat sat on the mat [SEP] Le chat était assis sur le tapis [SEP]"
Mask:  "[CLS] The [MASK] sat on the mat [SEP] Le chat était assis sur le [MASK] [SEP]"
```

Predicting "cat" benefits from seeing "chat" in French. Predicting "tapis" benefits
from seeing "mat" in English. This forces the model to develop cross-lingual
representations.

**LASER (Language-Agnostic SEntence Representations)**
An encoder-decoder architecture trained on parallel corpus translation:

```
Encoder:   fixed encoder producing sentence embeddings
Decoder:   reconstructs the translation from the embedding
Loss:      reconstruction quality of translation
```

Produces sentence embeddings where translations have similar embeddings.
Supports 93 languages.

### Approach 3 - Contrastive cross-lingual training (LaBSE, mE5, mGTE)

Train using translation pairs as positive pairs in contrastive learning:

```
Positive pair: (English sentence, its French translation)
Negatives:     all other sentences in the batch
Loss:          InfoNCE - pull translation pairs together, push others apart
```

This directly optimizes for cross-lingual alignment in embedding space -
translations should produce nearby embeddings, non-translations should not.

LaBSE (Language-agnostic BERT Sentence Encoder) uses this approach with
additive margin softmax loss and achieves state-of-the-art on cross-lingual
semantic textual similarity across 109 languages.

## Key Multilingual Embedding Models

### mBERT (Multilingual BERT, Google, 2019)

```
Languages:     104 languages
Architecture:  BERT-base (110M parameters)
Training:      MLM on multilingual Wikipedia
Vocabulary:    119,500 WordPiece tokens
Embedding dim: 768
Strengths:     Good zero-shot cross-lingual transfer for classification
Weaknesses:    Not optimized for retrieval/sentence similarity
               Weak on low-resource languages
               Unequal language coverage (English over-represented)
```

### XLM-R (Cross-lingual RoBERTa, Facebook, 2020)

```
Languages:     100 languages
Architecture:  RoBERTa-base (270M) or RoBERTa-large (560M)
Training:      MLM on 2.5TB of filtered CommonCrawl (100 languages)
Vocabulary:    250,002 SentencePiece tokens
Embedding dim: 768 (base) / 1024 (large)
Strengths:     Much stronger than mBERT on all cross-lingual tasks
               Better low-resource language coverage
               Standard base for fine-tuning multilingual retrievers
```

### LaBSE (Language-agnostic BERT Sentence Encoder, Google, 2020)

```
Languages:     109 languages
Architecture:  BERT-base with dual encoder
Training:      Contrastive on 6B translation pairs
Embedding dim: 768
Strengths:     Strongest zero-shot cross-lingual sentence similarity
               Directly optimized for cross-lingual retrieval
               Works well for bitext mining (finding translations)
Weaknesses:    Older model, more recent models outperform on some tasks
```

### mE5 (Multilingual E5, Microsoft, 2023)

```
Languages:     ~100 languages
Architecture:  XLM-R-base / XLM-R-large
Training:      Weakly supervised on multilingual text pairs
               + fine-tuned on multilingual MSMARCO
Embedding dim: 768 (base) / 1024 (large)
Use prefix:    "query: " for queries, "passage: " for documents
Strengths:     Strong multilingual retrieval
               Consistent with E5 monolingual counterpart
               Good BEIR multilingual performance
```

### mGTE (Multilingual GTE, Alibaba, 2023)

```
Languages:     ~70 languages
Architecture:  XLM-R-base
Training:      Large-scale multilingual contrastive training
Embedding dim: 768
Strengths:     Strong performance on MIRACL (multilingual retrieval benchmark)
               Good balance of language coverage and quality
```

### Cohere Embed Multilingual v3

```
Languages:     100+ languages
Architecture:  Proprietary (API access only)
Training:      Large-scale multilingual training
Embedding dim: 1024
Strengths:     State-of-the-art on multilingual MTEB
               Strong on both retrieval and classification
Weaknesses:    Commercial API, not self-hostable
```

### SONAR (Meta, 2023)

```
Languages:     200 languages (including many low-resource)
Architecture:  Encoder-decoder with language tokens
Training:      Massive parallel corpus across 200 languages
Embedding dim: 1024
Strengths:     Widest language coverage of any public model
               Designed for language-agnostic sentence representation
               Strong on truly low-resource languages
```

## The Language Imbalance Problem

Multilingual models are not equally good across all languages:

```
High-resource (strong performance):
  English, French, German, Spanish, Chinese, Japanese, Korean
  → Large Wikipedia, abundant web text, many parallel corpora

Medium-resource (moderate performance):
  Arabic, Hindi, Portuguese, Russian, Turkish, Polish
  → Moderate Wikipedia, some parallel corpora

Low-resource (weak performance):
  Swahili, Nepali, Tagalog, Yoruba, Welsh, hundreds of others
  → Small Wikipedia or none, few parallel corpora
```

This imbalance is fundamental - a model can only learn representations as
good as its training data. For low-resource languages, embeddings may not
properly align with higher-resource languages in cross-lingual retrieval.

Strategies for improving low-resource language performance:

- Domain-specific parallel data collection
- Machine translation augmentation (translate queries to high-resource language)
- Language-specific fine-tuning with even small amounts of labeled data
- Models specifically designed for low-resource languages (SONAR)

## Multilingual Tokenization

Tokenization for multilingual models has additional complexity:

### Shared subword vocabulary

All languages share a single vocabulary. Common patterns across languages
(numbers, punctuation, some shared roots in related languages) are tokenized
as single tokens. Language-specific characters get their own tokens.

### The fertility problem

Some languages produce many more tokens per word than others:

```
English: "information retrieval" → 2 tokens (base vocabulary well-covered)
Finnish: "tiedonhakujärjestelmä" → 8 tokens (agglutinative, fewer in vocab)
Chinese: "信息检索系统" → 4 tokens (each character often = 1 token, efficient)
Arabic:  "نظام استرجاع المعلومات" → 12 tokens (script, diacritics)
```

High-fertility languages get less of the model's context window for the same
semantic content. A 512-token limit represents much less text in Finnish than
in English. This directly harms retrieval quality for these languages.

Modern multilingual models use larger shared vocabularies (250K SentencePiece
tokens in XLM-R) to reduce the fertility problem for diverse scripts.

## MIRACL - The Multilingual Retrieval Benchmark

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is
the standard benchmark for evaluating multilingual retrieval:

```
Languages:  18 languages across diverse language families
  Tier 1:  Arabic, Bengali, English, Finnish, Indonesian, Korean, Russian, Swahili,
           Telugu, Thai (well-resourced, primary evaluation)
  Tier 2:  Chinese, French, German, Hindi, Japanese, Persian, Portuguese, Yoruba

Corpus:    Wikipedia in each language (varies: 100K to 6M articles)
Queries:   800+ queries per language from Wikipedia editors
Judgments: Human-labeled relevant passages per query, per language

Primary metric: nDCG@10
```

MIRACL evaluates monolingual retrieval (same language query and corpus) rather
than cross-lingual. This is important: monolingual multilingual models vs
cross-lingual models test different capabilities.

## mMARCO - Multilingual MS MARCO

MS MARCO translated into 13 languages using machine translation:

```
Languages:  Arabic, Chinese, Dutch, French, German, Hindi, Indonesian,
            Italian, Japanese, Portuguese, Russian, Spanish, Vietnamese
            + English (original)

Corpus:     MS MARCO passages translated into each language
Queries:    MS MARCO queries translated into each language

Use:        Training multilingual retrieval models
            Zero-shot evaluation of cross-lingual retrieval
```

mMARCO enables training dense retrievers for languages where no native labeled
data exists - train on translated MS MARCO, evaluate on native MIRACL.

## Choosing a Multilingual Embedding Model

| Scenario                             | Recommended model                     |
| ------------------------------------ | ------------------------------------- |
| General multilingual retrieval       | intfloat/multilingual-e5-base         |
| High quality, GPU available          | intfloat/multilingual-e5-large        |
| 109 languages, cross-lingual STS     | sentence-transformers/LaBSE           |
| Fast inference, constrained hardware | paraphrase-multilingual-MiniLM-L12-v2 |
| Low-resource languages (200+)        | SONAR (Meta)                          |
| Managed API, maximum performance     | Cohere Embed Multilingual v3          |
| Retrieval benchmark (MIRACL)         | Alibaba-NLP/gte-multilingual-base     |

## Common Pitfalls in Multilingual IR

| Pitfall                                         | Symptom                                  | Fix                                     |
| ----------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| Not adding E5 prefix                            | Poor retrieval quality with mE5 models   | Add "query: " and "passage: " prefixes  |
| Same model for high- and low-resource languages | Low-resource languages retrieve poorly   | Use SONAR or provide translated queries |
| Not normalizing embeddings before indexing      | Similarity scores wrong                  | Always normalize to unit length         |
| Language filtering too strict                   | Misses relevant cross-lingual documents  | Only filter when explicitly needed      |
| Assuming equal quality across all languages     | Over-trust low-resource language results | Test per-language metrics separately    |
| Using monolingual model for multilingual corpus | Zero cross-lingual matching              | Use multilingual model or translate     |

## My Summary

Multilingual embeddings map text from multiple languages into a shared vector
space where semantically equivalent sentences across languages are geometrically
close. Three training paradigms exist: multilingual MLM pretraining (mBERT, XLM-R)
learns cross-lingual transfer through shared vocabulary and parameter sharing;
parallel corpus training (LaBSE, LASER) directly aligns translations in embedding
space through contrastive objectives; and modern models like mE5 and mGTE combine
large-scale multilingual pretraining with contrastive fine-tuning on translated
retrieval pairs. Language imbalance is a fundamental limitation - high-resource
languages (English, French, German) consistently outperform low-resource languages
due to training data scarcity. MIRACL is the standard benchmark for multilingual
retrieval across 18 languages. For production systems, translation fallback for
low-confidence retrievals and per-language quality monitoring are essential
operational practices. The choice between self-hosted multilingual models and
managed APIs depends on language coverage requirements, latency constraints, and
data privacy considerations.
