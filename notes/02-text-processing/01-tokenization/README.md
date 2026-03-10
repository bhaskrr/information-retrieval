# Tokenization in Information Retrieval

Tokenization is the first and most fundamental step in most Information Retrieval (IR) and Natural Language Processing (NLP) pipelines. It converts raw text into smaller units called tokens, which are easier for machines to process.

Without tokenization, tasks like search, indexing, ranking, text classification, and question answering would not be possible.

- [Tokenization in Information Retrieval](#tokenization-in-information-retrieval)
  - [1. What is Tokenization?](#1-what-is-tokenization)
  - [2. Why Tokenization is Important in IR](#2-why-tokenization-is-important-in-ir)
  - [4. Types of Tokenization](#4-types-of-tokenization)
    - [4.1 Word Tokenization](#41-word-tokenization)
    - [4.2 Sentence Tokenization](#42-sentence-tokenization)
    - [4.3 Character Tokenization](#43-character-tokenization)
    - [4.4 Subword Tokenization](#44-subword-tokenization)

## 1. What is Tokenization?

Tokenization is the process of breaking a stream of text into smaller units called tokens.

These tokens can be:

- Words
- Subwords
- Characters
- Phrases
- Sentences

Tokenization transforms unstructured text into structured units that algorithms can work with.

Example:

```bash
Text: Information retrieval is fascinating.

Tokens: ["information", "retrieval", "is", "fascinating"]
```

## 2. Why Tokenization is Important in IR

Information Retrieval systems (such as search engines) depend on tokenization to:

1. **Build Indexes**: Documents are indexed based on tokens.

    ```bash
    Document:
     "Machine learning improves search systems"

    Tokens:
     ["machine", "learning", "improves", "search", "systems"]
    ```

2. **Process Queries**:

   ```bash
   User query:
    "machine learning"

   Tokens:
    ["machine", "learning"]
    ```

    The system matches these tokens with the index.

3. **Enable Efficient Searching**:

    Without tokenization:

    ```bash
    "machinelearning"
    ```

    Search systems cannot match relevant documents effectively.

## 3. Tokenization in an IR Pipeline

Typical IR pipeline:

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenization] --> C[Normalization] --> D[Stopword Removal] --> E[Stemming/Lemmatization] --> F[Indexing]
```

Tokenization must happen before almost every other text processing step.

## 4. Types of Tokenization

### 4.1 Word Tokenization

The most common approach. Split text based on spaces and punctuation.

Example:

```bash
Text:
"IR systems are powerful."

Tokens:
["IR", "systems", "are", "powerful"]```
```

Python example:

```python
text = "IR systems are powerful"
tokens = text.split()
```

### 4.2 Sentence Tokenization

Splitting text into sentences.

Example:

```bash
Text:
"IR is useful. Search engines rely on it."

Tokens:
[
"IR is useful.",
"Search engines rely on it."
]
```

Used in:

- Summarization
- QA systems
- Document processing

### 4.3 Character Tokenization

Splitting into individual characters.

Example:

```bash
Text:
"IR"

Tokens:
["I", "R"]
```

Used in:

- Character-level language models
- Spelling correction

### 4.4 Subword Tokenization

Modern NLP models use subword tokens to handle unknown words.

Example:

```bash
Word:
unhappiness

Tokens:
["un", "happi", "ness"]
```

**Advantages**:

1. Handles rare words
2. Reduces vocabulary size
3. Improves generalization

**Common algorithms**:

- Byte Pair Encoding (BPE)
- WordPiece
- SentencePiece

Used in **models** like:

- BERT
- GPT
- T5
