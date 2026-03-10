# Tokenization in Information Retrieval

Tokenization is the first and most fundamental step in most Information Retrieval (IR) and Natural Language Processing (NLP) pipelines. It converts raw text into smaller units called tokens, which are easier for machines to process.

Without tokenization, tasks like search, indexing, ranking, text classification, and question answering would not be possible.

- [Tokenization in Information Retrieval](#tokenization-in-information-retrieval)
  - [1. What is Tokenization?](#1-what-is-tokenization)
  - [2. Why Tokenization is Important in IR](#2-why-tokenization-is-important-in-ir)
  - [3. Tokenization in an IR Pipeline](#3-tokenization-in-an-ir-pipeline)
  - [4. Types of Tokenization](#4-types-of-tokenization)
    - [4.1 Word Tokenization](#41-word-tokenization)
    - [4.2 Sentence Tokenization](#42-sentence-tokenization)
    - [4.3 Character Tokenization](#43-character-tokenization)
    - [4.4 Subword Tokenization](#44-subword-tokenization)
  - [5. Tokenization Challenges in IR](#5-tokenization-challenges-in-ir)
    - [5.1 Punctuation](#51-punctuation)
    - [5.2 Contractions](#52-contractions)
    - [5.3 Hyphenated Words](#53-hyphenated-words)
    - [5.4 Email Addresses](#54-email-addresses)
    - [5.5 URLs](#55-urls)
    - [5.6 Numbers](#56-numbers)
  - [6. Tokenization Rules in IR Systems](#6-tokenization-rules-in-ir-systems)
  - [7. Good Tokenization Practices](#7-good-tokenization-practices)
  - [8 Limitations of Tokenization](#8-limitations-of-tokenization)

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

## 5. Tokenization Challenges in IR

Tokenization may seem simple but has many edge cases.

### 5.1 Punctuation

Example:

```bash
Hello, world!
```

Possible tokens:

```bash
["hello", "world"]
```

But naive tokenization might produce:

```bash
["hello,", "world!"]
```

### 5.2 Contractions

Example:

```bash
don't
```

Possible tokenizations:

```bash
["do", "not"]
```

or

```bash
["don't"]
```

### 5.3 Hyphenated Words

Example:

```bash
state-of-the-art
```

Possible tokens:

```bash
["state", "of", "the", "art"]
```

or

```bash
["state-of-the-art"]
```

### 5.4 Email Addresses

Example:

```bash
support@example.com
```

Should ideally stay as one token.

### 5.5 URLs

Example:

```bash
https://example.com
```

Usually treated as single tokens.

### 5.6 Numbers

Example:

```bash
Price is $100.50
```

Tokenization options:

```bash
["price", "is", "100.50"]
```

or

```bash
["price", "is", "$", "100.50"]
```

## 6. Tokenization Rules in IR Systems

IR systems often apply rules like:

1. **Case Folding**

    Convert to lowercase.

    ```bash
    Apple → apple
    ```

2. **Remove Punctuation**

    ```bash
    "hello!" → "hello"
    ```

3. **Normalize Numbers**

    Example:

    2025 --> <25> or |NUM|

4. **Special Token Handling**

    Keep certain patterns intact:

    - emails
    - URLs
    - hashtags
    - mentions

## 7. Good Tokenization Practices

Best practices:

- Always convert to lowercase
- Remove unnecessary punctuation
- Handle URLs and emails carefully
- Keep numbers meaningful
- Avoid overly aggressive splitting

## 8 Limitations of Tokenization

Tokenization alone cannot handle:

- synonyms
- spelling variations
- semantics
- context

Example:

**car** and **automobile** both refer to the same concept but tokenization does not capture this.
