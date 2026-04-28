# Query Tokenization

Query tokenization is the process of splitting a raw user query string into a
sequence of meaningful units - tokens - that can be processed by a retrieval
system. While document tokenization and query tokenization share the same
fundamental goal, they have different constraints, failure modes, and design
priorities. Query tokenization must handle misspellings, abbreviations, mixed
languages, special characters, URLs, code snippets, and natural language fragments, often without the surrounding context that makes document tokenization easier. Getting query tokenization right is a prerequisite for every subsequent processing step in the retrieval pipeline.

## Intuition

Tokenization looks trivial - split on whitespace, done. In practice it is where
retrieval systems encounter the full messiness of human language. Users type fast,
make mistakes, use domain-specific notation, mix languages, and express the same
information need in unpredictable ways.

Consider these real-world query inputs:

```bash
"pytorch>=2.0 cuda11.8 installation"
"what's the difference btw BERT vs GPT"
"C++ std::vector<int> iteration"
"co-occurrence matrix NLP"
"don't understand backprop"
"résumé parsing NLP"
"新型语言模型 transformer"
"http://arxiv.org/abs/1706.03762 summary"
```

Each of these presents a different tokenization challenge. A naive whitespace
splitter gets some of them partially right and completely destroys others. A
retrieval system that cannot tokenize "pytorch>=2.0" into meaningful units
will fail to match it against any document, regardless of how good the
ranking model is.

Tokenization errors compound downstream - a poorly tokenized query produces
bad term vectors, bad dense embeddings, and bad retrieval regardless of how
sophisticated the retrieval model is. A 5% improvement in tokenization often
produces larger downstream gains than a 5% improvement in the ranking model.

## Query Tokenization vs Document Tokenization

Both split text into tokens but with different priorities:

| Property           | Document tokenization | Query tokenization           |
| ------------------ | --------------------- | ---------------------------- |
| Text length        | Sentences to pages    | 1-20 words typically         |
| Context available  | Full document context | Isolated query string        |
| Error rate         | Low (edited text)     | High (typos, abbreviations)  |
| Special chars      | Rare                  | Common (code, URLs, symbols) |
| Processing time    | Offline, batch        | Online, latency-sensitive    |
| Retry possible     | No need               | Query reformulation likely   |
| Domain consistency | Usually consistent    | Highly variable              |

Document tokenization can afford to be slower and more conservative because
it happens offline. Query tokenization must be fast (< 1ms), robust to unusual
inputs, and handle the full diversity of things users type.

## Tokenization Strategies

### 1. Whitespace tokenization

Split on whitespace characters. The simplest approach:

```
"dense retrieval BM25 comparison" → ["dense", "retrieval", "BM25", "comparison"]
```

Works perfectly for well-formed natural language queries. Fails immediately on:

- Hyphenated terms: "co-occurrence" → ["co-occurrence"] (one token, correct)
  or "co-occurrence" → ["co", "occurrence"] (two tokens, loses meaning)
- Punctuation: "don't" → ["don't"] (correct) or ["don", "t"] (wrong)
- Code/technical: "std::vector<int>" → one token (should be multiple)

### 2. Punctuation-aware tokenization

Extends whitespace tokenization to handle punctuation:

```
Rules:
  Split on: spaces, commas, semicolons, parentheses
  Keep together: hyphens within words, apostrophes within words
  Split but preserve: periods (sentence boundaries vs abbreviations)
```

Handles most natural language well. Still fails on technical notation.

### 3. Rule-based tokenization

A hand-crafted set of rules for domain-specific patterns:

```
Pattern: version numbers         → keep together: "python3.11", "v2.0.1"
Pattern: email addresses         → keep together: "user@domain.com"
Pattern: URLs                    → keep together or split at /, ., ?
Pattern: programming syntax      → split at ::, ->, <>, []
Pattern: mathematical notation   → split at +, -, ×, =, ≥, ≤
Pattern: hashtags                → keep together: "#NLP", "#bert"
Pattern: camelCase               → optionally split: "BertModel" → "Bert", "Model"
```

Effective for known domains. Requires ongoing maintenance as new patterns emerge.

### 4. WordPiece / BPE (Subword tokenization)

Learned tokenization used by BERT, GPT, and most modern transformers. Splits
words into frequently-occurring subword units:

```
"unhappiness" → ["un", "##happiness"]
"backpropagation" → ["back", "##pro", "##pagation"]
"pytorch" → ["py", "##torch"]
"BERT" → ["B", "##ER", "##T"]   ← poor for abbreviations
```

Handles out-of-vocabulary words gracefully by decomposing them into known
subwords. The key property: no word is ever completely unknown.

For queries, subword tokenization is used implicitly when queries are passed
to BERT-based bi-encoders or cross-encoders. The tokenizer handles everything
before the query reaches the neural model.

### 5. Hybrid tokenization

Combine rule-based preprocessing with a subword tokenizer:

```
Raw query: "pytorch>=2.0 cuda11.8 installation"

Step 1 - Rule-based normalization:
  "pytorch >= 2.0 cuda 11.8 installation"

Step 2 - Subword tokenization (WordPiece):
  ["py", "##torch", ">=", "2", ".", "0", "cu", "##da", "11", ".", "8", "installation"]
```

The rule-based step handles domain-specific patterns, the subword tokenizer
handles the rest. This is the approach used in most production retrieval systems.

## Special Token Handling

### Hyphenated terms

Hyphens create a genuine ambiguity:

```
"co-occurrence"  → compound word, should stay together
"state-of-the-art" → multi-word phrase, split into components
"2023-2024"      → date range, keep as single unit or split
"e-mail"         → synonym for "email", normalize and keep
```

Strategy: keep hyphenated terms as single tokens for indexing, but also index
component words to support partial matching. "co-occurrence" is indexed as both
"co-occurrence" and "co" + "occurrence".

### Apostrophes and contractions

```
"don't"  → keep as "don't" or expand to "do not"
"it's"   → "it's" or "it is"
"I've"   → "I've" or "I have"
"NLP's"  → "NLP" (possessive - strip apostrophe-s)
```

For retrieval, expand contractions to increase matching recall:
"don't understand" → "do not understand" covers both query variants.

### Numbers and units

```
"10GB"       → "10" + "GB" or keep as "10gb"
"2.5x faster" → "2.5" + "x" + "faster" or keep as "2.5x"
"top-5"      → "top" + "5" or keep as "top-5"
"bert-base"  → "bert" + "base" (model name component splitting)
```

Domain-specific rules are required. In technical IR, "bert-base" splitting
into components helps - users may query "bert" or "base" separately.

### URLs

```
"https://arxiv.org/abs/1706.03762"
→ Option A: keep as single token (exact URL matching)
→ Option B: split into ["arxiv", "1706.03762"] (component matching)
→ Option C: extract meaningful parts: ["arxiv", "attention", "transformer"] (requires lookup)
```

For retrieval, option B is usually best - split at / and . to enable partial
URL matching.

### Code and technical notation

```
"std::vector<int>"  → ["std", "vector", "int"]
"O(n log n)"        → ["O", "n", "log", "n"]
"@staticmethod"     → ["staticmethod"] or ["@", "staticmethod"]
"#include <stdio>"  → ["include", "stdio"]
"git push --force"  → ["git", "push", "force"]
```

Strip programming punctuation, split at language-specific boundaries, normalize
common patterns.

### Multilingual queries

```
"transformer 注意力机制"         → handle both English and Chinese characters
"résumé parsing"                  → normalize accents or keep original
"über efficient model"            → handle unicode normalization
```

Two approaches:

- Language detection → route to language-specific tokenizer
- Universal tokenizer that handles multiple scripts (sentencepiece, mBART tokenizer)

### CamelCase and concatenated terms

```
"BertModel"        → ["Bert", "Model"] or keep as-is
"GPT4"             → ["GPT", "4"] or keep as-is
"NLPtasks"         → ["NLP", "tasks"]
"spaCy"            → keep as-is (proper noun)
```

CamelCase splitting improves recall for code search and technical IR. Risk:
"iPhone" should not be split into "i" + "Phone".

## Query-Specific Tokenization Considerations

### Short query length

Most queries are 2-5 tokens. Short queries mean:

- Every token matters more - losing one token is catastrophic
- Less context for disambiguation
- Higher sensitivity to tokenization errors

A tokenization error that converts a 4-token query into 3 tokens loses 25% of
the query signal. The same error in a 400-word document loses 0.25%.

### Query intent preservation

Tokenization should preserve semantic units that carry intent:

```
"best python libraries for machine learning"
→ Good: ["best", "python", "libraries", "machine learning"]
→ Bad:  ["best", "python", "libraries", "machine", "learning"]

"machine learning" is a semantic unit - splitting it loses the compound meaning.
Bigram preservation is important for multi-word entity recognition.
```

### Query segmentation vs tokenization

Query segmentation (covered in 02-query-segmentation.md) is distinct from
tokenization. Tokenization splits into atomic units. Segmentation groups tokens
into meaningful phrases:

```
Tokenization:  "best python libraries machine learning"
               → ["best", "python", "libraries", "machine", "learning"]

Segmentation:  ["best", "python libraries", "machine learning"]
               → identifies meaningful phrases within the token sequence
```

Both happen in the query processing pipeline, with tokenization first.

## Code

```python
# pip install spacy nltk transformers
# python -m spacy download en_core_web_sm
# python -m nltk.downloader punkt

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


# ── Part 1: Rule-based query tokenizer ────────────────────────────────────

@dataclass
class TokenizerConfig:
    """Configuration for query tokenization behavior."""
    lowercase:              bool = True
    normalize_unicode:      bool = True
    expand_contractions:    bool = True
    split_camelcase:        bool = True
    split_hyphens:          bool = False   # keep hyphenated terms by default
    handle_urls:            bool = True
    handle_code:            bool = True
    preserve_numbers:       bool = True
    min_token_length:       int  = 1
    max_token_length:       int  = 100


class QueryTokenizer:
    """
    Rule-based query tokenizer designed for IR.
    Handles the common special cases encountered in real query logs.
    """

    CONTRACTIONS = {
        "don't":    "do not",
        "doesn't":  "does not",
        "didn't":   "did not",
        "can't":    "cannot",
        "won't":    "will not",
        "isn't":    "is not",
        "aren't":   "are not",
        "wasn't":   "was not",
        "weren't":  "were not",
        "haven't":  "have not",
        "hasn't":   "has not",
        "hadn't":   "had not",
        "i'm":      "i am",
        "i've":     "i have",
        "i'll":     "i will",
        "i'd":      "i would",
        "it's":     "it is",
        "that's":   "that is",
        "there's":  "there is",
        "they're":  "they are",
        "we're":    "we are",
        "you're":   "you are",
        "what's":   "what is",
        "how's":    "how is",
        "where's":  "where is",
    }

    URL_PATTERN    = re.compile(
        r'https?://[^\s]+|www\.[^\s]+',
        re.IGNORECASE
    )
    VERSION_PATTERN = re.compile(
        r'\b\d+\.\d+(?:\.\d+)*\b'
    )
    CAMELCASE_PATTERN = re.compile(
        r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])'
    )
    CODE_OPERATORS = re.compile(
        r'(::|->|<[^>]+>|\[\]|[<>{}()\[\]])'
    )

    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters - remove accents, normalize forms."""
        text = unicodedata.normalize("NFKD", text)
        text = "".join(
            c for c in text
            if not unicodedata.combining(c)
        )
        return text

    def _expand_contractions(self, text: str) -> str:
        """Expand common contractions for better term matching."""
        text_lower = text.lower()
        for contraction, expansion in self.CONTRACTIONS.items():
            if contraction in text_lower:
                text = re.sub(
                    re.escape(contraction),
                    expansion,
                    text,
                    flags=re.IGNORECASE
                )
        return text

    def _handle_url(self, text: str) -> str:
        """
        Process URLs - extract meaningful components.
        e.g. https://arxiv.org/abs/1706.03762 → arxiv 1706.03762
        """
        def extract_url_parts(match):
            url      = match.group(0)
            url      = re.sub(r'https?://', '', url)
            url      = re.sub(r'www\.', '', url)
            parts    = re.split(r'[/\.\?&=]', url)
            meaningful = [
                p for p in parts
                if p and len(p) > 1 and not p.isdigit() or len(p) > 4
            ]
            return " ".join(meaningful)

        return self.URL_PATTERN.sub(extract_url_parts, text)

    def _handle_code(self, text: str) -> str:
        """
        Handle code-like syntax - split at operators, remove brackets.
        e.g. std::vector<int> → std vector int
        """
        text = self.CODE_OPERATORS.sub(' ', text)
        text = re.sub(r'[;{}()\[\]]', ' ', text)
        return text

    def _split_camelcase(self, token: str) -> list[str]:
        """
        Split camelCase tokens into component words.
        e.g. BertModel → Bert Model
        """
        parts = self.CAMELCASE_PATTERN.sub(' ', token).split()
        return parts if len(parts) > 1 else [token]

    def _handle_hyphens(self, token: str) -> list[str]:
        """
        Handle hyphenated tokens.
        Returns both the full form and components for dual indexing.
        """
        if '-' not in token:
            return [token]

        components = token.split('-')
        components = [c for c in components if c]

        if self.config.split_hyphens:
            return components
        else:
            return [token]   # keep hyphenated form

    def _clean_token(self, token: str) -> Optional[str]:
        """
        Clean and validate a single token.
        Returns None if token should be discarded.
        """
        # Remove leading/trailing punctuation (but not internal)
        token = token.strip('.,;:!?\'"')

        # Validate length
        if len(token) < self.config.min_token_length:
            return None
        if len(token) > self.config.max_token_length:
            return None

        # Remove pure punctuation tokens
        if all(not c.isalnum() for c in token):
            return None

        return token

    def tokenize(self, query: str) -> list[str]:
        """
        Tokenize a query string into cleaned tokens.

        Args:
            query: raw user query string

        Returns:
            list of cleaned tokens
        """
        if not query or not query.strip():
            return []

        text = query.strip()

        # Step 1 - unicode normalization
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)

        # Step 2 - lowercase (before contraction expansion)
        if self.config.lowercase:
            text = text.lower()

        # Step 3 - expand contractions
        if self.config.expand_contractions:
            text = self._expand_contractions(text)

        # Step 4 - handle URLs (before general punctuation removal)
        if self.config.handle_urls:
            text = self._handle_url(text)

        # Step 5 - handle code syntax
        if self.config.handle_code:
            text = self._handle_code(text)

        # Step 6 - split on whitespace
        raw_tokens = text.split()

        # Step 7 - per-token processing
        final_tokens = []
        for token in raw_tokens:
            # CamelCase splitting
            if self.config.split_camelcase and any(c.isupper() for c in token[1:]):
                sub_tokens = self._split_camelcase(token)
            else:
                sub_tokens = [token]

            for sub_token in sub_tokens:
                # Hyphen handling
                hyph_tokens = self._handle_hyphens(sub_token)

                for t in hyph_tokens:
                    cleaned = self._clean_token(t)
                    if cleaned is not None:
                        final_tokens.append(cleaned)

        return final_tokens

    def tokenize_with_positions(self,
                                  query: str) -> list[tuple[str, int]]:
        """
        Tokenize preserving approximate original positions.
        Useful for phrase detection and span-based processing.

        Returns:
            list of (token, approximate_start_position) tuples
        """
        tokens  = self.tokenize(query)
        query_l = query.lower()
        result  = []
        pos     = 0

        for token in tokens:
            idx = query_l.find(token, pos)
            if idx >= 0:
                result.append((token, idx))
                pos = idx + len(token)
            else:
                result.append((token, pos))

        return result


# ── Part 2: Subword-aware query tokenizer ─────────────────────────────────

class SubwordQueryTokenizer:
    """
    Combines rule-based preprocessing with subword tokenization.
    Used when queries are passed to transformer-based retrievers.
    Ensures consistent tokenization between queries and the neural model.
    """
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 rule_config: TokenizerConfig = None):
        from transformers import AutoTokenizer

        self.rule_tokenizer = QueryTokenizer(rule_config or TokenizerConfig(
            split_camelcase=True,
            handle_urls=True,
            handle_code=True,
            expand_contractions=True
        ))

        try:
            self.subword_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_name        = model_name
            print(f"Loaded subword tokenizer: {model_name}")
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            self.subword_tokenizer = None

    def tokenize_for_retrieval(self, query: str) -> list[str]:
        """
        Tokenize for sparse retrieval (BM25, TF-IDF).
        Uses rule-based tokenizer - produces clean, readable tokens.
        """
        return self.rule_tokenizer.tokenize(query)

    def tokenize_for_neural(self,
                              query: str,
                              max_length: int = 64,
                              return_tensors: str = "pt") -> dict:
        """
        Tokenize for neural retrieval (bi-encoders, cross-encoders).
        Uses subword tokenizer with rule-based preprocessing.

        Args:
            query:          raw query string
            max_length:     maximum token sequence length
            return_tensors: "pt" for PyTorch, "np" for NumPy, None for lists
        """
        if self.subword_tokenizer is None:
            raise RuntimeError("Subword tokenizer not available")

        # Apply rule-based preprocessing first
        preprocessed = query.strip()
        if self.rule_tokenizer.config.expand_contractions:
            preprocessed = self.rule_tokenizer._expand_contractions(preprocessed)
        if self.rule_tokenizer.config.handle_urls:
            preprocessed = self.rule_tokenizer._handle_url(preprocessed)

        # Subword tokenization
        return self.subword_tokenizer(
            preprocessed,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=return_tensors
        )

    def get_subword_tokens(self, query: str) -> list[str]:
        """
        Get human-readable subword tokens for inspection.
        Useful for debugging tokenization behavior.
        """
        if self.subword_tokenizer is None:
            return self.tokenize_for_retrieval(query)

        ids    = self.subword_tokenizer.encode(query, add_special_tokens=False)
        tokens = self.subword_tokenizer.convert_ids_to_tokens(ids)
        return tokens


# ── Part 3: Query tokenization analyzer ───────────────────────────────────

class QueryTokenizationAnalyzer:
    """
    Analyzes and compares tokenization behavior across different strategies.
    Useful for understanding tokenization decisions and debugging.
    """
    def __init__(self):
        self.simple    = QueryTokenizer(TokenizerConfig(
            lowercase=True,
            normalize_unicode=False,
            expand_contractions=False,
            split_camelcase=False,
            handle_urls=False,
            handle_code=False,
        ))
        self.full      = QueryTokenizer(TokenizerConfig(
            lowercase=True,
            normalize_unicode=True,
            expand_contractions=True,
            split_camelcase=True,
            handle_urls=True,
            handle_code=True,
        ))
        self.subword   = SubwordQueryTokenizer()

    def analyze(self, queries: list[str]) -> None:
        """
        Analyze tokenization of a list of queries across all strategies.
        Shows how each strategy handles different query types.
        """
        print(f"{'Query':<40} {'Strategy':<12} {'Tokens'}")
        print("─" * 80)

        for query in queries:
            simple_toks  = self.simple.tokenize(query)
            full_toks    = self.full.tokenize(query)
            subword_toks = self.subword.get_subword_tokens(query)

            print(f"{query:<40} {'simple':<12} {simple_toks}")
            print(f"{'':40} {'full':<12} {full_toks}")
            print(f"{'':40} {'subword':<12} {subword_toks}")
            print()

    def tokenization_diff(self,
                           query: str) -> dict:
        """
        Show detailed differences between tokenization strategies for a query.
        """
        simple  = self.simple.tokenize(query)
        full    = self.full.tokenize(query)
        subword = self.subword.get_subword_tokens(query)

        return {
            "query":           query,
            "simple":          simple,
            "full":            full,
            "subword":         subword,
            "simple_n_tokens": len(simple),
            "full_n_tokens":   len(full),
            "subword_n_tokens": len(subword),
            "full_vs_simple":  set(full) - set(simple),   # tokens added by full
        }


# ── Part 4: Tokenization evaluation ───────────────────────────────────────

def evaluate_tokenization_impact(
        tokenizer: QueryTokenizer,
        queries: list[dict],
        documents: dict[str, str]) -> dict:
    """
    Evaluate how tokenization affects retrieval using BM25.
    Compares retrieval quality with simple vs full tokenization.

    Args:
        tokenizer:  tokenizer to evaluate
        queries:    list of {"query": str, "relevant": [doc_id]}
        documents:  {doc_id: doc_text}

    Returns:
        retrieval metrics
    """
    from rank_bm25 import BM25Okapi
    import math

    doc_ids   = list(documents.keys())
    doc_texts = list(documents.values())

    # Tokenize documents with same tokenizer
    tokenized_docs = [tokenizer.tokenize(text) for text in doc_texts]
    bm25           = BM25Okapi(tokenized_docs)

    ndcg_scores = []

    for q_data in queries:
        query_tokens = tokenizer.tokenize(q_data["query"])
        relevant     = set(q_data["relevant"])

        if not query_tokens:
            ndcg_scores.append(0.0)
            continue

        scores   = bm25.get_scores(query_tokens)
        ranked   = [doc_ids[i] for i in sorted(
            range(len(scores)), key=lambda x: scores[x], reverse=True
        )]

        # NDCG@5
        dcg  = sum(
            1 / math.log2(i + 2)
            for i, doc_id in enumerate(ranked[:5])
            if doc_id in relevant
        )
        idcg = sum(
            1 / math.log2(i + 2)
            for i in range(min(len(relevant), 5))
        )
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    import numpy as np
    return {
        "mean_ndcg@5": float(np.mean(ndcg_scores)),
        "n_queries":   len(queries)
    }


# ── Run demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyzer = QueryTokenizationAnalyzer()

    test_queries = [
        "pytorch>=2.0 cuda installation",
        "what's the difference btw BERT vs GPT",
        "std::vector<int> iteration C++",
        "co-occurrence matrix NLP",
        "don't understand backpropagation",
        "http://arxiv.org/abs/1706.03762 summary",
        "BertModel fine-tuning classification",
        "résumé parsing NLP pipeline",
        "top-5 python libraries 2024",
    ]

    print("=" * 80)
    print("QUERY TOKENIZATION ANALYSIS")
    print("=" * 80)
    print()

    analyzer.analyze(test_queries)

    # Detailed diff for one query
    print("=" * 80)
    print("DETAILED DIFF - 'pytorch>=2.0 cuda installation'")
    print("=" * 80)

    diff = analyzer.tokenization_diff("pytorch>=2.0 cuda installation")
    for k, v in diff.items():
        print(f"  {k:<20}: {v}")

    # Retrieval impact evaluation
    print("\n" + "=" * 80)
    print("RETRIEVAL IMPACT - simple vs full tokenization")
    print("=" * 80)

    documents = {
        "D1": "PyTorch installation guide cuda 11.8 GPU setup",
        "D2": "BERT versus GPT comparison language models",
        "D3": "C++ standard library vector iterator performance",
        "D4": "co-occurrence matrix word embeddings NLP",
        "D5": "backpropagation tutorial neural network training",
        "D6": "attention is all you need transformer architecture arxiv",
        "D7": "BERT model fine-tuning text classification tutorial",
        "D8": "resume parsing named entity recognition pipeline",
        "D9": "top python libraries data science 2024",
        "D10": "information retrieval evaluation NDCG MAP metrics",
    }

    queries = [
        {"query": "pytorch cuda installation",     "relevant": ["D1"]},
        {"query": "BERT GPT comparison",           "relevant": ["D2"]},
        {"query": "backpropagation tutorial",       "relevant": ["D5"]},
        {"query": "attention transformer arxiv",    "relevant": ["D6"]},
        {"query": "BERT fine-tuning classification","relevant": ["D7"]},
    ]

    simple_tokenizer = QueryTokenizer(TokenizerConfig(
        lowercase=True,
        normalize_unicode=False,
        expand_contractions=False,
        split_camelcase=False,
        handle_urls=False,
        handle_code=False,
    ))
    full_tokenizer = QueryTokenizer()

    simple_metrics = evaluate_tokenization_impact(
        simple_tokenizer, queries, documents
    )
    full_metrics = evaluate_tokenization_impact(
        full_tokenizer, queries, documents
    )

    print(f"\n  Simple tokenizer NDCG@5: {simple_metrics['mean_ndcg@5']:.4f}")
    print(f"  Full tokenizer NDCG@5:   {full_metrics['mean_ndcg@5']:.4f}")
    print(f"  Improvement:             "
          f"{full_metrics['mean_ndcg@5'] - simple_metrics['mean_ndcg@5']:+.4f}")
```

## Common Tokenization Mistakes in IR Systems

```
Mistake                         Impact                    Fix
────────────────────────────────────────────────────────────────────────
Naive whitespace split          Misses punctuation         Use rule-based tokenizer
Not expanding contractions      "don't" ≠ "do not"         Contraction expansion
Stripping all punctuation       "C++" → "C"                Preserve meaningful symbols
Not handling URLs               URL becomes long garbage    URL-specific rules
Inconsistent doc/query tokenize Index-query mismatch       Same tokenizer for both
Aggressive stopword removal     "not relevant" → "relevant" Careful stopword lists
Ignoring case inconsistently    "BERT" ≠ "bert"            Consistent lowercasing
Splitting version numbers       "2.0" → "2", "0"           Preserve version patterns
```

## Query Tokenization vs Indexing Consistency

The most critical rule in query tokenization: **the same tokenization must be
applied to both queries and documents**. A mismatch means the query tokens will
never match document tokens in the index:

```
Document tokenization: "don't" → ["don't"]
Query tokenization:    "don't" → ["do", "not"]

Query "do not" → looks for tokens "do" and "not"
Index contains: "don't" (not "do" or "not")
Result: zero matches despite obvious relevance
```

Always use the same tokenizer instance for both indexing and querying. Store
the tokenizer configuration alongside the index and reload it at query time.

## Where This Fits in the Progression

```
Raw query input
    ↓
Query tokenization     → split into meaningful units  ← you are here
    ↓
Query normalization    → lowercase, unicode, punctuation
    ↓
Stopword handling      → remove or downweight common terms
    ↓
Query segmentation     → identify meaningful phrases
    ↓
Query expansion        → add related terms
    ↓
Retrieval system       → BM25, dense, hybrid
```

Query tokenization is the first step in the pipeline. Every subsequent processing
step operates on the output of the tokenizer. Errors here propagate and amplify
through every downstream component - a dropped token in tokenization can never
be recovered by even the most sophisticated retrieval model.

## My Summary

Query tokenization splits raw user queries into meaningful units for retrieval
processing. While document tokenization handles well-formed text in controlled
conditions, query tokenization must handle the full messiness of real user input -
typos, abbreviations, URLs, code snippets, mixed languages, and domain-specific
notation. Whitespace splitting is a starting point but fails systematically on
technical queries. A robust query tokenizer uses rule-based preprocessing for
known patterns (contractions, URLs, code operators, CamelCase) followed by
subword tokenization for neural retrieval components. The most critical constraint
is consistency - the same tokenizer must be applied to both queries and documents,
otherwise index and query tokens will never match. Query tokenization errors compound
downstream: a token dropped during tokenization is lost to every subsequent processing
step regardless of how sophisticated the retrieval model is.
