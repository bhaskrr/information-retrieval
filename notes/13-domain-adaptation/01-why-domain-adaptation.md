# Why Domain Adaptation

Domain adaptation in IR is the set of techniques used to make retrieval models
trained on general-purpose datasets perform well on a specific target domain -
such as biomedical literature, legal documents, source code, financial reports,
or customer support conversations - where the vocabulary, query style, document
structure, and relevance criteria differ significantly from the training distribution.
A model trained on web search queries from MS MARCO may work reasonably well out
of the box on general topics but fail systematically on specialized domains where
terminology, phrasing, and relevance judgments diverge from what it learned. Domain
adaptation bridges this gap without requiring a complete retraining from scratch.

## Intuition

Consider a dense retrieval model trained entirely on MS MARCO - web search queries
and passages from Bing. The training distribution looks like:

```
Queries: "what is gradient descent", "how to fix a leaky faucet",
         "best python libraries for data analysis", "who invented the telephone"

Documents: Wikipedia passages, news articles, general web pages
```

Now deploy this model on a biomedical retrieval task:

```
Queries: "ACE2 receptor SARS-CoV-2 binding affinity", "remdesivir cytokine
         storm immunopathology", "BRCA1 germline mutation penetrance"

Documents: PubMed abstracts, clinical trial reports, molecular biology papers
```

The model has never seen these terms. "ACE2", "SARS-CoV-2", "remdesivir", and
"BRCA1" are not in MS MARCO's vocabulary. The query style is completely different -
MS MARCO queries are conversational and natural, biomedical queries are technical
and dense with domain-specific terminology. The document structure is different -
PubMed abstracts follow a strict IMRAD format (Introduction, Methods, Results,
Discussion) that the model has never learned to parse.

BEIR research confirms this empirically. Models that score NDCG@10 = 0.45 on
MS MARCO often score 0.25-0.35 on biomedical, legal, and scientific retrieval
tasks - a 30-40% relative decline. BM25, which has no learned parameters, often
outperforms these neural models on out-of-domain tasks because it generalizes
perfectly by design.

Domain adaptation addresses this performance gap using techniques that range from
no additional data (prompting, zero-shot transfer) to small labeled datasets
(few-shot fine-tuning) to full domain-specific training pipelines.

## The Domain Gap in Practice

The domain gap manifests through several distinct mechanisms:

### 1. Vocabulary mismatch at the token level

Domain-specific terms are split into uninformative subword tokens by general
tokenizers:

```
BERT tokenizer on biomedical terms:
  "immunopathology"  → ["immune", "##path", "##ology"]   (3 tokens, partial)
  "desulfurization"  → ["de", "##sulfur", "##ization"]   (3 tokens, fragmented)
  "BRCA1"            → ["BR", "##CA", "##1"]             (meaningless fragments)
  "clopidogrel"      → ["cl", "##op", "##id", "##og", "##rel"] (5 tokens, destroyed)
```

The model has learned rich representations for "immune", "path", and "ology" as
individual concepts - but "immunopathology" as a compound medical term has no
learned representation beyond the sum of its fragments. This is fundamentally
different from how a domain expert understands it.

Domain-adapted tokenizers trained on domain-specific text learn compound terms
as single units, producing much more informative representations.

### 2. Query style mismatch

Different domains have fundamentally different query conventions:

```
Web search (MS MARCO style):
  "how does attention mechanism work"
  "best practices for machine learning"
  "what is transformer architecture"

Biomedical queries:
  "SARS-CoV-2 spike protein ACE2 receptor binding"
  "randomized controlled trial metformin type 2 diabetes HbA1c"

Legal queries:
  "damages for breach of implied warranty of merchantability"
  "Fourth Amendment reasonable expectation of privacy digital"

Code search queries:
  "parse JSON nested objects Python"
  "async await Promise.all JavaScript concurrency"
```

Models trained on web search learn to handle natural language questions.
They struggle with the dense, technical, noun-phrase-heavy style of biomedical
or legal queries.

### 3. Relevance criteria mismatch

What counts as relevant differs across domains:

```
Web search: any page discussing the topic is potentially relevant
Biomedical: only peer-reviewed papers with controlled methodology are relevant
Legal:      only documents from the same jurisdiction and time period are relevant
Code search: only code that actually solves the problem is relevant (not tutorials)
Enterprise: only documents the user has access rights to are relevant
```

A model trained on web search relevance may surface Wikipedia pages and news
articles for a biomedical query when only PubMed abstracts are relevant.

### 4. Document structure mismatch

Domain-specific document structure carries semantic signal not present in web pages:

```
Scientific paper:
  Abstract: one sentence per section (background, methods, results, conclusion)
  Methods section: critical for understanding experimental validity
  Figure captions: often contain key quantitative results

Legal document:
  Headings: jurisdictional markers, citation formats
  Citations: [2024] EWCA Civ 123 → specific court, year, case number
  Whereas clauses: critical for contract meaning

Code:
  Function signatures: parameter types, return types (semantic content)
  Comments: intent and constraints
  Import statements: dependencies and context
```

Models that treat all text as equivalent flat prose miss these structural signals.

## The Cost-Benefit Analysis

Domain adaptation exists on a spectrum from zero cost (use the pretrained model
as-is) to high cost (retrain from scratch on domain data). Every point on the
spectrum involves a tradeoff:

| Approach                   | Cost          | Expected gain                  |
| -------------------------- | ------------- | ------------------------------ |
| Zero-shot transfer         | $0            | Baseline; sometimes sufficient |
| Prompt engineering         | Hours         | 5-15% improvement              |
| Unsupervised domain tuning | Days (GPU)    | 10-20% improvement             |
| Few-shot fine-tuning       | Days + data   | 15-30% improvement             |
| Full domain fine-tuning    | Weeks + data  | 25-50% improvement             |
| Retrain from scratch       | Months + data | Maximum; rarely justified      |

The right approach depends on how large the domain gap is, how much labeled data
exists, and how much the performance gap costs in your application.

## When Domain Adaptation is Necessary

### Strong domain adaptation signal (high priority)

**Highly specialized vocabulary** - more than 30% of domain queries contain terms
absent from general vocabulary. Biomedical, legal, chemical, and financial domains
typically qualify.

**Different relevance criteria** - domain experts disagree with general relevance
judgments. If you showed retrieved results to a domain expert and they consistently
said "this is not relevant in my field," the model needs adaptation.

**Systematic BM25 outperformance** - if BM25 beats your neural model on your domain,
the neural model has not transferred. BM25 outperforming a neural model is a strong
signal that the neural model has not learned domain-appropriate representations.

**User query patterns differ from MS MARCO** - if your users type dense technical
noun phrases rather than natural language questions, the model needs adaptation.

### Weak domain adaptation signal (lower priority)

**General topics with specialized content** - "machine learning research" or
"software engineering best practices" may not require adaptation since these topics
appear in MS MARCO's training distribution.

**Short factual queries** - "who is the CEO of Apple" style queries are well-handled
by zero-shot transfer.

**Already strong zero-shot performance** - if BEIR scores on domain-similar datasets
are strong (NDCG@10 > 0.45), adaptation may not be worth the investment.

## Measuring the Domain Gap

Before investing in domain adaptation, measure the gap explicitly:

### Step 1 - Establish BM25 baseline

BM25 is the zero-effort, domain-agnostic baseline. If your neural model cannot
beat BM25 on your domain, neural adaptation is necessary:

```python
from rank_bm25 import BM25Okapi

# Tokenize and index domain corpus
tokenized = [doc.lower().split() for doc in corpus]
bm25      = BM25Okapi(tokenized)

# Score domain queries
bm25_ndcg = evaluate_ndcg(bm25, domain_queries, domain_qrels)
```

### Step 2 - Evaluate pretrained neural model

Use a strong zero-shot model (E5-base, GTE-base) and measure NDCG@10:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/e5-base-v2")
neural_ndcg = evaluate_ndcg(model, domain_queries, domain_qrels)
```

### Step 3 - Measure the gap

```
Gap = bm25_ndcg - neural_ndcg   ← positive means BM25 is better
Relative gap = (bm25_ndcg - neural_ndcg) / bm25_ndcg

Gap > 0:         neural model fails to generalize, adaptation needed
Gap = -0.02 to 0: marginal, adaptation may help slightly
Gap < -0.05:     neural model already generalizes well, adaptation optional
```

### Step 4 - Estimate adaptation return

Sample 50-100 domain queries. For each, manually assess whether the top-5 results
are relevant. Measure precision@5 manually:

```
Manual precision@5 < 0.4: severe domain gap, high adaptation priority
Manual precision@5 0.4-0.6: moderate gap, adaptation beneficial
Manual precision@5 > 0.6: mild gap, adaptation optional
```

## The Domain Adaptation Decision Tree

```
Start: Deploy pretrained model on new domain
              ↓
     Measure BM25 vs neural NDCG
              ↓
     ┌────────────────────┐
     │ BM25 > neural?     │
     └────────────────────┘
           │          │
          Yes         No
           ↓          ↓
   How large is     Gap < 0.05:
   the gap?        monitor only
           │
    ┌──────┴──────┐
    │             │
  Small         Large
  (< 0.05)     (> 0.05)
    ↓             ↓
  Few-shot      Do you have
  fine-tune     labeled data?
    ↓              │
  100-500       ┌──┴──┐
  examples      │     │
               Yes    No
                ↓     ↓
           Full    Synthetic
           fine-   data or
           tuning  UDAP/GPL
```

## Domain Adaptation Techniques Overview

The broader toolkit includes:

**Unsupervised Domain Adaptive Pre-training (UDAP/TSDAE)**
Continue BERT/RoBERTa pretraining on domain text without any labels.
Updates token representations to reflect domain vocabulary and style.
Cost: days of GPU time. Gain: 5-15% improvement.

**GPL (Generative Pseudo-Labeling)**
Use an LLM to generate synthetic (query, document) pairs from your corpus.
Then fine-tune a retrieval model on synthetic pairs.
Requires no human labeling. Cost: LLM API calls + fine-tuning GPU time.

**Domain-specific tokenizers**
Retrain WordPiece/BPE tokenizer on domain text before training the model.
Improves subword segmentation for domain terminology.
Example: BioBERT uses a tokenizer trained on PubMed.

**Hybrid retrieval as a domain adaptation hedge**
Combine BM25 (domain-agnostic) with neural retrieval (domain-limited).
BM25 handles domain terminology through exact matching; neural handles
semantics where it has learned representations.
No training required. Immediate improvement.

## The Pragmatic First Steps

Before investing in complex domain adaptation, two cheap interventions
often close most of the gap:

**Step 1 - Switch to hybrid search immediately**
BM25 outperforming your neural model is a symptom with an immediate cure:
combine both. Hybrid search is free, takes an hour to implement, and typically
recovers 50-70% of the performance gap without any training.

**Step 2 - Use a better zero-shot base model**
Not all pretrained models generalize equally. E5-large, GTE-large, and
INSTRUCTOR generally outperform all-MiniLM on out-of-domain tasks. Before
fine-tuning, try a larger or better-trained zero-shot model:

```
Model comparison on biomedical (approximate BEIR TREC-COVID NDCG@10):
  all-MiniLM-L6-v2:  0.54
  msmarco-bert-base: 0.53
  e5-base-v2:        0.58
  e5-large-v2:       0.63
  gtr-t5-large:      0.62
```

A model swap alone can close 30-50% of the domain gap before any fine-tuning.

## Where This Fits in the Progression

```
All prior modules:              learn how IR works in general
Practical systems:              deploy what you learned
Domain adaptation:              make it work on your specific domain
  01-why-domain-adaptation.md   measure the gap  ← you are here
  02-few-shot-retrieval.md      close the gap with minimal data
  03-domain-specific-datasets.md close the gap with domain resources
```

## My Summary

Domain adaptation addresses the systematic performance gap that occurs when
retrieval models trained on general-purpose datasets are deployed on specialized
domains with different vocabulary, query style, document structure, and relevance
criteria. The gap is measured by comparing BM25 and neural retrieval NDCG@10 on
the target domain - when BM25 wins, the neural model has failed to generalize and
adaptation is needed. The domain gap has four root causes: vocabulary mismatch
at the token level (domain terms become uninformative subword fragments), query
style mismatch (technical vs conversational queries), relevance criteria mismatch
(domain-specific notions of relevance), and document structure mismatch (domain
formats not seen during training). Before investing in domain-specific training,
two cheap interventions often close most of the gap: switching to hybrid search
(BM25 + neural) and switching to a larger, better-generalizing base model like E5
or GTE. When these are insufficient, the adaptation toolkit spans from synthetic
data generation to few-shot fine-tuning to full domain pretraining.
