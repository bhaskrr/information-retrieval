# Domain-Specific Datasets

Domain-specific datasets are curated collections of queries, documents, and
relevance judgments specifically designed or adapted for a particular subject area -
biomedical literature, legal documents, scientific papers, financial reports, source
code, or any other specialized domain. They serve two distinct roles in domain
adaptation: as training resources to fine-tune retrieval models on domain-appropriate
examples, and as evaluation benchmarks to measure whether a model actually performs
well on domain-specific retrieval tasks. Using the right domain-specific dataset
as both a training source and evaluation benchmark is the highest-leverage action
available when deploying retrieval systems in specialized domains.

## Intuition

MS MARCO is the training dataset that made modern dense retrieval possible. Every
major bi-encoder - DPR, E5, GTE, ColBERT v2 - was either trained on or evaluated
against MS MARCO. This is why they all generalize well to web search but struggle
with biomedical, legal, and scientific retrieval: they have never seen training
examples that reward domain-specific relevance judgments.

Domain-specific datasets are the MS MARCOs of their respective fields. They provide
the labeled signal that teaches a model what "relevant" means in a particular domain.
A model fine-tuned on TREC-COVID (biomedical COVID-19 retrieval) learns that a paper
about "cytokine storm" is relevant to a query about "COVID-19 severity factors" even
when the exact query terms do not appear in the paper. A model fine-tuned on CUAD
(contract understanding) learns that a clause about "liquidated damages" is relevant
to a query about "penalty for contract breach."

The challenge is that most domain-specific datasets are small, covering hundreds or
thousands of queries rather than the millions in MS MARCO. This makes them valuable
for evaluation but often insufficient for training alone - which is why they are
typically combined with general training data (MS MARCO) and domain-specific
pretraining (unsupervised on domain text) before final fine-tuning on domain labels.

## Major Domain-Specific Datasets and Benchmarks

### Biomedical IR

**TREC-COVID**

```
Task:        Ad-hoc retrieval over COVID-19 literature
Corpus:      CORD-19 (171,000 scientific papers)
Queries:     50 topics with detailed information needs
Judgments:   Graded (0-2), pooled from multiple systems
Split:       30 queries for training, 20 for evaluation
Primary metric: NDCG@10, MAP
Notes:       Best benchmark for pandemic/infectious disease retrieval
```

**NFCorpus (NF, Nutritional Facts)**

```
Task:        Medical information retrieval
Corpus:      3,633 medical documents from NutritionFacts.org
Queries:     3,237 queries from user questions
Judgments:   Binary relevance
Notes:       Conversational health queries over curated medical content
             Good for consumer health IR
```

**BioASQ**

```
Task:        Biomedical question answering and document retrieval
Corpus:      PubMed abstracts (14.9M articles)
Queries:     Biomedical questions from biomedical experts
Judgments:   Expert-curated relevant article lists
Notes:       Very large corpus, high-quality expert judgments
             Available in BEIR as BioASQ subset
```

**MedQA**

```
Task:        Medical licensing exam question answering
Source:      USMLE (United States Medical Licensing Exam)
Queries:     Medical multiple-choice questions
Notes:       Tests deep clinical reasoning, not just document retrieval
             Good for medical RAG evaluation
```

### Legal IR

**BEIR Legal Subsets**

```
Datasets: FIQA-Law, SCIDOCS-Law (community-created)
Task:     Legal question answering and document retrieval
Notes:    Small, primarily for zero-shot evaluation
```

**CUAD (Contract Understanding Atticus Dataset)**

```
Task:     Contract clause retrieval and extraction
Corpus:   510 legal contracts (13,000+ clause annotations)
Queries:  41 clause types (e.g., "governing law", "liquidated damages")
Notes:    Best available dataset for contract retrieval
          Available on HuggingFace: cuad
```

**LeCaRD (Legal Case Retrieval Dataset)**

```
Task:     Legal case retrieval (find similar cases)
Corpus:   55,000 Chinese legal cases
Queries:  107 query cases
Judgments: Graded relevance by legal experts
Notes:    Chinese-language legal retrieval, not directly applicable to English
```

**EUR-Lex**

```
Task:     EU legal document classification and retrieval
Corpus:   65K EU legislation documents
Notes:    In BEIR, used for zero-shot legal retrieval evaluation
```

### Scientific / Academic IR

**SCIDOCS**

```
Task:     Scientific paper retrieval and recommendation
Corpus:   25,657 papers from Semantic Scholar
Queries:  Based on paper citation, co-citation, co-view signals
Judgments: Implicit from citation patterns
Notes:    In BEIR, widely used for scientific retrieval evaluation
```

**SciFact**

```
Task:     Scientific claim verification
Corpus:   5,183 scientific abstracts
Queries:  1,409 scientific claims requiring evidence
Judgments: Graded (supports/contradicts/not enough info)
Notes:    Tests fine-grained scientific understanding
```

**QASPER**

```
Task:     QA over scientific papers (full-text)
Corpus:   1,585 NLP papers
Queries:  5,049 questions about paper content
Notes:    Tests passage retrieval within long scientific papers
          Good for scientific RAG evaluation
```

**DORIS-MAE (Domain-specific IR for Multi-Attribute Entity retrieval)**

```
Task:     Complex multi-criteria scientific paper retrieval
Queries:  3,388 natural language queries with multiple constraints
Notes:    Tests whether retrieval handles compound requirements
          Example: "papers on BERT for low-resource settings published after 2020"
```

### Code Search

**CodeSearchNet**

```
Task:     Code search - find code given natural language description
Corpus:   2M code functions in 6 languages (Python, Java, Go, PHP, Ruby, JS)
Queries:  Natural language function descriptions
Judgments: Implicit (docstring → function correspondence)
Notes:    Standard benchmark for code retrieval
          Available on HuggingFace: code_search_net
```

**CoNaLa (Code/Natural Language)**

```
Task:     Code search with short natural language intents
Corpus:   Python code snippets
Queries:  One-line natural language programming intents
Notes:    More realistic than CodeSearchNet - shorter, noisier queries
```

**SWE-Bench**

```
Task:     GitHub issue → relevant code file retrieval
Corpus:   GitHub repositories
Queries:  Real GitHub issues
Notes:    Tests whether a retrieval system can find the right files to fix a bug
          Critical for AI coding assistants
```

### Financial IR

**FIQA (Financial QA)**

```
Task:     Financial question answering
Corpus:   57,638 passages from financial web sources
Queries:  6,648 financial questions
Judgments: Crowd-sourced relevance judgments
Notes:    In BEIR, widely used for financial IR evaluation
          Queries are conversational financial questions
```

**FinanceBench**

```
Task:     Financial document QA (SEC filings, earnings reports)
Corpus:   Financial reports (10-K, 10-Q, earnings call transcripts)
Queries:  Quantitative and qualitative questions about company financials
Notes:    Tests RAG systems over structured financial documents
```

### News / Multimedia

**TREC News (Washington Post)**

```
Task:     Background linking - find articles providing context for a news story
Corpus:   Washington Post articles (600K documents)
Queries:  Current news articles (need background context)
Notes:    In BEIR, tests whether a system understands narrative context
```

**Robust04**

```
Task:     Ad-hoc retrieval on news articles
Corpus:   TREC Disks 4 and 5 (~500K news articles from TREC)
Queries:  250 topics with difficult, ambiguous information needs
Notes:    Classic difficult ad-hoc retrieval benchmark
          In BEIR as "robust04"
```

### Multi-Domain / General

**BEIR (18 datasets)**
The meta-benchmark covered throughout this repo. Includes a curated subset of
domain-specific datasets for zero-shot evaluation:

```
Dataset          Domain            Corpus size    Task type
────────────────────────────────────────────────────────────────────
TREC-COVID       Biomedical        171K           Ad-hoc
NFCorpus         Medical           3.6K           Ad-hoc
NQ               Wikipedia         2.7M           Open-domain QA
HotpotQA         Wikipedia         5.2M           Multi-hop QA
FiQA             Financial         57K            QA
Arguana          Counterargument   8.7K           Argument retrieval
Touché-2020      Controversial     382K           Argument retrieval
DBPedia          Entity            4.6M           Entity retrieval
SCIDOCS          Scientific        25K            Citation retrieval
FEVER            Fact verification 5.4M           Fact verification
Climate-FEVER    Climate           5.4M           Fact verification
SciFact          Scientific claims 5K             Claim verification
Quora            Duplicate QA      523K           Duplicate detection
CQADupStack      Community QA      457K           Duplicate detection
TREC-NEWS        News              595K           Background linking
Robust04         News              528K           Ad-hoc
BioASQ           Biomedical        14.9M          QA
MS MARCO         Web               8.8M           Passage retrieval
```

## Building Your Own Domain Dataset

When no suitable public dataset exists, building a domain-specific dataset
from scratch is necessary. The process has three components:

### Component 1 - Document corpus collection

```
Sources:
  Industry databases:  PubMed, LexisNexis, Bloomberg, GitHub, Arxiv
  Internal documents:  company wikis, support tickets, product documentation
  Web scraping:        domain-specific websites, forums (Stack Overflow subsets)
  Licensed data:       commercial data providers for regulated domains

Processing:
  Deduplication:       remove near-duplicate documents
  Quality filtering:   remove low-quality documents (too short, malformed)
  Chunking:            split long documents into retrievable passages
  Metadata extraction: title, date, author, category for filtering
```

### Component 2 - Query collection

```
Sources:
  User query logs:     if a search system already exists
  Domain expert input: ask experts what they search for
  LLM generation:      generate synthetic queries per document (GPL)
  Existing QA:         convert QA datasets to retrieval queries
  FAQ mining:          extract questions from FAQ pages

Quality criteria:
  Information need clarity: clear what the user is looking for
  Domain specificity:       uses domain-appropriate vocabulary
  Retrievability:           at least one document in corpus answers it
  Diversity:                covers different aspects of the domain
```

### Component 3 - Relevance judgments

```
Approaches:
  Expert annotation:   gold standard, expensive (~$2-10 per judgment)
  Crowdsourcing:       cheaper, requires multiple annotators per item
  Pooling:             run multiple systems, judge pooled top-k results
  LLM-as-judge:        automated relevance scoring with GPT-4/Claude
  Implicit feedback:   clickthrough data from production system

Inter-annotator agreement (IAA):
  Measure Cohen's kappa between annotators
  κ > 0.6: substantial agreement (acceptable)
  κ > 0.8: near-perfect agreement (excellent)
  κ < 0.4: poor agreement (revisit guidelines)
```

## Using Domain Datasets for Training

### Pretraining + fine-tuning hierarchy

The standard training hierarchy for strong domain-adapted retrieval:

```
Stage 1 - General dense retrieval pretraining
  Dataset: MS MARCO (529K training triples)
  Goal:    Learn general retrieval behavior
  Model:   E5-base or GTE-base (already done - use pretrained)

Stage 2 - Domain-adaptive pretraining (optional but helpful)
  Dataset: Unlabeled domain text (PubMed, legal corpus, code repos)
  Goal:    Update token representations for domain vocabulary
  Method:  TSDAE, UDAP, or continued MLM pretraining
  Cost:    ~1-3 days GPU

Stage 3 - Domain-specific fine-tuning
  Dataset: Domain-specific labeled pairs (TREC-COVID, CUAD, CodeSearchNet)
  Goal:    Learn domain-specific relevance judgments
  Method:  MultipleNegativesRankingLoss or MarginMSE with hard negatives
  Cost:    ~2-8 hours GPU with 1K-10K examples

Stage 4 - Task-specific fine-tuning (optional)
  Dataset: Your specific application's labeled data
  Goal:    Final calibration to your exact use case
  Cost:    ~1-4 hours GPU with 100-500 examples
```

### Data mixing ratios

When training with both general and domain-specific data:

```
Conservative (preserve generalization):
  70% MS MARCO + 30% domain-specific data

Balanced:
  50% MS MARCO + 50% domain-specific data

Domain-focused:
  20% MS MARCO + 80% domain-specific data
  (risk: may degrade on general queries)
```

Start with the balanced ratio and evaluate on both general (BEIR subset) and
domain benchmarks. Adjust the ratio based on the performance tradeoff.

## Dataset Selection Guide

```
Your domain                    Best training data            Best evaluation benchmark
───────────────────────────────────────────────────────────────────────────────────────
COVID-19 / Infectious disease  TREC-COVID training splits    TREC-COVID test set
General biomedical             BioASQ training data          BioASQ, NFCorpus
Contract / Legal               CUAD training data            CUAD test set, EUR-Lex
Scientific papers              SCIDOCS, SciFact              BEIR SCIDOCS, SciFact
Python code search             CodeSearchNet Python          CodeSearchNet test
General code search            CodeSearchNet all             SWE-Bench
Financial QA                   FiQA training data            FiQA, FinanceBench
News retrieval                 MS MARCO + TREC-NEWS          Robust04, TREC-NEWS
General multi-domain           MS MARCO (primary)            BEIR 18-dataset suite
No public dataset exists       GPL synthetic + labeled       Your own held-out set
```

## The Training Data Hierarchy in Practice

A general recipe for most domain adaptation scenarios:

```
1. Start with a strong zero-shot base (E5-large, GTE-large)
   → Evaluate on your domain

2. If gap is small (neural > BM25): done, deploy

3. If gap is moderate (BM25 wins by 0-10%):
   → Few-shot fine-tune with 100-300 domain examples
   → Evaluate, likely sufficient

4. If gap is large (BM25 wins by >10%):
   → TSDAE/UDAP unsupervised pretraining on domain text (1-3 days)
   → Fine-tune on domain-specific labeled dataset (TREC-COVID, CUAD, etc.)
   → Optional: mix with MS MARCO (30% MS MARCO, 70% domain)
   → Evaluate on both domain benchmark and BEIR general subset
   → Few-shot fine-tune on your specific application data if still insufficient

5. Ongoing:
   → Monitor production quality (see 11-practical-systems/04-monitoring)
   → Collect implicit feedback (clicks) to augment labeled data
   → Retrain quarterly with accumulated domain signal
```

## My Summary

Domain-specific datasets provide the labeled training and evaluation resources needed
to adapt retrieval models beyond the general web search distribution of MS MARCO. For
biomedical retrieval, TREC-COVID and BioASQ offer expert-curated relevance judgments
over PubMed-scale corpora. For legal retrieval, CUAD provides clause-level annotations
over contracts. For scientific retrieval, SCIDOCS and SciFact offer citation-based and
expert-annotated benchmarks. For code search, CodeSearchNet is the standard. When no
public dataset exists, domain datasets can be constructed from industry databases with
expert annotation, implicit clickthrough signals, or LLM-generated synthetic queries
via GPL. The standard training hierarchy combines general MS MARCO pretraining,
unsupervised domain adaptive pretraining on unlabeled domain text, and fine-tuning on
domain-specific labeled data - with a 70%/30% domain/MS MARCO data mix providing the
best balance between domain performance and generalization. The BEIR benchmark serves
as the meta-evaluation framework, testing whether domain-adapted models generalize
across the full diversity of domain-specific retrieval tasks.
