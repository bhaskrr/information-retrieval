# Query Reformulation

Query reformulation is the process of transforming a conversational query
one that depends on prior context, contains pronouns, or omits information
stated earlier into a standalone query that fully expresses the user's current
information need without requiring any conversation history to understand. It is
the single most impactful technique in conversational IR because it converts the
multi-turn retrieval problem back into a standard single-turn retrieval problem
that any existing retrieval system can handle without modification.

## Intuition

Multi-turn retrieval is hard because retrieval systems are built for standalone
queries. The simplest way to solve this is not to build a context-aware retrieval
system it is to rewrite the context-dependent query into a standalone one and
use existing retrieval infrastructure unchanged.

Consider:

```bash
Turn 1: "What is BERT?"
Turn 2: "How was it trained?"
Turn 3: "Is it better than GPT?"
Turn 4: "What about for text classification?"
```

A human reading Turn 4 in isolation has no idea what "it" refers to or what
"for text classification" is compared to. But the reformulation is obvious
with context:

```bash
Turn 4 reformulated: "Is BERT better than GPT for text classification?"
```

This standalone query can be sent to any retrieval system BM25, dense
retrieval, hybrid with no modification. The entire context-handling burden
is absorbed by the reformulation step, leaving retrieval systems free to do
what they do best.

## Why Reformulation is the Dominant Approach

Query reformulation has become the standard approach in production conversational
IR systems for several reasons:

**Modularity** reformulation cleanly separates context understanding from
retrieval. Each component can be developed, tested, and improved independently.

**Reuse existing infrastructure** no changes to retrieval indexes, ranking
models, or evaluation pipelines. The reformulated query drops in as a standard
query.

**Interpretability** the reformulated query is human-readable. You can inspect
it, debug it, and verify it captures the user's intent before retrieval.

**Flexibility** the same reformulation component works with BM25, dense
retrieval, and hybrid systems. Switching the retrieval backend requires no
changes to reformulation.

**LLM compatibility** modern LLMs are excellent at query reformulation with
minimal prompting. This makes high-quality reformulation accessible without
training a dedicated model.

## Types of Reformulation

### Coreference resolution

Replace pronouns and demonstratives with their referents:

```bash
History:  "What is dense retrieval?"
Current:  "How does it work?"
Rewritten: "How does dense retrieval work?"
```

The simplest and most common form. Works well when the referent is unambiguous
and recent in the conversation.

### Ellipsis completion

Restore omitted information that was stated in prior turns:

```bash
History:  "What are the advantages of BM25?"
Current:  "And the disadvantages?"
Rewritten: "What are the disadvantages of BM25?"
```

Requires understanding that the current query is an incomplete parallel
construction and filling in the missing elements from context.

### Topic carryforward

Explicitly state the topic being discussed when the current query omits it:

```bash
History:  "Tell me about transformer architecture."
Current:  "What are the computational requirements?"
Rewritten: "What are the computational requirements of transformer architecture?"
```

The current query is grammatically complete but semantically incomplete without
knowing the topic.

### Disambiguation

When the current query is ambiguous, select the most plausible interpretation
given the conversation context:

```bash
History:  "We have been discussing Python web frameworks."
Current:  "What about performance?"
Rewritten: "What is the performance of Python web frameworks?"
           (not "What about performance [of snakes]")
```

### Full query reconstruction

For complex dependencies, reconstruct the query entirely:

```bash
History:  "Compare BERT and GPT-3 for sentiment analysis."
         "BERT seems better. Why?"
Current:  "What would make the other one competitive?"
Rewritten: "What improvements would make GPT-3 competitive with BERT
            for sentiment analysis?"
```

## Reformulation Approaches

### Rule-based reformulation

Apply hand-crafted rules for common patterns:

```python
PRONOUN_MAP = {
    "it":   last_mentioned_entity,
    "they": last_mentioned_entities,
    "this": last_mentioned_concept,
    "that": last_mentioned_concept,
}

if current_query.startswith("And"):
    # Ellipsis prepend topic from last turn
    reformulated = topic_from_last_turn + current_query[4:]
```

Fast, predictable, and interpretable. Fails on complex or unexpected patterns.
Useful as a fallback when LLM reformulation is too expensive.

### Seq2seq model reformulation

Fine-tune a sequence-to-sequence model (T5, BART) on (history, current_query)
-> standalone_query pairs:

```bash
Input:  "conversation: What is dense retrieval? [SEP]
         How does it compare to BM25? [SEP]
         current: Which is faster?"

Output: "Which is faster, dense retrieval or BM25?"
```

Training data: TREC CAsT provides manually rewritten queries for each turn.
QReCC provides 81,000 conversational QA instances with reformulations.

T5-based models trained on QReCC produce strong reformulations and are the
standard approach in academic multi-turn IR systems.

### LLM-based reformulation

Use a large language model with a carefully crafted prompt:

```
System: "You are a query reformulation assistant. Given a conversation
         history and the current query, rewrite the current query as a
         complete standalone query that captures the user's information
         need without requiring the conversation history to understand.
         Output only the rewritten query, nothing else."

User:   "History:
         User: What is dense retrieval?
         User: How does it compare to BM25?

         Current query: Which is faster?"

Model:  "Which is faster, dense retrieval or BM25?"
```

**Advantages**: no training data needed, handles complex cases, can be improved
by prompt engineering alone.

**Disadvantages**: adds LLM latency to every query, more expensive than a
dedicated seq2seq model.

In production, LLM reformulation is the default choice when a capable LLM
is already in the stack (e.g. in a RAG pipeline). A dedicated T5 model is
preferred when latency or cost are the primary constraints.

### Hybrid reformulation

Use a lightweight seq2seq model for simple cases and fall back to an LLM
for complex ones:

```bash
Classify reformulation difficulty:
  Simple (pronoun replacement only) -> rule-based or T5
  Medium (ellipsis, topic carry)    -> T5
  Complex (full reconstruction)     -> LLM
```

Balances quality and cost by routing to the appropriate reformulation approach.

## Training Data for Reformulation Models

### TREC CAsT

30-50 conversation sessions per year with manually written rewritten queries
for each turn. Small but high quality. Standard for academic evaluation.

### QReCC (Open-Domain Question Rewriting for Conversational QA)

81,000 question-answer pairs across 14,000 conversations. Includes both
in-context questions and standalone reformulations. Larger and more diverse
than CAsT the standard training set for seq2seq reformulation models.

### CANARD (Context-Dependent Question Disambiguation)

40,527 questions paired with their conversational context from QuAC. Each
question has a manually reformulated standalone version. Good for training
coreference resolution in questions.

### OR-QuAC (Open-Domain Retrieval Question Answering under Conversational setting)

Extends QuAC with retrieved passages and reformulations. Useful for training
end-to-end conversational retrieval systems.

## Evaluating Reformulation Quality

### Intrinsic metrics reformulation quality

Measure how well the reformulated query matches a human reference:

**BLEU** n-gram overlap between reformulated and reference query.
Commonly used but poorly correlated with actual retrieval improvement.

**ROUGE-L** longest common subsequence between reformulated and reference.
Similar limitations to BLEU.

**BERTScore** contextual embedding similarity between reformulated and
reference. More semantic than BLEU/ROUGE.

**Limitation**: these metrics measure similarity to a single reference reformulation.

Multiple valid reformulations may exist, "Is BERT faster than GPT for text
classification?" is as valid as "Between BERT and GPT, which is faster for text
classification?" BLEU penalizes both equally.

### Extrinsic metrics downstream retrieval quality

The most meaningful evaluation: does the reformulated query improve retrieval?

```bash
NDCG@k on raw utterances -> baseline retrieval quality
NDCG@k on reformulations -> quality with reformulation
Gap = improvement from reformulation
```

A reformulation model that produces fluent standalone queries but does not
improve retrieval NDCG is not useful. Extrinsic evaluation on CAsT or QReCC
retrieval tasks is the gold standard.

### Human evaluation

For production systems, human evaluation of reformulation quality on sampled
conversations. Judges assess whether the reformulation correctly captures
intent and resolves all context dependencies.

## Reformulation Failure Modes

Understanding where reformulation fails helps build better systems:

### Topic drift

The reformulation incorrectly carries forward irrelevant context:

```bash
Turn 1: "Compare BERT and GPT."
Turn 2: "What about latency?"
Bad reformulation: "What about the latency of BERT and GPT?"
                   (correct)
Turn 3: "And memory?"
Bad reformulation: "What about the memory of BERT and GPT?"
                   (still correct)
Turn 4: [user actually wants to ask about something unrelated]
         "What are the training costs?"
Bad reformulation: "What are the training costs of BERT and GPT?"
                   (incorrect if user moved on)
```

### Overly aggressive expansion

The reformulation adds too much context, creating an unnatural query:

```bash
Current: "What about speed?"
Bad:     "What about the inference speed of BERT-base compared to GPT-3
          for sentiment classification tasks on short text?"
Good:    "What about the speed of BERT?"
```

### Missing context

The reformulation fails to include necessary context:

```bash
History: "We discussed BERT for text classification."
Current: "What is the best learning rate?"
Bad:     "What is the best learning rate?"   (unchanged missing BERT context)
Good:    "What is the best learning rate for fine-tuning BERT for text classification?"
```

### Hallucinated context

The reformulation introduces information not present in the conversation:

```bash
History: "Compare BERT and RoBERTa."
Current: "Which handles longer sequences better?"
Bad:     "Which model, BERT or RoBERTa, handles longer sequences better for
          document classification?"  ← "document classification" hallucinated
Good:    "Which handles longer sequences better, BERT or RoBERTa?"
```

## Reformulation Quality vs Retrieval Improvement

A common finding in conversational IR research is that reformulation quality
metrics (BLEU, ROUGE) are weakly correlated with retrieval improvement:

| Reformulation type        | BLEU-1 | NDCG@3 improvement |
| ------------------------- | ------ | ------------------ |
| Raw utterance (no reform) | -      | baseline           |
| Rule-based                | 0.61   | +2.1 points        |
| T5 (CANARD)               | 0.72   | +4.3 points        |
| T5 (QReCC)                | 0.74   | +5.1 points        |
| LLM (GPT-4)               | 0.69   | +6.2 points        |
| Manual rewrite            | 1.00   | +8.7 points        |

Note: LLM reformulation achieves lower BLEU than T5 but higher NDCG improvement.
The LLM produces more creative reformulations that diverge from the reference but
better capture user intent. This highlights why extrinsic evaluation on retrieval
quality is the true measure of reformulation usefulness.

## My Summary

Query reformulation converts context-dependent conversational queries into
standalone queries by resolving pronouns, completing ellipsis, and making
implicit topics explicit. It is the most practical approach to multi-turn
retrieval because it decouples context resolution from retrieval existing
retrieval systems work unchanged on reformulated queries. Three approaches
exist in order of increasing quality and cost: rule-based pattern matching
(fast, fragile), T5 seq2seq models fine-tuned on CANARD or QReCC (balanced),
and LLM reformulation via prompting (highest quality, most expensive). The
key insight from evaluation is that BLEU and ROUGE scores correlate poorly
with retrieval improvement extrinsic evaluation using NDCG on CAsT or
QReCC retrieval tasks is the only reliable measure of reformulation usefulness.
Reformulation quality directly caps the upper bound of conversational retrieval
quality: if the reformulation is wrong, even a perfect retrieval system fails.
