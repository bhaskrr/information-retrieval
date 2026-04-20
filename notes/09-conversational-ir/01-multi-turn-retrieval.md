# Multi-Turn Retrieval

Multi-turn retrieval is the task of finding relevant documents given a conversation history rather than a single standalone query. In a multi-turn interaction, the current query is often incomplete, ambiguous, or dependent on what was said in previous turns it cannot be understood or retrieved against in isolation. A multi-turn retrieval system must understand the full conversational context and use it to identify what the user is actually looking for at this point in the dialogue.

## Intuition

Single-turn retrieval is well-defined: one query, one retrieval. Multi-turn
retrieval is messier because human conversation is inherently context-dependent.

Consider this exchange:

```bash
Turn 1: "What is dense retrieval?"
Turn 2: "How does it compare to BM25?"
Turn 3: "Which one should I use for my project?"
Turn 4: "What about combining them?"
```

Turn 2 contains "it" which refers to dense retrieval from Turn 1. Turn 3 has
no explicit topic "which one" refers to the comparison from Turn 2. Turn 4
is completely opaque without the full history "combining them" means
combining dense retrieval and BM25 from context.

None of Turns 2, 3, or 4 are retrievable as standalone queries. A retrieval
system that treats each turn independently will fail on most of them. Multi-turn
retrieval systems address this by maintaining and using the conversation context
when formulating retrieval queries.

## The Core Challenge Context Dependency

Context dependency in conversations takes several forms:

### Coreference

Pronouns and demonstratives refer to entities mentioned earlier:

```bash
Turn 1: "Tell me about BERT."
Turn 2: "When was it released?"    ← "it" = BERT
Turn 3: "Who created it?"          ← "it" = BERT
```

### Ellipsis

Parts of the query are omitted because they were stated earlier:

```bash
Turn 1: "What are the best practices for dense retrieval?"
Turn 2: "And for sparse?"          ← full query: "best practices for sparse retrieval"
Turn 3: "What about hybrid?"       ← full query: "best practices for hybrid retrieval"
```

### Topic continuation

The query implicitly continues a topic without stating it:

```bash
Turn 1: "Explain transformer architecture."
Turn 2: "What are the computational requirements?"   ← of transformers, not stated
Turn 3: "Are there more efficient alternatives?"     ← to transformers, not stated
```

### Topic shift

The user changes topics mid-conversation a different kind of context dependency:

```bash
Turn 1: "What is BM25?"
Turn 2: "How is it different from TF-IDF?"
Turn 3: "Actually, tell me about ColBERT."   ← topic shift, new context
Turn 4: "What hardware does it need?"        ← "it" = ColBERT, not BM25
```

The system must track when topics shift and update the context accordingly
not naively appending all history but understanding what is currently relevant.

## Approaches to Multi-Turn Retrieval

### Approach 1 Concatenate history with current query

The simplest approach. Prepend the conversation history to the current query
and treat the concatenation as a single long query:

```bash
History:
  U: "What is dense retrieval?"
  A: "Dense retrieval uses neural encoders..."
  U: "How does it compare to BM25?"

Current query: "Which one is faster?"

Concatenated input:
  "What is dense retrieval? Dense retrieval uses neural encoders...
   How does it compare to BM25? Which one is faster?"
```

Problems:

- Input becomes very long may exceed model token limits
- Early turns dilute the current query signal
- Equal weight given to all history regardless of relevance
- Does not explicitly resolve pronouns or ellipsis

Works reasonably well for short conversations but degrades as history grows.

### Approach 2 Query rewriting

Rewrite the current query into a standalone, context-resolved query:

```bash
Turn history:
  "What is dense retrieval?"
  "How does it compare to BM25?"

Current: "Which one is faster?"
Rewritten: "Which is faster, dense retrieval or BM25?"
```

The rewritten query can be sent to any standard single-turn retrieval system
no changes to the retrieval pipeline are needed.

Two rewriting strategies:

**Rule-based rewriting** replace pronouns and ellipsis using pattern matching.
Fast but fragile. Fails on complex coreference.

**Neural rewriting** fine-tune a seq2seq model (T5, GPT) to produce standalone
queries from (history, current_query) inputs. More robust but requires labeled
training data.

This is the most practical approach for production systems it decouples context
resolution from retrieval, letting each component focus on what it does best.
Covered in depth in 02-query-reformulation.md.

### Approach 3 Context-aware dense retrieval

Encode the conversation history alongside the current query using a context-aware
encoder. The query embedding reflects not just the current turn but the full
conversational context:

```bash
Standard bi-encoder:
  q_vec = encoder(current_query)

Conversational bi-encoder:
  q_vec = encoder([CLS] history_summary [SEP] current_query [SEP])
```

Fine-tune the bi-encoder on conversational retrieval datasets (TREC CAsT, QReCC)
where positive/negative passage pairs are labeled with full conversation context.

The document encoder remains unchanged only the query encoder changes. Document
embeddings are still precomputed without conversational context.

### Approach 4 Fusion-in-Decoder with conversation

Extends Fusion-in-Decoder (FiD) for conversational settings. Retrieve passages
independently for each conversation turn, then fuse all retrieved passages along
with the conversation history when generating the answer:

```bash
Turn 1 query → retrieve passages_1
Turn 2 query → retrieve passages_2
Turn 3 query → retrieve passages_3 (current)

Generate answer: FiD encoder encodes (history + [passages_1, passages_2, passages_3])
```

Expensive but produces the most context-aware answers. Used in open-domain
conversational QA research.

## TREC CAsT The Multi-Turn Retrieval Benchmark

TREC Conversational Assistance Track (CAsT) is the standard benchmark for
evaluating multi-turn retrieval:

```bash
Format:
  Multi-turn conversations (5-13 turns each)
  Each turn: raw utterance + manually rewritten standalone query
  Relevant passages labeled for each turn

Scale:
  CAsT 2019: 30 topics, 300 turns
  CAsT 2020: 25 topics, 216 turns (with passage-level answer labels)
  CAsT 2021: 25 topics, harder context dependencies
  CAsT 2022: 23 topics, mixed initiative (system can ask clarifying questions)

Primary metric: NDCG@3 (users want the answer quickly in conversational IR)
```

CAsT provides both the raw conversational utterances and manually rewritten
standalone queries enabling evaluation of automatic query rewriting quality
separately from retrieval quality.

## Session Context Management

As conversations grow longer, naive concatenation of all history degrades
performance. Effective multi-turn systems manage context selectively:

### Fixed window

Keep only the last K turns in context:

```bash
window_size = 3
context = history[-3:]   # only use last 3 turns
```

Simple, prevents context explosion. Loses early relevant context.

### Relevance-weighted history

Weight earlier turns by their semantic similarity to the current query:

```bash
for each prior turn t:
    weight(t) = cosine_similarity(encode(t), encode(current_query))
context = weighted_combination(history, weights)
```

Preserves relevant earlier context while downweighting irrelevant turns.

### Topic segmentation

Detect when topics shift and reset context at boundaries:

```bash
Turn 1-3: topic "dense retrieval"
Turn 4:   "Actually, let me ask about ColBERT."  ← topic shift detected
Turn 5-7: topic "ColBERT"                         ← fresh context from Turn 4
```

Prevents prior topic context from contaminating current topic retrieval.

### Summarization

Periodically summarize older history into a compact context representation:

```bash
Turns 1-5: detailed history
           ↓ (LLM summarization)
           "User asked about dense retrieval, BM25 comparison, and hybrid search.
            Currently discussing efficiency tradeoffs."
Turns 6-10: detailed history
Current turn + summary → retrieval
```

Compresses long histories without losing key information. Adds LLM inference cost.

## Evaluation Challenges in Multi-Turn Retrieval

### Turn-level vs session-level evaluation

Single-turn metrics (NDCG@K, MAP) applied per turn miss session-level quality:

- A system that retrieves correctly for early turns but fails on dependent later
  turns may score well per-turn but produce a poor user experience
- Session-level metrics evaluate the full conversation trajectory

### Raw vs rewritten query evaluation

Two evaluation modes:

**Raw utterance evaluation** use the original unmodified user query.
Tests the system's ability to handle natural conversational language.

**Rewritten query evaluation** use manually rewritten standalone queries.
Tests retrieval quality independent of context resolution.

Good systems should perform well on both. The gap between raw and rewritten
performance quantifies how well the system handles context dependency.

### Passage-level vs turn-level relevance

Some passages are relevant across multiple turns. Relevance judgments in CAsT
are turn-specific a passage relevant for Turn 2 may not be relevant for Turn 4
even in the same session.

## Multi-Turn Retrieval in Production

Modern conversational AI systems handle multi-turn retrieval at different
levels of sophistication:

| System type           | Approach                   | Notes          |
| --------------------- | -------------------------- | -------------- |
| Simple chatbots       | Fixed-window concatenation | Fast, baseline |
| RAG assistants        | Query rewriting (LLM)      | Most common    |
| Conversational search | Context-aware bi-encoder   | More accurate  |
| Research systems      | FiD + full history fusion  | Most powerful  |

Most production RAG systems today use LLM-based query rewriting feeding the
conversation history to the LLM and asking it to produce a standalone retrieval
query. This is covered in depth in 02-query-reformulation.md.

## The Degradation Problem

A critical observation in multi-turn retrieval: performance tends to degrade
as conversations get longer. This happens for several reasons:

```bash
Turn 1:  Standalone query → standard retrieval, no context needed
Turn 2:  Single reference → easy to resolve
Turn 3:  Multiple references → harder to track
Turn 4+: Accumulated context → noise from irrelevant history
Turn 8+: Long history → context window limits, topic drift
```

Measuring the gap between early-turn and late-turn NDCG scores
(context_degradation in the evaluation code above) is a diagnostic metric
for multi-turn system quality. A well-designed system should show minimal
degradation across conversation length.

## My Summary

Multi-turn retrieval finds relevant documents given a conversation history where
current queries are typically context-dependent containing pronouns, ellipsis,
or topic references that are only meaningful in light of prior turns. Three main
approaches exist: concatenating recent history with the current query (simple
but noisy), rewriting queries into standalone form (most practical for production),
and context-aware bi-encoders that incorporate history into query encoding. The
TREC CAsT benchmark provides the standard evaluation framework with NDCG@3 as
the primary metric. A consistent challenge is context degradation retrieval
quality tends to decline in longer conversations as accumulated history introduces
noise. Managing context selectively through fixed windows, relevance weighting,
or topic shift detection mitigates this degradation. Query reformulation covered
next is the most impactful single technique for improving multi-turn retrieval
quality in practice.
