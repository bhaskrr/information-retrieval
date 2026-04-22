# Conversational RAG

Conversational RAG (Retrieval-Augmented Generation) extends the standard single-turn
RAG pipeline to handle multi-turn dialogues. In a conversational setting, each user
turn may depend on prior context, the system must decide what to retrieve given both
the current query and the conversation history, and the generated response must be
coherent with what was said before. Conversational RAG combines query reformulation,
context-aware retrieval, memory management, and grounded generation into a unified
dialogue system that can answer follow-up questions, handle topic shifts, and
maintain coherent multi-turn interactions over long conversations.

## Intuition

Standard RAG is stateless — each query is independent, each retrieval is fresh,
each response is generated without memory of prior exchanges. This works perfectly
for single questions but fails for dialogue.

Consider a user exploring a topic across multiple turns:

```
Turn 1: "What is ColBERT?"
        → retrieve docs about ColBERT → generate explanation
Turn 2: "How does it differ from a standard bi-encoder?"
        → "it" is ambiguous without context
        → must retrieve docs about bi-encoder comparison with ColBERT
        → must generate response that connects to the Turn 1 explanation
Turn 3: "Which would you recommend for a corpus of 10 million documents?"
        → recommendation depends on knowing both ColBERT and bi-encoders
        → response should reference what was established in Turns 1-2
Turn 4: "What about memory requirements?"
        → "what about" is elliptical — memory of which system?
        → response should clarify which system is being asked about
```

Each turn has retrieval dependencies (what to search for), generation dependencies
(what prior context to include), and coherence requirements (the response must fit
naturally in the ongoing dialogue). Conversational RAG manages all three.

## How Conversational RAG Extends Standard RAG

Standard RAG pipeline (stateless):

```
Query → Retrieve → Assemble prompt → Generate → Response
```

Conversational RAG pipeline (stateful):

```
[History, Current query]
    ↓
Query reformulation      → standalone query from context
    ↓
Context-aware retrieval  → retrieve relevant passages
    ↓
Memory management        → select which prior context to include
    ↓
Prompt assembly          → combine history + retrieved + current
    ↓
Grounded generation      → LLM generates coherent response
    ↓
Response + memory update → store response for next turn
```

## Core Components

### Component 1 — Query Reformulation

Covered in depth in 02-query-reformulation.md. In the conversational RAG context,
reformulation is the first processing step — before any retrieval occurs:

```
Turn 3 raw: "Which would you recommend for a corpus of 10 million documents?"

Reformulated: "Which is better for large-scale retrieval — ColBERT or
               bi-encoder — for a corpus of 10 million documents?"
```

The reformulated query is sent to the retrieval system. The raw query is preserved
for the generation prompt so the LLM sees what the user actually said.

### Component 2 — Context-Aware Retrieval

Retrieve passages relevant to the current turn's information need. Three strategies:

**Reformulation-based** — use the reformulated standalone query for retrieval.
Simple and effective. Standard for most systems.

**History-augmented** — concatenate recent history with current query for retrieval.
Captures multi-aspect information needs that span turns.

**Multi-query retrieval** — generate multiple retrieval queries from the current
context and retrieve for each:

```
Turn: "Which would you recommend for a large corpus?"
Queries: ["ColBERT large-scale retrieval", "bi-encoder efficiency large corpus",
          "comparison ColBERT bi-encoder memory requirements"]
→ retrieve for each, merge results
```

More expensive but captures different aspects of the information need.

### Component 3 — Conversation Memory Management

The conversation history grows with each turn. Including everything in every prompt
quickly exceeds context window limits and degrades generation quality. Memory
management selects what to include:

**Full history** — include all prior turns. Simple but context-window-limited.
Suitable for short conversations (< 5 turns).

**Windowed history** — include last K turns. Fast and predictable. Loses early
relevant context.

**Summarized history** — periodically summarize older turns into a compact
representation. Preserves key information from long conversations.

**Entity memory** — maintain a structured record of entities mentioned and
facts established:

```
Entities discussed:
  ColBERT: late interaction model, good for accuracy
  Bi-encoder: fast retrieval, precomputable, less accurate
  User's corpus: 10 million documents
  User constraint: memory is a concern
```

This structured memory is more compact and targeted than raw conversation history.

**Retrieved passage memory** — track which passages were retrieved in prior turns
to avoid re-retrieving identical content:

```
Already retrieved: ["ColBERT paper abstract", "HNSW index overview"]
Current retrieval: exclude already-shown passages
```

### Component 4 — Prompt Assembly

Combine all relevant context into the generation prompt:

```
System: "You are a helpful IR assistant. Answer based on the provided
         context and maintain consistency with the conversation history.
         If context is insufficient, say so."

Conversation history (last 2 turns):
  User: What is ColBERT?
  Assistant: ColBERT is a neural retrieval model that...
  User: How does it differ from a standard bi-encoder?
  Assistant: Unlike bi-encoders which encode query and document into single vectors...

Retrieved passages (for current turn):
  [1] "ColBERT requires storing n×128 vectors per document vs one 768-dim
      vector for bi-encoders. Storage is 6-10x higher..."
  [2] "For 10M documents, ColBERT requires ~90GB storage vs ~15GB for
      bi-encoders..."

Current query: "Which would you recommend for a corpus of 10 million documents?"

Answer:
```

### Component 5 — Grounded Generation

The LLM generates a response grounded in retrieved passages while maintaining
conversational coherence. Two challenges specific to conversational generation:

**Consistency** — the response must not contradict earlier turns. If Turn 1
established that ColBERT is more accurate, Turn 3 cannot suddenly claim it is
less accurate without acknowledging the inconsistency.

**Coherence** — the response should feel like part of a natural dialogue, not
a disconnected answer to an isolated question. References to prior context ("As
I mentioned earlier...") and acknowledgment of the conversation flow matter.

**Grounding** — the response should cite the retrieved passages rather than
hallucinating. In multi-turn settings, there is additional risk of the LLM
confusing what was established from retrieved text versus what was said in
prior generated turns.

## Advanced Conversational RAG Patterns

### Clarification seeking

When the current query is ambiguous and retrieval cannot resolve the ambiguity,
the system asks a clarifying question instead of retrieving:

```
User: "What about memory?"
System detects ambiguity: ColBERT memory? Bi-encoder memory? Training memory?
System: "Are you asking about memory requirements for ColBERT, bi-encoders,
         or both during training or inference?"
```

TREC CAsT 2022 introduced mixed-initiative conversations where the system can
ask clarifying questions — an active research area.

### Iterative retrieval

After generating an initial response, detect information gaps and retrieve
additional passages to fill them:

```
Generate initial response
↓
Detect: "I don't have specific numbers for memory at 10M documents"
↓
Re-retrieve: "ColBERT memory requirements large scale deployment"
↓
Augment response with newly retrieved information
```

### Conversational reranking

Rerank retrieved passages based not just on the current query but on what the
conversation has established:

```
Retrieved passages ranked by query relevance:
  1. ColBERT general overview (score: 0.92)
  2. ColBERT memory requirements (score: 0.87)
  3. Bi-encoder comparison (score: 0.84)

After conversational reranking (user already saw overview in Turn 1):
  1. ColBERT memory requirements (score: 0.91)  ← boosted (new information)
  2. Bi-encoder comparison (score: 0.87)
  3. ColBERT general overview (score: 0.41)     ← penalized (already covered)
```

### Session summarization

For long conversations, periodically compress history into a structured summary:

```
After every 5 turns:
  LLM generates: "The user is exploring retrieval architectures for a 10M
                  document corpus. They understand ColBERT (late interaction,
                  high accuracy, high storage) and bi-encoders (fast, lower
                  storage, lower accuracy). Current concern: memory requirements."

Summary replaces raw turns in subsequent prompts.
```

## Code

```python
# pip install sentence-transformers faiss-cpu rank-bm25 transformers torch

import numpy as np
import faiss
from dataclasses import dataclass, field
from collections import defaultdict, deque
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ── Part 1: Conversation memory ────────────────────────────────────────────

@dataclass
class Message:
    role:    str   # "user" or "assistant"
    content: str
    turn_id: int


@dataclass
class RetrievedPassage:
    doc_id:  str
    text:    str
    score:   float
    turn_id: int   # which turn retrieved this passage


class ConversationMemory:
    """
    Manages conversation history and retrieved passage tracking
    for a conversational RAG session.
    """
    def __init__(self,
                 max_turns: int = 10,
                 max_passages_per_turn: int = 3,
                 summarize_after: int = 8):
        self.messages:          list[Message]          = []
        self.retrieved_passages: list[RetrievedPassage] = []
        self.entity_memory:     dict[str, str]          = {}
        self.max_turns          = max_turns
        self.max_passages       = max_passages_per_turn
        self.summarize_after    = summarize_after
        self.summary:           str                     = ""
        self.turn_id:           int                     = 0

    def add_user_turn(self, content: str) -> int:
        turn_id = self.turn_id
        self.messages.append(Message(
            role="user",
            content=content,
            turn_id=turn_id
        ))
        self.turn_id += 1
        return turn_id

    def add_assistant_turn(self, content: str):
        self.messages.append(Message(
            role="assistant",
            content=content,
            turn_id=self.turn_id - 1
        ))

    def add_retrieved_passages(self,
                                passages: list[dict],
                                turn_id: int):
        """Track which passages were retrieved for each turn."""
        for p in passages[:self.max_passages]:
            self.retrieved_passages.append(RetrievedPassage(
                doc_id=p["doc_id"],
                text=p["text"],
                score=p.get("score", 0.0),
                turn_id=turn_id
            ))

    def get_already_retrieved(self) -> set[str]:
        """Get set of doc_ids already retrieved in this session."""
        return {p.doc_id for p in self.retrieved_passages}

    def get_history_text(self,
                          window_size: int = None,
                          include_summary: bool = True) -> str:
        """
        Get formatted conversation history for prompt assembly.

        Args:
            window_size:      only include last N messages
            include_summary:  prepend session summary if available

        Returns:
            formatted history string
        """
        messages = self.messages
        if window_size:
            messages = messages[-(window_size * 2):]   # *2 for user+assistant

        formatted = []
        if include_summary and self.summary:
            formatted.append(f"[Session summary: {self.summary}]")

        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{prefix}: {msg.content}")

        return "\n".join(formatted)

    def get_user_history(self) -> list[str]:
        """Get list of user utterances only."""
        return [m.content for m in self.messages if m.role == "user"]

    def update_entity_memory(self, entities: dict[str, str]):
        """Update structured entity memory with new facts."""
        self.entity_memory.update(entities)

    def get_entity_context(self) -> str:
        """Format entity memory for prompt inclusion."""
        if not self.entity_memory:
            return ""
        facts = [f"{k}: {v}" for k, v in self.entity_memory.items()]
        return "Known context:\n" + "\n".join(facts)

    def should_summarize(self) -> bool:
        """Check if conversation is long enough to warrant summarization."""
        user_turns = sum(1 for m in self.messages if m.role == "user")
        return user_turns >= self.summarize_after

    def set_summary(self, summary: str):
        """Store session summary and trim old messages."""
        self.summary  = summary
        # Keep only recent messages after summarization
        recent        = self.messages[-(self.max_turns * 2):]
        self.messages = recent


# ── Part 2: Context-aware retrieval ───────────────────────────────────────

class ConversationalRetriever:
    """
    Retrieval system for conversational RAG.
    Supports reformulation-based and history-augmented retrieval.
    Tracks already-retrieved passages to avoid repetition.
    """
    def __init__(self,
                 model_name: str = "msmarco-distilbert-base-v4"):
        self.encoder     = SentenceTransformer(model_name)
        self.doc_ids     = []
        self.doc_texts   = []
        self.bm25        = None
        self.faiss_index = None

    def build_index(self, documents: dict[str, str]):
        self.doc_ids   = list(documents.keys())
        self.doc_texts = list(documents.values())

        tokenized = [t.lower().split() for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)

        embeddings = self.encoder.encode(
            self.doc_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        d = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(embeddings)

    def retrieve(self,
                  query: str,
                  top_k: int = 5,
                  exclude_doc_ids: set[str] = None,
                  boost_factor: float = 0.2) -> list[dict]:
        """
        Hybrid retrieval with optional passage exclusion.

        Args:
            query:           retrieval query (typically reformulated)
            top_k:           results to return
            exclude_doc_ids: passages already shown, penalize in ranking
            boost_factor:    penalty multiplier for already-retrieved docs

        Returns:
            list of {doc_id, text, score} dicts
        """
        if exclude_doc_ids is None:
            exclude_doc_ids = set()

        # BM25
        bm25_scores   = self.bm25.get_scores(query.lower().split())
        bm25_ranking  = np.argsort(bm25_scores)[::-1][:top_k * 3]

        # Dense
        q_vec = self.encoder.encode(
            [query],
            normalize_embeddings=True
        )
        q_vec = np.array(q_vec, dtype=np.float32)
        _, dense_indices = self.faiss_index.search(q_vec, top_k * 3)

        # RRF fusion with penalization for already-retrieved
        rrf_scores = defaultdict(float)
        for rank, idx in enumerate(bm25_ranking, 1):
            doc_id = self.doc_ids[idx]
            score  = 1.0 / (60 + rank)
            if doc_id in exclude_doc_ids:
                score *= boost_factor   # penalize already-retrieved
            rrf_scores[idx] += score

        for rank, idx in enumerate(dense_indices[0], 1):
            doc_id = self.doc_ids[idx]
            score  = 1.0 / (60 + rank)
            if doc_id in exclude_doc_ids:
                score *= boost_factor
            rrf_scores[idx] += score

        top_indices = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )[:top_k]

        return [
            {
                "doc_id": self.doc_ids[i],
                "text":   self.doc_texts[i],
                "score":  float(rrf_scores[i])
            }
            for i in top_indices
        ]


# ── Part 3: Prompt assembly ────────────────────────────────────────────────

class PromptAssembler:
    """
    Assembles generation prompts for conversational RAG.
    Manages the tradeoff between context richness and prompt length.
    """

    SYSTEM_PROMPT = """You are a helpful information retrieval assistant.
Answer the user's question based on the provided context and conversation history.
Follow these guidelines:
- Ground your answer in the retrieved passages when possible
- Maintain consistency with what was established in prior turns
- If the context does not contain the answer, say so clearly
- Keep responses focused and conversational in tone
- Reference earlier discussion when relevant"""

    def __init__(self,
                 max_history_turns: int = 4,
                 max_passage_chars: int = 300):
        self.max_history_turns = max_history_turns
        self.max_passage_chars = max_passage_chars

    def assemble(self,
                  memory: ConversationMemory,
                  retrieved_passages: list[dict],
                  current_query: str,
                  reformulated_query: str = None) -> str:
        """
        Build the full generation prompt.

        Args:
            memory:             conversation memory object
            retrieved_passages: passages retrieved for current turn
            current_query:      original user query
            reformulated_query: standalone reformulated query (if used)

        Returns:
            complete prompt string
        """
        parts = [self.SYSTEM_PROMPT, ""]

        # Entity memory (compact structured context)
        entity_ctx = memory.get_entity_context()
        if entity_ctx:
            parts.append(entity_ctx)
            parts.append("")

        # Conversation history
        history = memory.get_history_text(
            window_size=self.max_history_turns,
            include_summary=True
        )
        if history:
            parts.append("Conversation so far:")
            parts.append(history)
            parts.append("")

        # Retrieved passages
        if retrieved_passages:
            parts.append("Relevant information:")
            for i, passage in enumerate(retrieved_passages, 1):
                truncated = passage["text"][:self.max_passage_chars]
                if len(passage["text"]) > self.max_passage_chars:
                    truncated += "..."
                parts.append(f"[{i}] {truncated}")
            parts.append("")

        # Current query
        if reformulated_query and reformulated_query != current_query:
            parts.append(f"User's question (interpreted as): {reformulated_query}")
        else:
            parts.append(f"User: {current_query}")

        parts.append("Assistant:")

        return "\n".join(parts)


# ── Part 4: Mock LLM for demonstration ────────────────────────────────────

class MockLLM:
    """
    Mock LLM for demonstration purposes.
    Generates simple template-based responses using retrieved context.
    Replace with real LLM API (Anthropic, OpenAI) in production.
    """
    def generate(self,
                  prompt: str,
                  max_tokens: int = 200) -> str:
        """Generate a mock response from the prompt."""
        # Extract retrieved passages from prompt
        lines       = prompt.split("\n")
        passages    = []
        in_passages = False

        for line in lines:
            if line.startswith("Relevant information:"):
                in_passages = True
            elif in_passages and line.startswith("["):
                content = line.split("]", 1)[-1].strip()
                passages.append(content)
            elif in_passages and line == "":
                in_passages = False

        if passages:
            first_passage = passages[0][:100]
            return (
                f"Based on the available information: {first_passage}... "
                f"This relates to your question. "
                f"[Note: Replace MockLLM with a real LLM API for actual responses]"
            )
        else:
            return (
                "I don't have specific information about that in my context. "
                "[Note: Replace MockLLM with a real LLM API for actual responses]"
            )


# ── Part 5: Full Conversational RAG pipeline ───────────────────────────────

class ConversationalRAG:
    """
    Complete conversational RAG pipeline.
    Manages multi-turn retrieval, memory, prompt assembly, and generation.
    """
    def __init__(self,
                 documents: dict[str, str],
                 retrieval_model: str = "all-MiniLM-L6-v2",
                 top_k: int = 3,
                 window_size: int = 4,
                 use_reformulation: bool = True):
        self.retriever    = ConversationalRetriever(retrieval_model)
        self.retriever.build_index(documents)

        self.assembler    = PromptAssembler(max_history_turns=window_size)
        self.llm          = MockLLM()   # replace with real LLM
        self.top_k        = top_k
        self.use_reform   = use_reformulation

        # Simple reformulator (replace with T5 or LLM reformulator)
        from collections import deque
        self.recent_queries: deque = deque(maxlen=5)

    def _simple_reformulate(self,
                              history: list[str],
                              current: str) -> str:
        """
        Simple reformulation: prepend recent context to resolve pronouns.
        Replace with T5Reformulator or LLMReformulator from 02-query-reformulation.md
        """
        pronouns = {"it", "its", "they", "them", "this", "that", "these",
                    "those", "which", "what about"}

        tokens   = current.lower().split()
        has_ref  = any(t in pronouns for t in tokens)

        if not has_ref or not history:
            return current

        last_query = history[-1] if history else ""
        words      = last_query.split()
        topic      = " ".join(words[:4]) if words else ""

        return f"{topic} — {current}" if topic else current

    def chat(self,
              memory: ConversationMemory,
              user_query: str,
              verbose: bool = True) -> dict:
        """
        Process one conversational turn.

        Args:
            memory:     session memory object
            user_query: raw user input
            verbose:    print intermediate steps

        Returns:
            dict with response and retrieval details
        """
        # Step 1 — add user turn to memory
        turn_id     = memory.add_user_turn(user_query)
        user_history = memory.get_user_history()[:-1]   # exclude current

        # Step 2 — reformulate query
        if self.use_reform and user_history:
            reformulated = self._simple_reformulate(user_history, user_query)
        else:
            reformulated = user_query

        if verbose and reformulated != user_query:
            print(f"  Reformulated: '{reformulated}'")

        # Step 3 — retrieve with novelty penalty
        already_retrieved = memory.get_already_retrieved()
        passages = self.retriever.retrieve(
            query=reformulated,
            top_k=self.top_k,
            exclude_doc_ids=already_retrieved
        )

        memory.add_retrieved_passages(passages, turn_id)

        if verbose:
            print(f"  Retrieved: {[p['doc_id'] for p in passages]}")

        # Step 4 — check if summarization needed
        if memory.should_summarize():
            history_text = memory.get_history_text(window_size=None)
            summary      = (
                f"Multi-turn conversation about IR topics. "
                f"Key points: {history_text[:150]}..."
            )
            memory.set_summary(summary)

        # Step 5 — assemble prompt
        prompt = self.assembler.assemble(
            memory=memory,
            retrieved_passages=passages,
            current_query=user_query,
            reformulated_query=reformulated
        )

        # Step 6 — generate response
        response = self.llm.generate(prompt)

        # Step 7 — store response in memory
        memory.add_assistant_turn(response)

        return {
            "user_query":    user_query,
            "reformulated":  reformulated,
            "passages":      passages,
            "response":      response,
            "turn_id":       turn_id
        }

    def run_session(self,
                     conversation: list[str],
                     verbose: bool = True) -> list[dict]:
        """
        Run a full conversation session.

        Args:
            conversation: list of user utterances
            verbose:      print turn details

        Returns:
            list of turn result dicts
        """
        memory  = ConversationMemory()
        results = []

        print("=" * 70)
        print("CONVERSATIONAL RAG SESSION")
        print("=" * 70)

        for turn_idx, utterance in enumerate(conversation):
            print(f"\nTurn {turn_idx + 1} — User: '{utterance}'")

            result = self.chat(memory, utterance, verbose=verbose)
            results.append(result)

            print(f"  Response: '{result['response'][:100]}...'")

        return results


# ── Part 6: Evaluation ─────────────────────────────────────────────────────

def evaluate_conversational_rag(
        rag: ConversationalRAG,
        test_sessions: list[dict],
        k: int = 3) -> dict:
    """
    Evaluate conversational RAG across multiple test sessions.

    Args:
        rag:           ConversationalRAG instance
        test_sessions: list of sessions, each with turns and relevant doc IDs
        k:             retrieval cutoff for NDCG

    Returns:
        evaluation metrics dict
    """
    import math

    all_ndcg       = []
    all_recall     = []
    turn_ndcg      = defaultdict(list)   # ndcg by turn position

    for session in test_sessions:
        memory = ConversationMemory()

        for turn_idx, turn in enumerate(session["turns"]):
            result   = rag.chat(memory, turn["utterance"], verbose=False)
            relevant = set(turn.get("relevant", []))

            if not relevant:
                continue

            retrieved_ids = [p["doc_id"] for p in result["passages"]]

            # NDCG@k
            dcg  = sum(
                1 / math.log2(i + 2)
                for i, doc_id in enumerate(retrieved_ids[:k])
                if doc_id in relevant
            )
            idcg = sum(
                1 / math.log2(i + 2)
                for i in range(min(len(relevant), k))
            )
            ndcg = dcg / idcg if idcg > 0 else 0.0

            # Recall@k
            recall = len(set(retrieved_ids[:k]) & relevant) / len(relevant)

            all_ndcg.append(ndcg)
            all_recall.append(recall)
            turn_ndcg[turn_idx].append(ndcg)

    # Context degradation — later turns harder?
    early_ndcg = [v for idx, vals in turn_ndcg.items()
                  if idx < 2 for v in vals]
    late_ndcg  = [v for idx, vals in turn_ndcg.items()
                  if idx >= 2 for v in vals]

    return {
        "ndcg@k":              float(np.mean(all_ndcg)) if all_ndcg else 0.0,
        "recall@k":            float(np.mean(all_recall)) if all_recall else 0.0,
        "early_turn_ndcg":     float(np.mean(early_ndcg)) if early_ndcg else 0.0,
        "late_turn_ndcg":      float(np.mean(late_ndcg)) if late_ndcg else 0.0,
        "context_degradation": (
            float(np.mean(early_ndcg)) - float(np.mean(late_ndcg))
            if early_ndcg and late_ndcg else 0.0
        )
    }


# ── Run demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    documents = {
        "D1":  "ColBERT is a neural retrieval model that uses token-level late "
               "interaction — each query token finds its best matching document token.",
        "D2":  "Bi-encoders encode queries and documents independently into single "
               "dense vectors and compute relevance as vector similarity.",
        "D3":  "ColBERT requires storing n×128 token vectors per document compared "
               "to one 768-dimensional vector for bi-encoders, making it 6-10x larger.",
        "D4":  "For 10 million documents, ColBERT requires roughly 90GB of storage "
               "while bi-encoders require approximately 15GB.",
        "D5":  "ColBERT v2 uses residual compression to reduce storage by 6-10x "
               "with minimal accuracy loss, making large-scale deployment practical.",
        "D6":  "Bi-encoders are faster at query time than ColBERT because document "
               "vectors are single fixed-size vectors enabling simple ANN search.",
        "D7":  "ColBERT consistently outperforms bi-encoders on retrieval accuracy "
               "benchmarks due to its fine-grained token-level matching.",
        "D8":  "FAISS IVF-PQ index can store bi-encoder embeddings for 10M documents "
               "in approximately 200MB using product quantization compression.",
        "D9":  "Cross-encoder rerankers are the most accurate but cannot precompute "
               "document representations making them unsuitable for first-stage retrieval.",
        "D10": "Hybrid search combines sparse BM25 retrieval with dense neural "
               "retrieval and consistently outperforms either approach alone.",
    }

    rag = ConversationalRAG(
        documents=documents,
        top_k=3,
        window_size=4,
        use_reformulation=True
    )

    conversation = [
        "What is ColBERT?",
        "How does it compare to bi-encoders?",
        "Which is better for a 10 million document corpus?",
        "What about memory requirements specifically?",
        "Is there a way to reduce that?",
    ]

    results = rag.run_session(conversation)

    # Evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    test_sessions = [
        {
            "turns": [
                {"utterance": "What is ColBERT?",
                 "relevant":  ["D1", "D7"]},
                {"utterance": "How does it compare to bi-encoders?",
                 "relevant":  ["D2", "D3", "D6", "D7"]},
                {"utterance": "Which is better for large scale?",
                 "relevant":  ["D4", "D5", "D8"]},
                {"utterance": "What about memory requirements?",
                 "relevant":  ["D3", "D4", "D5", "D8"]},
            ]
        }
    ]

    metrics = evaluate_conversational_rag(rag, test_sessions, k=3)
    print("\nConversational RAG metrics:")
    for metric, value in metrics.items():
        print(f"  {metric:<30}: {value:.4f}")
```

## Conversational RAG Failure Modes and Mitigations

Understanding where conversational RAG fails is as important as understanding
how it works:

```
Failure mode              Cause                         Mitigation
───────────────────────────────────────────────────────────────────────────
Context window overflow   Long history exceeds LLM      Summarization,
                          context limit                  windowing

Retrieval drift           Reformulation introduces       Better reformulation
                          wrong context                  model, fallback to
                                                         raw query

Contradiction             LLM generates response that    Consistency check,
                          contradicts prior turns        include full history

Passage repetition        Same passages retrieved        Already-retrieved
                          across multiple turns          penalization

Pronoun misresolution     Wrong entity resolved          Explicit entity
                          from history                   tracking, better
                                                         coreference model

Topic confusion           Old topic context bleeds       Topic shift detection,
                          into new topic                 context reset
```

Conversational RAG is the integration point of the conversational IR module.
It combines every technique from this module — multi-turn retrieval, query
reformulation, context management — into a complete dialogue system. It also
connects back to the standard RAG pipeline from 07-advanced/04-rag.md and
extends it with the statefulness needed for real dialogue.

## My Summary

Conversational RAG extends standard stateless RAG into a stateful dialogue system
by adding five components: query reformulation (converting context-dependent queries
into standalone ones), context-aware retrieval (using history to guide what to
retrieve), conversation memory management (selecting which prior context to include
in prompts), prompt assembly (combining history, retrieved passages, and current
query coherently), and grounded generation (producing responses consistent with
prior turns). The key engineering challenge is managing growing conversation state
within fixed LLM context windows — strategies include fixed windowing, periodic
summarization, and structured entity memory. Common failure modes include context
overflow, retrieval drift from poor reformulation, and cross-turn contradiction.
Conversational RAG is the foundation of most production AI assistants — virtually
every deployed RAG-based chat system implements some variant of this architecture.
