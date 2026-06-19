# Why Graphs for Retrieval

Graph-based retrieval is an approach to information retrieval that represents
documents, entities, and their relationships as nodes and edges in a graph
structure, then exploits that graph structure - through propagation, traversal,
or graph neural network message passing - to improve relevance estimation beyond
what text similarity alone can capture. Standard retrieval, whether sparse (BM25)
or dense (bi-encoders), treats each document as an isolated unit: relevance is
computed purely from the textual content of the query and the document, with no
consideration of how documents relate to each other or to the broader entity
landscape they describe. Graph-based retrieval explicitly models these relationships -
citation networks between papers, hyperlink structures between web pages, co-
purchase patterns between products, knowledge graph relations between entities -
and uses this relational structure as an additional, often highly informative,
signal for relevance estimation. This note motivates why relational structure
matters for retrieval and introduces the conceptual foundations that the rest of
this module builds on.

## Intuition

Consider trying to find the most authoritative paper on transformer architectures.
Pure text similarity would retrieve any paper whose abstract uses similar
vocabulary - "attention mechanism," "self-attention," "sequence transduction."
But text similarity cannot distinguish between a foundational paper that
introduced these concepts and a minor paper that merely uses the same terminology
in passing. The original "Attention Is All You Need" paper might not even use
the word "transformer" in its abstract in the same density as papers written
five years later that extensively reference and build on it.

What text similarity misses, the citation graph captures directly: papers that
many other important papers cite are likely to be important themselves. This is
exactly the insight behind PageRank, originally developed for ranking web pages
by their link structure rather than just their content. The same insight extends
naturally to scientific literature (citation graphs), product recommendation
(co-purchase graphs), code search (import/dependency graphs), and entity-centric
question answering (knowledge graphs).

The relational structure is not just a tiebreaker for cases where text similarity
is ambiguous - it often carries information that text similarity cannot represent
at all. Two documents about the same topic, written by completely different
authors using completely different vocabulary, may be more meaningfully related
through their citation patterns (they both cite the same foundational works, or
one explicitly builds on the other) than through any text-based similarity measure.
Graph-based retrieval is the family of techniques that captures and exploits
this relational signal.

## What Text-Based Retrieval Cannot Capture

Standard retrieval methods - BM25, dense bi-encoders, learned sparse models -
share a fundamental limitation regardless of their internal sophistication:
they compute relevance from the textual content of a single query and a single
document in isolation. This creates several specific blind spots that graph
structure can address.

### Blind spot 1 - Authority and importance

Text similarity cannot distinguish a seminal, highly-cited work from a minor
paper that happens to use similar vocabulary:

```
Query: "transformer architecture for sequence modeling"

Document A: "Attention Is All You Need" (2017)
  Cited by 100,000+ papers, foundational to the field
  Text similarity to query: moderate (introduces the term "transformer"
  but predates much of the now-standard terminology)

Document B: A 2023 workshop paper applying transformers to a niche task
  Cited by 12 papers, minor contribution
  Text similarity to query: potentially HIGHER than Document A
  (uses more current "transformer architecture" terminology directly)

Pure text retrieval may rank B above A. Citation-graph-aware retrieval
correctly identifies A's foundational importance through its citation count
and the importance of papers that cite it.
```

### Blind spot 2 - Implicit relevance through relationship

Two documents can be highly relevant to each other without sharing significant
vocabulary, if their relationship is established through structural connections
rather than textual overlap:

```
Document A: A paper proposing a new optimization algorithm,
            written in highly mathematical notation with minimal prose
Document B: A blog post explaining "why this algorithm works" in
            plain language with different terminology

Text similarity(A, B): potentially low (different vocabulary, different style)
Graph relationship: B explicitly cites and explains A
  → A hyperlink or citation edge directly establishes their relationship
  → This relationship is invisible to pure text similarity but
    directly visible in the graph structure
```

### Blind spot 3 - Entity disambiguation and relation context

When a query mentions an entity, text-based retrieval cannot easily distinguish
between different entities sharing the same name, but a knowledge graph can
use relational context to disambiguate:

```
Query: "Cambridge research output in machine learning"

Text retrieval: matches documents containing "Cambridge" and "machine learning"
  → ambiguous between Cambridge, UK and Cambridge, Massachusetts (MIT/Harvard)
  → ambiguous between Cambridge University and Cambridge, the city

Knowledge graph retrieval:
  → Cambridge University node has explicit edges: located_in(UK),
    has_department(Computer_Laboratory), affiliated_with(researchers...)
  → Graph traversal can correctly resolve which "Cambridge" based on
    connected entity types and relation patterns
```

### Blind spot 4 - Multi-hop reasoning requirements

Some queries require connecting information across multiple documents or
entities that no single document contains:

```
Query: "Which company did the founder of company X previously work at
        before starting a competitor to company Y?"

This requires:
  1. Identify founder of company X
  2. Identify their previous employer
  3. Verify that employer is a competitor to company Y

No single document is likely to contain all three facts connected together.
A knowledge graph with entity and relation edges can traverse:
  X --founded_by--> Person --previously_worked_at--> Company Z
  Company Z --competitor_of--> Y

Text-based retrieval over individual documents struggles with this kind
of compositional, multi-hop information need.
```

## The Spectrum of Graph Structures in IR

Graph-based retrieval is not a single technique but a family of approaches
that differ in what the graph nodes and edges represent:

### Citation graphs (academic literature)

```
Nodes: papers
Edges: "cites" relationships (directed)

Properties exploited:
  In-degree (citation count): proxy for importance/authority
  Co-citation: papers cited together are often related
  Bibliographic coupling: papers that cite the same sources are often related
  Citation recency and context: recent citations may matter more than old ones
```

### Hyperlink graphs (web search)

```
Nodes: web pages
Edges: hyperlinks (directed)

Properties exploited:
  PageRank: importance propagates through link structure
  HITS (hubs and authorities): distinguishes pages that link to good
    content (hubs) from pages that are good content (authorities)
  Anchor text: the link text itself provides a relevance signal for
    the target page, often more descriptive than the page's own content
```

### Knowledge graphs (entity-centric retrieval)

```
Nodes: entities (people, places, organizations, concepts)
Edges: typed relations (born_in, founded, located_in, employed_by, ...)

Properties exploited:
  Entity disambiguation: relational context resolves ambiguous mentions
  Multi-hop traversal: answer composite queries through relation chains
  Entity linking: connect mentions in text to graph entities
  Structured relevance: relation type matching provides precise filtering
```

### Co-occurrence and co-interaction graphs (recommendation, e-commerce)

```
Nodes: products, items, users
Edges: co-purchase, co-view, co-click relationships

Properties exploited:
  Collaborative signal: items frequently purchased together are related
  User behavior propagation: user preferences propagate through interaction graph
  Cold-start mitigation: new items can inherit relevance signal from
    structurally similar items even before direct interaction data accumulates
```

### Document similarity graphs (constructed for retrieval)

```
Nodes: documents or passages
Edges: similarity above a threshold (constructed from embeddings)

Properties exploited:
  Graph-based reranking: propagate relevance scores through similar documents
  Diversity-aware retrieval: avoid retrieving multiple documents from a
    densely connected cluster (redundant content)
  Pseudo-relevance feedback: expand the candidate set through graph neighbors
    of initially retrieved high-scoring documents
```

## How Graph Structure Improves Relevance Estimation

The core mechanisms by which graph-based methods enhance retrieval quality,
each covered in depth in subsequent notes:

### Mechanism 1 - Authority propagation

Algorithms like PageRank propagate "importance" scores through the graph
structure, independent of any specific query. A document's static importance
score can be combined with query-specific text relevance to produce a final
ranking:

```
final_score(q, d) = α × text_relevance(q, d) + (1-α) × authority_score(d)

Where authority_score(d) is computed once from the graph structure
(not dependent on the specific query)
```

### Mechanism 2 - Relevance propagation (graph-based reranking)

Rather than a static, query-independent score, relevance propagation spreads
query-specific relevance signal through the graph after initial retrieval:

```
Step 1: Standard retrieval identifies initial relevant documents (seed set)
Step 2: Propagate relevance through graph edges to neighboring documents
Step 3: Documents connected to high-relevance seeds receive boosted scores,
        even if their direct text relevance to the query is lower
```

This is the mechanism behind pseudo-relevance feedback approaches that use
document similarity graphs, and behind entity-centric approaches that use
knowledge graph traversal from initially identified entities.

### Mechanism 3 - Structural embedding (graph neural networks)

Rather than treating graph structure as a separate signal combined with text
relevance, graph neural networks (GNNs) directly incorporate graph structure
into the learned representation of each node, producing embeddings that encode
both textual content and relational context simultaneously:

```
node_embedding(d) = GNN(text_embedding(d), {neighbor_embeddings})

The final embedding for document d incorporates information from its
neighbors in the graph, which themselves incorporate information from
their neighbors - multi-hop relational context is baked directly into
the representation used for retrieval.
```

This is the most powerful and most computationally expensive mechanism,
covered in depth in the GNN-focused note of this module.

### Mechanism 4 - Multi-hop traversal for compositional queries

For queries requiring connecting multiple facts across entities, explicit
graph traversal algorithms (not learned representations) can directly
answer compositional queries that text retrieval cannot:

```
Query decomposition: identify the entities and relation types implied
                     by the query
Graph traversal: follow relation edges to connect the required entities
Answer construction: synthesize the traversal path into a coherent answer
```

This mechanism is the foundation of knowledge-graph-based question answering
and is covered in depth in the knowledge graph retrieval note.

## When Graph Structure Helps vs When It Does Not

Graph-based retrieval is not universally beneficial. Understanding when
relational structure provides genuine signal versus when it adds complexity
without corresponding quality improvement is essential for principled
application.

### Graph structure helps when

**Rich, meaningful relational structure exists in the domain:**
Academic literature (citations), web pages (hyperlinks), e-commerce
(co-purchase patterns), and structured knowledge domains (Wikipedia,
biomedical literature with curated ontologies) all have naturally occurring,
information-rich graph structure.

**Authority or importance is a meaningful relevance dimension:**
For queries seeking authoritative or foundational sources rather than merely
topically matching content, graph-derived authority signals provide value
text similarity cannot.

**Queries require multi-hop or compositional reasoning:**
Questions that cannot be answered from a single document but require
connecting facts across multiple entities benefit from explicit graph
traversal capability.

**Cold-start scenarios for new content:**
New documents or products with limited text history can inherit relevance
signal from their structural position in the graph (citing well-established
work, co-located with related products) even before sufficient direct
interaction or citation data accumulates.

### Graph structure does not help (or adds unnecessary complexity) when

**No meaningful relational structure exists:**
A collection of independent customer support tickets, isolated product
reviews, or standalone FAQ entries typically lacks the kind of rich
relational structure that citation or hyperlink graphs provide. Constructing
an artificial graph (e.g., via embedding similarity) may add complexity
without corresponding signal.

**Queries are purely topical/factual with no authority dimension:**
"What is the boiling point of water?" does not benefit from authority
propagation - any document stating the correct fact is equally valid,
regardless of its position in any citation or link graph.

**Graph construction or maintenance cost exceeds the value gained:**
Building and maintaining an accurate, up-to-date graph (especially knowledge
graphs requiring entity extraction, disambiguation, and relation extraction)
is a significant engineering investment. For applications where simpler
methods already achieve acceptable quality, this investment may not be
justified.

**Latency requirements are very tight:**
Graph traversal, especially multi-hop traversal or GNN inference, typically
adds latency compared to standard vector similarity search. For applications
with strict sub-50ms latency budgets, graph-based methods may be impractical
without significant precomputation and caching infrastructure.

## Connecting to This Repository's Research Context

Graph-based retrieval connects directly to the recommender systems research
covered earlier in this repository - particularly the UIFRS-HAN architecture
and the broader exploration of heterogeneous graph neural networks for
recommendation. The conceptual machinery is shared: both graph-based retrieval
and graph-based recommendation represent entities (documents/items) and their
relationships as a heterogeneous graph, then use graph neural networks or
graph traversal to produce relevance or preference scores that incorporate
relational structure beyond what content alone provides.

The key distinction is the task framing: recommendation typically predicts
user-item affinity from interaction graphs (user nodes, item nodes, interaction
edges), while graph-based retrieval typically predicts query-document relevance
from content and relational graphs (document nodes, citation/link/entity edges).
The underlying GNN architectures, training objectives (often contrastive,
connecting back to the dense retrieval training module), and evaluation
challenges are substantially shared between these two areas - making this
module a natural extension for someone with a recommender systems research
background moving into broader IR system design.

## The Structure of This Module

The subsequent notes build from foundational graph algorithms through to
state-of-the-art graph-augmented retrieval and generation systems:

```
02-graph-neural-networks-for-ir.md
  GNN architectures (GCN, GAT, GraphSAGE) applied to retrieval tasks
  How message passing produces structure-aware document/entity embeddings

03-knowledge-graph-retrieval.md
  Entity-centric retrieval over structured knowledge graphs
  Entity linking, relation extraction, graph query answering

04-heterogeneous-graph-retrieval.md
  Multi-type node and edge graphs (documents, entities, users, attributes)
  Directly connects to UIFRS-HAN and heterogeneous GNN research background

05-graph-rag.md
  Combining graph traversal with retrieval-augmented generation
  Graph-structured context construction for LLM grounding

06-entity-linking-for-retrieval.md
  Connecting text mentions to knowledge graph entities
  The preprocessing step that enables knowledge-graph-aware retrieval
```

## My Summary

Graph-based retrieval addresses a fundamental limitation shared by all text-only
retrieval methods - BM25, dense bi-encoders, learned sparse models - which compute
relevance purely from the textual content of a query and document in isolation,
ignoring the relational structure that connects documents, entities, and their
context. This blind spot manifests in four ways: inability to distinguish authoritative
foundational sources from minor works using similar vocabulary, inability to capture
implicit relevance established through structural relationships rather than textual
overlap, inability to disambiguate entities using relational context, and inability
to answer compositional queries requiring multi-hop reasoning across multiple
documents or entities. Different domains provide different graph structures to
exploit: citation graphs for academic literature, hyperlink graphs for web search,
knowledge graphs for entity-centric retrieval, and co-occurrence graphs for
recommendation and e-commerce. The mechanisms by which graph structure improves
relevance estimation range from simple authority propagation (PageRank-style static
importance scores) through relevance propagation (spreading query-specific signal
through graph neighbors after initial retrieval) to structural embedding via graph
neural networks (baking relational context directly into learned representations)
and explicit multi-hop traversal for compositional question answering. Graph-based
methods provide the most value when rich relational structure naturally exists in
the domain, when authority or importance is a meaningful relevance dimension, and
when queries require connecting facts across multiple entities - and provide little
value when no meaningful relational structure exists or when graph construction and
maintenance costs exceed the quality benefit gained. This module connects directly
to heterogeneous graph neural network research in recommendation systems, sharing
much of its underlying architectural and training methodology.
