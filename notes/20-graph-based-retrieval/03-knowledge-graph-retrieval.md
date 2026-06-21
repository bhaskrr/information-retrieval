# Knowledge Graph Retrieval

Knowledge graph retrieval is the practice of answering information needs by
querying, traversing, or reasoning over a structured knowledge graph - a graph
where nodes represent entities (people, places, organizations, concepts, products)
and edges represent typed relations between them (born_in, founded, located_in,
treats, causes, is_a) - rather than searching over unstructured text documents.
Where text-based retrieval finds documents that are textually similar to a query,
knowledge graph retrieval finds and connects entities and facts that satisfy the
precise structural and semantic constraints implied by a query, often answering
questions that no single document in a text corpus directly states but that follow
logically from combining multiple facts. This note covers the construction of
knowledge graphs, the mechanisms for querying them (structured query languages,
embedding-based retrieval, and traversal-based reasoning), and how knowledge
graph retrieval integrates with and complements text-based retrieval in hybrid
systems.

## Intuition

Consider the query: "Which Nobel laureates in Physics were born in countries
that no longer exist?" No single Wikipedia article or document is likely to
contain this exact compound fact. Answering it requires: (1) identifying all
Nobel Physics laureates, (2) identifying their birth countries, (3) checking
which of those countries no longer exist (East Germany, Yugoslavia, USSR,
Austria-Hungary), and (4) returning the laureates whose birth country matches.

Text-based retrieval, however sophisticated, struggles with this query because
it requires synthesizing facts from many separate sources and applying logical
filtering across them - not finding a single passage that states the answer.
A knowledge graph, by contrast, represents each laureate as a node with an
edge to their birth country node, and represents historical countries with
explicit existence date ranges. The query becomes a graph traversal: find all
nodes of type "Person" with edge "won_prize"→"Nobel Prize in Physics", follow
their "born_in"→Country edges, and filter to countries with an "end_date"
attribute before the present.

This is the fundamental value proposition of knowledge graph retrieval: it
excels precisely where text retrieval struggles - compositional queries that
require connecting multiple discrete facts through explicit, typed relationships,
where the relationships themselves (not just textual similarity) carry the
core of the information need.

## Knowledge Graph Construction

A knowledge graph must be built before it can be queried. Construction
involves three core tasks, often performed in sequence over a text corpus
or structured data sources.

### Entity extraction and typing

Identify mentions of entities in source documents and assign them a type:

```
Source text: "Albert Einstein developed the theory of general relativity
              while working at the Prussian Academy of Sciences."

Extracted entities:
  "Albert Einstein"  → type: Person
  "general relativity" → type: Theory/Concept
  "Prussian Academy of Sciences" → type: Organization
```

Entity extraction uses named entity recognition (NER) models, increasingly
LLM-based extraction for nuanced or domain-specific entity types beyond the
standard PERSON/ORG/LOCATION categories.

### Entity resolution and disambiguation

Map extracted entity mentions to canonical entity nodes, merging different
surface forms that refer to the same real-world entity and disambiguating
surface forms that could refer to multiple entities:

```
Surface forms mapping to the same canonical entity:
  "Einstein", "Albert Einstein", "A. Einstein" → Entity: Q937 (Wikidata ID)

Disambiguation challenge:
  "Cambridge" → could be Entity: Q350 (Cambridge, UK)
                or Entity: Q49111 (Cambridge, Massachusetts)
  Resolved using surrounding context (mentioned alongside "University"
  vs "MIT" vs other disambiguating signals)
```

This is the entity linking problem, covered in depth in note 06 of this module.
It is foundational infrastructure for knowledge graph construction and retrieval.

### Relation extraction

Identify the typed relationships between extracted entities:

```
Source text: "Albert Einstein developed the theory of general relativity
              while working at the Prussian Academy of Sciences."

Extracted relations (triples):
  (Albert Einstein, developed, general relativity)
  (Albert Einstein, worked_at, Prussian Academy of Sciences)
```

Relation extraction methods range from rule-based pattern matching (high
precision, low recall, for well-defined relation types) to supervised neural
classifiers trained on labeled relation examples, to LLM-based extraction
(prompting an LLM to identify and structure relations from text, increasingly
the dominant approach for building new domain-specific knowledge graphs).

### Knowledge graph schemas

A schema defines the allowed entity types and relation types in the graph,
providing structure that constrains and validates extraction:

```
Example schema fragment (academic domain):
  Entity types: Person, Paper, Institution, Concept, Venue
  Relation types:
    Person --authored--> Paper
    Paper --cites--> Paper
    Paper --published_at--> Venue
    Person --affiliated_with--> Institution
    Paper --introduces--> Concept
    Concept --related_to--> Concept
```

Well-designed schemas are critical - overly permissive schemas (allowing
arbitrary free-text relations) sacrifice the structural precision that makes
knowledge graphs valuable, while overly rigid schemas may fail to capture
nuanced relationships present in the source data.

## Pre-Built Knowledge Graphs

Building a knowledge graph from scratch is expensive. For many applications,
existing large-scale knowledge graphs provide substantial coverage without
requiring custom construction:

```
Wikidata:
  Collaborative, structured knowledge graph derived from Wikipedia and
  curated directly. ~100 million entities, billions of facts. General-purpose
  coverage across nearly all domains.

DBpedia:
  Knowledge graph extracted from Wikipedia infoboxes and structured content.
  Strong coverage of well-documented entities (people, places, organizations).

YAGO:
  Combines Wikipedia, WordNet, and GeoNames. Strong on taxonomic
  (is-a) relationships and temporal facts.

Domain-specific knowledge graphs:
  UMLS / SNOMED CT (medical): diseases, symptoms, treatments, drug interactions
  PubChem / ChEMBL (chemistry/pharmacology): molecular structures, drug targets
  Patents/USPTO graphs: patent citation and inventor networks
  Enterprise knowledge graphs: internal company knowledge bases, often built
    from CRM, documentation, and organizational data
```

For applications in well-covered domains (general knowledge, biomedical,
chemistry), starting from an existing knowledge graph and extending it with
domain-specific or proprietary facts is almost always more practical than
building from scratch.

## Querying Knowledge Graphs

Three distinct paradigms exist for extracting information from a knowledge
graph, each suited to different query types.

### Structured query languages (SPARQL, Cypher)

For users or systems that can express their information need as a precise
structured query, graph query languages provide exact, deterministic answers:

```
SPARQL example (querying Wikidata-style graph):
  SELECT ?laureate ?country WHERE {
    ?laureate wdt:P166 wd:Q38104 .          # won Nobel Prize in Physics
    ?laureate wdt:P19 ?birthplace .         # born in
    ?birthplace wdt:P17 ?country .          # country
    ?country wdt:P576 ?dissolution_date .   # has dissolution date
  }
```

This directly answers the compositional query from the introduction - but
requires the query to already be expressed in structured form, with knowledge
of the exact schema (property IDs, entity IDs) used by the graph.

**The natural-language-to-query gap:** Most users (and most retrieval
applications receiving free-text queries) cannot directly author SPARQL or
Cypher queries. This motivates either (a) natural-language-to-query translation
(an LLM or trained semantic parser converts the free-text query into a
structured query) or (b) embedding-based and traversal-based approaches that
do not require an exact structured query.

### Embedding-based knowledge graph retrieval

Rather than requiring an exact structured query, knowledge graph embedding
methods learn vector representations for entities and relations such that
graph structure is preserved in the embedding space, enabling similarity-based
retrieval and even prediction of missing facts:

```
TransE (foundational KG embedding method):
  Learns embeddings such that: entity_emb(head) + relation_emb(r) ≈ entity_emb(tail)
  For triple (Einstein, born_in, Germany):
    emb(Einstein) + emb(born_in) ≈ emb(Germany)

Retrieval application:
  Given a partial query "(Einstein, born_in, ?)", compute:
    candidate_emb = emb(Einstein) + emb(born_in)
  Retrieve entities whose embedding is nearest to candidate_emb
  → directly answers "where was Einstein born" without explicit graph traversal

More expressive KG embedding methods:
  RotatE: represents relations as rotations in complex vector space,
          better captures relation patterns like symmetry and inversion
  ComplEx: uses complex-valued embeddings, handles asymmetric relations
           more effectively than TransE
  Knowledge graph embeddings via GNNs (R-GCN, CompGCN):
          incorporate the relational graph structure through message passing
          rather than only pairwise translation/rotation operations
```

KG embedding methods are particularly valuable for **link prediction** - inferring
likely missing facts that were never explicitly stated in the source data but
are implied by the graph's overall structure. This capability has no direct
analogue in text retrieval, which can only surface facts that are explicitly
present in some document.

### Traversal-based retrieval and multi-hop reasoning

For compositional queries requiring connecting multiple entities through
relation chains, explicit graph traversal - guided by query understanding -
provides interpretable, precise answers:

```
Query: "What university did the founder of Tesla attend?"

Traversal process:
  1. Entity linking: identify "Tesla" → Entity Q478214 (Tesla, Inc.)
  2. Relation identification: "founder of" → relation type "founded_by"
  3. Traverse: Tesla --founded_by--> Elon Musk
  4. Relation identification: "attend" (university) → relation type "educated_at"
  5. Traverse: Elon Musk --educated_at--> University of Pennsylvania
  6. Return: "University of Pennsylvania"
```

Multi-hop traversal requires query decomposition (breaking a complex natural
language query into a sequence of entity and relation identification steps)
and is increasingly implemented using LLMs as the query decomposition and
traversal-planning component, with the actual graph lookups executed against
a structured graph database.

## Hybrid Text-Knowledge Graph Retrieval

In practice, knowledge graph retrieval is rarely deployed in isolation - most
production systems combine it with text-based retrieval, exploiting the
complementary strengths of each approach.

### Entity-augmented text retrieval

Use knowledge graph entity context to enrich text retrieval rather than
replacing it:

```
Standard text retrieval:
  query_embedding = encode(query_text)
  document_embedding = encode(document_text)
  score = similarity(query_embedding, document_embedding)

Entity-augmented retrieval:
  query_entities = entity_link(query_text)
  document_entities = entity_link(document_text)
  entity_overlap_score = jaccard_similarity(query_entities, document_entities)
  graph_distance_score = shortest_path_distance(query_entities, document_entities)

  final_score = α × text_similarity + β × entity_overlap_score
              + γ × (1 / graph_distance_score)
```

This approach uses the knowledge graph as an additional signal layered on
top of standard text retrieval, providing entity-level precision (does this
document actually discuss the same entity the query asks about, not just
similar vocabulary) without requiring the query to be expressed as a structured
graph query.

### Knowledge graph as a structured filter

Use graph queries to narrow a candidate set before applying text-based ranking:

```
Step 1: Parse query to identify structural constraints
        "papers about transformers published after 2020 by authors
         affiliated with Google"
        → structural constraint: topic=transformers, year>2020,
          author_affiliation=Google

Step 2: Query knowledge graph to identify the candidate document set
        satisfying these structural constraints
        candidates = KG_query(topic=transformers, year>2020,
                               author_affiliation=Google)

Step 3: Apply text-based ranking (BM25, dense retrieval, reranking)
        within this pre-filtered candidate set
        ranked_results = text_rank(query, candidates)
```

This pattern is especially valuable for queries with explicit structural
constraints (dates, categorical attributes, named entities with specific
relationships) that text similarity alone cannot reliably enforce - a dense
retrieval model might retrieve a highly topically relevant paper from 2018
despite the explicit "after 2020" constraint, because topical similarity
does not encode the date filter.

### Knowledge graph for query expansion

Use the knowledge graph's relational structure to expand a query with related
entities and concepts before applying standard text retrieval:

```
Original query: "treatments for myocardial infarction"

Knowledge graph expansion:
  myocardial infarction --is_a--> heart attack (synonym/related term)
  myocardial infarction --treated_by--> [thrombolytics, PCI, aspirin, ...]

Expanded query: "treatments for myocardial infarction heart attack
                 thrombolytics percutaneous coronary intervention aspirin"

This expanded query is then used for standard BM25 or dense retrieval,
benefiting from the precise, curated relational knowledge in the graph
rather than relying solely on the embedding model's learned associations
```

This is a structured alternative (or complement) to the LLM-based query
expansion techniques covered in the upcoming LLM-augmented retrieval module -
knowledge-graph-based expansion provides curated, precise relation knowledge,
while LLM-based expansion provides more flexible but potentially noisier
generative expansion.

## Knowledge Graph Retrieval for Question Answering

Knowledge graph question answering (KGQA) is one of the most mature
applications of knowledge graph retrieval, with a well-established
processing pipeline:

```
Pipeline stages:

1. Question understanding:
   Parse the natural language question to identify the question type
   (factual lookup, comparison, aggregation, multi-hop) and the entities
   and relations it references

2. Entity linking:
   Map question entity mentions to knowledge graph entity nodes

3. Query construction:
   Convert the parsed question into a structured query (SPARQL, graph
   traversal plan, or embedding-based nearest-neighbor lookup)

4. Query execution:
   Execute the structured query against the knowledge graph

5. Answer extraction and verbalization:
   Convert the structured query result (entity IDs, numeric values)
   back into a natural language answer
```

### Simple vs complex KGQA

```
Simple factual questions (single-hop):
  "When was Einstein born?"
  → Direct lookup: Einstein --born_on--> [date]
  → High accuracy achievable with entity linking + single relation lookup

Multi-hop questions:
  "What is the population of the country where Einstein was born?"
  → Requires: Einstein --born_in--> Germany --has_population--> [number]
  → Requires correct decomposition into a chain of relation lookups

Comparative/aggregation questions:
  "Which Nobel laureate in Physics has the most citations?"
  → Requires: identify all laureates, retrieve citation counts for each,
    compute maximum
  → Requires aggregation logic beyond simple traversal

Questions requiring graph + text:
  "Why did Einstein develop the theory of relativity?"
  → "Why" questions typically require explanatory text, not just
    structured facts - knowledge graph alone is insufficient,
    text retrieval (or graph-to-text generation) is needed
```

The progression from simple to complex KGQA reveals an important limitation:
knowledge graphs excel at factual, structural questions but are poorly suited
to explanatory, narrative, or nuanced questions where the "answer" is not a
discrete fact or entity but a textual explanation. This is precisely where
hybrid graph + text approaches, and ultimately Graph RAG (covered in note 05
of this module), become necessary.

## Limitations of Knowledge Graph Retrieval

Understanding when knowledge graph retrieval falls short is as important as
understanding its strengths.

### Coverage gaps

Knowledge graphs are inherently incomplete - they only contain facts that
were explicitly extracted and curated. Text corpora, by contrast, contain
the full richness of natural language expression, including facts, nuances,
and context that have never been formally extracted into graph triples.

```
A knowledge graph might capture:
  (Aspirin, treats, headache)
  (Aspirin, treats, fever)

But miss the nuanced clinical context available in a medical text:
  "Aspirin is generally effective for tension headaches but is contraindicated
   in patients with active gastrointestinal bleeding and should be used with
   caution in combination with anticoagulant therapy."

The knowledge graph triple captures the basic relation but loses the
contraindications, dosage nuance, and conditional logic present in the
source text.
```

### Schema rigidity

A knowledge graph's schema constrains what kinds of facts can be represented.
Novel relationship types not anticipated by the schema cannot be naturally
incorporated without schema extension - a non-trivial engineering task,
especially for graphs with millions of existing facts conforming to the
prior schema.

### Extraction errors propagate

Errors introduced during entity extraction, entity resolution, or relation
extraction become embedded as incorrect facts in the graph, and these errors
can compound through multi-hop traversal - an incorrect single-hop fact at
the start of a reasoning chain produces an incorrect final answer, with no
mechanism within the graph itself to flag the uncertainty.

### Temporal and contextual nuance

Facts often have temporal validity (a person's employer changes over time,
a country's borders change) and contextual qualifications (a treatment is
effective "in most cases" or "for patients without contraindication X") that
simple triple-based graph representations struggle to capture without
significant schema complexity (temporal knowledge graphs, qualified statements
as in Wikidata's qualifier system).

## Where This Fits in the Progression

```
01-why-graphs-for-retrieval         → the motivation and landscape
02-graph-neural-networks-for-ir     → GNN architectures for retrieval
03-knowledge-graph-retrieval        → entity-centric retrieval  ← you are here
04-heterogeneous-graph-retrieval    → multi-type graph retrieval
05-graph-rag                        → graph-augmented generation
06-entity-linking-for-retrieval     → connecting text to graph entities
```

## My Summary

Knowledge graph retrieval answers information needs by querying or traversing
a structured graph of typed entities and relations rather than searching
unstructured text, excelling precisely where text retrieval struggles: compositional
queries that require connecting multiple discrete facts through explicit
relationships. Construction requires entity extraction (identifying entity
mentions and their types), entity resolution (mapping mentions to canonical
entities and disambiguating ambiguous references), and relation extraction
(identifying typed relationships between entities), increasingly performed
using LLM-based extraction for flexibility across domains. Pre-built knowledge
graphs (Wikidata, DBpedia, domain-specific graphs like UMLS for medicine) provide
substantial coverage without requiring custom construction from scratch. Three
querying paradigms serve different needs: structured query languages (SPARQL,
Cypher) provide exact deterministic answers but require the query already expressed
in structured form; embedding-based methods (TransE, RotatE, ComplEx) learn
vector representations enabling similarity-based retrieval and crucially link
prediction - inferring facts never explicitly stated; and traversal-based
multi-hop reasoning, increasingly LLM-guided, decomposes natural language
questions into sequences of relation lookups. In practice, knowledge graph
retrieval is rarely deployed alone - hybrid patterns include entity-augmented
text retrieval (adding entity overlap as a ranking signal), knowledge graph as
structured pre-filter (narrowing candidates by exact attribute constraints before
text ranking), and knowledge-graph-based query expansion (adding curated related
terms before standard retrieval). Knowledge graph question answering reveals
a clear capability gradient from simple single-hop factual lookups through
multi-hop reasoning to aggregation questions, with explanatory "why" questions
exposing the fundamental limitation that knowledge graphs capture discrete facts
but not the narrative nuance, conditional logic, and contextual qualification
present in natural text - motivating the hybrid graph-plus-text approaches
covered in subsequent notes.
