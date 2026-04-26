# Personalization in IR

Personalization in IR is the practice of tailoring retrieval results to individual
users based on their past behavior, stated preferences, demographic attributes, or
inferred interests. Instead of returning the same ranked list to every user for a
given query, a personalized retrieval system produces different rankings for
different users - boosting documents that align with a user's history and
preferences while demoting those that do not. Personalization is the bridge between
pure information retrieval and recommendation systems, and is a standard component
of production search engines, e-commerce platforms, and AI assistants.

## Intuition

The same query from two different users can have completely different ideal answers.

```
Query: "python"

User A: a data scientist who has been reading machine learning papers
  → ideal results: Python for data science, NumPy, Pandas, scikit-learn

User B: a biology student who was just reading about reptiles
  → ideal results: Python (snake) species, habitat, care guides
```

Without personalization, the search engine must guess which interpretation is
more common and return the same ranked list to both. With personalization, it
knows User A's history suggests the programming language and User B's suggests
the animal - and ranks accordingly.

Personalization improves two things simultaneously:

- **Precision** - results are more likely to match what this specific user wants
- **Efficiency** - users find relevant documents faster, reducing reformulations

The challenge is doing this without violating privacy, without overfitting to
stale history, and without creating a filter bubble where users only ever see
content confirming their existing interests.

## Sources of Personalization Signal

### Explicit feedback

Users directly express preferences:

```
Star ratings, thumbs up/down, bookmarks, saved queries,
explicit interest declarations ("I'm interested in machine learning")
```

Clean signal but sparse - most users rarely provide explicit feedback.

### Implicit feedback

Behavioral signals inferred from user actions:

```
Click-through:    which results did the user click?
Dwell time:       how long did the user spend on a clicked result?
Scroll depth:     how far did the user read?
Return visits:    did the user come back to a document?
Query reformulation: did the user refine their query (bad result)?
```

Noisy but abundant. Click does not always imply relevance (clickbait) and
no-click does not always imply non-relevance (user found answer in snippet).

### Session context

What has the user been doing in the current session?

```
Recent queries, recently viewed documents, current conversation history,
items in cart (for e-commerce), recently read articles
```

Short-term signal, highly predictive for the current task.

### Long-term user profile

Aggregated history across sessions:

```
Frequently searched topics, preferred document types (papers vs tutorials),
reading level, language preferences, domain expertise level
```

Captures stable user interests. Requires storage and privacy consideration.

### Collaborative filtering signals

What did similar users prefer for this query?

```
Users who searched "transformer architecture" also clicked papers on
attention mechanisms and BERT - boost these for new users with similar profiles
```

Fills gaps in individual history by borrowing signal from similar users.

## Personalization Approaches

### Approach 1 - Query expansion with user profile

Expand the query with terms from the user's interest profile before retrieval:

```
Query:       "transformer"
User profile: frequent queries about machine learning, NLP, neural networks

Expanded:    "transformer neural network architecture attention mechanism"
             → retrieves ML documents rather than electrical transformers
```

Simple, interpretable, and compatible with any retrieval backend. The
personalization happens at the query level - the retrieval system itself
is unchanged.

### Approach 2 - Result reranking with user signals

Retrieve a standard result set, then rerank it using user-specific signals:

```
Standard retrieval:     [D1, D2, D3, D4, D5, ...]
User history signals:   D3 and D5 are on topics the user frequently reads
Personalized reranking: [D3, D5, D1, D2, D4, ...]
```

Flexible and modular - the retrieval system provides recall, personalization
provides precision. The reranker can incorporate any combination of signals.

### Approach 3 - User-conditioned retrieval models

Learn a retrieval model that takes the user representation as input alongside
the query:

```
score(q, d, u) = f(query_features, document_features, user_features)
```

The user representation u encodes the user's history and preferences. During
inference, the model scores documents based on both query-document relevance
and user-document affinity.

Neural approaches:

```
q_vec  = query_encoder(q)
d_vec  = doc_encoder(d)
u_vec  = user_encoder(user_history)

score  = (q_vec + α × u_vec) · d_vec    ← linear user query modulation
      or MLP(concat(q_vec, d_vec, u_vec)) ← nonlinear combination
```

### Approach 4 - Collaborative filtering integration

Use matrix factorization or neural collaborative filtering to learn user and
document embeddings jointly from click/rating data:

```
Users × Documents interaction matrix
    ↓ matrix factorization
User embeddings:     u_i ∈ ℝ^d
Document embeddings: v_j ∈ ℝ^d

Relevance for new query:
  score(q, d, u) = query_score(q, d) + β × u_i · v_j
```

Captures long-term user preferences beyond the current query context.

## User Modeling

### Short-term user model (session context)

Represents what the user wants right now, based on current session:

```python
class SessionModel:
    recent_queries:   deque  # last 5 queries
    viewed_docs:      deque  # last 10 documents viewed
    current_intent:   str    # inferred intent (informational/navigational)
    session_topic:    str    # current topic of interest
```

High recency weight - recent clicks matter much more than clicks from last week.

### Long-term user model (persistent profile)

Represents stable user interests across sessions:

```python
class UserProfile:
    interest_topics:  dict[str, float]  # topic → weight from history
    preferred_types:  list[str]         # papers, tutorials, docs, news
    expertise_level:  str               # beginner, intermediate, expert
    language_pref:    str               # preferred language
    click_history:    list[str]         # doc_ids of previously clicked docs
```

Updated incrementally as the user interacts with the system.

### Hybrid model

Combine short-term session context with long-term profile:

```
final_query_representation = λ × session_vector + (1 - λ) × profile_vector
```

λ close to 1 emphasizes current session context.
λ close to 0 emphasizes long-term stable preferences.

Dynamically adjust λ based on session length - early in a session use more
long-term profile, later in a session use more session context.

## Personalized Reranking

The most practical personalization approach for a retrieval system you are
already running:

```
Step 1: Retrieve top-k candidates using standard retrieval (BM25 or dense)
Step 2: Score each candidate for this specific user
Step 3: Combine retrieval score with personalization score
Step 4: Return reranked results
```

### Scoring function for personalized reranking

```
final_score(d, u, q) = α × relevance_score(d, q)
                     + β × user_affinity(d, u)
                     + γ × freshness_score(d)
                     + δ × authority_score(d)
```

Where:

- relevance_score: standard retrieval score (BM25, dense, cross-encoder)
- user_affinity: how much this user tends to like documents like d
- freshness_score: recency boost for time-sensitive content
- authority_score: document quality signal (PageRank, citation count)
- α, β, γ, δ: weights learned from user feedback data

### User affinity computation

```
user_affinity(d, u) = cosine_similarity(
    document_embedding(d),
    user_interest_embedding(u)
)
```

Where user_interest_embedding is computed from the user's click history:

```
user_interest_embedding = mean(document_embeddings of clicked documents)
```

Or weighted by recency:

```
user_interest_embedding = Σ recency_weight(t) × doc_embedding(clicked_doc_t)
```

## Filter Bubble Problem

Personalization creates a real risk of reinforcing existing preferences without
exposing users to new relevant content. A user who has only read BM25 papers
will never be shown neural IR papers if the system only surfaces content similar
to past behavior.

Mitigations:

### Diversity injection

Explicitly inject diverse results into personalized rankings:

```
Top-10 personalized results → replace positions 3, 6, 9 with
diverse recommendations (similar topic, different perspective)
```

### Exploration vs exploitation

Balance showing known-good content (exploitation) vs new content (exploration):

```
ε-greedy: with probability ε, show a random relevant document
          instead of the personalized top result

UCB (Upper Confidence Bound): boost documents with high uncertainty
in user preference - documents the user has not interacted with much
```

### Serendipity score

Add a serendipity component to the reranking formula that rewards surprising
but relevant documents:

```
serendipity(d, u) = relevance(d, q) × (1 - similarity(d, user_history))
```

High serendipity = highly relevant but different from what the user usually reads.

## Privacy Considerations

Personalization requires storing user data, which raises privacy concerns:

### On-device personalization

Store user profile on the user's device, not on a central server. Personalization
happens locally - no user data leaves the device. Used by Apple's on-device search.

### Federated learning

Train personalization models across many users without centralizing individual
data. Each device trains locally, only model updates (gradients) are shared:

```
Central server:   sends global model to devices
Each device:      trains on local user data → sends gradient update
Central server:   aggregates gradients → updates global model
No raw user data ever leaves the device
```

### Differential privacy

Add calibrated noise to user profiles before storing or aggregating to prevent
individual user data from being identifiable:

```
noisy_profile = true_profile + Laplace(0, sensitivity/ε)
```

Where ε controls the privacy-utility tradeoff.

### Data minimization

Only collect signals necessary for personalization. Dwell time is often more
privacy-preserving than storing the full document history - you know the user
spent 5 minutes on a topic without knowing exactly what they read.

## Personalization in Different IR Contexts

```
Context                  Signals used              Primary approach
────────────────────────────────────────────────────────────────────
Web search               Long-term query history   Query expansion +
                         click-through data        result reranking
E-commerce               Purchase history,         Collaborative filtering
                         wishlist, browsing        + content-based
Academic search          Paper reading history,    User-conditioned
                         citation patterns         retrieval model
Code search              Language preferences,     Query expansion
                         repository context        with project context
Conversational AI        Current session,          Short-term model
                         conversation history      + context injection
Enterprise search        Role, department,         Access-control aware
                         document access rights    retrieval + reranking
```

## Personalization vs Recommendation - The Distinction

A common point of confusion:

```
Personalization in IR:
  - User provides an explicit query
  - System returns relevant documents for that query
  - Personalization re-orders results based on user profile
  - Goal: same relevant documents, better ranked for this user

Recommendation:
  - No explicit query
  - System proactively suggests items
  - Based purely on user history and collaborative filtering
  - Goal: predict what the user wants without being asked
```

In practice, modern systems blend both. A search that returns product listings
both matches the query (IR) and considers purchase history (recommendation).
RAG systems that proactively surface related documents the user has not asked
for are performing recommendation. The boundary is blurry and increasingly
irrelevant in production systems.

## Where This Fits in the Progression

```
Standard retrieval   → same results for all users
Query understanding  → better understand what the user asked
LTR                  → learn to combine signals
Personalization      → tailor results to this specific user  ← you are here
Conversational IR    → maintain context across turns
RAG                  → generate grounded answers
```

Personalization sits at the intersection of retrieval and user modeling. It
takes the ranked list produced by all prior techniques and adjusts it based
on who is asking. Every technique covered before this - BM25, dense retrieval,
LTR, reranking - determines relevance in the abstract. Personalization
determines relevance for a specific user.

## My Summary

Personalization tailors retrieval rankings to individual users by incorporating
their interaction history, stated preferences, and inferred interests alongside
standard query-document relevance. The user model has two components: a long-term
profile capturing stable interests across sessions (updated with recency decay
from click history) and a short-term session context capturing the current task.
Personalized reranking combines standard retrieval scores with user affinity
scores - cosine similarity between the user interest vector and document
embeddings. The filter bubble risk is mitigated through MMR diversification, which
iteratively selects documents that are both relevant and dissimilar to already
selected ones. Privacy considerations include on-device personalization, federated
learning, and differential privacy. Personalization is most impactful for
ambiguous queries where the same query means different things to different
users - the classic example being "python" for a programmer versus a biologist.
