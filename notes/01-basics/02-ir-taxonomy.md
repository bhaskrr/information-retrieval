# Taxonomy of Information Retrieval

## Components of Information Retrieval/ IR Model

The Information Retrieval (IR) model can be broken down into key components that involve both the system and the user.

### User Side (Search Process)

1. **Problem Identification**: A student wants to learn about machine learning and types a query into a search engine.

2. **Representation**: The user converts their need into a search query using keywords or phrases like instead of asking "How do machines learn?" the student types "machine learning basics" into Google and the problem is converted into a query (keywords or phrases).
3. **Query**: The user submits the search query into IR system.
4. **Feedback**: User can refine or modify the search based on the retrieved results.

### System Side (Retrieval Process)

1. **Acquisition**: The system collects and stores a large number of documents or data sources. It can includes web pages, books, research papers or any text-based information.

2. **Representation**: Each document in the system is analyzed and represented in a structured way using keywords (terms). Example: If the document talks about "machine learning" it is tagged with relevant terms like "AI, deep learning, algorithms, models" to help retrieval.
3. **File Organization**: The documents are indexed and stored efficiently so the system can quickly find relevant ones. Like organizing a library so books can be found easily based on topics.
4. Matching: The system compares the user's search query with stored documents to find the best matches. It uses matching functions that rank documents based on relevance.
5. Retrieved Object: The system returns the most relevant documents to the user. These documents are ranked so the most useful ones appear at the top.

### Interaction Between User & System

The user reviews the retrieved results and may provide feedback to refine the search. The system then processes the updated query and retrieves better results.

## Classification of IR Models

Information Retrieval models can be classified into the following categories:

1. **Set-Theoritic Models**
   1. Fuzzy Set model
   2. Boolean Model
   3. Extended-Boolean Model
2. **Algebric Models**
   1. Vector Space Model
   2. Generalized Vector Space Model
   3. Latent Sematic Analysis(LSA)
   4. Support Vector Machines
   5. Neural Networks
3. **Probabilistic Models**
   1. Best Match Models
   2. Language Models
   3. Bayesian Networks
