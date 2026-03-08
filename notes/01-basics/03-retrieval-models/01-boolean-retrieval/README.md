# Boolean Retrieval Model

The Boolean Retrieval Model is a classical information retrieval framework based on Set Theory and Boolean Algebra. It treats documents as sets of terms and processes queries as logical expressions, returning a binary result: a document either matches the query (relevant) or it does not (not relevant).

## Core Concept

In Boolean retrieval, a document is either relevant (1) or non-relevant (0) based on the presence of terms. There is no concept of "partial matches" or "ranking".

### The Logic

- AND: Documents must contain both terms. (Set Intersection)
- OR: Documents can contain either term. (Set Union)
- NOT: Documents must not contain the specified term. (Set Complement)

## Components

1. **Term-Document Matrix**  
A conceptual binary matrix where rows represent Terms and columns represent Documents.
   - 1 = Term is present in the document.
   - 0 = Term is absent.

    |Term|Doc 1|Doc 2|Doc 3|
    |---|---|---|---|
    |Apple|1|0|1|
    |Banana|0|1|1|

2. **Inverted Index** (The Industry Standard)  
Since matrices are too large and sparse for real-world data, we use an Inverted Index. It maps each term to **a list of document IDs** (**Postings List**) where it appears.

   - Apple: [1, 3]
   - Banana: [2, 3]

## The Retrieval Process

1. **Tokenization**: Break document text into individual words (tokens).
2. **Normalization**: Standardise words (e.g., lowercasing, stemming, lemmatization etc.).
3. **Indexing**: Create the Inverted Index.
4. **Query Processing**:
    - Retrieve the postings lists for each term in the query.
    - Apply logical operations (Intersections or Unions).
    - Example: `Apple AND Banana` results in the intersection of [1, 3] and [2, 3], which is [3].

## Pros and Cons

|Pros|Cons|
|----|----|
|**Simple & Efficient**: Easy to implement and fast for straightforward queries.|**Binary Results**: No partial matching; a document is either 100% relevant or not at all.|
|**User Contro**l: Expert users can precisely define their search criteria.|**No Ranking**: Results are not ordered by relevance, which can lead to "information overload" or zero results.|
|**Predictable**: The logic is clear, making it obvious why a document was retrieved.|**Complex Querying**: Requires users to understand Boolean logic, which can be unintuitive.|
