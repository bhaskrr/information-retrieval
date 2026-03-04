# What is Information Retrieval

Information Retrieval(IR) is the process of searching for, finding, and ranking **relevant** information (**documents, images, text**) from large, often unstructured, digital collections based on a user's query.

## The IR Pipeline

A standard IR system follows a specific flow to move from raw data to a ranked list of results:

1. **Collection**: Gathering the documents (web crawling, local files, etc.).

2. **Preprocessing**:
   1. Tokenization: Breaking text into individual words (tokens).
   2. Normalization: Lowercasing, removing punctuation.
   3. Stop-word Removal: Getting rid of "the," "is," "at" etc. to focus on meaningful terms.
   4. Stemming/Lemmatization: Reducing words to their root form.

3. **Indexing**: Creating an Inverted Index - a data structure that maps words to the documents they appear in.

4. **Querying**: The user provides a "query." The system processes this query just like it processed the documents.

5. **Ranking**: Calculating a score for each document and sorting them from most to least relevant.

6. **Evaluation**: Measuring how well the system did using metrics.

## Information Retrieval vs Data Retrieval

|Feature|Data Retrieval (Database)|Information Retrieval (Search)|
|-----|----|---|
|Data|Structured (Tables/Schema)|Unstructured (Text/Web)|
|Matching|Exact Match (SQL)|Partial Match / Similarity|
|Model|Deterministic|Probabilistic|
|Goal|Find all items matching logic|Rank items by relevance|
