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

|Feature|Information Retrieval (Search)|Data Retrieval (Database)|
|-----|----|---|
|Scope|The software program that deals with the organization, storage, retrieval and evaluation of information from document repositories particularly textual information.|Data retrieval deals with obtaining data from a database management system such as ODBMS. It is A process of identifying and retrieving the data from the database based on the query provided by user or application.|
|Retrieval|Retrieves information about a subject.|Determines the keywords in the user query and retrieves the data.|
|Error Tolerance|Small errors are likely to go unnoticed.|A single error object means total failure.|
|Data|Not always well structured and is semantically ambiguous.|Has a well-defined structure and semantics.|
|Matching|The results obtained are approximate matches.|The results obtained are exact matches.|
|Relevance|Results are ordered by relevance.|Results are unordered by relevance.|
|Model|It is a probabilistic model.|It is a deterministic model.|
