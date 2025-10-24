## WHY QUERY, KEY, AND VALUE?
- The terms "key," "query," and "value" in the context of attention mechanisms are
borrowed from the domain of information retrieval and databases, where similar
concepts are used to store, search, and retrieve information.
- A "query" is analogous to a search query in a database. It represents the current
item (e.g., a word or token in a sentence) the model focuses on or tries to
understand. The query is used to probe the other parts of the input sequence to
determine how much attention to pay to them.
- The "key" is like a database key used for indexing and searching. In the attention
mechanism, each item in the input sequence (e.g., each word in a sentence) has an
associated key. These keys are used to match with the query.
- The "value" in this context is similar to the value in a key-value pair in a database. It
represents the actual content or representation of the input items. Once the model
determines which keys (and thus which parts of the input) are most relevant to the
query (the current focus item), it retrieves the corresponding values.
