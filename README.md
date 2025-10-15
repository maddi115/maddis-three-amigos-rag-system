<<<<<<< HEAD
# Probability-Stasis RAG

A RAG (Retrieval-Augmented Generation) system that uses probability-stasis filtering to ensure stable, reliable search results.

## What is Probability-Stasis?

Probability-stasis filters retrieval results by cross-checking multiple scoring methods (semantic similarity, keyword matching, length scoring), keeping only chunks where **all signals consistently agree** they're relevant. This reduces hallucination and improves answer quality.

**Formula:** `StasisScore = mean(p_i) / (1 + variance(p_i))`

- High mean + low variance = stable, reliable result
- Filters out chunks with contradictory signals

## Installation
```bash
pip install -e .


Quick Start


from probability_stasis_rag import ProbabilityStasisRAG

# Initialize
rag = ProbabilityStasisRAG(
    collection_name="my_docs",
    stasis_threshold=0.05,
    top_k=5
)

# Add documents
documents = [
    "Python is a programming language.",
    "Machine learning uses neural networks.",
    "RAG combines retrieval with generation.",
]
rag.add_documents(documents)

# Query with filtering
results = rag.query("What is Python?", use_cross_reference=True)

# Display results
rag.display_results(results, "What is Python?")


Command-Line Usage

python3 examples/cli_query.py "your question here"

Optional: Create an alias
Add to ~/.bashrc:

alias ask="python3 ~/probability-stasis-rag/examples/cli_query.py"

Then use: ask "your question"
Architecture

Vector Database: ChromaDB
Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)
No LLM required - Pure retrieval, zero API costs

How It Works

Retrieves top N candidates from vector database
Cross-references using: semantic similarity, keyword overlap, length scoring
Calculates StasisScore for each result
Filters out unstable results (high variance)
Returns only stable, cross-verified chunks

Use Cases

Documentation Search
Code RAG
Q&A Systems
Research Tools

Theory
Probability-stasis isolates situations where probability remains stable across multiple checks, filtering out everything that doesn't fit consistent probability patterns.
Author
agentmaddi
License
MIT
=======
# maddis-rag-system
>>>>>>> 2c935add9e37bf8a54c4b3fa6eed742fe270d1e3
