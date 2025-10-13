# Probability-Stasis RAG

A RAG (Retrieval-Augmented Generation) system that uses probability-stasis filtering to ensure stable, reliable search results.

## What is Probability-Stasis?

Probability-stasis filters retrieval results by cross-checking multiple scoring methods (semantic similarity, keyword matching, length scoring), keeping only chunks where **all signals consistently agree** they're relevant. This reduces hallucination and improves answer quality.

**Formula:** `StasisScore = mean(p_i) / (1 + variance(p_i))`

- High mean + low variance = stable, reliable result
- Filters out chunks with contradictory signals

## Architecture

- **Vector Database**: ChromaDB (persistent storage with HNSW indexing)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2` by default)
- **Similarity Metric**: Cosine similarity
- **Cross-Reference Scoring**:
  - Semantic similarity (via embeddings)
  - Keyword overlap (lexical matching)
  - Length-based relevance
- **Filtering**: Probability-stasis score calculation with variance threshold

### Technical Stack
```python
chromadb>=1.1.1           # Vector database
sentence-transformers>=5.1.1  # Embedding models
numpy>=2.0.0              # Numerical computations
## Installation
```bash
pip install -e .
Quick Start
pythonfrom probability_stasis_rag import ProbabilityStasisRAG

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
How It Works

Retrieves top N candidates from vector database
Cross-references using multiple signals:

Semantic similarity (embeddings)
Keyword overlap
Length-based relevance


Calculates StasisScore for each result
Filters out unstable results (high variance)
Returns only stable, cross-verified chunks

Use Cases

Documentation Search - Find reliable technical answers
Code RAG - Retrieve relevant code snippets
Q&A Systems - Reduce hallucination in responses
Research Tools - Cross-verify information sources

Parameters

stasis_threshold: Minimum StasisScore (0.05-0.15 recommended)
top_k: Maximum results to return (3-5 recommended)
use_cross_reference: Enable multi-signal verification (True recommended)

Theory
Probability-stasis applies a method to isolate situations where the probability of an outcome remains stable across multiple checks, filtering out everything that doesn't fit consistent probability patterns. It cross-references left, right, top, bottom—using all available signals to find the highest quality results.
Author
Created by agentmaddi
License
MIT
