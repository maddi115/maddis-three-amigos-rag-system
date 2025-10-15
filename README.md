# The Three Amigos RAG System

A multi-method RAG (Retrieval-Augmented Generation) system featuring three complementary search algorithms working together for optimal retrieval.

## The Three Amigos 🔥🎯🔬

### 1. 🔥 Gradient Proximity Search (Chaining Algorithm)
Novel algorithm that finds rare anchor words and radiates outward bidirectionally, with exponential strength decay and boosting when finding related terms.

**How it works:**
- Identifies rarest query word as anchor
- Radiates left/right with decaying strength
- **BOOSTS strength** when finding other query words (chaining effect)
- Scores by physical proximity of terms in text
- Perfect for finding clustered concepts

**Best for:** Finding specific term combinations, clustered discussions

### 2. 🎯 Vector Search (Semantic Similarity)
Pure semantic search using sentence transformers for meaning-based retrieval.

**How it works:**
- Encodes query into embedding vector
- Finds semantically similar documents
- Cosine similarity scoring
- Understanding intent, not just keywords

**Best for:** Understanding meaning, sentiment, conceptual queries

### 3. 🔬 Stasis (Probability Filtering)
Cross-references multiple scoring methods to find stable, reliable results.

**How it works:**
- Retrieves candidates from vector database
- Cross-references: semantic similarity, keyword overlap, length scoring
- Calculates StasisScore = `mean(p_i) / (1 + variance(p_i))`
- Filters unstable results (high variance)

**Best for:** High-confidence results, reducing hallucination

## Installation
```bash
pip install -e .
```

## Quick Start

### Compare All Three Methods
```python
from probability_stasis_rag import ProbabilityStasisRAG, VectorSearch, GradientProximitySearch

# Initialize
rag = ProbabilityStasisRAG(
    collection_name="my_docs",
    persist_directory="./chroma_db",
    stasis_threshold=0.05,
    top_k=3
)

# Get documents
collection = rag.collection
all_docs = collection.get()
documents = all_docs['documents']

# Method 1: Gradient Proximity (Chaining)
gradient = GradientProximitySearch(documents)
gradient_results = gradient.search("your query", top_k=3)

# Method 2: Vector Search (Semantic)
vector = VectorSearch(rag.collection, rag.embedder)
vector_results = vector.search("your query", top_k=3)

# Method 3: Stasis (Probability Filtering)
stasis_results = rag.query("your query", use_cross_reference=True)
```

### Command-Line Comparison Tool
```bash
python3 query_all.py "What food does agentmaddi like?"
```

**Output includes:**
- 🔥 Gradient results with confidence tiers (high/medium/low)
- 🎯 Vector semantic matches
- 🔬 Stasis filtered results
- Beautiful formatted output with threading visualization

## Architecture

**Vector Database:** ChromaDB  
**Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)  
**No LLM required** - Pure retrieval, zero API costs

## Features

✅ **Three complementary search methods** for different use cases  
✅ **Beautiful formatted output** with confidence scoring  
✅ **Threading visualization** (→→) for conversation flow  
✅ **Anchor word highlighting** (>>>word<<<)  
✅ **Rarity-based boosting** in Gradient search  
✅ **Cross-reference validation** in Stasis  
✅ **No API costs** - runs entirely locally

## Use Cases

- 📚 Documentation Search
- 💬 Conversational Data Analysis
- 🔍 Code RAG
- ❓ Q&A Systems
- 🔬 Research Tools

## Advanced: Gradient Proximity Deep Dive

For detailed analysis with all results:
```bash
python3 newv2Gradient_Proximity_Search_Rarity_Based_Chaining.py "your query"
```

**Features:**
- Narrative summary of findings
- Confidence distribution (high/medium/low)
- Complete result set with threading
- Score explanations

## Theory

**Gradient Proximity:** Chains discoveries through text using exponential decay and rarity-based boosting, naturally prioritizing documents where query terms cluster together.

cool but this is unreadable and it also leaks into other unrelated conversations we should a stop word if it encounters it going from left to right this would be agentmaddi then you get me what im saying dont update anything yet----add threading too because which adds that is right after you can tell because of the timestamp in the csv this unentintional result describe my idea perfectly its just that it wasnt intended and it looks messy : 1 │ 1.147 │ ...of course with rice agentmaddi when my family folks make carne asada and pollo asada we like giving >>>food<<< the homeless people passing by our house agentmaddi i usually give food to homeless people only agentmaddi i might strreamsnipe and give u one when i get my third job so represent the treads in a clean way 1 │ 1.147 │ [context] of course with rice │ → when my family folks make carne asada and pollo asada we like giving >>>food<<< the homeless people passing by our house │ → i usually give food to homeless people only │ → i might strreamsnipe and give u one when i get my third job yup perfect also do this : cool but this is unreadable and it also leaks into other unrelated conversations we should a stop word if it encounters it going from left to right this would be agentmaddi then, articulate what i want updated before you implement and return the entire codebase back with its updates --- Anchors on the rarest query term, propagates outward with gradient decay, chains through related words with dynamic rarity-based boosts, respects conversation boundaries, and reveals complete threaded context—achieving rich, explainable search without vectors.

**Vector Search:** Pure semantic similarity using learned embeddings to understand meaning beyond keywords.

**Probability Stasis:** Isolates situations where probability remains stable across multiple scoring methods, filtering contradictory signals.

## Author

agentmaddi

## License

MIT
