#!/usr/bin/env python3
"""
Command-line query tool for Probability-Stasis RAG

Usage:
    python3 cli_query.py "your question here"

Example:
    python3 cli_query.py "What is Python?"
"""
import sys
from probability_stasis_rag import ProbabilityStasisRAG

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cli_query.py \"your question\"")
        print("Example: python3 cli_query.py \"What is Python?\"")
        sys.exit(1)

    # Get query from command line
    query = " ".join(sys.argv[1:])

    # Initialize RAG (users should adjust these to their database)
    rag = ProbabilityStasisRAG(
        collection_name="my_docs",
        persist_directory="./my_chroma_db",
        stasis_threshold=0.05,
        top_k=3
    )

    print(f"\nQuerying: '{query}'\n")

    # Query with probability-stasis filtering
    results = rag.query(query, use_cross_reference=True)

    # Display results
    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. [StasisScore: {result['stasis_score']:.3f}]")
            print(f"   {result['document'][:200]}...")
            print()
    else:
        print("No stable results found.")

if __name__ == "__main__":
    main()
