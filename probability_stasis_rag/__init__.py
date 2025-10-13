"""
Probability-Stasis RAG Library
Filters retrieval results by cross-checking multiple scoring methods,
keeping only chunks where all signals consistently agree they're relevant.
"""

from .filter import ProbabilityStasisFilter
from .rag import ProbabilityStasisRAG

__version__ = "0.1.0"
__author__ = "agentmaddi"

__all__ = [
    "ProbabilityStasisFilter",
    "ProbabilityStasisRAG",
]
