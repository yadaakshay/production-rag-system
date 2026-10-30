"""Embeddings module: Provider-agnostic text vectorization."""

from .base import BaseEmbedder
from .local_embedder import SentenceTransformerEmbedder
from .api_embedder import MockEmbedder, APIEmbedder
from .factory import get_embedder

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "MockEmbedder",
    "APIEmbedder",
    "get_embedder",
]
