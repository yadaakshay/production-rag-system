"""Abstract base interface for text embedding providers."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Abstract interface for dense embedding generation."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vectors for a list of string passages."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate a dense vector for a search query string."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return dimensionality of the embedding vector space."""
        pass
