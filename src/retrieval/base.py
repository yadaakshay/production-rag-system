"""Abstract interface for information retrieval components."""

from abc import ABC, abstractmethod
from typing import List
from ..core.models import RetrievalResult


class BaseRetriever(ABC):
    """Abstract interface defining the retrieval contract."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Fetch top_k candidate chunks matching the query string."""
        pass
