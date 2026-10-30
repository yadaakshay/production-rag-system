"""Dense semantic vector retriever."""

from typing import List
from .base import BaseRetriever
from ..core.models import RetrievalResult
from ..indexing.vector_store import VectorStore
from ..embeddings.base import BaseEmbedder


class DenseRetriever(BaseRetriever):
    """Retrieves chunks by mapping queries to latent vector space and measuring semantic similarity."""

    def __init__(self, vector_store: VectorStore, embedder: BaseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        query_vector = self.embedder.embed_query(query)
        raw_matches = self.vector_store.search(query_vector, top_k=top_k)

        results = []
        for rank, (chunk, score) in enumerate(raw_matches, start=1):
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    retriever_name="dense",
                )
            )
        return results
