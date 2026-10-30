"""Sparse keyword lexical retriever based on BM25 Okapi."""

from typing import List
from .base import BaseRetriever
from ..core.models import RetrievalResult
from ..indexing.bm25_index import BM25Index


class SparseRetriever(BaseRetriever):
    """Retrieves chunks using BM25 inverted index term matching."""

    def __init__(self, bm25_index: BM25Index):
        self.bm25_index = bm25_index

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        raw_matches = self.bm25_index.search(query, top_k=top_k)

        results = []
        for rank, (chunk, score) in enumerate(raw_matches, start=1):
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    retriever_name="sparse_bm25",
                )
            )
        return results
