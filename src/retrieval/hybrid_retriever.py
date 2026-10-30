"""Hybrid retriever combining dense semantic search and sparse lexical search.

Algorithms implemented:
1. Reciprocal Rank Fusion (RRF):
   RRF_Score(d) = sum_{m in models} [ 1 / (k + rank_m(d)) ]
   Robust against scale discrepancies between BM25 scores (unbounded) and cosine similarities ([0, 1]).

2. Convex Linear Combination:
   Score(d) = alpha * norm(Dense_Score(d)) + (1 - alpha) * norm(Sparse_Score(d))
"""

from typing import List, Dict, Optional
from .base import BaseRetriever
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from ..core.models import RetrievalResult, Chunk


class HybridRetriever(BaseRetriever):
    """Fuses results from dense and sparse retrievers using RRF or linear combination."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        mode: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.mode = mode.lower()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Combine ranks using standard RRF formula."""
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        # Accumulate dense ranks
        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Accumulate sparse ranks
        for rank, res in enumerate(sparse_results, start=1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort descending by fused score
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        final_results = []
        for new_rank, cid in enumerate(sorted_cids[:top_k], start=1):
            final_results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=fused_scores[cid],
                    rank=new_rank,
                    retriever_name="hybrid_rrf",
                )
            )
        return final_results

    def _linear_combination(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Combine min-max normalized scores using weighted average."""
        chunk_map: Dict[str, Chunk] = {}
        dense_scores_map: Dict[str, float] = {}
        sparse_scores_map: Dict[str, float] = {}

        # Extract and normalize dense scores
        d_scores = [r.score for r in dense_results] or [0.0]
        min_d, max_d = min(d_scores), max(d_scores)
        d_range = (max_d - min_d) if (max_d - min_d) > 0 else 1.0

        for r in dense_results:
            cid = r.chunk.chunk_id
            chunk_map[cid] = r.chunk
            dense_scores_map[cid] = (r.score - min_d) / d_range

        # Extract and normalize sparse scores
        s_scores = [r.score for r in sparse_results] or [0.0]
        min_s, max_s = min(s_scores), max(s_scores)
        s_range = (max_s - min_s) if (max_s - min_s) > 0 else 1.0

        for r in sparse_results:
            cid = r.chunk.chunk_id
            chunk_map[cid] = r.chunk
            sparse_scores_map[cid] = (r.score - min_s) / s_range

        # Compute weighted sum
        all_cids = set(dense_scores_map.keys()) | set(sparse_scores_map.keys())
        combined_scores: Dict[str, float] = {}

        for cid in all_cids:
            d_norm = dense_scores_map.get(cid, 0.0)
            s_norm = sparse_scores_map.get(cid, 0.0)
            combined_scores[cid] = (self.dense_weight * d_norm) + (self.sparse_weight * s_norm)

        sorted_cids = sorted(all_cids, key=lambda cid: combined_scores[cid], reverse=True)

        final_results = []
        for new_rank, cid in enumerate(sorted_cids[:top_k], start=1):
            final_results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=combined_scores[cid],
                    rank=new_rank,
                    retriever_name="hybrid_linear",
                )
            )
        return final_results

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        # Overfetch candidates from both sub-retrievers to maximize fusion coverage
        fetch_k = max(top_k * 2, 20)
        dense_results = self.dense_retriever.retrieve(query, top_k=fetch_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=fetch_k)

        if self.mode == "linear":
            return self._linear_combination(dense_results, sparse_results, top_k=top_k)
        return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)
