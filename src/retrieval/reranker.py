"""Cross-Encoder Reranker for high-precision two-stage retrieval.

Unlike Bi-Encoders that embed queries and passages independently, Cross-Encoders pass
both (query, passage) simultaneously through full cross-attention layers, computing
token-level interactions that drastically reduce False Positives.
"""

from typing import List, Optional
import numpy as np
from ..core.models import RetrievalResult


class CrossEncoderReranker:
    """Reranks candidate chunks using a trained Cross-Encoder classification model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            if self.model_name == "heuristic":
                self._model = "heuristic"
                return
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                # Log and fallback to heuristic
                print(f"Warning: Could not load CrossEncoder model ({e}). Using lexical-overlap heuristic.")
                self._model = "heuristic"

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: int = 4,
    ) -> List[RetrievalResult]:
        """Score candidate chunks against query and return top_n reranked results."""
        if not candidates:
            return []

        self._load_model()

        if self._model != "heuristic":
            pairs = [[query, res.chunk.text] for res in candidates]
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            # Sigmoid transformation if raw logits
            scores = [1.0 / (1.0 + np.exp(-s)) if abs(s) > 1.0 else float(s) for s in scores]
        else:
            # Heuristic token overlap scoring fallback
            q_terms = set(query.lower().split())
            scores = []
            for res in candidates:
                c_terms = set(res.chunk.text.lower().split())
                overlap = len(q_terms.intersection(c_terms)) / (len(q_terms) + 1e-6)
                scores.append(overlap)

        # Pair scores with candidate results and sort
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        reranked_results: List[RetrievalResult] = []
        for new_rank, (res, score) in enumerate(scored_candidates[:top_n], start=1):
            reranked_results.append(
                RetrievalResult(
                    chunk=res.chunk,
                    score=float(score),
                    rank=new_rank,
                    retriever_name="cross_encoder_reranker",
                )
            )
        return reranked_results
