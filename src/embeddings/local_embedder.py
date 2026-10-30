"""Local HuggingFace SentenceTransformer embedder implementation."""

from typing import List, Optional
import numpy as np
from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Generates dense embeddings locally using HuggingFace SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
        self._dim: Optional[int] = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dim = self._model.get_embedding_dimension()
                elif hasattr(self._model, "get_sentence_embedding_dimension"):
                    self._dim = self._model.get_sentence_embedding_dimension()
                else:
                    self._dim = 384
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SentenceTransformer model '{self.model_name}': {e}"
                )

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load_model()
        return self._dim or 384

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        self._load_model()
        embedding = self._model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding.tolist()
