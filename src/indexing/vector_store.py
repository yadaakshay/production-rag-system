"""High-performance in-memory Vector Store with disk persistence and metric options."""

from typing import List, Tuple, Optional
import json
from pathlib import Path
import numpy as np
from ..core.models import Chunk


class InMemoryVectorStore:
    """Vector database implementation using normalized matrix multiplication.
    
    Supports cosine similarity, Euclidean (L2) distance, and inner dot product.
    Zero external C++ dependencies required, ensuring seamless deployment.
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric.lower()
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, D)

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks alongside their pre-computed dense embedding vectors."""
        if not chunks or not embeddings:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count and embeddings count must match.")

        new_mat = np.array(embeddings, dtype=np.float32)

        # Normalize if cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(new_mat, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            new_mat = new_mat / norms

        if self.embeddings is None:
            self.embeddings = new_mat
        else:
            self.embeddings = np.vstack([self.embeddings, new_mat])

        for c, emb in zip(chunks, embeddings):
            c.embedding = emb
            self.chunks.append(c)

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k nearest chunks sorted by similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)

        if self.metric == "cosine":
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
            # Dot product of normalized vectors equals cosine similarity
            scores = np.dot(self.embeddings, q_vec)
        elif self.metric == "l2":
            diff = self.embeddings - q_vec
            dist = np.linalg.norm(diff, axis=1)
            # Invert distance to score: higher is closer
            scores = 1.0 / (1.0 + dist)
        else:  # dot product
            scores = np.dot(self.embeddings, q_vec)

        # Retrieve top indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    def save(self, directory: str) -> None:
        """Persist embeddings matrix and chunk metadata to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(str(path / "vector_embeddings.npy"), self.embeddings)

        # Store chunks without duplicate embedding inside JSON
        chunks_data = []
        for c in self.chunks:
            dump = c.model_dump()
            dump.pop("embedding", None)
            chunks_data.append(dump)

        with open(path / "vector_chunks.json", "w", encoding="utf-8") as f:
            json.dump({"metric": self.metric, "chunks": chunks_data}, f)

    @classmethod
    def load(cls, directory: str) -> "InMemoryVectorStore":
        """Load vector store from disk."""
        path = Path(directory)
        json_file = path / "vector_chunks.json"
        npy_file = path / "vector_embeddings.npy"

        if not json_file.exists() or not npy_file.exists():
            return cls()

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls(metric=data.get("metric", "cosine"))
        store.embeddings = np.load(str(npy_file))
        store.chunks = [Chunk(**c) for c in data["chunks"]]
        return store


# Alias for default vector store
VectorStore = InMemoryVectorStore
