"""Remote API and deterministic mock embedder implementations."""

from typing import List, Optional
import hashlib
import numpy as np
from .base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Deterministic hash-based pseudo-embedder for zero-dependency unit tests and offline demos."""

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_to_vector(self, text: str) -> List[float]:
        # Generate pseudo-vector from text hash
        vec = []
        for i in range(self._dim):
            token = f"{text}:{i}".encode("utf-8")
            h = int(hashlib.md5(token).hexdigest()[:8], 16)
            vec.append((h / 0xFFFFFFFF) * 2.0 - 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = (np.array(vec) / norm).tolist()
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._hash_to_vector(query)


class APIEmbedder(BaseEmbedder):
    """Generic HTTP embedder supporting OpenAI or Ollama endpoints."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: int = 1536,
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or ("https://api.openai.com/v1" if provider == "openai" else "http://localhost:11434/api")
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        import httpx
        if self.provider == "openai":
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {"input": texts, "model": self.model_name}
            resp = httpx.post(f"{self.base_url}/embeddings", json=payload, headers=headers, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        elif self.provider == "ollama":
            embeddings = []
            for t in texts:
                resp = httpx.post(f"{self.base_url}/embeddings", json={"model": self.model_name, "prompt": t}, timeout=30.0)
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            return embeddings
        else:
            raise ValueError(f"Unsupported API embedder provider: {self.provider}")

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]
