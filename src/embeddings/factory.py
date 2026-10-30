"""Factory for instantiating text embedders based on configuration."""

from typing import Optional
from .base import BaseEmbedder
from .local_embedder import SentenceTransformerEmbedder
from .api_embedder import MockEmbedder, APIEmbedder


def get_embedder(
    provider: str = "sentence_transformers",
    model_name: str = "all-MiniLM-L6-v2",
    dimension: int = 384,
    device: str = "cpu",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseEmbedder:
    """Instantiate and return an embedder matching the requested provider."""
    provider = provider.lower()
    if provider in ("sentence_transformers", "hf", "huggingface", "local"):
        return SentenceTransformerEmbedder(
            model_name=model_name,
            device=device,
            **kwargs,
        )
    elif provider in ("mock", "test"):
        return MockEmbedder(dimension=dimension)
    elif provider in ("openai", "ollama"):
        return APIEmbedder(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            dimension=dimension,
        )
    else:
        # Fallback to local sentence-transformers
        return SentenceTransformerEmbedder(model_name=model_name, device=device)
