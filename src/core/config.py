"""Centralized configuration loader and settings management."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class SystemConfig(BaseModel):
    project_name: str = "production-rag-system"
    version: str = "1.0.0"
    random_seed: int = 42
    log_level: str = "INFO"


class StorageConfig(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    index_dir: str = "data/indices"
    eval_dir: str = "data/evaluation"


class IngestionConfig(BaseModel):
    chunking_strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", "? ", "! ", " ", ""])
    preserve_headers: bool = True


class EmbeddingConfig(BaseModel):
    provider: str = "sentence_transformers"
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"


class BM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75


class IndexingConfig(BaseModel):
    vector_backend: str = "in_memory"
    similarity_metric: str = "cosine"
    bm25: BM25Config = Field(default_factory=BM25Config)


class RetrievalConfig(BaseModel):
    top_k_retrieval: int = 10
    hybrid_mode: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 4


class QueryTransformConfig(BaseModel):
    enable_hyde: bool = False
    enable_multi_query: bool = False
    enable_router: bool = True


class GenerationConfig(BaseModel):
    provider: str = "mock"
    model_name: str = "mock-model"
    temperature: float = 0.1
    max_tokens: int = 1024
    query_transformation: QueryTransformConfig = Field(default_factory=QueryTransformConfig)


class EvaluationConfig(BaseModel):
    sample_size: int = 20
    metrics: List[str] = Field(default_factory=lambda: [
        "faithfulness", "answer_relevance", "context_precision", "context_recall", "hit_rate_at_k", "mrr"
    ])


class RAGConfig(BaseModel):
    """Root configuration object containing all sub-configs."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # API keys loaded from environment variables
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @classmethod
    def load_from_yaml(cls, path: str = "configs/config.yaml") -> "RAGConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# Global default configuration instance
app_config = RAGConfig.load_from_yaml()
