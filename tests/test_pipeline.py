"""Unit tests for the end-to-end RAG pipeline."""

import pytest
from src.core.models import Document
from src.core.config import RAGConfig
from src.embeddings.api_embedder import MockEmbedder
from src.generation.llm_client import MockLLMClient
from src.pipeline.rag_pipeline import RAGPipeline, PipelineMode


@pytest.fixture
def test_pipeline():
    config = RAGConfig()
    from src.retrieval.reranker import CrossEncoderReranker
    pipeline = RAGPipeline(
        config=config,
        embedder=MockEmbedder(dimension=64),
        llm_client=MockLLMClient(),
        reranker=CrossEncoderReranker(model_name="heuristic"),
    )
    docs = [
        Document(
            text="Apache Kafka uses partitioned topics with replication factor 3 for durability.",
            metadata={"filename": "kafka_guide.md"},
        ),
        Document(
            text="The circuit breaker monitors failure rate and trips to OPEN state after 50% failures.",
            metadata={"filename": "resilience.md"},
        ),
    ]
    pipeline.ingest_documents(docs)
    return pipeline


def test_end_to_end_pipeline_query(test_pipeline):
    response = test_pipeline.run(
        query="What is the Kafka replication factor?",
        mode=PipelineMode.PRODUCTION,
        top_k=2,
        top_n=2,
    )

    assert response.query == "What is the Kafka replication factor?"
    assert response.answer is not None
    assert len(response.retrieved_chunks) > 0
    assert "total_ms" in response.latency_breakdown
    assert response.faithfulness_score is not None
    assert 0.0 <= response.faithfulness_score <= 1.0


def test_conversational_routing(test_pipeline):
    response = test_pipeline.run(
        query="Hello, good morning!",
        mode=PipelineMode.PRODUCTION,
    )

    assert "assist you" in response.answer.lower()
    assert len(response.retrieved_chunks) == 0  # Retrieval bypassed
