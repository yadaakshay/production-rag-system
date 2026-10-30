"""Unit tests for the Cross-Encoder reranker."""

import pytest
from src.core.models import Chunk, RetrievalResult
from src.retrieval.reranker import CrossEncoderReranker


def test_cross_encoder_reranking_heuristic_and_order():
    reranker = CrossEncoderReranker(model_name="heuristic")

    c1 = Chunk(doc_id="d1", text="Weather forecast sunny clear skies 75 degrees")
    c2 = Chunk(doc_id="d2", text="Circuit breaker resilience trips into OPEN state")
    c3 = Chunk(doc_id="d3", text="Kafka event streaming partitioned topic replication")

    candidates = [
        RetrievalResult(chunk=c1, score=0.9, rank=1, retriever_name="mock"),
        RetrievalResult(chunk=c2, score=0.8, rank=2, retriever_name="mock"),
        RetrievalResult(chunk=c3, score=0.7, rank=3, retriever_name="mock"),
    ]

    # Query clearly matching candidate 2
    query = "When does the circuit breaker trip into OPEN state?"
    reranked = reranker.rerank(query, candidates, top_n=2)

    assert len(reranked) == 2
    # Candidate 2 should be boosted to rank 1
    assert "circuit breaker" in reranked[0].chunk.text.lower()
    assert reranked[0].rank == 1
