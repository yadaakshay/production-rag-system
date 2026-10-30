"""Unit tests for Dense, Sparse (BM25), and Hybrid RRF retrievers."""

import pytest
from src.core.models import Document, Chunk
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import InMemoryVectorStore
from src.embeddings.api_embedder import MockEmbedder
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_chunks():
    chunks = [
        Chunk(doc_id="d1", text="Kafka distributed streaming partitioned topics replication factor", chunk_index=0),
        Chunk(doc_id="d1", text="Circuit breaker resilience trips into OPEN state when failures occur", chunk_index=1),
        Chunk(doc_id="d2", text="Financial earnings consolidated revenue and Adjusted EBITDA margins", chunk_index=0),
    ]
    return chunks


def test_bm25_index(sample_chunks):
    index = BM25Index()
    index.add_chunks(sample_chunks)

    results = index.search("Kafka replication", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "Kafka" in top_chunk.text
    assert score > 0


def test_vector_store(sample_chunks):
    embedder = MockEmbedder(dimension=64)
    store = InMemoryVectorStore(metric="cosine")

    texts = [c.text for c in sample_chunks]
    embs = embedder.embed_texts(texts)
    store.add_chunks(sample_chunks, embs)

    q_vec = embedder.embed_query("EBITDA financial earnings")
    results = store.search(q_vec, top_k=2)

    assert len(results) == 2
    top_chunk, score = results[0]
    assert 0.0 <= score <= 1.0


def test_hybrid_rrf_retrieval(sample_chunks):
    embedder = MockEmbedder(dimension=64)
    store = InMemoryVectorStore(metric="cosine")
    embs = embedder.embed_texts([c.text for c in sample_chunks])
    store.add_chunks(sample_chunks, embs)

    bm25 = BM25Index()
    bm25.add_chunks(sample_chunks)

    dense = DenseRetriever(store, embedder)
    sparse = SparseRetriever(bm25)
    hybrid = HybridRetriever(dense, sparse, mode="rrf", rrf_k=60)

    results = hybrid.retrieve("Kafka streaming", top_k=2)
    assert len(results) <= 2
    assert results[0].retriever_name == "hybrid_rrf"
    assert results[0].score > 0
