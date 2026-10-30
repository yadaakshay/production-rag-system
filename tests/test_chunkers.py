"""Unit tests for document chunking algorithms."""

import pytest
from src.core.models import Document
from src.ingestion.chunkers import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    HierarchicalChunker,
    get_chunker,
)


def test_fixed_size_chunker():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    doc = Document(text=text, metadata={"source": "test.txt"})
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[0].doc_id == doc.doc_id
    assert chunks[0].metadata["strategy"] == "fixed_size"


def test_recursive_character_chunker_paragraphs():
    p1 = "First paragraph with introductory concepts."
    p2 = "Second paragraph explaining the core methodology."
    p3 = "Third paragraph summarizing the final conclusions."
    doc = Document(text=f"{p1}\n\n{p2}\n\n{p3}")

    chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=10)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3
    # Check paragraphs are preserved cleanly rather than split mid-word
    assert any("First paragraph" in c.text for c in chunks)
    assert any("Second paragraph" in c.text for c in chunks)


def test_hierarchical_chunker():
    doc = Document(text="Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. " * 10)
    chunker = HierarchicalChunker(parent_chunk_size=200, child_chunk_size=70)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    assert "parent_id" in chunks[0].metadata
    assert "parent_text" in chunks[0].metadata
    assert len(chunks[0].text) <= len(chunks[0].metadata["parent_text"])


def test_chunker_factory():
    c1 = get_chunker("recursive")
    assert isinstance(c1, RecursiveCharacterChunker)
    c2 = get_chunker("fixed")
    assert isinstance(c2, FixedSizeChunker)
    c3 = get_chunker("hierarchical")
    assert isinstance(c3, HierarchicalChunker)
