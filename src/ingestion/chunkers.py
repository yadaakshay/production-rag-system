"""Advanced chunking strategies for document processing.

Includes:
1. Fixed-size chunking (naive baseline)
2. Recursive character chunking (production standard)
3. Semantic chunking (embedding-distance boundary detection)
4. Hierarchical / Parent-Child chunking (fine-grained retrieval with broad context)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import numpy as np
from ..core.models import Document, Chunk


class BaseChunker(ABC):
    """Abstract base class for chunking implementations."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a Document into an ordered list of Chunks."""
        pass

    def estimate_tokens(self, text: str) -> int:
        """Heuristic estimation of tokens (~4 chars per token in English)."""
        return max(1, len(text) // 4)


class FixedSizeChunker(BaseChunker):
    """Naive fixed-character chunker with sliding window overlap.
    
    Serves as an instructional baseline to demonstrate context truncation issues.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.text
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        index = 0

        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = Chunk(
                    doc_id=document.doc_id,
                    text=chunk_text,
                    chunk_index=index,
                    token_count=self.estimate_tokens(chunk_text),
                    metadata={
                        **document.metadata,
                        "strategy": "fixed_size",
                        "char_start": start,
                        "char_end": end,
                    },
                )
                chunks.append(chunk)
                index += 1
            if end >= len(text):
                break

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """Production standard: recursively splits on paragraph, sentence, and word boundaries.
    
    Tries each separator in order of semantic preservation (double newline -> newline -> sentence -> word).
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return list(text)

        splits = text.split(separator)
        final_splits = []

        for piece in splits:
            if len(piece) <= self.chunk_size:
                final_splits.append(piece)
            else:
                if remaining_separators:
                    sub_splits = self._split_text(piece, remaining_separators)
                    final_splits.extend(sub_splits)
                else:
                    final_splits.append(piece)

        return final_splits

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        merged_chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            if not split.strip():
                continue

            piece_len = len(split) + (len(separator) if current_chunk else 0)

            if current_length + piece_len <= self.chunk_size:
                current_chunk.append(split)
                current_length += piece_len
            else:
                if current_chunk:
                    doc = separator.join(current_chunk).strip()
                    if doc:
                        merged_chunks.append(doc)

                    # Build overlap window from tail
                    overlap_chunk: List[str] = []
                    overlap_len = 0
                    for rev_piece in reversed(current_chunk):
                        rev_len = len(rev_piece) + (len(separator) if overlap_chunk else 0)
                        if overlap_len + rev_len <= self.chunk_overlap:
                            overlap_chunk.insert(0, rev_piece)
                            overlap_len += rev_len
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_len

                current_chunk.append(split)
                current_length += len(split)

        if current_chunk:
            doc = separator.join(current_chunk).strip()
            if doc:
                merged_chunks.append(doc)

        return merged_chunks

    def chunk(self, document: Document) -> List[Chunk]:
        raw_splits = self._split_text(document.text, self.separators)
        merged_texts = self._merge_splits(raw_splits, separator=" ")

        chunks: List[Chunk] = []
        for i, text in enumerate(merged_texts):
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    text=text,
                    chunk_index=i,
                    token_count=self.estimate_tokens(text),
                    metadata={
                        **document.metadata,
                        "strategy": "recursive_character",
                        "char_count": len(text),
                    },
                )
            )
        return chunks


class SemanticChunker(BaseChunker):
    """Splits text dynamically based on embedding distance between consecutive sentences.
    
    Detects thematic shifts rather than relying on arbitrary character counts.
    """

    def __init__(
        self,
        embedder: Any = None,
        similarity_threshold_percentile: float = 25.0,
        max_chunk_size: int = 800,
    ):
        self.embedder = embedder
        self.percentile = similarity_threshold_percentile
        self.max_chunk_size = max_chunk_size

    def _split_into_sentences(self, text: str) -> List[str]:
        # Regex-based sentence boundary detection
        sentences = re.split(r"(?<=[.?!])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, document: Document) -> List[Chunk]:
        sentences = self._split_into_sentences(document.text)
        if len(sentences) <= 1:
            return [
                Chunk(
                    doc_id=document.doc_id,
                    text=document.text.strip(),
                    chunk_index=0,
                    token_count=self.estimate_tokens(document.text),
                    metadata={**document.metadata, "strategy": "semantic"},
                )
            ]

        # If embedder is provided, compute semantic distances
        if self.embedder is not None:
            try:
                embeddings = self.embedder.embed_texts(sentences)
                # Compute cosine similarities between consecutive sentences
                similarities = []
                for i in range(len(embeddings) - 1):
                    v1, v2 = np.array(embeddings[i]), np.array(embeddings[i + 1])
                    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    sim = np.dot(v1, v2) / (norm1 * norm2 + 1e-9)
                    similarities.append(float(sim))

                threshold = float(np.percentile(similarities, self.percentile))
            except Exception:
                threshold = 0.5
                similarities = [1.0] * (len(sentences) - 1)
        else:
            # Fallback heuristic: sentence punctuation boundary
            threshold = 0.5
            similarities = [0.8] * (len(sentences) - 1)

        # Segment sentences where similarity drops below threshold
        chunks_text: List[str] = []
        current_cluster: List[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            next_sent = sentences[i + 1]
            curr_len = sum(len(s) for s in current_cluster)

            if sim < threshold or (curr_len + len(next_sent) > self.max_chunk_size):
                chunks_text.append(" ".join(current_cluster))
                current_cluster = [next_sent]
            else:
                current_cluster.append(next_sent)

        if current_cluster:
            chunks_text.append(" ".join(current_cluster))

        chunks: List[Chunk] = []
        for i, text in enumerate(chunks_text):
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    text=text,
                    chunk_index=i,
                    token_count=self.estimate_tokens(text),
                    metadata={
                        **document.metadata,
                        "strategy": "semantic",
                        "sentence_count": len(self._split_into_sentences(text)),
                    },
                )
            )
        return chunks


class HierarchicalChunker(BaseChunker):
    """Parent-Child chunking:
    Produces small child chunks for fine-grained retrieval, while keeping parent chunk ID
    to supply rich surrounding context to the generation phase.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1200,
        parent_overlap: int = 150,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ):
        self.parent_chunker = RecursiveCharacterChunker(
            chunk_size=parent_chunk_size, chunk_overlap=parent_overlap
        )
        self.child_chunker = RecursiveCharacterChunker(
            chunk_size=child_chunk_size, chunk_overlap=child_overlap
        )

    def chunk(self, document: Document) -> List[Chunk]:
        parents = self.parent_chunker.chunk(document)
        all_children: List[Chunk] = []
        child_counter = 0

        for parent in parents:
            parent_doc = Document(
                text=parent.text,
                metadata={
                    **document.metadata,
                    "parent_chunk_id": parent.chunk_id,
                },
            )
            children = self.child_chunker.chunk(parent_doc)
            for child in children:
                child.chunk_index = child_counter
                child.metadata["parent_id"] = parent.chunk_id
                child.metadata["parent_text"] = parent.text
                child.metadata["strategy"] = "hierarchical_child"
                all_children.append(child)
                child_counter += 1

        return all_children


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """Factory function for instantiating chunking algorithms."""
    strategy = strategy.lower()
    if strategy in ("fixed", "fixed_size"):
        return FixedSizeChunker(**kwargs)
    elif strategy in ("semantic",):
        return SemanticChunker(**kwargs)
    elif strategy in ("hierarchical", "parent_child"):
        return HierarchicalChunker(**kwargs)
    else:
        return RecursiveCharacterChunker(**kwargs)
