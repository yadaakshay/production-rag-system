"""BM25 sparse lexical inverted index for keyword search."""

from typing import List, Tuple, Optional
import re
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from ..core.models import Chunk


class BM25Index:
    """Tokenized BM25 Okapi index for exact keyword matching and lexical retrieval."""

    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
        "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
        "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
        "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
        "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
        "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
        "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
        "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
        "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
        "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def tokenize(self, text: str) -> List[str]:
        """Convert text into cleaned, lowercase alphanumeric tokens filtering stopwords."""
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS and len(w) > 1]

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the corpus and build/update the BM25 index."""
        if not chunks:
            return
        self.chunks.extend(chunks)
        for chunk in chunks:
            self.tokenized_corpus.append(self.tokenize(chunk.text))
        self._bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Query the index and return top-k matching chunks with BM25 scores."""
        if self._bm25 is None or not self.chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            # Query only has stopwords; fallback to simple tokenization
            tokenized_query = query.lower().split()

        scores = self._bm25.get_scores(tokenized_query)
        # Pair with chunks and sort descending
        scored_pairs = list(zip(self.chunks, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return [(chunk, float(score)) for chunk, score in scored_pairs[:top_k]]

    def save(self, directory: str) -> None:
        """Persist BM25 chunks and corpus to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [c.model_dump() for c in self.chunks],
            "tokenized_corpus": self.tokenized_corpus,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path / "bm25_index.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, directory: str) -> "BM25Index":
        """Load persisted BM25 index from disk."""
        path = Path(directory) / "bm25_index.json"
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        idx.chunks = [Chunk(**c) for c in data["chunks"]]
        idx.tokenized_corpus = data["tokenized_corpus"]
        if idx.tokenized_corpus:
            idx._bm25 = BM25Okapi(idx.tokenized_corpus, k1=idx.k1, b=idx.b)
        return idx
