"""Evaluation module: RAG Triad, Retrieval Metrics, and Benchmarking."""

from .metrics import (
    RAGEvaluator,
    calculate_faithfulness,
    calculate_answer_relevance,
    calculate_context_precision,
    calculate_context_recall,
    calculate_mrr,
    calculate_hit_rate,
)

__all__ = [
    "RAGEvaluator",
    "calculate_faithfulness",
    "calculate_answer_relevance",
    "calculate_context_precision",
    "calculate_context_recall",
    "calculate_mrr",
    "calculate_hit_rate",
]
