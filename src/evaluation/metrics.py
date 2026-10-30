"""Quantitative evaluation metrics for RAG pipelines (RAG Triad + Retrieval Metrics).

Metrics covered:
1. Faithfulness (Hallucination Detection): Verifies whether generated claims are grounded in context.
2. Answer Relevance: Semantic alignment between user query and generated response.
3. Context Precision: Evaluates whether relevant signal chunks rank higher than noise.
4. Context Recall: Proportion of ground-truth statements present in retrieved context.
5. MRR (Mean Reciprocal Rank) & Hit Rate@K: Classic Information Retrieval metrics.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import re
from ..core.models import EvaluationMetricResult, RAGBenchmarkResult


def _tokenize_terms(text: str) -> set:
    words = re.findall(r"\b\w{3,}\b", text.lower())
    return set(words)


def calculate_faithfulness(generated_answer: str, retrieved_contexts: List[str]) -> float:
    """Estimates the fraction of claims in generated_answer supported by retrieved_contexts.
    
    Returns float in range [0.0, 1.0].
    """
    if not generated_answer.strip() or not retrieved_contexts:
        return 0.0

    combined_context = " ".join(retrieved_contexts).lower()
    # Break answer into sentences / claims
    sentences = [s.strip() for s in re.split(r"[.?!]\s*", generated_answer) if len(s.strip()) > 5]
    if not sentences:
        return 1.0

    supported_count = 0
    for s in sentences:
        s_terms = _tokenize_terms(s)
        if not s_terms:
            supported_count += 1
            continue
        # Check term overlap against context
        overlap = len(s_terms.intersection(_tokenize_terms(combined_context)))
        overlap_ratio = overlap / len(s_terms)
        if overlap_ratio >= 0.5:
            supported_count += 1

    return float(round(supported_count / len(sentences), 4))


def calculate_answer_relevance(query: str, generated_answer: str, embedder = None) -> float:
    """Computes semantic alignment between user question and generated answer."""
    if not query.strip() or not generated_answer.strip():
        return 0.0

    if embedder is not None:
        try:
            q_emb = embedder.embed_query(query)
            a_emb = embedder.embed_query(generated_answer)
            v1, v2 = np.array(q_emb), np.array(a_emb)
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            return float(round(max(0.0, min(1.0, float(sim))), 4))
        except Exception:
            pass

    # Fallback to Jaccard token similarity
    q_terms = _tokenize_terms(query)
    a_terms = _tokenize_terms(generated_answer)
    if not q_terms or not a_terms:
        return 0.5
    jaccard = len(q_terms.intersection(a_terms)) / len(q_terms.union(a_terms))
    return float(round(min(1.0, jaccard * 2.5), 4))


def calculate_context_precision(ground_truth: str, retrieved_contexts: List[str]) -> float:
    """Evaluates whether chunks containing ground truth appear near the top of retrieved ranks."""
    if not retrieved_contexts or not ground_truth.strip():
        return 0.0

    gt_terms = _tokenize_terms(ground_truth)
    if not gt_terms:
        return 0.0

    relevant_indicator = []
    for ctx in retrieved_contexts:
        ctx_terms = _tokenize_terms(ctx)
        overlap = len(gt_terms.intersection(ctx_terms))
        is_rel = 1 if (overlap / len(gt_terms)) >= 0.4 else 0
        relevant_indicator.append(is_rel)

    total_relevant = sum(relevant_indicator)
    if total_relevant == 0:
        return 0.0

    # Calculate average precision
    cum_score = 0.0
    running_rel = 0
    for k, is_rel in enumerate(relevant_indicator, start=1):
        if is_rel:
            running_rel += 1
            precision_at_k = running_rel / k
            cum_score += precision_at_k

    return float(round(cum_score / total_relevant, 4))


def calculate_context_recall(ground_truth: str, retrieved_contexts: List[str]) -> float:
    """Fraction of ground-truth statements/terms present within retrieved contexts."""
    if not ground_truth.strip() or not retrieved_contexts:
        return 0.0

    combined_context = " ".join(retrieved_contexts)
    gt_terms = _tokenize_terms(ground_truth)
    ctx_terms = _tokenize_terms(combined_context)

    if not gt_terms:
        return 1.0

    overlap = len(gt_terms.intersection(ctx_terms))
    return float(round(overlap / len(gt_terms), 4))


def calculate_mrr(ground_truth: str, retrieved_contexts: List[str]) -> float:
    """Mean Reciprocal Rank: reciprocal of the rank of the first relevant chunk."""
    if not retrieved_contexts or not ground_truth.strip():
        return 0.0

    gt_terms = _tokenize_terms(ground_truth)
    for rank, ctx in enumerate(retrieved_contexts, start=1):
        ctx_terms = _tokenize_terms(ctx)
        overlap = len(gt_terms.intersection(ctx_terms))
        if (overlap / max(1, len(gt_terms))) >= 0.4:
            return float(round(1.0 / rank, 4))
    return 0.0


def calculate_hit_rate(ground_truth: str, retrieved_contexts: List[str]) -> int:
    """1 if at least one retrieved context contains ground truth, 0 otherwise."""
    return 1 if calculate_mrr(ground_truth, retrieved_contexts) > 0 else 0


class RAGEvaluator:
    """Batch evaluator for running evaluation suites against RAG pipelines."""

    def __init__(self, embedder = None):
        self.embedder = embedder

    def evaluate_sample(
        self,
        query: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationMetricResult:
        faithfulness = calculate_faithfulness(generated_answer, retrieved_contexts)
        relevance = calculate_answer_relevance(query, generated_answer, embedder=self.embedder)
        precision = calculate_context_precision(ground_truth or "", retrieved_contexts) if ground_truth else None
        recall = calculate_context_recall(ground_truth or "", retrieved_contexts) if ground_truth else None
        mrr = calculate_mrr(ground_truth or "", retrieved_contexts) if ground_truth else None
        hit_rate = calculate_hit_rate(ground_truth or "", retrieved_contexts) if ground_truth else None

        return EvaluationMetricResult(
            query=query,
            generated_answer=generated_answer,
            ground_truth=ground_truth,
            contexts=retrieved_contexts,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            context_precision=precision,
            context_recall=recall,
            reciprocal_rank=mrr,
            hit_rate=hit_rate,
        )

    def benchmark_pipeline(
        self,
        pipeline,
        dataset: List[Dict[str, str]],
        pipeline_mode: str = "Production RAG",
    ) -> RAGBenchmarkResult:
        import time
        results: List[EvaluationMetricResult] = []
        latencies: List[float] = []

        for item in dataset:
            q = item["question"]
            gt = item.get("ground_truth")
            t0 = time.perf_counter()
            resp = pipeline.run(q)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            contexts = [r.chunk.text for r in resp.retrieved_chunks]
            eval_res = self.evaluate_sample(
                query=q,
                generated_answer=resp.answer,
                retrieved_contexts=contexts,
                ground_truth=gt,
            )
            results.append(eval_res)

        n = max(1, len(results))
        return RAGBenchmarkResult(
            pipeline_mode=pipeline_mode,
            num_samples=len(results),
            avg_latency_ms=float(round(np.mean(latencies), 2)),
            mean_faithfulness=float(round(np.mean([r.faithfulness for r in results if r.faithfulness is not None]), 4)),
            mean_answer_relevance=float(round(np.mean([r.answer_relevance for r in results if r.answer_relevance is not None]), 4)),
            mean_context_precision=float(round(np.mean([r.context_precision for r in results if r.context_precision is not None] or [0.0]), 4)),
            mean_context_recall=float(round(np.mean([r.context_recall for r in results if r.context_recall is not None] or [0.0]), 4)),
            mean_mrr=float(round(np.mean([r.reciprocal_rank for r in results if r.reciprocal_rank is not None] or [0.0]), 4)),
            mean_hit_rate=float(round(np.mean([r.hit_rate for r in results if r.hit_rate is not None] or [0.0]), 4)),
        )
