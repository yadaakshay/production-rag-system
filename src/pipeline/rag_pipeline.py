"""Production-grade modular RAG Pipeline orchestrator."""

from enum import Enum
from typing import List, Dict, Any, Optional
import time
import re
from pathlib import Path

from ..core.models import (
    Document,
    Chunk,
    RetrievalResult,
    RAGResponse,
    Citation,
    QueryTransformationResult,
)
from ..core.config import RAGConfig, app_config
from ..ingestion.chunkers import get_chunker, BaseChunker
from ..embeddings.factory import get_embedder
from ..embeddings.base import BaseEmbedder
from ..indexing.vector_store import VectorStore
from ..indexing.bm25_index import BM25Index
from ..retrieval.dense_retriever import DenseRetriever
from ..retrieval.sparse_retriever import SparseRetriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import CrossEncoderReranker
from ..generation.prompt_templates import RAGPromptTemplate
from ..generation.query_transforms import HyDETransformer, MultiQueryExpander, QueryRouter
from ..generation.llm_client import get_llm_client, BaseLLMClient
from ..evaluation.metrics import calculate_faithfulness


class PipelineMode(str, Enum):
    NAIVE = "naive"                    # Dense only, fixed chunking, no reranker
    DENSE_ONLY = "dense_only"          # Dense vector search only
    SPARSE_ONLY = "sparse_only"        # BM25 keyword search only
    HYBRID = "hybrid"                  # Dense + BM25 with Reciprocal Rank Fusion
    PRODUCTION = "production"          # Hybrid + Cross-Encoder Reranking + Query Transforms


class RAGPipeline:
    """End-to-end RAG orchestrator supporting ingestion, hybrid retrieval, reranking, and generation."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        chunker: Optional[BaseChunker] = None,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.config = config or app_config

        # Ingestion & Embeddings
        self.chunker = chunker or get_chunker(
            strategy=self.config.ingestion.chunking_strategy,
            chunk_size=self.config.ingestion.chunk_size,
            chunk_overlap=self.config.ingestion.chunk_overlap,
        )
        self.embedder = embedder or get_embedder(
            provider=self.config.embeddings.provider,
            model_name=self.config.embeddings.model_name,
            dimension=self.config.embeddings.dimension,
            device=self.config.embeddings.device,
        )

        # Indexing & Stores
        self.vector_store = vector_store or VectorStore(metric=self.config.indexing.similarity_metric)
        self.bm25_index = bm25_index or BM25Index(
            k1=self.config.indexing.bm25.k1, b=self.config.indexing.bm25.b
        )

        # Retrievers
        self.dense_retriever = DenseRetriever(self.vector_store, self.embedder)
        self.sparse_retriever = SparseRetriever(self.bm25_index)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            mode=self.config.retrieval.hybrid_mode,
            rrf_k=self.config.retrieval.rrf_k,
            dense_weight=self.config.retrieval.dense_weight,
            sparse_weight=self.config.retrieval.sparse_weight,
        )

        # Reranker
        self.reranker = reranker or CrossEncoderReranker(
            model_name=self.config.reranker.model_name
        )

        # LLM Generation & Query transformations
        self.llm = llm_client or get_llm_client(
            provider=self.config.generation.provider,
            model_name=self.config.generation.model_name,
            api_key=self.config.openai_api_key or self.config.groq_api_key or self.config.gemini_api_key,
        )
        self.router = QueryRouter(self.llm if self.config.generation.query_transformation.enable_router else None)
        self.hyde = HyDETransformer(self.llm) if self.config.generation.query_transformation.enable_hyde else None
        self.multi_query = MultiQueryExpander(self.llm) if self.config.generation.query_transformation.enable_multi_query else None

        self.indexed_chunks_count = 0

    def ingest_documents(self, documents: List[Document], persist: bool = False) -> int:
        """Chunk, embed, and index incoming documents into both vector and BM25 stores."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Compute dense embeddings
        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Populate vector store and BM25 index
        self.vector_store.add_chunks(all_chunks, embeddings)
        self.bm25_index.add_chunks(all_chunks)

        self.indexed_chunks_count += len(all_chunks)

        if persist:
            self.persist_indices(self.config.storage.index_dir)

        return len(all_chunks)

    def persist_indices(self, directory: str) -> None:
        """Save indices to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(str(path))
        self.bm25_index.save(str(path))

    def load_indices(self, directory: str) -> None:
        """Load persisted indices from disk."""
        self.vector_store = VectorStore.load(directory)
        self.bm25_index = BM25Index.load(directory)
        self.dense_retriever = DenseRetriever(self.vector_store, self.embedder)
        self.sparse_retriever = SparseRetriever(self.bm25_index)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            mode=self.config.retrieval.hybrid_mode,
            rrf_k=self.config.retrieval.rrf_k,
        )
        self.indexed_chunks_count = len(self.vector_store.chunks)

    def run(
        self,
        query: str,
        mode: PipelineMode = PipelineMode.PRODUCTION,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> RAGResponse:
        """Execute end-to-end RAG query flow with latency tracking and provenance citations."""
        latencies: Dict[str, float] = {}
        top_k = top_k or self.config.retrieval.top_k_retrieval
        top_n = top_n or self.config.reranker.top_n

        # Stage 1: Query Intent Routing
        t0 = time.perf_counter()
        route_decision, route_reason = self.router.route(query)
        latencies["router_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if route_decision == "CONVERSATIONAL":
            return RAGResponse(
                query=query,
                answer="Hello! How can I assist you with your documents today?",
                latency_breakdown=latencies,
            )

        # Stage 2: Query Transformation (HyDE or Multi-Query if enabled)
        t0 = time.perf_counter()
        search_queries = [query]
        q_transform_res = None

        if mode == PipelineMode.PRODUCTION:
            if self.hyde is not None:
                q_transform_res = self.hyde.transform(query)
                search_queries = q_transform_res.transformed_queries
            elif self.multi_query is not None:
                q_transform_res = self.multi_query.transform(query)
                search_queries = q_transform_res.transformed_queries
        latencies["query_transform_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Stage 3: Retrieval
        t0 = time.perf_counter()
        all_retrieved: List[RetrievalResult] = []

        for q in search_queries:
            if mode == PipelineMode.NAIVE or mode == PipelineMode.DENSE_ONLY:
                results = self.dense_retriever.retrieve(q, top_k=top_k)
            elif mode == PipelineMode.SPARSE_ONLY:
                results = self.sparse_retriever.retrieve(q, top_k=top_k)
            else:  # HYBRID or PRODUCTION
                results = self.hybrid_retriever.retrieve(q, top_k=top_k)
            all_retrieved.extend(results)

        # Deduplicate retrieved chunks while preserving order
        seen_ids = set()
        deduped_results: List[RetrievalResult] = []
        for r in all_retrieved:
            if r.chunk.chunk_id not in seen_ids:
                seen_ids.add(r.chunk.chunk_id)
                deduped_results.append(r)
        latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Stage 4: Cross-Encoder Reranking
        t0 = time.perf_counter()
        if mode == PipelineMode.PRODUCTION and self.config.reranker.enabled:
            final_chunks = self.reranker.rerank(query, deduped_results, top_n=top_n)
        else:
            final_chunks = deduped_results[:top_n]
        latencies["reranking_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Stage 5: Prompt Formatting & Context Synthesis
        formatted_prompt = RAGPromptTemplate.format_rag_prompt(query, final_chunks)

        # Stage 6: Generation
        t0 = time.perf_counter()
        raw_answer = self.llm.generate(
            prompt=formatted_prompt,
            system_prompt=RAGPromptTemplate.SYSTEM_PROMPT,
            temperature=self.config.generation.temperature,
            max_tokens=self.config.generation.max_tokens,
        )
        latencies["generation_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Stage 7: Citations extraction
        citations: List[Citation] = []
        found_sources = set(re.findall(r"\[Source (\d+)\]", raw_answer))
        for src_str in found_sources:
            idx = int(src_str) - 1
            if 0 <= idx < len(final_chunks):
                matched = final_chunks[idx]
                citations.append(
                    Citation(
                        source_chunk_id=matched.chunk.chunk_id,
                        source_document=matched.chunk.metadata.get("filename", "unknown"),
                        chunk_index=matched.chunk.chunk_index,
                        snippet=matched.chunk.text[:120] + "...",
                    )
                )

        # Stage 8: Faithfulness Verification
        faithfulness = calculate_faithfulness(raw_answer, [r.chunk.text for r in final_chunks])

        total_latency = sum(latencies.values())
        latencies["total_ms"] = round(total_latency, 2)

        return RAGResponse(
            query=query,
            answer=raw_answer,
            retrieved_chunks=final_chunks,
            citations=citations,
            latency_breakdown=latencies,
            faithfulness_score=faithfulness,
            query_transformations=q_transform_res,
        )
