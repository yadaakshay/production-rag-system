"""Script to generate educational Jupyter notebooks for complete RAG mastery."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook(filename: str, cells: list):
    nb_data = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.12.0"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(NOTEBOOKS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2)
    print(f"Created notebook: {filename}")


# ---------------- 01_naive_rag_limitations.ipynb ----------------
cells_01 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Module 1: Naive RAG & Its Real-World Failure Modes\n",
            "\n",
            "### Learning Objectives:\n",
            "1. Understand the architecture of Naive RAG (Load -> Split -> Vector Embed -> Cosine Top-K -> Prompt).\n",
            "2. Identify why Naive RAG fails in production on technical and enterprise documents.\n",
            "3. Witness specific failure modes: exact keyword misses, false positive semantic drift, and chunk boundary truncation.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "sys.path.append('..')\n",
            "\n",
            "from src.core.models import Document\n",
            "from src.ingestion.chunkers import FixedSizeChunker\n",
            "from src.embeddings.api_embedder import MockEmbedder\n",
            "from src.indexing.vector_store import InMemoryVectorStore\n",
            "\n",
            "# Ingest a realistic technical document\n",
            "sample_doc = Document(\n",
            "    text='Our API gateway enforces asymmetric RS256 JWT tokens. Redis L2 cache uses a 24-hour TTL with idempotency keys to avoid duplicate transaction processing.',\n",
            "    metadata={'filename': 'arch_spec.md'}\n",
            ")\n",
            "print('Original Document:', sample_doc)\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Naive Step 1: Fixed Character Chunking\n",
            "Naive RAG typically splits text at arbitrary fixed intervals (e.g. 50 characters). Observe how this severs critical phrases across boundaries.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)\n",
            "chunks = chunker.chunk(sample_doc)\n",
            "for i, c in enumerate(chunks):\n",
            "    print(f'Chunk #{i}: \"{c.text}\"')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Failure Mode Analysis:\n",
            "- **Boundary Mutilation**: 'asymmetric RS256 JWT' is broken midway, ruining exact term search.\n",
            "- **Lost Semantic Anchor**: Without paragraph awareness, the context of 'Redis L2 cache' is separated from 'idempotency keys'.\n",
            "- **Next Step**: In Module 2, we implement Recursive and Semantic Chunking to solve this!\n",
        ],
    },
]
create_notebook("01_naive_rag_limitations.ipynb", cells_01)


# ---------------- 02_advanced_chunking_strategies.ipynb ----------------
cells_02 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Module 2: Advanced Chunking Strategies\n",
            "\n",
            "### Comparing 4 Chunking Methodologies:\n",
            "1. **Fixed Size**: Naive slicing by character count.\n",
            "2. **Recursive Character**: Separator hierarchy (`\\n\\n` -> `\\n` -> `. ` -> ` `) preserving paragraph and sentence integrity.\n",
            "3. **Semantic Chunking**: Dynamic boundary detection using sentence embedding distance thresholds.\n",
            "4. **Hierarchical (Parent-Child)**: Small child chunks for vector indexing; large parent chunks for LLM context synthesis.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "\n",
            "from src.core.models import Document\n",
            "from src.ingestion.chunkers import RecursiveCharacterChunker, HierarchicalChunker\n",
            "\n",
            "text = '''## Distributed Caching\n",
            "The platform deploys a two-tier caching hierarchy to protect backend relational datastores.\n",
            "L1 Cache is an in-process Caffeine cache with a 60-second TTL.\n",
            "\n",
            "## Messaging Tier\n",
            "All asynchronous event processing utilizes Apache Kafka with partitioned topics.\n",
            "Replication factor is set to 3 to guarantee cross-AZ durability.'''\n",
            "\n",
            "doc = Document(text=text, metadata={'source': 'architecture.md'})\n",
            "\n",
            "# 1. Recursive Chunking\n",
            "rec_chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30)\n",
            "rec_chunks = rec_chunker.chunk(doc)\n",
            "print(f'Recursive Chunks generated: {len(rec_chunks)}')\n",
            "for i, c in enumerate(rec_chunks):\n",
            "    print(f'-- Chunk {i} ({len(c.text)} chars):\\n{c.text}\\n')\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Hierarchical (Parent-Child) Chunking\n",
            "hier_chunker = HierarchicalChunker(parent_chunk_size=250, child_chunk_size=80)\n",
            "child_chunks = hier_chunker.chunk(doc)\n",
            "print(f'Hierarchical Child Chunks: {len(child_chunks)}')\n",
            "first_child = child_chunks[0]\n",
            "print('Child Chunk Text:', first_child.text)\n",
            "print('Associated Parent Text:', first_child.metadata.get('parent_text'))\n",
        ],
    },
]
create_notebook("02_advanced_chunking_strategies.ipynb", cells_02)


# ---------------- 03_hybrid_retrieval_and_fusion.ipynb ----------------
cells_03 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Module 3: Hybrid Retrieval & Reciprocal Rank Fusion (RRF)\n",
            "\n",
            "### Why Dense-Only Search Is Not Enough:\n",
            "- Dense embeddings excel at conceptual, semantic queries (*'how do we handle server crashes'* -> *'circuit breaker pattern'*).\n",
            "- BUT dense embeddings struggle with exact keywords, serial numbers, codes, or domain acronyms (*'RS256'*, *'CVE-2024-1234'*, *'TTL=60s'*).\n",
            "- **Solution**: Hybrid search fusing **BM25 (Sparse Lexical)** and **Dense Vector Search**.\n",
            "\n",
            "### The Reciprocal Rank Fusion (RRF) Formula:\n",
            "$$\\text{RRF Score}(d) = \\sum_{m \\in M} \\frac{1}{k + \\text{rank}_m(d)}$$\n",
            "where $k$ is typically 60. RRF eliminates the scale disparity between unbounded BM25 scores and [0,1] cosine similarities.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "\n",
            "from src.core.models import Chunk\n",
            "from src.indexing.bm25_index import BM25Index\n",
            "from src.indexing.vector_store import InMemoryVectorStore\n",
            "from src.embeddings.api_embedder import MockEmbedder\n",
            "from src.retrieval.dense_retriever import DenseRetriever\n",
            "from src.retrieval.sparse_retriever import SparseRetriever\n",
            "from src.retrieval.hybrid_retriever import HybridRetriever\n",
            "\n",
            "chunks = [\n",
            "    Chunk(doc_id='d1', text='External client auth uses asymmetric RS256 JWT tokens with OAuth 2.0.'),\n",
            "    Chunk(doc_id='d2', text='The circuit breaker trips into OPEN state when failure rate exceeds 50%.'),\n",
            "    Chunk(doc_id='d3', text='Kafka partitioned topics use replication factor 3 for high availability.'),\n",
            "]\n",
            "\n",
            "embedder = MockEmbedder(dimension=64)\n",
            "vector_store = InMemoryVectorStore()\n",
            "vector_store.add_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))\n",
            "\n",
            "bm25 = BM25Index()\n",
            "bm25.add_chunks(chunks)\n",
            "\n",
            "dense = DenseRetriever(vector_store, embedder)\n",
            "sparse = SparseRetriever(bm25)\n",
            "hybrid = HybridRetriever(dense, sparse, mode='rrf', rrf_k=60)\n",
            "\n",
            "query = 'What is the RS256 token policy?'\n",
            "results = hybrid.retrieve(query, top_k=3)\n",
            "for r in results:\n",
            "    print(f'Rank {r.rank} | RRF Score: {r.score:.5f} | Snippet: {r.chunk.text}')\n",
        ],
    },
]
create_notebook("03_hybrid_retrieval_and_fusion.ipynb", cells_03)


# ---------------- 04_cross_encoder_reranking.ipynb ----------------
cells_04 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Module 4: Precision Boost with Cross-Encoder Reranking\n",
            "\n",
            "### Bi-Encoders vs Cross-Encoders:\n",
            "| Feature | Bi-Encoder (Embedding Search) | Cross-Encoder (Reranker) |\n",
            "|---|---|---|\n",
            "| Architecture | Query and Passage encoded independently | Full cross-attention over `[CLS] Query [SEP] Passage [SEP]` |\n",
            "| Speed | Sub-millisecond (ANN vector search over millions) | Slower (requires Transformer forward pass for each pair) |\n",
            "| Precision | Medium | High (captures intricate token interactions) |\n",
            "| Best Role | First-stage candidate generation (top 20-50) | Second-stage reranker (top 3-5) |\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "\n",
            "from src.core.models import Chunk, RetrievalResult\n",
            "from src.retrieval.reranker import CrossEncoderReranker\n",
            "\n",
            "query = 'How long does the circuit breaker remain open?'\n",
            "\n",
            "# Candidates retrieved by stage 1\n",
            "candidates = [\n",
            "    RetrievalResult(chunk=Chunk(doc_id='1', text='Envoy gateway handles client rate-limiting and mTLS certificates.'), score=0.75, rank=1),\n",
            "    RetrievalResult(chunk=Chunk(doc_id='2', text='When failure rate exceeds 50%, the circuit breaker trips to OPEN for 30 seconds.'), score=0.72, rank=2),\n",
            "    RetrievalResult(chunk=Chunk(doc_id='3', text='Redis cache invalidation uses Pub/Sub to purge stale local entries.'), score=0.68, rank=3),\n",
            "]\n",
            "\n",
            "reranker = CrossEncoderReranker(model_name='heuristic')\n",
            "reranked = reranker.rerank(query, candidates, top_n=2)\n",
            "\n",
            "print('Top candidate after Cross-Encoder reranking:')\n",
            "print(f'Chunk: {reranked[0].chunk.text}')\n",
            "print(f'Rerank Score: {reranked[0].score:.4f}')\n",
        ],
    },
]
create_notebook("04_cross_encoder_reranking.ipynb", cells_04)


# ---------------- 05_query_transformations_hyde.ipynb ----------------
cells_05 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Module 5: Query Transformations (HyDE & Query Routing)\n",
            "\n",
            "### 1. HyDE (Hypothetical Document Embeddings - Gao et al.):\n",
            "User questions are often short and asymmetric compared to document passages (*'Kafka failover?'* vs 500-word paragraph detailing zookeeper leader election).\n",
            "HyDE uses an LLM to generate a hypothetical passage that would answer the question, and searches using that passage vector!\n",
            "\n",
            "### 2. Query Routing:\n",
            "Not every query requires vector retrieval. Conversational chit-chat (*'Hello!'*) or general arithmetic (*'What is 100 * 5?'*) should be routed to direct response to eliminate latency and avoid retrieving random noise.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "\n",
            "from src.generation.query_transforms import QueryRouter, HyDETransformer\n",
            "from src.generation.llm_client import MockLLMClient\n",
            "\n",
            "llm = MockLLMClient()\n",
            "router = QueryRouter(llm)\n",
            "\n",
            "queries = [\n",
            "    'Hi there, how are you today?',\n",
            "    'What is the replication factor for partitioned Kafka topics?',\n",
            "]\n",
            "\n",
            "for q in queries:\n",
            "    decision, reason = router.route(q)\n",
            "    print(f'Query: \"{q}\" -> Route: {decision} ({reason})')\n",
        ],
    },
]
create_notebook("05_query_transformations_hyde.ipynb", cells_05)


# ---------------- 06_rag_evaluation_triad.ipynb ----------------
cells_06 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Module 6: Evaluating RAG with the RAG Triad (Ragas Style)\n",
            "\n",
            "### The RAG Triad Metrics:\n",
            "1. **Faithfulness (Anti-Hallucination)**: Is the answer strictly derived from retrieved context?\n",
            "2. **Answer Relevance**: Does the answer directly address the user query?\n",
            "3. **Context Precision**: Are the relevant context chunks ranked higher than irrelevant ones?\n",
            "4. **Context Recall**: Are all key ground-truth facts retrieved?\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "\n",
            "from src.evaluation.metrics import (\n",
            "    calculate_faithfulness,\n",
            "    calculate_answer_relevance,\n",
            "    calculate_context_precision,\n",
            "    calculate_context_recall,\n",
            ")\n",
            "\n",
            "query = 'What is the circuit breaker reset timeout?'\n",
            "ground_truth = 'The circuit breaker stays in OPEN state for 30 seconds before testing downstream health.'\n",
            "retrieved_context = ['The circuit breaker monitors call failure rates. When tripped into OPEN state, it waits 30 seconds before transitioning to HALF-OPEN.']\n",
            "answer = 'According to the documentation, the circuit breaker remains in the OPEN state for 30 seconds before switching to HALF-OPEN.'\n",
            "\n",
            "faith = calculate_faithfulness(answer, retrieved_context)\n",
            "relevance = calculate_answer_relevance(query, answer)\n",
            "precision = calculate_context_precision(ground_truth, retrieved_context)\n",
            "recall = calculate_context_recall(ground_truth, retrieved_context)\n",
            "\n",
            "print(f'Faithfulness: {faith * 100:.1f}%')\n",
            "print(f'Answer Relevance: {relevance * 100:.1f}%')\n",
            "print(f'Context Precision: {precision * 100:.1f}%')\n",
            "print(f'Context Recall: {recall * 100:.1f}%')\n",
        ],
    },
]
create_notebook("06_rag_evaluation_triad.ipynb", cells_06)

print("All 6 educational notebooks created successfully.")
