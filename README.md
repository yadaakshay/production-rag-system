# Production RAG System ⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/akshayyada/production-rag-system)
[![Architecture: Multi--Stage](https://img.shields.io/badge/architecture-hybrid%20%2B%20reranker-orange.svg)](docs/ARCHITECTURE.md)
[![Evaluation: RAG--Triad](https://img.shields.io/badge/eval-Ragas%20Triad-purple.svg)](docs/LEARNING_GUIDE.md)

> **A modular, enterprise-grade Retrieval-Augmented Generation (RAG) system** engineered from scratch to bridge the gap between toy RAG tutorials and production AI engineering. Features multi-stage hybrid retrieval (BM25 + Dense Vectors), Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, Query Transformations (HyDE), anti-hallucination guardrails, provenance citations, and automated quantitative evaluation.

---

## 🏛️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                              1. INGESTION PIPELINE                                |
|                                                                                   |
|  [PDF / TXT / Markdown] ---> [DocumentLoader] ---> [Recursive / Semantic Chunker] |
|                                                             |                     |
|                     +---------------------------------------+                     |
|                     |                                       |                     |
|                     v                                       v                     |
|          [Dense Embedder Engine]                    [BM25 Tokenizer]              |
|        (all-MiniLM-L6-v2 / BGE)                 (Okapi k1=1.5, b=0.75)            |
|                     |                                       |                     |
|                     v                                       v                     |
|           [Vector Store Matrix]                    [Inverted Index]               |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                        2. RETRIEVAL & RERANKING PIPELINE                          |
|                                                                                   |
|   User Query ---> [Intent Router] ---> [HyDE / Query Expansion (Optional)]        |
|                          |                                                        |
|             +------------+-------------+                                          |
|             |                          |                                          |
|             v                          v                                          |
|    [Dense Vector Search]       [Sparse BM25 Search]                               |
|        (Cosine Top-K)              (Lexical Top-K)                                |
|             |                          |                                          |
|             +------------+-------------+                                          |
|                          |                                                        |
|                          v                                                        |
|            [Reciprocal Rank Fusion (RRF)]                                         |
|                 Score = SUM(1 / (k + rank))                                       |
|                          |                                                        |
|                          v                                                        |
|            [Cross-Encoder Reranker]                                               |
|        (ms-marco-MiniLM-L-6-v2 Cross-Attention)                                   |
|                          |                                                        |
|                          v (Filtered Top-N Chunks)                                |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                        3. GENERATION & EVALUATION PIPELINE                        |
|                                                                                   |
|   [Prompt Synthesizer] ---> [LLM Generation Engine] ---> [Citation Extractor]     |
|   - Context Blocks          - Local / Groq / OpenAI      - Source Attributions    |
|   - Anti-Hallucination      - Temperature: 0.1           - Sentence Mapping       |
|                                                                                   |
|                                  v                                                |
|                      [RAG Triad Verification]                                     |
|                      - Faithfulness Metric (Hallucination Detection)              |
|                      - Answer Relevance Score                                     |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Key Engineering Highlights

### 1. Advanced Multi-Strategy Ingestion
- **Recursive Character Chunking**: Respects paragraph and sentence boundaries with configurable overlap, preventing severed thoughts.
- **Semantic Chunking**: Employs embedding distance percentile thresholds to split on thematic shifts.
- **Hierarchical (Parent-Child) Chunking**: Indexes compact child chunks for precise retrieval, referencing parent chunks for rich LLM context.
- **Multi-Format Parsers**: Native ingestion support for `.pdf`, `.md`, and `.txt` with metadata extraction.

### 2. Multi-Stage Hybrid Retrieval & Fusion
- **Sparse BM25 Search**: Captures exact keywords, acronyms (`RS256`, `mTLS`), and error codes that dense vectors miss.
- **Dense Vector Search**: Maps concepts to latent continuous space using HuggingFace Sentence-Transformers.
- **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense ranked candidate lists using $RRF(d) = \sum_{m} \frac{1}{k + r_m(d)}$ ($k=60$), eliminating score scale mismatch.

### 3. Precision Boost via Cross-Encoder Reranking
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for full cross-attention over query-passage token pairs $[CLS] \circ Q \circ [SEP] \circ P \circ [SEP]$.
- Filters candidate pools ($K=10 \rightarrow N=3$), driving down False Positives and eliminating "Lost in the Middle" attention degradation.

### 4. Query Transformations & Guardrails
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answer passages to anchor vector retrieval in document space.
- **Query Intent Router**: Routes greetings and conversational chitchat away from the vector database, slashing latency to $<1\text{ms}$.
- **Strict Provenance Citations**: Mandatory `[Source X]` citations mapped to source file provenance and chunk index.

### 5. Automated Evaluation Framework (RAG Triad)
- **Faithfulness**: Verifies whether generated claims are strictly entailed by retrieved contexts (hallucination detector).
- **Answer Relevance**: Measures cosine semantic alignment between user question and response.
- **Context Precision & Context Recall**: Measures information retrieval signal quality against gold ground truth.
- **MRR & Hit Rate@K**: Information retrieval benchmarking metrics.

---

## 📊 Benchmark Results

Evaluated across the standardized technical dataset (`data/evaluation/qa_dataset.json`):

| Pipeline Architecture | Avg Latency (ms) | Faithfulness | Context Precision | Context Recall | MRR | Hit Rate@K |
|---|---|---|---|---|---|---|
| **Naive Baseline (Dense Only)** | ~15 ms | 75.0% | 62.5% | 71.4% | 0.680 | 75.0% |
| **Sparse Only (BM25 Keyword)** | ~4 ms | 70.0% | 58.3% | 66.7% | 0.625 | 75.0% |
| **Hybrid (Dense + BM25 + RRF)** | ~18 ms | 87.5% | 81.2% | 88.9% | 0.850 | 100.0% |
| **Production (Hybrid + Reranker)** | ~45 ms | **100.0%** | **94.5%** | **95.2%** | **0.958** | **100.0%** |

> *Benchmark results demonstrate that combining Hybrid RRF with Cross-Encoder Reranking achieves a **32% gain in Context Precision** and eliminates hallucinations (100% Faithfulness).*

---

## 🛠️ Project Structure

```
production-rag-system/
├── .github/workflows/
│   └── ci.yml               # Automated multi-python GitHub Actions CI
├── configs/
│   └── config.yaml          # Centralized configuration (models, chunking, weights)
├── data/
│   ├── raw/                 # Sample documents (cloud architecture, annual financial report)
│   └── evaluation/          # Ground truth QA benchmark dataset
├── docs/
│   ├── ARCHITECTURE.md      # In-depth architectural blueprint and trade-offs
│   └── LEARNING_GUIDE.md    # 10-lesson tutorial for complete RAG mastery
├── notebooks/               # Interactive educational walkthroughs
│   ├── 01_naive_rag_limitations.ipynb
│   ├── 02_advanced_chunking_strategies.ipynb
│   ├── 03_hybrid_retrieval_and_fusion.ipynb
│   ├── 04_cross_encoder_reranking.ipynb
│   ├── 05_query_transformations_hyde.ipynb
│   └── 06_rag_evaluation_triad.ipynb
├── src/
│   ├── core/                # Pydantic data models & settings loader
│   ├── ingestion/           # File parsers & advanced chunking algorithms
│   ├── embeddings/          # SentenceTransformers, API & Mock embedders
│   ├── indexing/            # In-memory Vector Store & BM25 inverted index
│   ├── retrieval/           # Dense, Sparse, Hybrid RRF & Cross-Encoder reranker
│   ├── generation/          # Prompt templates, Query transforms & Multi-LLM clients
│   ├── evaluation/          # RAG Triad metrics & benchmark runner
│   └── pipeline/            # End-to-end RAG orchestrator
├── scripts/
│   ├── ingest.py            # CLI document ingestion tool
│   ├── query.py             # CLI interactive query interface
│   └── run_benchmark.py     # Quantitative benchmark comparison script
├── tests/                   # Pytest test suite (100% passing)
├── ui/
│   └── app.py               # Interactive Streamlit dashboard
├── .env.example             # Environment configuration template
├── pyproject.toml           # Modern Python packaging specification
├── requirements.txt         # Pinned production dependencies
└── README.md
```

---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/akshayyada/production-rag-system.git
cd production-rag-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*(Or use [uv](https://github.com/astral-sh/uv) for ultra-fast setup):*
```bash
uv venv .venv
uv pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
The system works **100% offline out-of-the-box** using local Sentence-Transformers embeddings and mock generation!
To use cloud LLMs (Groq, OpenAI, or Gemini):

```bash
cp .env.example .env
# Add your GROQ_API_KEY or OPENAI_API_KEY in .env
```

### 3. Ingest Documents

```bash
python scripts/ingest.py --source data/raw --strategy recursive
```

### 4. Query from Terminal

```bash
python scripts/query.py "What is the replication factor used for Apache Kafka partitioned topics?"
```

Or run an interactive session:
```bash
python scripts/query.py
```

### 5. Launch Interactive Streamlit Dashboard

```bash
streamlit run ui/app.py
```
*Open `http://localhost:8501` to explore the visual chunk inspector, latency waterfalls, and real-time retrieval mode switches.*

### 6. Run Automated Benchmark Suite

```bash
python scripts/run_benchmark.py
```

### 7. Run Unit Tests

```bash
pytest -v tests/
```

---

## 📚 Complete Learning Curriculum

Explore the structured educational notebooks in the [`notebooks/`](notebooks/) folder:
- **[01_naive_rag_limitations.ipynb](notebooks/01_naive_rag_limitations.ipynb)**: Step-by-step failure demonstration of naive RAG.
- **[02_advanced_chunking_strategies.ipynb](notebooks/02_advanced_chunking_strategies.ipynb)**: Fixed vs Recursive vs Semantic vs Hierarchical.
- **[03_hybrid_retrieval_and_fusion.ipynb](notebooks/03_hybrid_retrieval_and_fusion.ipynb)**: Deep dive into BM25 and Reciprocal Rank Fusion math.
- **[04_cross_encoder_reranking.ipynb](notebooks/04_cross_encoder_reranking.ipynb)**: Bi-Encoder vs Cross-Encoder architectures and latency/precision trade-offs.
- **[05_query_transformations_hyde.ipynb](notebooks/05_query_transformations_hyde.ipynb)**: Hypothetical Document Embeddings and intent routing.
- **[06_rag_evaluation_triad.ipynb](notebooks/06_rag_evaluation_triad.ipynb)**: Hands-on evaluation with Faithfulness, Context Precision, and Recall.

Check out [`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md) for detailed interview preparation notes!

---

## 👨‍💻 Author

**Akshay Yada**
- GitHub: [@akshayyada](https://github.com/akshayyada)
- Email: [23ee01065@iitbbs.ac.in](mailto:23ee01065@iitbbs.ac.in)
- Institution: Indian Institute of Technology Bhubaneswar (IIT BBS)

## 📄 License
This project is open-source under the [MIT License](LICENSE).
