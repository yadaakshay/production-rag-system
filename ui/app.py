"""Interactive Streamlit Dashboard for the Production RAG System.

Allows users to:
1. Query the RAG system under different modes (Naive vs Hybrid vs Cross-Encoder Reranked).
2. Inspect retrieved chunks, fusion scores, and cross-encoder logits.
3. Visualize end-to-end latency breakdowns (Retrieval vs Reranking vs Generation).
4. Run live evaluation and benchmark pipeline performance.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import time
import json
import pandas as pd
from src.core.config import RAGConfig
from src.core.models import Document
from src.ingestion.parsers import DocumentLoader
from src.pipeline.rag_pipeline import RAGPipeline, PipelineMode
from src.evaluation.metrics import RAGEvaluator

st.set_page_config(
    page_title="Production RAG System | Portfolio Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { color: #64748b; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }
    .chunk-box { border-left: 4px solid #3b82f6; background-color: #f1f5f9; padding: 10px; margin-bottom: 10px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_initialized_pipeline():
    """Initializes and pre-indexes sample documents."""
    config = RAGConfig.load_from_yaml(str(PROJECT_ROOT / "configs" / "config.yaml"))
    pipeline = RAGPipeline(config=config)

    # Ingest default raw files if available
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if raw_dir.exists():
        docs = DocumentLoader.load_directory(str(raw_dir))
        if docs:
            pipeline.ingest_documents(docs)

    return pipeline


pipeline = get_initialized_pipeline()

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.header("⚙️ RAG Configuration")

    pipeline_mode_choice = st.selectbox(
        "Retrieval Pipeline Mode",
        options=[
            PipelineMode.PRODUCTION.value,
            PipelineMode.HYBRID.value,
            PipelineMode.DENSE_ONLY.value,
            PipelineMode.SPARSE_ONLY.value,
            PipelineMode.NAIVE.value,
        ],
        format_func=lambda x: {
            "production": "🚀 Production (Hybrid + Reranker + Router)",
            "hybrid": "⚖️ Hybrid (BM25 + Dense + RRF)",
            "dense_only": "🔍 Dense Only (Vector Search)",
            "sparse_only": "📝 Sparse Only (BM25 Keyword)",
            "naive": "🌱 Naive Baseline (Single Stage)",
        }[x],
        index=0,
    )

    st.subheader("Retrieval Hyperparameters")
    top_k = st.slider("Initial Candidates (Top-K)", min_value=2, max_value=20, value=8)
    top_n = st.slider("Final Context Chunks (Top-N)", min_value=1, max_value=8, value=3)

    st.subheader("Advanced Enhancements")
    enable_hyde = st.checkbox("Hypothetical Document Embeddings (HyDE)", value=False)
    enable_reranker = st.checkbox("Cross-Encoder Reranking", value=True)

    # Update pipeline config dynamically
    pipeline.config.reranker.enabled = enable_reranker
    pipeline.config.generation.query_transformation.enable_hyde = enable_hyde

    st.divider()
    st.markdown(f"**Indexed Chunks:** `{pipeline.indexed_chunks_count}`")
    st.markdown(f"**Embedding Model:** `{pipeline.config.embeddings.model_name}`")
    st.markdown(f"**Reranker Model:** `{pipeline.config.reranker.model_name}`")

    st.divider()
    st.markdown("👨‍💻 **Author:** [Akshay Yada](https://github.com/akshayyada)")


# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-title">⚡ Production RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Stage Retrieval Engine with Reciprocal Rank Fusion, Cross-Encoder Reranking & Ragas Evaluation</div>', unsafe_allow_html=True)

tab_chat, tab_evaluation, tab_ingest = st.tabs(["💬 Query & Inspection", "📊 Benchmark & Metrics", "📁 Document Management"])

# ----------------- TAB 1: QUERY & INSPECTION -----------------
with tab_chat:
    # Example questions for fast demonstration
    st.markdown("**Sample Questions (Click to test):**")
    cols = st.columns(3)
    sample_q = None
    if cols[0].button("What is the Kafka replication factor?"):
        sample_q = "What is the replication factor used for Apache Kafka partitioned topics?"
    if cols[1].button("How does the circuit breaker trip?"):
        sample_q = "When does the circuit breaker trip into the OPEN state and for how long?"
    if cols[2].button("What was Nexa's 2025 revenue?"):
        sample_q = "What was the total consolidated revenue of Nexa in fiscal year 2025?"

    query = st.text_input("Enter your question:", value=sample_q or "", placeholder="Ask a factual question about the ingested documents...")

    if st.button("Generate Answer", type="primary") and query:
        with st.spinner("Processing through multi-stage RAG pipeline..."):
            response = pipeline.run(
                query=query,
                mode=PipelineMode(pipeline_mode_choice),
                top_k=top_k,
                top_n=top_n,
            )

        # Main Answer Box
        st.markdown("### 🤖 Synthesized Answer")
        st.info(response.answer)

        # Verification metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Latency", f"{response.latency_breakdown.get('total_ms', 0)} ms")
        m2.metric("Retrieval Latency", f"{response.latency_breakdown.get('retrieval_ms', 0)} ms")
        m3.metric("Reranking Latency", f"{response.latency_breakdown.get('reranking_ms', 0)} ms")
        faith_val = f"{int((response.faithfulness_score or 0) * 100)}%"
        m4.metric("Faithfulness Score", faith_val, help="Anti-hallucination metric checking context grounding")

        st.divider()

        # Detailed Inspection Columns
        col_chunks, col_latency = st.columns([1.5, 1])

        with col_chunks:
            st.markdown("### 📄 Retrieved Context Chunks (Top-N)")
            if not response.retrieved_chunks:
                st.warning("No context chunks were retrieved.")
            for i, res in enumerate(response.retrieved_chunks, start=1):
                with st.expander(f"Chunk #{i} | Score: {res.score:.4f} | Source: {res.chunk.metadata.get('filename', 'Unknown')}", expanded=(i == 1)):
                    st.markdown(f"**Strategy:** `{res.chunk.metadata.get('strategy', 'default')}` | **Retriever:** `{res.retriever_name}`")
                    st.markdown(f"**Content Snippet:**\n> {res.chunk.text}")
                    if "parent_text" in res.chunk.metadata:
                        st.markdown(f"**Parent Context:**\n> {res.chunk.metadata['parent_text'][:200]}...")

        with col_latency:
            st.markdown("### ⏱️ Latency Distribution")
            chart_data = {
                k.replace("_ms", "").capitalize(): v
                for k, v in response.latency_breakdown.items()
                if k != "total_ms"
            }
            if chart_data:
                st.bar_chart(chart_data)

            st.markdown("### 📌 Provenance Citations")
            if response.citations:
                for c in response.citations:
                    st.success(f"**[{c.source_document}]** Chunk #{c.chunk_index}: {c.snippet}")
            else:
                st.write("No explicit citations triggered.")


# ----------------- TAB 2: BENCHMARK & METRICS -----------------
with tab_evaluation:
    st.markdown("### 📊 Automated RAG Triad Evaluation & Benchmarking")
    st.markdown("Run a quantitative evaluation across the standard ground-truth evaluation dataset.")

    eval_file = PROJECT_ROOT / "data" / "evaluation" / "qa_dataset.json"

    if st.button("Run Full Benchmark Suite", type="primary"):
        if not eval_file.exists():
            st.error("Evaluation dataset not found at data/evaluation/qa_dataset.json")
        else:
            with open(eval_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            evaluator = RAGEvaluator(embedder=pipeline.embedder)
            with st.spinner("Evaluating Naive RAG vs Production RAG..."):
                naive_bench = evaluator.benchmark_pipeline(pipeline, dataset, pipeline_mode="Naive (Dense Only)")
                prod_bench = evaluator.benchmark_pipeline(pipeline, dataset, pipeline_mode="Production (Hybrid + Rerank)")

            st.success("Benchmark completed successfully!")

            df_comparison = pd.DataFrame([
                {
                    "Pipeline Mode": naive_bench.pipeline_mode,
                    "Avg Latency (ms)": naive_bench.avg_latency_ms,
                    "Faithfulness": f"{naive_bench.mean_faithfulness * 100:.1f}%",
                    "Context Precision": f"{naive_bench.mean_context_precision * 100:.1f}%",
                    "Context Recall": f"{naive_bench.mean_context_recall * 100:.1f}%",
                    "MRR": round(naive_bench.mean_mrr, 3),
                    "Hit Rate@K": f"{naive_bench.mean_hit_rate * 100:.1f}%",
                },
                {
                    "Pipeline Mode": prod_bench.pipeline_mode,
                    "Avg Latency (ms)": prod_bench.avg_latency_ms,
                    "Faithfulness": f"{prod_bench.mean_faithfulness * 100:.1f}%",
                    "Context Precision": f"{prod_bench.mean_context_precision * 100:.1f}%",
                    "Context Recall": f"{prod_bench.mean_context_recall * 100:.1f}%",
                    "MRR": round(prod_bench.mean_mrr, 3),
                    "Hit Rate@K": f"{prod_bench.mean_hit_rate * 100:.1f}%",
                },
            ])

            st.table(df_comparison)


# ----------------- TAB 3: DOCUMENT INGESTION -----------------
with tab_ingest:
    st.markdown("### 📁 Upload Custom Documents")
    uploaded_files = st.file_uploader(
        "Upload text, markdown, or PDF files:",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process & Index Uploaded Documents"):
        new_docs = []
        for file in uploaded_files:
            content = file.read().decode("utf-8", errors="replace")
            doc = Document(
                text=content,
                metadata={"filename": file.name, "source": "user_upload"},
            )
            new_docs.append(doc)

        with st.spinner(f"Chunking and embedding {len(new_docs)} document(s)..."):
            count = pipeline.ingest_documents(new_docs)

        st.success(f"Successfully chunked and indexed {count} new chunks into the vector store and BM25 index!")
