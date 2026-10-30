"""CLI benchmark comparison tool across RAG architectural modes."""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table
from src.core.config import RAGConfig
from src.ingestion.parsers import DocumentLoader
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.metrics import RAGEvaluator

console = Console()


def main():
    console.print("[bold cyan]==================================================[/bold cyan]")
    console.print("[bold cyan]       Production RAG System - Benchmark Suite    [/bold cyan]")
    console.print("[bold cyan]==================================================[/bold cyan]")

    eval_path = PROJECT_ROOT / "data" / "evaluation" / "qa_dataset.json"
    if not eval_path.exists():
        console.print(f"[bold red]Evaluation dataset not found at {eval_path}[/bold red]")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    config = RAGConfig.load_from_yaml(str(PROJECT_ROOT / "configs" / "config.yaml"))
    pipeline = RAGPipeline(config=config)

    # Ingest documents
    raw_dir = PROJECT_ROOT / "data" / "raw"
    docs = DocumentLoader.load_directory(str(raw_dir))
    pipeline.ingest_documents(docs)

    evaluator = RAGEvaluator(embedder=pipeline.embedder)

    modes = [
        ("Naive (Dense Baseline)", "naive"),
        ("Sparse BM25 Keyword", "sparse_only"),
        ("Hybrid (Dense + BM25 + RRF)", "hybrid"),
        ("Production (Hybrid + Cross-Encoder)", "production"),
    ]

    benchmark_results = []
    for mode_name, mode_val in modes:
        console.print(f"Evaluating mode: [bold green]{mode_name}[/bold green]...")
        res = evaluator.benchmark_pipeline(pipeline, dataset, pipeline_mode=mode_name)
        benchmark_results.append(res)

    table = Table(title="RAG Architecture Benchmark Comparison", show_header=True)
    table.add_column("Pipeline Architecture", style="cyan", no_wrap=True)
    table.add_column("Avg Latency (ms)", style="magenta")
    table.add_column("Faithfulness", style="green")
    table.add_column("Context Precision", style="yellow")
    table.add_column("Context Recall", style="blue")
    table.add_column("MRR", style="white")
    table.add_column("Hit Rate@K", style="bold green")

    for b in benchmark_results:
        table.add_row(
            b.pipeline_mode,
            str(b.avg_latency_ms),
            f"{b.mean_faithfulness * 100:.1f}%",
            f"{b.mean_context_precision * 100:.1f}%",
            f"{b.mean_context_recall * 100:.1f}%",
            f"{b.mean_mrr:.3f}",
            f"{b.mean_hit_rate * 100:.1f}%",
        )

    console.print("\n")
    console.print(table)
    console.print("\n[bold green][OK] Benchmark run completed successfully![/bold green]")


if __name__ == "__main__":
    main()
