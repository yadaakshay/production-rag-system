"""CLI tool to query the RAG system and inspect retrieval latency and citations."""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.core.config import RAGConfig
from src.ingestion.parsers import DocumentLoader
from src.pipeline.rag_pipeline import RAGPipeline, PipelineMode

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline from terminal.")
    parser.add_argument("query", type=str, nargs="?", help="Question to ask.")
    parser.add_argument(
        "--mode",
        type=str,
        default="production",
        choices=["naive", "dense_only", "sparse_only", "hybrid", "production"],
        help="Pipeline retrieval mode.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Initial retrieval count.")
    parser.add_argument("--top-n", type=int, default=3, help="Reranked context count.")

    args = parser.parse_args()

    config = RAGConfig.load_from_yaml(str(PROJECT_ROOT / "configs" / "config.yaml"))
    pipeline = RAGPipeline(config=config)

    # Ingest default documents if index empty
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if raw_dir.exists():
        docs = DocumentLoader.load_directory(str(raw_dir))
        pipeline.ingest_documents(docs)

    def process_query(q: str):
        console.print(f"\n[bold yellow]User Question:[/bold yellow] {q}")
        with console.status("[bold green]Executing multi-stage RAG pipeline...[/bold green]"):
            response = pipeline.run(
                query=q,
                mode=PipelineMode(args.mode),
                top_k=args.top_k,
                top_n=args.top_n,
            )

        console.print(Panel(response.answer, title="[bold green]Synthesized Answer[/bold green]", border_style="green"))

        # Citations
        if response.citations:
            console.print("[bold cyan]Citations:[/bold cyan]")
            for c in response.citations:
                console.print(f"  * [{c.source_document}] Chunk {c.chunk_index}: [italic]{c.snippet}[/italic]")

        # Latency breakdown table
        lat_table = Table(title="Latency Breakdown (ms)", show_header=True)
        lat_table.add_column("Stage", style="cyan")
        lat_table.add_column("Latency (ms)", style="magenta")
        for k, v in response.latency_breakdown.items():
            lat_table.add_row(k.replace("_ms", "").title(), str(v))
        console.print(lat_table)

        faith = response.faithfulness_score
        console.print(f"[bold]Faithfulness Grounding:[/bold] [{'green' if (faith or 0) > 0.7 else 'yellow'}]{int((faith or 0) * 100)}%[/{'green' if (faith or 0) > 0.7 else 'yellow'}]")

    if args.query:
        process_query(args.query)
    else:
        console.print("[bold green]Interactive RAG Query Session (type 'exit' or 'q' to quit)[/bold green]")
        while True:
            try:
                user_q = console.input("\n[bold cyan]Ask a question > [/bold cyan]").strip()
                if user_q.lower() in ("exit", "q", "quit"):
                    break
                if user_q:
                    process_query(user_q)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
