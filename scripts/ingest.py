"""CLI script to ingest documents and persist vector and BM25 indices."""

import sys
import argparse
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

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw",
        help="Path to file or directory to ingest.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="recursive",
        choices=["fixed", "recursive", "semantic", "hierarchical"],
        help="Chunking strategy to apply.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk character size.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk character overlap.",
    )

    args = parser.parse_args()

    console.print(f"[bold cyan]Initiating Ingestion Pipeline[/bold cyan]")
    console.print(f"Source: [yellow]{args.source}[/yellow] | Strategy: [green]{args.strategy}[/green]")

    config = RAGConfig.load_from_yaml(str(PROJECT_ROOT / "configs" / "config.yaml"))
    config.ingestion.chunking_strategy = args.strategy
    config.ingestion.chunk_size = args.chunk_size
    config.ingestion.chunk_overlap = args.chunk_overlap

    pipeline = RAGPipeline(config=config)

    src_path = Path(args.source)
    if not src_path.is_absolute():
        src_path = PROJECT_ROOT / src_path

    if src_path.is_file():
        documents = [DocumentLoader.load_file(str(src_path))]
    elif src_path.is_dir():
        documents = DocumentLoader.load_directory(str(src_path))
    else:
        console.print(f"[bold red]Error:[/bold red] Source path '{src_path}' does not exist.")
        sys.exit(1)

    console.print(f"Loaded [bold]{len(documents)}[/bold] document(s). Processing chunking & embeddings...")
    num_chunks = pipeline.ingest_documents(documents, persist=True)

    table = Table(title="Ingestion Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Documents", str(len(documents)))
    table.add_row("Total Chunks Created", str(num_chunks))
    table.add_row("Chunking Strategy", args.strategy)
    table.add_row("Embedding Model", config.embeddings.model_name)
    table.add_row("Index Directory", config.storage.index_dir)

    console.print(table)
    console.print("[bold green][OK] Ingestion and indexing completed successfully![/bold green]")


if __name__ == "__main__":
    main()
