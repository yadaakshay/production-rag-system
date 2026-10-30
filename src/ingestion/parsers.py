"""Document parsers for extracting structured text and metadata from files."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import re
from ..core.models import Document


class BaseParser(ABC):
    """Abstract base class for all file parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> Document:
        """Parse file content and return a Document instance with metadata."""
        pass


class TextParser(BaseParser):
    """Parses raw plain text files (.txt)."""

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        metadata = {
            "source": str(path.resolve()),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
        }
        return Document(text=text, metadata=metadata)


class MarkdownParser(BaseParser):
    """Parses Markdown files (.md) with header and structural extraction."""

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        # Extract markdown headers to understand document structure
        headers = re.findall(r"^(#{1,6})\s+(.*)$", text, flags=re.MULTILINE)
        extracted_headers = [{"level": len(h[0]), "title": h[1].strip()} for h in headers]

        metadata = {
            "source": str(path.resolve()),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "headers": extracted_headers,
            "file_size_bytes": path.stat().st_size,
        }
        return Document(text=text, metadata=metadata)


class PDFParser(BaseParser):
    """Parses PDF documents using pypdf."""

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages_text = []
            for i, page in enumerate(reader.pages):
                page_content = page.extract_text() or ""
                pages_text.append(f"[Page {i + 1}]\n{page_content}")

            combined_text = "\n\n".join(pages_text)
            metadata = {
                "source": str(path.resolve()),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "total_pages": len(reader.pages),
                "file_size_bytes": path.stat().st_size,
            }
            return Document(text=combined_text, metadata=metadata)
        except Exception as e:
            # Fallback for plain text or failure
            return Document(
                text=f"Failed to parse PDF {path.name}: {str(e)}",
                metadata={"source": str(path.resolve()), "error": str(e)},
            )


class DocumentLoader:
    """Multi-format document loader with recursive directory scanning."""

    PARSERS = {
        ".txt": TextParser,
        ".md": MarkdownParser,
        ".pdf": PDFParser,
    }

    @classmethod
    def load_file(cls, file_path: str) -> Document:
        path = Path(file_path)
        suffix = path.suffix.lower()
        parser_cls = cls.PARSERS.get(suffix, TextParser)
        parser = parser_cls()
        return parser.parse(str(path))

    @classmethod
    def load_directory(cls, dir_path: str, recursive: bool = True) -> List[Document]:
        directory = Path(dir_path)
        if not directory.exists() or not directory.is_dir():
            return []

        pattern = "**/*" if recursive else "*"
        documents = []
        for path in directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in cls.PARSERS:
                try:
                    doc = cls.load_file(str(path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")
        return documents
