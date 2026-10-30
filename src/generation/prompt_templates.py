"""Standardized prompt templates with strict citation constraints and anti-hallucination guardrails."""

from typing import List
from ..core.models import RetrievalResult, Chunk


class RAGPromptTemplate:
    """Formats retrieved context and instructions for strict grounded generation."""

    SYSTEM_PROMPT = """You are an expert, highly reliable factual AI assistant.
Answer the user's question STRICTLY based on the provided retrieved context below.

CRITICAL INSTRUCTIONS:
1. ONLY make claims directly supported by the context snippets.
2. For EVERY key factual claim, append a citation referencing the source snippet number, formatted as [Source X].
3. If the provided context does NOT contain sufficient information to answer the question faithfully, state clearly: "Based on the provided documents, I do not have enough information to answer this question." Do NOT fabricate facts.
4. Keep answers concise, objective, and well-structured."""

    HYDE_PROMPT = """Write a hypothetical, authoritative passage that directly answers the following user query.
The passage does not need to be factually verified; it will be used as a dense retrieval anchor to search our document database.

Query: {query}

Hypothetical Document:"""

    MULTI_QUERY_PROMPT = """You are an AI language model assistant. Your task is to generate 3 different versions of the given user query to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.
Provide these alternative questions separated by newlines. Do not number them.

Original question: {query}"""

    ROUTER_PROMPT = """Analyze the following user input and classify it into exactly one category:
- RETRIEVE: Specific domain questions, factual inquiries, or requests requiring document lookup.
- CONVERSATIONAL: Greetings, pleasantries, or general chit-chat.
- DIRECT: General knowledge, definitions, or simple math that do not require specialized external documents.

User Query: "{query}"
Category (RETRIEVE / CONVERSATIONAL / DIRECT):"""

    @classmethod
    def format_rag_prompt(cls, query: str, results: List[RetrievalResult]) -> str:
        """Format retrieved chunks into indexed context blocks."""
        context_blocks = []
        for i, res in enumerate(results, start=1):
            source_file = res.chunk.metadata.get("filename", "document")
            chunk_id = res.chunk.chunk_id[:8]
            block = f"[Source {i}] (File: {source_file}, Chunk: {chunk_id}):\n{res.chunk.text.strip()}"
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        return f"""Retrieved Context:
---------------------
{context_str}
---------------------

User Question: {query}

Helpful, Grounded Answer (with [Source X] citations):"""
