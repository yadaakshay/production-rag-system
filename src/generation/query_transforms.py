"""Query rewriting, expansion, and intent routing strategies.

Techniques:
1. HyDE (Hypothetical Document Embeddings - Gao et al., 2022): Generates a fake answer passage, embeds it to retrieve semantically close true passages.
2. Multi-Query Expansion: Explores vocabulary variations to overcome lexical mismatches.
3. Query Router: Bypasses unnecessary retrieval for conversational inputs.
"""

from typing import List, Tuple
from abc import ABC, abstractmethod
from ..core.models import QueryTransformationResult


class QueryTransformer(ABC):
    @abstractmethod
    def transform(self, query: str) -> QueryTransformationResult:
        pass


class HyDETransformer(QueryTransformer):
    """Generates a hypothetical answer passage using an LLM to guide dense retrieval."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def transform(self, query: str) -> QueryTransformationResult:
        from .prompt_templates import RAGPromptTemplate
        prompt = RAGPromptTemplate.HYDE_PROMPT.format(query=query)
        hypothetical_doc = self.llm.generate(prompt, temperature=0.3, max_tokens=256)
        return QueryTransformationResult(
            original_query=query,
            transformed_queries=[hypothetical_doc],
            transformation_type="hyde",
            metadata={"hypothetical_document": hypothetical_doc},
        )


class MultiQueryExpander(QueryTransformer):
    """Expands a single user question into multiple distinct query variations."""

    def __init__(self, llm_client, num_queries: int = 3):
        self.llm = llm_client
        self.num_queries = num_queries

    def transform(self, query: str) -> QueryTransformationResult:
        from .prompt_templates import RAGPromptTemplate
        prompt = RAGPromptTemplate.MULTI_QUERY_PROMPT.format(query=query)
        response = self.llm.generate(prompt, temperature=0.5, max_tokens=256)
        variations = [line.strip("- *").strip() for line in response.split("\n") if line.strip()]
        queries = [query] + variations[: self.num_queries]

        return QueryTransformationResult(
            original_query=query,
            transformed_queries=queries,
            transformation_type="multi_query",
            metadata={"variations_count": len(queries)},
        )


class QueryRouter:
    """Classifies user queries to decide whether information retrieval is required."""

    def __init__(self, llm_client = None):
        self.llm = llm_client

    def route(self, query: str) -> Tuple[str, str]:
        """Returns tuple (category, explanation): RETRIEVE, CONVERSATIONAL, or DIRECT."""
        lower_q = query.strip().lower()
        # Rule-based fast paths
        if lower_q in ["hi", "hello", "hey", "good morning", "good evening", "how are you", "who are you"]:
            return "CONVERSATIONAL", "User greeted or initiated conversation."
        if lower_q.startswith("what is 2 +") or lower_q == "help":
            return "DIRECT", "Standard direct query."

        if self.llm is not None:
            from .prompt_templates import RAGPromptTemplate
            prompt = RAGPromptTemplate.ROUTER_PROMPT.format(query=query)
            decision = self.llm.generate(prompt, temperature=0.0, max_tokens=20).strip().upper()
            if "CONVERSATIONAL" in decision:
                return "CONVERSATIONAL", "Classified by LLM as conversational."
            if "DIRECT" in decision:
                return "DIRECT", "Classified by LLM as direct reasoning."
            return "RETRIEVE", "Classified by LLM as requiring retrieval."

        return "RETRIEVE", "Default to retrieval."
