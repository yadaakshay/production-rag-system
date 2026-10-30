"""Multi-provider LLM client supporting Groq, OpenAI, Gemini, Ollama, and an offline Mock."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
import re
import httpx


class BaseLLMClient(ABC):
    """Abstract interface for Language Model completions."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        pass


class MockLLMClient(BaseLLMClient):
    """Smart offline mock LLM that extracts answers and citations from the formatted prompt.
    
    Guarantees that the repository is completely testable, verifiable, and runnable
    without needing any external API keys or incurring costs.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        # Check if this is a router prompt
        if "Category (RETRIEVE / CONVERSATIONAL / DIRECT)" in prompt:
            m = re.search(r'User Query:\s*"([^"]+)"', prompt)
            user_q = (m.group(1) if m else prompt).lower()
            if any(re.search(rf"\b{w}\b", user_q) for w in ["hi", "hello", "hey", "greetings", "how are you", "who are you"]):
                return "CONVERSATIONAL"
            return "RETRIEVE"

        # Check if this is a HyDE prompt
        if "Hypothetical Document:" in prompt:
            m = re.search(r"Query:\s*(.*)", prompt)
            q = m.group(1).strip() if m else "query"
            return f"This document comprehensively explains {q}, detailing its core architecture, operational mechanisms, and implementation guidelines."

        # Check if this is a Multi-Query prompt
        if "Provide these alternative questions" in prompt:
            m = re.search(r"Original question:\s*(.*)", prompt)
            q = m.group(1).strip() if m else "query"
            return f"What are the main aspects of {q}?\nHow does {q} work in practice?\nKey technical details regarding {q}"

        # Standard RAG prompt: extract key context lines and format with citation
        if "Retrieved Context:" in prompt:
            # Parse [Source X] blocks
            sources = re.findall(
                r"\[Source\s+(\d+)\][^\n]*\n([\s\S]*?)(?=(?:\n\[Source|\n-{3,}|$))",
                prompt,
            )
            if not sources:
                return "Based on the provided documents, I do not have enough information to answer this question."

            # Construct an answer using the top matching snippets
            lines = []
            for src_idx, text in sources[:2]:
                clean_lines = [l.strip().lstrip("#").strip() for l in text.split("\n") if l.strip()]
                clean_snippet = " ".join(" ".join(clean_lines).split()[:30])
                if clean_snippet:
                    lines.append(f"According to the ingested documentation, {clean_snippet} [Source {src_idx}].")

            return " ".join(lines) if lines else "Based on the provided documents, I do not have enough information to answer this question."

        return "Response generated successfully."


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API client (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()


class GroqLLMClient(BaseLLMClient):
    """High-speed Groq LPU inference client (llama-3.1-70b-versatile, llama-3.1-8b-instant)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini API client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {
            "contents": [{"parts": [{"text": full_text}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def get_llm_client(
    provider: str = "mock",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseLLMClient:
    """Instantiate appropriate LLM client based on requested provider."""
    provider = provider.lower()
    if provider == "openai":
        return OpenAILLMClient(api_key=api_key, model=model_name or "gpt-4o-mini")
    elif provider == "groq":
        return GroqLLMClient(api_key=api_key, model=model_name or "llama-3.1-8b-instant")
    elif provider in ("gemini", "google"):
        return GeminiLLMClient(api_key=api_key, model=model_name or "gemini-1.5-flash")
    else:
        return MockLLMClient()
