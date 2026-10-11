"""LLM client abstraction: Ollama for real runs, a scripted fake for tests and offline demos."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from graphrag_engine.config import GraphRAGSettings, settings


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class FakeLLM:
    """Returns scripted responses in order; repeats the last one when exhausted."""

    def __init__(self, responses: Sequence[str] | None = None) -> None:
        self.responses = list(
            responses
            or [
                "1. Review the retrieved functions.\n2. Apply the smallest safe change.",
                "```python\ndef refactored_function() -> None:\n    return None\n```",
            ]
        )
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]


class OllamaLLM:
    def __init__(self, model: str, host: str) -> None:
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )
        return str(resp["message"]["content"])


def get_llm(cfg: GraphRAGSettings | None = None, backend: str | None = None) -> LLMClient:
    cfg = cfg or settings
    backend = backend or cfg.llm_backend
    if backend == "ollama":
        return OllamaLLM(cfg.llm_model, cfg.ollama_host)
    return FakeLLM()


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the first fenced code block, or the raw text if none is present."""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip() + "\n"
