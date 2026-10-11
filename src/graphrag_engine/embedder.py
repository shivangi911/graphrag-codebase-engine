"""Pluggable text embedders: Ollama, sentence-transformers, and a deterministic hash fallback."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from graphrag_engine.config import GraphRAGSettings, settings

log = logging.getLogger(__name__)
Matrix = NDArray[np.float32]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> Matrix: ...


def _normalize(m: Matrix) -> Matrix:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Matrix = (m / norms).astype(np.float32)
    return out


_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize_identifiers(text: str) -> list[str]:
    """Split text into lowercase sub-tokens (snake_case, camelCase, digits)."""
    return [t.lower() for t in _SPLIT.findall(text)]


class HashEmbedder:
    """Deterministic signed feature-hashing bag-of-subtokens. Offline, dependency-free."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: list[str]) -> Matrix:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = tokenize_identifiers(text)
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            for g in grams:
                h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[i, h % self.dim] += sign
        return _normalize(out)


class SBertEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = f"sbert-{model_name}"

    def embed(self, texts: list[str]) -> Matrix:
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class OllamaEmbedder:
    def __init__(self, model: str, host: str) -> None:
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model
        self.name = f"ollama-{model}"
        self.client.show(model)  # raises if server/model unavailable

    def embed(self, texts: list[str]) -> Matrix:
        resp = self.client.embed(model=self.model, input=texts)
        return _normalize(np.asarray(resp["embeddings"], dtype=np.float32))


def get_embedder(cfg: GraphRAGSettings | None = None, backend: str | None = None) -> Embedder:
    cfg = cfg or settings
    backend = backend or cfg.embedding_backend
    if backend == "hash":
        return HashEmbedder()
    if backend == "sbert":
        return SBertEmbedder(cfg.embedding_model)
    if backend == "ollama":
        return OllamaEmbedder(cfg.ollama_embedding_model, cfg.ollama_host)
    # auto
    for factory in (
        lambda: OllamaEmbedder(cfg.ollama_embedding_model, cfg.ollama_host),
        lambda: SBertEmbedder(cfg.embedding_model),
    ):
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001
            log.info("embedder unavailable (%s), trying next", exc)
    log.warning("falling back to hash embedder")
    return HashEmbedder()


def node_text(info: dict[str, object]) -> str:
    """Text representation of a graph node used for embedding."""
    code = str(info.get("code", ""))[:1500]
    return f"{info.get('type')} {info.get('name')}\n{info.get('docstring', '')}\n{code}"
