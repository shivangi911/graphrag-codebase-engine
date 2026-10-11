"""Pydantic settings for paths, retrieval and agent parameters."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GraphRAGSettings(BaseSettings):
    app_name: str = "GraphRAG-Codebase-Engine"
    data_dir: Path = Field(default=Path("data"))

    # Vector store
    qdrant_collection: str = "code_nodes"
    qdrant_url: str | None = None  # None -> embedded local mode

    # Embeddings: auto | ollama | sbert | hash
    embedding_backend: str = "auto"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"

    # Retrieval
    top_k_vector: int = 5
    bfs_max_hops: int = 2
    rrf_k_constant: int = 60
    max_context_nodes: int = 20

    # Agent
    llm_backend: str = "fake"  # fake | ollama
    llm_model: str = "qwen2.5-coder:3b"
    max_refactor_attempts: int = 3

    # Sandbox: auto | docker | subprocess
    sandbox_backend: str = "auto"
    sandbox_timeout_s: int = 60
    sandbox_image: str = (
        "graphrag-sandbox:latest"  # build: docker build -f docker/sandbox.Dockerfile
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def graph_path(self) -> Path:
        return self.data_dir / "graph_store.json"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "index_meta.json"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"


settings = GraphRAGSettings()
