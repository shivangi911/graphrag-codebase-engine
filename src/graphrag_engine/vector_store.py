"""Qdrant-backed vector index over code nodes (embedded local mode by default)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from graphrag_engine.embedder import Embedder, node_text
from graphrag_engine.graph_builder import CodeKnowledgeGraph
from graphrag_engine.models import NodeType


class VectorStore:
    def __init__(
        self,
        embedder: Embedder,
        collection: str = "code_nodes",
        path: Path | None = None,
        url: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.collection = collection
        self._vecs: dict[str, NDArray[np.float32]] = {}
        if url:
            self.client = QdrantClient(url=url)
        elif path is not None:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(":memory:")

    def index_graph(self, ckg: CodeKnowledgeGraph, batch_size: int = 64) -> int:
        ids = ckg.nodes_of_type(NodeType.FUNCTION, NodeType.CLASS)
        if not ids:
            return 0
        infos = [ckg.node_info(i) for i in ids]
        first = self.embedder.embed([node_text(infos[0])])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=int(first.shape[1]), distance=Distance.COSINE),
        )
        for start in range(0, len(infos), batch_size):
            chunk = infos[start : start + batch_size]
            vecs = self.embedder.embed([node_text(c) for c in chunk])
            for j, c in enumerate(chunk):
                self._vecs[str(c["id"])] = vecs[j]
            points = [
                PointStruct(
                    id=start + j,
                    vector=vecs[j].tolist(),
                    payload={"node_id": chunk[j]["id"], "name": chunk[j]["name"]},
                )
                for j in range(len(chunk))
            ]
            self.client.upsert(self.collection, points=points)
        return len(infos)

    def load_cache(self) -> int:
        """Reload node vectors from a persisted collection into the in-memory cache."""
        if not self.client.collection_exists(self.collection):
            return 0
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_vectors=True, with_payload=True
            )
            for p in points:
                vec = p.vector
                if isinstance(vec, list):
                    self._vecs[str((p.payload or {})["node_id"])] = np.asarray(
                        vec, dtype=np.float32
                    )
            if offset is None:
                break
        return len(self._vecs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.client.collection_exists(self.collection):
            return []
        vec = self.embedder.embed([query])[0].tolist()
        res = self.client.query_points(self.collection, query=vec, limit=top_k).points
        return [(str((p.payload or {})["node_id"]), float(p.score)) for p in res]

    def score_nodes(
        self, query: str, node_ids: list[str], ckg: CodeKnowledgeGraph
    ) -> dict[str, float]:
        """Cosine similarity of the query to arbitrary nodes (cached vectors, else embedded)."""
        if not node_ids:
            return {}
        q = self.embedder.embed([query])[0]
        missing = [n for n in node_ids if n not in self._vecs]
        if missing:
            mat = self.embedder.embed([node_text(ckg.node_info(n)) for n in missing])
            for i, n in enumerate(missing):
                self._vecs[n] = mat[i]
        return {n: float(np.dot(self._vecs[n], q)) for n in node_ids}
