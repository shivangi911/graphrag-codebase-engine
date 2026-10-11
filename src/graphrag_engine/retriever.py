"""Hybrid retrieval: dense vector seeds, K-hop BFS expansion, Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Any

from graphrag_engine.config import GraphRAGSettings, settings
from graphrag_engine.graph_builder import CodeKnowledgeGraph
from graphrag_engine.models import NodeType
from graphrag_engine.vector_store import VectorStore


def ranks_from_scores(scores: dict[str, float]) -> dict[str, int]:
    """1-based ranks, highest score first (ties broken by id for determinism)."""
    ordered = sorted(scores, key=lambda n: (-scores[n], n))
    return {n: i + 1 for i, n in enumerate(ordered)}


def rrf_fuse(rank_maps: list[dict[str, int]], k: int = 60) -> dict[str, float]:
    """RRF(v) = sum_i 1 / (k + r_i(v)). Items missing from a list contribute nothing."""
    fused: dict[str, float] = {}
    for ranks in rank_maps:
        for node, r in ranks.items():
            fused[node] = fused.get(node, 0.0) + 1.0 / (k + r)
    return fused


class HybridGraphRAGRetriever:
    def __init__(
        self,
        ckg: CodeKnowledgeGraph,
        store: VectorStore,
        cfg: GraphRAGSettings | None = None,
        graph_signal: str = "pagerank",
    ) -> None:
        self.ckg = ckg
        self.store = store
        self.cfg = cfg or settings
        self.graph_signal = graph_signal

    def _graph_scores(self, node_ids: list[str]) -> dict[str, float]:
        base = (
            self.ckg.pagerank() if self.graph_signal == "pagerank" else self.ckg.degree_centrality()
        )
        return {n: base.get(n, 0.0) for n in node_ids}

    def retrieve_vector_only(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Baseline: dense retrieval only."""
        k = top_k or self.cfg.max_context_nodes
        out = []
        for rank, (nid, score) in enumerate(self.store.search(query, k), start=1):
            info = self.ckg.node_info(nid)
            info.update(score=score, dense_score=score, hops=0, rank=rank)
            out.append(info)
        return out

    def retrieve_context(self, query: str) -> list[dict[str, Any]]:
        seeds = self.store.search(query, self.cfg.top_k_vector)
        if not seeds:
            return []
        seed_ids = [s for s, _ in seeds]
        hops = self.ckg.k_hop_neighborhood(seed_ids, self.cfg.bfs_max_hops)
        allowed = {str(NodeType.FUNCTION), str(NodeType.CLASS)}
        candidates = [n for n in hops if self.ckg.graph.nodes[n].get("node_type") in allowed]

        dense = self.store.score_nodes(query, candidates, self.ckg)
        fused = rrf_fuse(
            [ranks_from_scores(dense), ranks_from_scores(self._graph_scores(candidates))],
            self.cfg.rrf_k_constant,
        )
        ordered = sorted(fused, key=lambda n: (-fused[n], n))[: self.cfg.max_context_nodes]

        out = []
        for rank, nid in enumerate(ordered, start=1):
            info = self.ckg.node_info(nid)
            info.update(score=fused[nid], dense_score=dense[nid], hops=hops[nid], rank=rank)
            out.append(info)
        return out
