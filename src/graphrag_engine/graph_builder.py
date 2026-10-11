"""Code Knowledge Graph (CKG): NetworkX multigraph with symbol resolution, BFS and centrality."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx

from graphrag_engine.ast_parser import EXTERNAL_PREFIX
from graphrag_engine.models import Edge, Node, NodeType, ParsedFile, Relation

_TYPE_TARGETS: dict[Relation, set[NodeType]] = {
    Relation.CALLS: {NodeType.FUNCTION, NodeType.CLASS},
    Relation.INHERITS_FROM: {NodeType.CLASS},
    Relation.USES_TYPE: {NodeType.CLASS},
}
_MAX_AMBIGUOUS = 3


class CodeKnowledgeGraph:
    """Directed multi-graph G = (V, E, tau_V, tau_E) over code entities."""

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph[str] = nx.MultiDiGraph()
        self._raw_edges: list[Edge] = []
        self._pagerank: dict[str, float] | None = None

    # -- construction ---------------------------------------------------
    def add_parsed_file(self, parsed: ParsedFile) -> None:
        for n in parsed.nodes:
            if n.id not in self.graph:
                self._add_node(n)
        self._raw_edges.extend(parsed.edges)
        self._pagerank = None

    def _add_node(self, n: Node) -> None:
        self.graph.add_node(
            n.id,
            node_type=str(n.type),
            name=n.name,
            code=n.code,
            file=n.file,
            start_line=n.start_line,
            end_line=n.end_line,
            docstring=n.docstring,
        )

    def build_graph_from_ast(self, ast_data: dict[str, Any]) -> None:
        """Compatibility entry point for the dict format returned by ``parse_to_dict``."""
        parsed = ParsedFile(
            nodes=[
                Node(
                    id=d["id"],
                    type=NodeType(d["type"]),
                    name=d["name"],
                    code=d.get("code", ""),
                    file=d.get("file", ""),
                    start_line=d.get("start_line", 0),
                    end_line=d.get("end_line", 0),
                    docstring=d.get("docstring", ""),
                )
                for d in ast_data["nodes"]
            ],
            edges=[
                Edge(d["source"], d["target"], Relation(d["relation"])) for d in ast_data["edges"]
            ],
        )
        self.add_parsed_file(parsed)
        self.resolve()

    @classmethod
    def from_parsed_files(cls, files: Iterable[ParsedFile]) -> CodeKnowledgeGraph:
        ckg = cls()
        for p in files:
            ckg.add_parsed_file(p)
        ckg.resolve()
        return ckg

    def resolve(self) -> None:
        """Resolve symbolic edge targets (calls, bases, types, imports) to defined nodes."""
        by_name: dict[str, list[str]] = defaultdict(list)
        file_ids: list[str] = []
        for nid, attr in self.graph.nodes(data=True):
            t = attr.get("node_type")
            if t in (NodeType.FUNCTION, NodeType.CLASS):
                by_name[attr["name"]].append(nid)
            elif t == NodeType.FILE:
                file_ids.append(nid)

        self.graph.remove_edges_from(list(self.graph.edges(keys=True)))
        for nid in [n for n, a in self.graph.nodes(data=True) if a.get("node_type") == "External"]:
            self.graph.remove_node(nid)

        for e in self._raw_edges:
            if e.source not in self.graph:
                continue
            if not e.target.startswith(EXTERNAL_PREFIX):
                if e.target in self.graph:
                    self.graph.add_edge(e.source, e.target, relation=str(e.relation))
                continue
            raw = e.target[len(EXTERNAL_PREFIX) :]
            targets = (
                self._resolve_import(raw, file_ids)
                if e.relation == Relation.IMPORTS
                else self._resolve_symbol(e, raw, by_name)
            )
            if not targets:
                ext_id = f"{EXTERNAL_PREFIX}{raw}"
                if ext_id not in self.graph:
                    self._add_node(Node(id=ext_id, type=NodeType.EXTERNAL, name=raw))
                targets = [ext_id]
            for t in targets:
                if t != e.source:
                    self.graph.add_edge(e.source, t, relation=str(e.relation))
        self._pagerank = None

    def _resolve_symbol(self, e: Edge, raw: str, by_name: dict[str, list[str]]) -> list[str]:
        simple = raw.split(".")[-1]
        wanted = _TYPE_TARGETS.get(e.relation, set())
        cands = [c for c in by_name.get(simple, []) if self.graph.nodes[c]["node_type"] in wanted]
        if not cands:
            return []
        src_file = self.graph.nodes[e.source].get("file", "")
        same_file = [c for c in cands if self.graph.nodes[c].get("file") == src_file]
        if same_file:
            return same_file[:1]
        return cands if len(cands) <= _MAX_AMBIGUOUS else []

    @staticmethod
    def _resolve_import(raw: str, file_ids: list[str]) -> list[str]:
        mod = raw.removeprefix("module:").lstrip(".")
        if not mod:
            return []
        rel = mod.replace(".", "/")
        suffixes = (f"{rel}.py", f"{rel}/__init__.py")
        hits = [f for f in file_ids if any(f == s or f.endswith("/" + s) for s in suffixes)]
        return sorted(hits, key=len)[:1]

    # -- queries --------------------------------------------------------
    def nodes_of_type(self, *types: NodeType) -> list[str]:
        wanted = {str(t) for t in types}
        return [n for n, a in self.graph.nodes(data=True) if a.get("node_type") in wanted]

    def node_info(self, node_id: str) -> dict[str, Any]:
        attr = self.graph.nodes[node_id]
        return {
            "id": node_id,
            "type": attr.get("node_type"),
            "name": attr.get("name"),
            "code": attr.get("code", ""),
            "file": attr.get("file", ""),
            "start_line": attr.get("start_line", 0),
            "end_line": attr.get("end_line", 0),
            "docstring": attr.get("docstring", ""),
        }

    def k_hop_neighborhood(
        self, seed_nodes: list[str], k_hops: int = 2, directed: bool = False
    ) -> dict[str, int]:
        """BFS to depth K. Returns {node_id: hop_distance} (seeds at 0)."""
        dist: dict[str, int] = {}
        queue: deque[str] = deque()
        for s in seed_nodes:
            if s in self.graph and s not in dist:
                dist[s] = 0
                queue.append(s)
        while queue:
            cur = queue.popleft()
            if dist[cur] >= k_hops:
                continue
            neighbors: Iterable[str] = self.graph.successors(cur)
            if not directed:
                neighbors = [*neighbors, *self.graph.predecessors(cur)]
            for nb in neighbors:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        return dist

    def get_k_hop_neighborhood(self, seed_nodes: list[str], k_hops: int = 2) -> list[str]:
        return list(self.k_hop_neighborhood(seed_nodes, k_hops))

    def pagerank(self) -> dict[str, float]:
        if self._pagerank is None:
            if self.graph.number_of_nodes() == 0:
                self._pagerank = {}
            else:
                simple = nx.DiGraph(self.graph)
                self._pagerank = nx.pagerank(simple, alpha=0.85)
        return self._pagerank

    def degree_centrality(self) -> dict[str, float]:
        return dict(nx.degree_centrality(nx.DiGraph(self.graph)))

    # -- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [{"id": n, **a} for n, a in self.graph.nodes(data=True)],
            "edges": [
                {"source": u, "target": v, "relation": d["relation"]}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> CodeKnowledgeGraph:
        data = json.loads(path.read_text())
        ckg = cls()
        for n in data["nodes"]:
            nid = n.pop("id")
            ckg.graph.add_node(nid, **n)
        for e in data["edges"]:
            ckg.graph.add_edge(e["source"], e["target"], relation=e["relation"])
        return ckg
