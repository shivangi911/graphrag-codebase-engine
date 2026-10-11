from pathlib import Path

from graphrag_engine.ast_parser import CodeASTParser
from graphrag_engine.graph_builder import CodeKnowledgeGraph


def build(files: dict[str, str]) -> CodeKnowledgeGraph:
    parser = CodeASTParser()
    return CodeKnowledgeGraph.from_parsed_files(
        parser.parse_source_code(p, s) for p, s in files.items()
    )


FILES = {
    "pkg/a.py": "from pkg.b import g\n\ndef f():\n    return g()\n",
    "pkg/b.py": "class Base: pass\n\ndef g():\n    return 1\n\ndef lonely():\n    return 2\n",
    "pkg/c.py": (
        "from pkg.b import Base\n\nclass K(Base):\n    def m(self):\n        return missing()\n"
    ),
}


def relations(ckg, src, dst):
    return [d["relation"] for _, v, d in ckg.graph.out_edges(src, data=True) if v == dst]


def test_cross_file_resolution():
    ckg = build(FILES)
    assert relations(ckg, "pkg/a.py::f", "pkg/b.py::g") == ["CALLS"]
    assert relations(ckg, "pkg/c.py::K", "pkg/b.py::Base") == ["INHERITS_FROM"]
    assert relations(ckg, "pkg/a.py", "pkg/b.py") == ["IMPORTS"]


def test_unresolved_becomes_external():
    ckg = build(FILES)
    assert "ext::missing" in ckg.graph
    assert ckg.graph.nodes["ext::missing"]["node_type"] == "External"


def test_resolve_is_idempotent():
    ckg = build(FILES)
    n_edges = ckg.graph.number_of_edges()
    ckg.resolve()
    assert ckg.graph.number_of_edges() == n_edges


def test_k_hop_bfs_depths():
    ckg = build(FILES)
    d1 = ckg.k_hop_neighborhood(["pkg/a.py::f"], 1)
    assert d1["pkg/a.py::f"] == 0
    assert d1["pkg/b.py::g"] == 1
    assert "pkg/b.py" not in d1  # two hops away
    d2 = ckg.k_hop_neighborhood(["pkg/a.py::f"], 2)
    assert d2["pkg/b.py"] == 2
    assert "pkg/b.py::lonely" not in d1


def test_directed_bfs_ignores_callers():
    ckg = build(FILES)
    undirected = ckg.k_hop_neighborhood(["pkg/b.py::g"], 1)
    directed = ckg.k_hop_neighborhood(["pkg/b.py::g"], 1, directed=True)
    assert "pkg/a.py::f" in undirected
    assert "pkg/a.py::f" not in directed


def test_bfs_unknown_seed_is_ignored():
    assert build(FILES).k_hop_neighborhood(["nope"], 2) == {}


def test_pagerank_sums_to_one_and_hub_ranks_high():
    ckg = build(FILES)
    pr = ckg.pagerank()
    assert abs(sum(pr.values()) - 1.0) < 1e-6
    assert pr["pkg/b.py::g"] > pr["pkg/b.py::lonely"]


def test_save_load_roundtrip(tmp_path: Path):
    ckg = build(FILES)
    path = tmp_path / "g.json"
    ckg.save(path)
    loaded = CodeKnowledgeGraph.load(path)
    assert set(loaded.graph.nodes) == set(ckg.graph.nodes)
    assert loaded.graph.number_of_edges() == ckg.graph.number_of_edges()


def test_build_graph_from_ast_dict_compat():
    ckg = CodeKnowledgeGraph()
    ckg.build_graph_from_ast(
        CodeASTParser().parse_to_dict("x.py", "def a():\n    b()\ndef b(): pass")
    )
    assert relations(ckg, "x.py::a", "x.py::b") == ["CALLS"]
