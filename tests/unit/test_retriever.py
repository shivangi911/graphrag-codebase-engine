import numpy as np
import pytest

from graphrag_engine.config import GraphRAGSettings
from graphrag_engine.embedder import HashEmbedder, tokenize_identifiers
from graphrag_engine.ingest import build_index
from graphrag_engine.retriever import HybridGraphRAGRetriever, ranks_from_scores, rrf_fuse


def test_tokenize_identifiers():
    assert tokenize_identifiers("verifyToken_jwt2HTTPServer") == [
        "verify", "token", "jwt", "2", "http", "server",
    ]  # fmt: skip


def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder()
    a = e.embed(["decode jwt token"])
    b = e.embed(["decode jwt token"])
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_hash_embedder_similar_text_is_closer():
    e = HashEmbedder()
    q, near, far = e.embed(
        ["parse json config file", "load_json_config_file", "render opengl mesh"]
    )
    assert float(q @ near) > float(q @ far)


def test_ranks_from_scores():
    assert ranks_from_scores({"a": 0.1, "b": 0.9, "c": 0.5}) == {"b": 1, "c": 2, "a": 3}


def test_rrf_matches_hand_computation():
    fused = rrf_fuse([{"x": 1, "y": 2}, {"y": 1, "x": 3}], k=60)
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["y"] > fused["x"]


def test_rrf_missing_from_one_list():
    fused = rrf_fuse([{"x": 1}, {"y": 1}], k=60)
    assert fused["x"] == fused["y"] == pytest.approx(1 / 61)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def decode_jwt(token):\n    return {}\n\n"
        "def verify_token(token):\n    return decode_jwt(token)\n"
    )
    (tmp_path / "render.py").write_text("def draw_mesh(m):\n    return m\n")
    return tmp_path


def make_retriever(repo, **overrides):
    cfg = GraphRAGSettings(top_k_vector=1, bfs_max_hops=2, **overrides)
    ckg, store = build_index(repo, HashEmbedder(), cfg)
    return HybridGraphRAGRetriever(ckg, store, cfg)


def test_retrieval_uses_query_and_expands_graph(repo):
    r = make_retriever(repo)
    ids = [n["id"] for n in r.retrieve_context("verify token")]
    assert "auth.py::verify_token" in ids
    assert "auth.py::decode_jwt" in ids  # reached by BFS, not by vector seed
    assert "render.py::draw_mesh" not in ids
    hops = {n["id"]: n["hops"] for n in r.retrieve_context("verify token")}
    assert hops["auth.py::decode_jwt"] >= 1


def test_vector_only_baseline_has_no_expansion(repo):
    r = make_retriever(repo)
    out = r.retrieve_vector_only("verify token", top_k=1)
    assert [n["id"] for n in out] == ["auth.py::verify_token"]


def test_empty_graph_returns_nothing(tmp_path):
    r = make_retriever(tmp_path)
    assert r.retrieve_context("anything") == []


def test_max_context_nodes_respected(repo):
    r = make_retriever(repo, max_context_nodes=1)
    assert len(r.retrieve_context("verify token")) == 1
