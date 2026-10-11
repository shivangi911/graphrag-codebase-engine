from pathlib import Path

from graphrag_engine.config import GraphRAGSettings
from graphrag_engine.embedder import HashEmbedder
from graphrag_engine.evaluation import evaluate_instances
from graphrag_engine.ingest import SWEInstance, build_index, load_index
from graphrag_engine.llm import FakeLLM
from graphrag_engine.orchestrator import CodeGraphOrchestrator
from graphrag_engine.retriever import HybridGraphRAGRetriever
from graphrag_engine.sandbox import SubprocessSandbox


def make_repo(root: Path) -> Path:
    (root / "app").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".git" / "ignored.py").write_text("def hidden(): pass\n")
    (root / "app" / "auth.py").write_text(
        "from app.tokens import decode_jwt\n\ndef verify_token(t):\n    return decode_jwt(t)\n"
    )
    (root / "app" / "tokens.py").write_text("def decode_jwt(t):\n    return {'user': t}\n")
    (root / "app" / "render.py").write_text("def draw_mesh(m):\n    return m\n")
    return root


def test_index_persist_load_and_query(tmp_path):
    repo = make_repo(tmp_path / "repo")
    cfg = GraphRAGSettings(data_dir=tmp_path / "data", top_k_vector=1)
    emb = HashEmbedder()
    ckg, store = build_index(repo, emb, cfg, persist=True)
    assert "hidden" not in " ".join(ckg.graph.nodes)
    del store  # release embedded qdrant lock

    ckg2, store2 = load_index(emb, cfg)
    r = HybridGraphRAGRetriever(ckg2, store2, cfg)
    ids = [n["id"] for n in r.retrieve_context("verify token")]
    assert "app/auth.py::verify_token" in ids
    assert "app/tokens.py::decode_jwt" in ids


def test_retrieval_to_agent_pipeline(tmp_path):
    repo = make_repo(tmp_path / "repo")
    cfg = GraphRAGSettings(top_k_vector=1)
    ckg, store = build_index(repo, HashEmbedder(), cfg)
    llm = FakeLLM(["plan", "```python\ndef verify(t: str) -> bool:\n    return bool(t)\n```"])
    orch = CodeGraphOrchestrator(
        HybridGraphRAGRetriever(ckg, store, cfg), llm, SubprocessSandbox(120)
    )
    state = orch.run_refactor_pipeline("simplify verify_token")
    assert state.verification_passed
    assert any(c["id"] == "app/auth.py::verify_token" for c in state.retrieved_context)


def test_eval_graphrag_beats_or_matches_vector_on_toy_instance(tmp_path):
    repo = make_repo(tmp_path / "repo")
    patch = "diff --git a/app/tokens.py b/app/tokens.py\n@@ -1,2 +1,2 @@ def decode_jwt(t):\n+x\n"
    inst = SWEInstance("toy-1", "o/r", "abc", "verify_token fails for expired token", patch)
    cfg = GraphRAGSettings(top_k_vector=1)
    report = evaluate_instances(
        [inst], HashEmbedder(), tmp_path, ks=(1, 3), cfg=cfg, checkout=lambda r, c, d: repo
    )
    m = report.mean()
    assert report.n == 1
    assert m["graphrag (BFS+RRF)"]["recall@3"] >= m["vector-only"]["recall@3"]
    assert m["graphrag (BFS+RRF)"]["recall@3"] == 1.0


def test_query_with_mismatched_embedder_gives_clear_error(tmp_path):
    import pytest

    repo = make_repo(tmp_path / "repo")
    cfg = GraphRAGSettings(data_dir=tmp_path / "data")
    _, store = build_index(repo, HashEmbedder(dim=384), cfg, persist=True)
    del store
    with pytest.raises(ValueError, match="re-run"):
        load_index(HashEmbedder(dim=768), cfg)
