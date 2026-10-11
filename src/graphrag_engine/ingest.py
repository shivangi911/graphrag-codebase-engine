"""Repository ingestion: walk Python files, build the CKG and vector index; SWE-bench helpers."""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphrag_engine.ast_parser import CodeASTParser
from graphrag_engine.config import GraphRAGSettings, settings
from graphrag_engine.embedder import Embedder
from graphrag_engine.graph_builder import CodeKnowledgeGraph
from graphrag_engine.vector_store import VectorStore

log = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", ".hg", ".venv", "venv", "env", "node_modules", "__pycache__", "build", "dist",
    ".tox", ".mypy_cache", ".pytest_cache", "site-packages", ".eggs",
}  # fmt: skip
MAX_FILE_BYTES = 1_000_000


def iter_python_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in rel_parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def build_graph_from_directory(root: Path) -> CodeKnowledgeGraph:
    parser = CodeASTParser()
    parsed = []
    for path in iter_python_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf8", errors="replace")
        except OSError as exc:
            log.warning("skip %s: %s", rel, exc)
            continue
        parsed.append(parser.parse_source_code(rel, source))
    return CodeKnowledgeGraph.from_parsed_files(parsed)


def build_index(
    root: Path,
    embedder: Embedder,
    cfg: GraphRAGSettings | None = None,
    persist: bool = False,
    collection: str | None = None,
) -> tuple[CodeKnowledgeGraph, VectorStore]:
    cfg = cfg or settings
    ckg = build_graph_from_directory(root)
    store = VectorStore(
        embedder,
        collection=collection or cfg.qdrant_collection,
        path=cfg.qdrant_path if persist else None,
        url=cfg.qdrant_url,
    )
    store.index_graph(ckg)
    if persist:
        ckg.save(cfg.graph_path)
        cfg.meta_path.write_text(json.dumps({"embedder": embedder.name}))
    return ckg, store


def stored_embedder_name(cfg: GraphRAGSettings | None = None) -> str | None:
    """Name of the embedder the persisted index was built with, if known."""
    cfg = cfg or settings
    if not cfg.meta_path.exists():
        return None
    name = json.loads(cfg.meta_path.read_text()).get("embedder")
    return str(name) if name else None


def load_index(
    embedder: Embedder, cfg: GraphRAGSettings | None = None
) -> tuple[CodeKnowledgeGraph, VectorStore]:
    cfg = cfg or settings
    if not cfg.graph_path.exists():
        raise FileNotFoundError(f"No index at {cfg.graph_path}. Run `graphrag index <path>` first.")
    built_with = stored_embedder_name(cfg)
    if built_with and built_with != embedder.name:
        raise ValueError(
            f"Index was built with '{built_with}' but the query embedder is '{embedder.name}'. "
            "Use the same --embedder, or re-run `graphrag index`."
        )
    ckg = CodeKnowledgeGraph.load(cfg.graph_path)
    store = VectorStore(
        embedder, collection=cfg.qdrant_collection, path=cfg.qdrant_path, url=cfg.qdrant_url
    )
    store.load_cache()
    return ckg, store


# -- SWE-bench ----------------------------------------------------------------
@dataclass
class SWEInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str


def load_swebench(
    repo: str | None = None,
    limit: int = 10,
    split: str = "test",
    dataset: str = "SWE-bench/SWE-bench",
) -> list[SWEInstance]:
    """Stream the dataset from Hugging Face, filtering by repo (e.g. ``psf/requests``)."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    out: list[SWEInstance] = []
    for row in ds:
        r: dict[str, Any] = dict(row)
        if repo and r["repo"] != repo:
            continue
        out.append(
            SWEInstance(
                instance_id=r["instance_id"],
                repo=r["repo"],
                base_commit=r["base_commit"],
                problem_statement=r["problem_statement"],
                patch=r["patch"],
            )
        )
        if len(out) >= limit:
            break
    return out


def _git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)  # noqa: S603, S607


def checkout_repo(repo: str, base_commit: str, repos_dir: Path) -> Path:
    """Clone ``owner/name`` once (blobless) and check out ``base_commit``."""
    dest = repos_dir / repo.replace("/", "__")
    if not dest.exists():
        repos_dir.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--filter=blob:none", f"https://github.com/{repo}.git", str(dest)])
    _git(["checkout", "-f", base_commit], cwd=dest)
    return dest
