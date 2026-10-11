"""Retrieval evaluation on SWE-bench: do retrieved files cover the gold patch files?"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from graphrag_engine.config import GraphRAGSettings, settings
from graphrag_engine.embedder import Embedder
from graphrag_engine.ingest import SWEInstance, build_index, checkout_repo
from graphrag_engine.retriever import HybridGraphRAGRetriever

log = logging.getLogger(__name__)

_DIFF_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)
_HUNK_SYM = re.compile(r"^@@[^@]*@@\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE)
MAX_QUERY_CHARS = 2000
BASELINE = "vector-only"
CHALLENGER = "graphrag (BFS+RRF)"
SUMMARY_COLS = (
    "hit@1",
    "hit@3",
    "hit@5",
    "hit@10",
    "recall@1",
    "recall@10",
    "mrr",
    "symbol_hit@10",
)


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap (1-alpha) CI over instances."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def _by_method(per_instance: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for rec in per_instance:
        out.setdefault(rec["method"], {})[rec["instance_id"]] = rec["metrics"]
    return out


def _columns(per_instance: Sequence[dict[str, Any]]) -> list[str]:
    keys = set(per_instance[0]["metrics"]) if per_instance else set()
    return [c for c in SUMMARY_COLS if c in keys]


def paired_delta_markdown(
    per_instance: Sequence[dict[str, Any]], n_boot: int = 10_000, seed: int = 0
) -> str:
    """Paired per-instance difference (challenger - baseline) with bootstrap 95% CI."""
    by = _by_method(per_instance)
    if BASELINE not in by or CHALLENGER not in by:
        return ""
    common = sorted(set(by[BASELINE]) & set(by[CHALLENGER]))
    if not common:
        return ""
    lines = [
        f"Paired difference ({CHALLENGER} minus {BASELINE}), n={len(common)}, "
        "95% bootstrap CI over instances:",
        "",
        "| metric | baseline | graphrag | delta | 95% CI | CI excludes 0 |",
        "|---|---|---|---|---|---|",
    ]
    for col in _columns(per_instance):
        base = [by[BASELINE][i][col] for i in common]
        chal = [by[CHALLENGER][i][col] for i in common]
        deltas = [c - b for b, c in zip(base, chal, strict=True)]
        mean, lo, hi = bootstrap_ci(deltas, n_boot, seed)
        sig = "yes" if lo > 0 or hi < 0 else "no"
        lines.append(
            f"| {col} | {np.mean(base):.3f} | {np.mean(chal):.3f} | {mean:+.3f} "
            f"| [{lo:+.3f}, {hi:+.3f}] | {sig} |"
        )
    return "\n".join(lines) + "\n"


def pooled_markdown(per_instance: Sequence[dict[str, Any]], n_boot: int = 10_000) -> str:
    """Pooled summary over every instance, with per-method CIs and paired deltas."""
    by = _by_method(per_instance)
    repos: dict[str, int] = {}
    for rec in per_instance:
        if rec["method"] == BASELINE:
            repos[rec.get("repo", "?")] = repos.get(rec.get("repo", "?"), 0) + 1
    cols = _columns(per_instance)
    lines = [
        "# Pooled retrieval evaluation",
        "",
        "Instances per repo: " + ", ".join(f"{r}: {n}" for r, n in sorted(repos.items())),
        f"Total: {sum(repos.values())}",
        "",
        "Mean (95% bootstrap CI):",
        "",
        "| method | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for method, rows in by.items():
        cells = []
        for col in cols:
            m, lo, hi = bootstrap_ci([r[col] for r in rows.values()], n_boot)
            cells.append(f"{m:.3f} ({lo:.2f}-{hi:.2f})")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", paired_delta_markdown(per_instance, n_boot)]
    return "\n".join(lines)


def save_per_instance(path: Path, meta: dict[str, Any], per_instance: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({**meta, "per_instance": per_instance}, indent=1))


def load_per_instance(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        out.extend(json.loads(p.read_text())["per_instance"])
    return out


def parse_patch_files(patch: str) -> list[str]:
    """Files modified by a unified diff (new-side paths, de-duplicated, order preserved)."""
    seen: dict[str, None] = {}
    for _, new in _DIFF_FILE.findall(patch):
        seen.setdefault(new, None)
    return list(seen)


def parse_patch_symbols(patch: str) -> list[str]:
    """Best-effort names of functions/classes touched, from hunk-header context."""
    return sorted(set(_HUNK_SYM.findall(patch)))


def ranked_files(nodes: Sequence[dict[str, Any]]) -> list[str]:
    """Unique files in order of best-ranked node."""
    seen: dict[str, None] = {}
    for n in nodes:
        f = n.get("file")
        if f:
            seen.setdefault(str(f), None)
    return list(seen)


def file_metrics(ranked: Sequence[str], gold: Sequence[str], ks: Sequence[int]) -> dict[str, float]:
    gold_set = set(gold)
    out: dict[str, float] = {}
    for k in ks:
        hits = len(gold_set & set(ranked[:k]))
        out[f"recall@{k}"] = hits / len(gold_set) if gold_set else 0.0
        out[f"hit@{k}"] = 1.0 if hits else 0.0
    rr = 0.0
    for i, f in enumerate(ranked, start=1):
        if f in gold_set:
            rr = 1.0 / i
            break
    out["mrr"] = rr
    return out


def symbol_hit(nodes: Sequence[dict[str, Any]], symbols: Sequence[str], k: int) -> float:
    if not symbols:
        return 0.0
    names = {str(n.get("name")) for n in nodes[:k]}
    return 1.0 if names & set(symbols) else 0.0


@dataclass
class EvalReport:
    n: int = 0
    totals: dict[str, dict[str, float]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    per_instance: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        method: str,
        metrics: dict[str, float],
        instance_id: str | None = None,
        repo: str | None = None,
    ) -> None:
        if instance_id is not None:
            self.per_instance.append(
                {"instance_id": instance_id, "repo": repo, "method": method, "metrics": metrics}
            )
        bucket = self.totals.setdefault(method, {})
        for k, v in metrics.items():
            bucket[k] = bucket.get(k, 0.0) + v

    def mean(self) -> dict[str, dict[str, float]]:
        return {
            m: {k: v / self.n for k, v in b.items()} if self.n else {}
            for m, b in self.totals.items()
        }

    def to_markdown(self, ks: Sequence[int]) -> str:
        means = self.mean()
        cols = [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + ["mrr", "symbol_hit@10"]
        lines = [
            f"Instances evaluated: {self.n}  (skipped: {len(self.skipped)})",
            "",
            "| method | " + " | ".join(cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        for method, m in means.items():
            lines.append(f"| {method} | " + " | ".join(f"{m.get(c, 0.0):.3f}" for c in cols) + " |")
        md = "\n".join(lines) + "\n"
        delta = paired_delta_markdown(self.per_instance)
        return md + ("\n" + delta if delta else "")


def evaluate_instances(
    instances: Sequence[SWEInstance],
    embedder: Embedder,
    repos_dir: Path,
    ks: Sequence[int] = (1, 3, 5, 10),
    cfg: GraphRAGSettings | None = None,
    checkout: Callable[[str, str, Path], Path] = checkout_repo,
) -> EvalReport:
    cfg = cfg or settings
    report = EvalReport()
    for inst in instances:
        gold = parse_patch_files(inst.patch)
        symbols = parse_patch_symbols(inst.patch)
        if not gold:
            report.skipped.append(inst.instance_id)
            continue
        try:
            root = checkout(inst.repo, inst.base_commit, repos_dir)
            ckg, store = build_index(root, embedder, cfg, persist=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", inst.instance_id, exc)
            report.skipped.append(inst.instance_id)
            continue
        query = inst.problem_statement[:MAX_QUERY_CHARS]
        retriever = HybridGraphRAGRetriever(ckg, store, cfg)
        variants = {
            "vector-only": retriever.retrieve_vector_only(query, cfg.max_context_nodes),
            "graphrag (BFS+RRF)": retriever.retrieve_context(query),
        }
        for method, nodes in variants.items():
            m = file_metrics(ranked_files(nodes), gold, ks)
            m["symbol_hit@10"] = symbol_hit(nodes, symbols, 10)
            report.add(method, m, inst.instance_id, inst.repo)
        report.n += 1
        log.info("evaluated %s", inst.instance_id)
    return report
