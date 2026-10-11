"""Command line interface: index, query, refactor, eval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from graphrag_engine.config import settings
from graphrag_engine.embedder import Embedder, get_embedder
from graphrag_engine.evaluation import (
    evaluate_instances,
    load_per_instance,
    pooled_markdown,
    save_per_instance,
)
from graphrag_engine.ingest import (
    build_index,
    load_index,
    load_swebench,
    stored_embedder_name,
)
from graphrag_engine.llm import get_llm
from graphrag_engine.orchestrator import CodeGraphOrchestrator
from graphrag_engine.retriever import HybridGraphRAGRetriever
from graphrag_engine.sandbox import get_sandbox


def _index_embedder(args: argparse.Namespace) -> Embedder:
    """Embedder for querying: explicit flag wins, else the one the index was built with."""
    backend = args.embedder
    if backend is None:
        built_with = stored_embedder_name()
        if built_with:
            backend = built_with.split("-")[0]
    return get_embedder(backend=backend)


def _cmd_index(args: argparse.Namespace) -> int:
    embedder = get_embedder(backend=args.embedder)
    ckg, store = build_index(Path(args.path), embedder, persist=True)
    g = ckg.graph
    print(f"embedder: {embedder.name}")
    print(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(f"saved to {settings.data_dir}/")
    del store
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    embedder = _index_embedder(args)
    ckg, store = load_index(embedder)
    retriever = HybridGraphRAGRetriever(ckg, store)
    nodes = (
        retriever.retrieve_vector_only(args.query)
        if args.vector_only
        else retriever.retrieve_context(args.query)
    )
    for n in nodes:
        print(f"{n['rank']:>2}. {n['id']}  score={n['score']:.4f} hops={n['hops']}")
    return 0


def _cmd_refactor(args: argparse.Namespace) -> int:
    embedder = _index_embedder(args)
    ckg, store = load_index(embedder)
    orch = CodeGraphOrchestrator(
        HybridGraphRAGRetriever(ckg, store),
        get_llm(backend=args.llm),
        get_sandbox(backend=args.sandbox),
    )
    tests = Path(args.tests).read_text() if args.tests else ""
    state = orch.run_refactor_pipeline(args.task, tests)
    print(f"passed={state.verification_passed} attempts={state.refactor_attempts}")
    print(state.generated_code)
    if state.error_trace:
        print(state.error_trace, file=sys.stderr)
    return 0 if state.verification_passed else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    embedder = get_embedder(backend=args.embedder)
    instances = load_swebench(repo=args.repo, limit=args.limit)
    ks = tuple(args.k)
    report = evaluate_instances(instances, embedder, settings.data_dir / "repos", ks)
    md = f"# Retrieval evaluation ({args.repo or 'all repos'}, embedder: {embedder.name})\n\n"
    md += report.to_markdown(ks)
    out = (
        Path(args.output)
        if args.output
        else Path("results") / ("eval_" + (args.repo or "all").replace("/", "__") + ".md")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    json_path = out.with_suffix(".json")
    save_per_instance(
        json_path,
        {"repo": args.repo, "embedder": embedder.name, "ks": list(ks)},
        report.per_instance,
    )
    print(f"saved to {out} and {json_path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    per_instance = load_per_instance([Path(f) for f in args.files])
    md = pooled_markdown(per_instance)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"saved to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="graphrag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    emb_choices = ["auto", "hash", "sbert", "ollama"]

    s = sub.add_parser("index", help="build the graph + vector index for a directory")
    s.add_argument("path")
    s.add_argument("--embedder", choices=emb_choices, default=None)
    s.set_defaults(fn=_cmd_index)

    s = sub.add_parser("query", help="retrieve context for a natural-language query")
    s.add_argument("query")
    s.add_argument("--vector-only", action="store_true")
    s.add_argument("--embedder", choices=emb_choices, default=None)
    s.set_defaults(fn=_cmd_query)

    s = sub.add_parser("refactor", help="run the planner/coder/referee loop")
    s.add_argument("task")
    s.add_argument("--tests", help="path to a pytest file (module under test is `solution`)")
    s.add_argument("--llm", choices=["fake", "ollama"], default=None)
    s.add_argument("--sandbox", choices=["auto", "docker", "subprocess"], default=None)
    s.add_argument("--embedder", choices=emb_choices, default=None)
    s.set_defaults(fn=_cmd_refactor)

    s = sub.add_parser("eval", help="SWE-bench retrieval evaluation")
    s.add_argument("--repo", default="psf/requests")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    s.add_argument("--output", default=None, help="default: results/eval_<repo>.md")
    s.add_argument("--embedder", choices=emb_choices, default=None)
    s.set_defaults(fn=_cmd_eval)

    s = sub.add_parser("report", help="pool several eval .json files into one summary with CIs")
    s.add_argument("files", nargs="+", help="results/eval_*.json")
    s.add_argument("--output", default="results/summary.md")
    s.set_defaults(fn=_cmd_report)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
