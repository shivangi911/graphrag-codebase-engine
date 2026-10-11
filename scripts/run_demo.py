"""End-to-end demo: AST -> graph -> hybrid retrieval -> agent loop.

Runs fully offline by default (hash embedder + scripted fake LLM).
Use --embedder sbert / --llm ollama for the real models.
"""

from __future__ import annotations

import argparse

from graphrag_engine.ast_parser import CodeASTParser
from graphrag_engine.config import settings
from graphrag_engine.embedder import get_embedder
from graphrag_engine.graph_builder import CodeKnowledgeGraph
from graphrag_engine.llm import FakeLLM, get_llm
from graphrag_engine.orchestrator import CodeGraphOrchestrator
from graphrag_engine.retriever import HybridGraphRAGRetriever
from graphrag_engine.sandbox import get_sandbox
from graphrag_engine.vector_store import VectorStore

SAMPLE_CODE = '''
import jwt


def decode_jwt(token: str) -> dict:
    """Decode a JWT token into its claims."""
    return jwt.decode(token, "secret", algorithms=["HS256"])


class AuthHandler:
    def verify_token(self, token: str) -> bool:
        claims = decode_jwt(token)
        return "user" in claims


def login(user: str) -> str:
    return "token-for-" + user
'''

SAFE_CODE = """```python
def verify_token(token: str) -> bool:
    return token.startswith("token-for-")
```"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hash", choices=["auto", "hash", "sbert", "ollama"])
    ap.add_argument("--llm", default="fake", choices=["fake", "ollama"])
    args = ap.parse_args()

    ast_data = CodeASTParser().parse_to_dict("auth.py", SAMPLE_CODE)
    print(f"parsed: {len(ast_data['nodes'])} nodes, {len(ast_data['edges'])} edges")

    ckg = CodeKnowledgeGraph()
    ckg.build_graph_from_ast(ast_data)
    for u, v, d in ckg.graph.edges(data=True):
        print(f"  {u} -[{d['relation']}]-> {v}")

    embedder = get_embedder(backend=args.embedder)
    store = VectorStore(embedder)
    print(f"indexed {store.index_graph(ckg)} nodes with {embedder.name}")

    retriever = HybridGraphRAGRetriever(ckg, store, settings)
    query = "How is the JWT token verified?"
    for n in retriever.retrieve_context(query):
        print(f"  {n['rank']}. {n['id']} (score={n['score']:.4f}, hops={n['hops']})")

    llm = (
        FakeLLM(["Plan: simplify token verification.", SAFE_CODE])
        if args.llm == "fake"
        else get_llm(backend=args.llm)
    )
    orch = CodeGraphOrchestrator(retriever, llm, get_sandbox(backend="subprocess"))
    state = orch.run_refactor_pipeline("Refactor verify_token to avoid decoding the JWT")
    print(f"agent passed={state.verification_passed} attempts={state.refactor_attempts}")
    print(state.generated_code)


if __name__ == "__main__":
    main()
