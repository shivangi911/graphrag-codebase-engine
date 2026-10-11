"""LangGraph cyclical state machine: retrieve, planner, coder, referee (retry on failure)."""

from __future__ import annotations

from typing import Any, Protocol

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from graphrag_engine.config import settings
from graphrag_engine.llm import LLMClient, extract_code
from graphrag_engine.sandbox import Sandbox

PLANNER_SYSTEM = (
    "You are a senior engineer. Given a task and code context from a repository knowledge "
    "graph, write a short numbered plan. If a previous attempt failed, address the error."
)
CODER_SYSTEM = (
    "You are an expert Python developer. Implement the plan. Reply with ONE complete, "
    "self-contained Python module in a ```python fenced block, typed, ruff and mypy clean."
)
MAX_NODE_CHARS = 800


class ContextRetriever(Protocol):
    def retrieve_context(self, query: str) -> list[dict[str, Any]]: ...


class AgentState(BaseModel):
    query: str
    tests: str = ""
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)
    generated_plan: str = ""
    generated_code: str = ""
    error_trace: str = ""
    verification_passed: bool = False
    refactor_attempts: int = 0
    history: list[dict[str, Any]] = Field(default_factory=list)


def format_context(context: list[dict[str, Any]]) -> str:
    parts = []
    for c in context:
        code = str(c.get("code", ""))[:MAX_NODE_CHARS]
        parts.append(f"### {c.get('type')} {c.get('id')}\n{code}")
    return "\n\n".join(parts) or "(no context retrieved)"


class CodeGraphOrchestrator:
    def __init__(
        self,
        retriever: ContextRetriever,
        llm: LLMClient,
        sandbox: Sandbox,
        max_attempts: int | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.sandbox = sandbox
        self.max_attempts = max_attempts or settings.max_refactor_attempts
        self.graph = self._build()

    # -- nodes ----------------------------------------------------------
    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        return {"retrieved_context": self.retriever.retrieve_context(state.query)}

    def _planner(self, state: AgentState) -> dict[str, Any]:
        prompt = f"Task: {state.query}\n\nContext:\n{format_context(state.retrieved_context)}\n"
        if state.error_trace:
            prompt += (
                f"\nPrevious code:\n{state.generated_code}\n"
                f"\nReferee error trace:\n{state.error_trace}\n"
            )
        return {"generated_plan": self.llm.complete(PLANNER_SYSTEM, prompt)}

    def _coder(self, state: AgentState) -> dict[str, Any]:
        prompt = f"Task: {state.query}\n\nPlan:\n{state.generated_plan}\n"
        if state.error_trace:
            prompt += f"\nFix these errors:\n{state.error_trace}\n"
        if state.tests:
            prompt += f"\nThe code must pass these tests (module is `solution`):\n{state.tests}\n"
        code = extract_code(self.llm.complete(CODER_SYSTEM, prompt))
        return {"generated_code": code, "refactor_attempts": state.refactor_attempts + 1}

    def _referee(self, state: AgentState) -> dict[str, Any]:
        res = self.sandbox.run(state.generated_code, state.tests)
        entry = {
            "attempt": state.refactor_attempts,
            "passed": res.passed,
            "steps": res.steps,
        }
        return {
            "verification_passed": res.passed,
            "error_trace": "" if res.passed else res.output,
            "history": [*state.history, entry],
        }

    def _route(self, state: AgentState) -> str:
        if state.verification_passed or state.refactor_attempts >= self.max_attempts:
            return "end"
        return "retry"

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("planner", self._planner)
        g.add_node("coder", self._coder)
        g.add_node("referee", self._referee)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "planner")
        g.add_edge("planner", "coder")
        g.add_edge("coder", "referee")
        g.add_conditional_edges("referee", self._route, {"retry": "planner", "end": END})
        return g.compile()

    # -- public ---------------------------------------------------------
    def run_refactor_pipeline(self, query: str, tests: str = "") -> AgentState:
        out = self.graph.invoke(AgentState(query=query, tests=tests))
        return AgentState(**out)
