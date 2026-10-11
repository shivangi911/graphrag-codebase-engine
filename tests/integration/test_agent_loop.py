from graphrag_engine.llm import FakeLLM
from graphrag_engine.orchestrator import CodeGraphOrchestrator
from graphrag_engine.sandbox import SandboxResult, SubprocessSandbox

TESTS = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
BAD = "```python\ndef add(a: int, b: int) -> int:\n    return a - b\n```"
GOOD = "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"


class StubRetriever:
    def retrieve_context(self, query):
        return [{"id": "m.py::add", "type": "Function", "code": "def add(a, b): ..."}]


class AlwaysFail:
    def run(self, code, tests=""):
        return SandboxResult(passed=False, steps={"pytest": False}, output="boom")


def test_loop_retries_with_error_trace_then_passes():
    llm = FakeLLM(["plan1", BAD, "plan2", GOOD])
    orch = CodeGraphOrchestrator(StubRetriever(), llm, SubprocessSandbox(120), max_attempts=3)
    state = orch.run_refactor_pipeline("implement add", TESTS)
    assert state.verification_passed
    assert state.refactor_attempts == 2
    assert [h["passed"] for h in state.history] == [False, True]
    # the second planner call must have seen the referee trace
    assert "pytest FAILED" in llm.calls[2][1]
    assert "m.py::add" in llm.calls[0][1]


def test_loop_stops_at_max_attempts():
    llm = FakeLLM(["plan", BAD])
    orch = CodeGraphOrchestrator(StubRetriever(), llm, AlwaysFail(), max_attempts=2)
    state = orch.run_refactor_pipeline("task")
    assert not state.verification_passed
    assert state.refactor_attempts == 2
    assert state.error_trace == "boom"


def test_loop_passes_first_try_without_retry():
    llm = FakeLLM(["plan", GOOD])
    orch = CodeGraphOrchestrator(StubRetriever(), llm, SubprocessSandbox(120), max_attempts=3)
    state = orch.run_refactor_pipeline("implement add", TESTS)
    assert state.verification_passed and state.refactor_attempts == 1
    assert len(llm.calls) == 2
