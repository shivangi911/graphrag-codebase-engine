from graphrag_engine.evaluation import (
    EvalReport,
    file_metrics,
    parse_patch_files,
    parse_patch_symbols,
    ranked_files,
)
from graphrag_engine.llm import extract_code
from graphrag_engine.sandbox import SubprocessSandbox

GOOD = "def add(a: int, b: int) -> int:\n    return a + b\n"
TESTS = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def test_sandbox_passes_clean_code():
    res = SubprocessSandbox(timeout_s=120).run(GOOD, TESTS)
    assert res.passed, res.output
    assert res.steps == {"ruff": True, "mypy": True, "pytest": True}


def test_sandbox_reports_failing_tests():
    bad = "def add(a: int, b: int) -> int:\n    return a - b\n"
    res = SubprocessSandbox(timeout_s=120).run(bad, TESTS)
    assert not res.passed
    assert res.steps["pytest"] is False
    assert "pytest FAILED" in res.output


def test_sandbox_reports_type_and_lint_errors():
    res = SubprocessSandbox(timeout_s=120).run("import os\nx: int = 'a'\n")
    assert res.steps["ruff"] is False
    assert res.steps["mypy"] is False


def test_extract_code():
    assert extract_code("hi\n```python\nx = 1\n```\nbye") == "x = 1\n"
    assert extract_code("x = 2") == "x = 2\n"


PATCH = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,3 +1,4 @@ def foo(x):
+    pass
diff --git a/src/b.py b/src/b.py
@@ -5,3 +5,4 @@ class Bar:
+    pass
@@ -9,3 +9,4 @@ async def baz():
"""


def test_patch_parsing():
    assert parse_patch_files(PATCH) == ["src/a.py", "src/b.py"]
    assert parse_patch_symbols(PATCH) == ["Bar", "baz", "foo"]


def test_file_metrics():
    m = file_metrics(["x.py", "src/b.py", "src/a.py"], ["src/a.py", "src/b.py"], [1, 3])
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 1.0
    assert m["hit@1"] == 0.0
    assert m["mrr"] == 0.5


def test_ranked_files_dedupes_in_order():
    nodes = [{"file": "a"}, {"file": "b"}, {"file": "a"}, {"file": ""}]
    assert ranked_files(nodes) == ["a", "b"]


def test_report_markdown():
    r = EvalReport(n=2)
    r.add("m", {"recall@1": 1.0, "mrr": 1.0})
    r.add("m", {"recall@1": 0.0, "mrr": 0.5})
    md = r.to_markdown([1])
    assert "| m |" in md and "0.500" in md and "0.750" in md


def _records(deltas):
    recs = []
    for i, d in enumerate(deltas):
        recs.append(
            {
                "instance_id": f"i{i}",
                "repo": "r",
                "method": "vector-only",
                "metrics": {"mrr": 0.4, "hit@1": 0.0},
            }
        )
        recs.append(
            {
                "instance_id": f"i{i}",
                "repo": "r",
                "method": "graphrag (BFS+RRF)",
                "metrics": {"mrr": 0.4 + d, "hit@1": 1.0 if d > 0 else 0.0},
            }
        )
    return recs


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    from graphrag_engine.evaluation import bootstrap_ci

    vals = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    a = bootstrap_ci(vals, n_boot=2000, seed=1)
    assert a == bootstrap_ci(vals, n_boot=2000, seed=1)
    assert a[1] <= a[0] <= a[2]
    assert bootstrap_ci([0.5] * 5, n_boot=500) == (0.5, 0.5, 0.5)
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_paired_delta_flags_significance():
    from graphrag_engine.evaluation import paired_delta_markdown

    always_better = paired_delta_markdown(_records([0.2] * 10), n_boot=500)
    assert "| mrr | 0.400 | 0.600 | +0.200 |" in always_better
    assert always_better.count("| yes |") == 2
    noisy = paired_delta_markdown(_records([0.3, -0.3] * 5), n_boot=500)
    assert "| mrr |" in noisy and "| no |" in noisy.split("| mrr |")[1].split("\n")[0]


def test_report_roundtrip_and_pooling(tmp_path):
    from graphrag_engine.evaluation import load_per_instance, pooled_markdown, save_per_instance

    p = tmp_path / "e.json"
    save_per_instance(p, {"repo": "r"}, _records([0.1] * 4))
    recs = load_per_instance([p])
    assert len(recs) == 8
    md = pooled_markdown(recs, n_boot=200)
    assert "Total: 4" in md and "Paired difference" in md
