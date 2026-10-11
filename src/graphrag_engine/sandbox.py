"""Referee sandbox: runs ruff, mypy and pytest against generated code (Docker or subprocess)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from graphrag_engine.config import GraphRAGSettings, settings

MAX_TRACE_CHARS = 3000


@dataclass
class SandboxResult:
    passed: bool
    steps: dict[str, bool] = field(default_factory=dict)
    output: str = ""


class Sandbox(Protocol):
    def run(self, code: str, tests: str = "") -> SandboxResult: ...


def _tool_cmds(has_tests: bool, python: str) -> list[tuple[str, list[str]]]:
    cmds = [
        ("ruff", [python, "-m", "ruff", "check", "solution.py", "--no-cache"]),
        ("mypy", [python, "-m", "mypy", "solution.py", "--ignore-missing-imports"]),
    ]
    if has_tests:
        cmds.append(("pytest", [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."]))
    return cmds


class SubprocessSandbox:
    """Isolated temp dir + timeout. Not a security boundary; prefer Docker for untrusted code."""

    def __init__(self, timeout_s: int = 60) -> None:
        self.timeout_s = timeout_s

    def run(self, code: str, tests: str = "") -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="graphrag_sbx_") as tmp:
            d = Path(tmp)
            (d / "solution.py").write_text(code)
            if tests:
                (d / "test_solution.py").write_text(tests)
            result = SandboxResult(passed=True)
            for name, cmd in _tool_cmds(bool(tests), sys.executable):
                try:
                    proc = subprocess.run(  # noqa: S603
                        cmd, cwd=d, capture_output=True, text=True, timeout=self.timeout_s
                    )
                    ok = proc.returncode == 0
                    out = proc.stdout + proc.stderr
                except subprocess.TimeoutExpired:
                    ok, out = False, f"{name} timed out after {self.timeout_s}s"
                result.steps[name] = ok
                if not ok:
                    result.passed = False
                    result.output += f"--- {name} FAILED ---\n{out}\n"
            return _trim(result)


class DockerSandbox:
    """Runs the checks in a network-less container built from docker/sandbox.Dockerfile."""

    def __init__(self, image: str, timeout_s: int = 60) -> None:
        self.image = image
        self.timeout_s = timeout_s

    def run(self, code: str, tests: str = "") -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="graphrag_sbx_") as tmp:
            d = Path(tmp)
            (d / "solution.py").write_text(code)
            if tests:
                (d / "test_solution.py").write_text(tests)
            result = SandboxResult(passed=True)
            for name, cmd in _tool_cmds(bool(tests), "python"):
                docker_cmd = [
                    "docker", "run", "--rm", "--network", "none", "--memory", "512m",
                    "--cpus", "1", "-v", f"{d}:/work", "-w", "/work", self.image, *cmd,
                ]  # fmt: skip
                try:
                    proc = subprocess.run(  # noqa: S603
                        docker_cmd, capture_output=True, text=True, timeout=self.timeout_s
                    )
                    ok = proc.returncode == 0
                    out = proc.stdout + proc.stderr
                except subprocess.TimeoutExpired:
                    ok, out = False, f"{name} timed out after {self.timeout_s}s"
                result.steps[name] = ok
                if not ok:
                    result.passed = False
                    result.output += f"--- {name} FAILED ---\n{out}\n"
            return _trim(result)


def _trim(result: SandboxResult) -> SandboxResult:
    if len(result.output) > MAX_TRACE_CHARS:
        result.output = result.output[:MAX_TRACE_CHARS] + "\n...[truncated]"
    return result


def _docker_image_ready(image: str) -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(  # noqa: S603
            ["docker", "image", "inspect", image], capture_output=True, timeout=15
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def get_sandbox(cfg: GraphRAGSettings | None = None, backend: str | None = None) -> Sandbox:
    cfg = cfg or settings
    backend = backend or cfg.sandbox_backend
    if backend == "docker":
        return DockerSandbox(cfg.sandbox_image, cfg.sandbox_timeout_s)
    if backend == "auto" and _docker_image_ready(cfg.sandbox_image):
        return DockerSandbox(cfg.sandbox_image, cfg.sandbox_timeout_s)
    return SubprocessSandbox(cfg.sandbox_timeout_s)
