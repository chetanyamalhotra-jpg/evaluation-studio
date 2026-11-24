"""Sandbox protocol for isolated Tier 3 code execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from eval_studio.sandbox.limits import SandboxLimits


@dataclass(frozen=True)
class ExecutionResult:
    """Primitive JSON-serializable verdict returned across the sandbox boundary."""

    passed: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@runtime_checkable
class Sandbox(Protocol):
    """Protocol for container/VM-isolated untrusted code execution."""

    name: str

    def run_code(
        self,
        code: str,
        *,
        test_harness: str,
        limits: SandboxLimits | None = None,
    ) -> ExecutionResult:
        """Execute untrusted code inside an isolated environment and return a primitive verdict."""
        ...
