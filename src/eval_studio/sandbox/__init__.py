"""Isolated Tier 3 code-execution sandboxes and resource limits."""

from eval_studio.sandbox.base import ExecutionResult, Sandbox
from eval_studio.sandbox.docker import DockerSandbox
from eval_studio.sandbox.limits import SandboxLimits

__all__ = [
    "DockerSandbox",
    "ExecutionResult",
    "Sandbox",
    "SandboxLimits",
]
