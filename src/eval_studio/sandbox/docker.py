"""Docker-backed Tier 3 sandbox with isolated offline subprocess fallback.

Never executes untrusted code in-process on the host. Always returns a primitive
JSON-serializable ExecutionResult to prevent `__eq__` spoofing across boundaries.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from eval_studio.sandbox.base import ExecutionResult
from eval_studio.sandbox.limits import SandboxLimits

_RUNNER_WRAPPER = """
import contextlib
import importlib
import json
import socket
import sys
import traceback

network_disabled = {network_disabled!r}
if network_disabled:
    def _blocked_network(*args, **kwargs):
        raise PermissionError("Network access is disabled inside eval-studio sandbox.")

    socket.socket = _blocked_network  # type: ignore[misc]
    socket.create_connection = _blocked_network  # type: ignore[assignment]

code_src = {code!r}
harness_src = {test_harness!r}
namespace = {{"__name__": "__main__"}}
try:
    exec(compile(code_src, "<untrusted_code>", "exec"), namespace)
    exec(compile(harness_src, "<test_harness>", "exec"), namespace)
except BaseException as exc:
    sys.stderr.write(traceback.format_exc())
    sys.stdout.write(json.dumps({{"passed": False, "error": str(exc)}}) + "\\n")
    sys.exit(1)
else:
    sys.stdout.write(json.dumps({{"passed": True}}) + "\\n")
    sys.exit(0)
"""


def _parse_verdict(stdout: str, stderr: str, exit_code: int) -> ExecutionResult:
    passed = False
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data: Any = json.loads(stripped)
                if isinstance(data, dict) and type(data.get("passed")) is bool:
                    passed = bool(data["passed"]) and exit_code == 0
                    break
            except json.JSONDecodeError:
                continue
    return ExecutionResult(
        passed=bool(passed),
        exit_code=int(exit_code),
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )


class DockerSandbox:
    """Container-isolated code runner with zero-network default and offline subprocess fallback."""

    name: str = "docker"

    def __init__(self, image: str = "python:3.11-slim", *, prefer_docker: bool = True) -> None:
        self.image = image
        self.prefer_docker = prefer_docker

    def _try_docker(
        self,
        script_text: str,
        limits: SandboxLimits,
    ) -> ExecutionResult | None:
        if not self.prefer_docker:
            return None
        try:
            docker = importlib.import_module("docker")
        except ImportError:
            return None
        try:
            client = docker.from_env()
            client.ping()
        except Exception:
            return None

        container = None
        try:
            container = client.containers.run(
                self.image,
                ["python", "-I", "-c", script_text],
                detach=True,
                network_disabled=limits.network_disabled,
                read_only=limits.read_only_rootfs,
                mem_limit=f"{limits.memory_mb}m",
                pids_limit=limits.pids_limit,
            )
            status = container.wait(timeout=limits.timeout_seconds)
            exit_code = int(status.get("StatusCode", 1))
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            return _parse_verdict(stdout, stderr, exit_code)
        except Exception as exc:
            if "timed out" in str(exc).casefold() or "ReadTimeout" in type(exc).__name__:
                return ExecutionResult(
                    passed=False,
                    exit_code=124,
                    stderr="Sandbox execution timed out.",
                    timed_out=True,
                )
            return None
        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.remove(force=True)

    def run_code(
        self,
        code: str,
        *,
        test_harness: str,
        limits: SandboxLimits | None = None,
    ) -> ExecutionResult:
        eff_limits = limits if limits is not None else SandboxLimits()
        script_text = _RUNNER_WRAPPER.format(
            network_disabled=eff_limits.network_disabled,
            code=code,
            test_harness=test_harness,
        )

        docker_result = self._try_docker(script_text, eff_limits)
        if docker_result is not None:
            return docker_result

        # Isolated subprocess fallback when Docker daemon or [sandbox] extra is unavailable
        with tempfile.TemporaryDirectory(prefix="eval_studio_sbx_") as tmpdir:
            script_path = Path(tmpdir) / "run_sandbox.py"
            script_path.write_text(script_text, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=eff_limits.timeout_seconds,
                    cwd=tmpdir,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                out_str = exc.stdout if isinstance(exc.stdout, str) else ""
                err_str = exc.stderr if isinstance(exc.stderr, str) else ""
                return ExecutionResult(
                    passed=False,
                    exit_code=124,
                    stdout=out_str,
                    stderr=err_str or "Sandbox execution timed out.",
                    timed_out=True,
                )
            return _parse_verdict(proc.stdout, proc.stderr, proc.returncode)
