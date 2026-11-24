"""Shared fixtures and marker registration for opaque-box E2E tests (offline CPU)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from eval_studio.gateway.secrets import SecretRole, clear_session_key

E2E_PROVIDER_ENV_VARS: tuple[str, ...] = (
    "TARGET_API_KEY",
    "JUDGE_API_KEY",
    "EXTRACTOR_API_KEY",
    "FALLBACK_JUDGE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "SARVAM_API_KEY",
)


def pytest_configure(config: pytest.Config) -> None:
    """Dynamically register the `e2e` marker without modifying pyproject.toml."""
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end opaque-box requirement tests (offline CPU)",
    )


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate cwd (.env), cache dir, UTF-8 mode, OS keyring, session keys, and API env vars."""
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "eval_studio_cache"
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("PYTHONUTF8", "1")
    for var in E2E_PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for role in SecretRole:
        clear_session_key(role)
    monkeypatch.setattr("keyring.get_password", lambda service, username: None)
    return cache_dir


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Typer CliRunner compatible with typer>=0.29 (no mix_stderr)."""
    return CliRunner()


@pytest.fixture
def run_eval_studio_subprocess(
    repo_root: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Return a helper executing `python -m eval_studio ...` with UTF-8 I/O and PYTHONPATH=src."""
    src_dir = str(repo_root / "src")

    def _run(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        proc_env = os.environ.copy()
        proc_env["PYTHONUTF8"] = "1"
        proc_env["PYTHONIOENCODING"] = "utf-8"
        existing_pp = proc_env.get("PYTHONPATH", "")
        parts = [p for p in existing_pp.split(os.pathsep) if p] if existing_pp else []
        if src_dir not in parts:
            proc_env["PYTHONPATH"] = os.pathsep.join([src_dir, *parts]) if parts else src_dir
        if env:
            proc_env.update(env)
        return subprocess.run(
            [sys.executable, "-m", "eval_studio", *args],
            cwd=str(cwd) if cwd is not None else None,
            env=proc_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    return _run


@pytest.fixture
def make_mock_transport() -> Callable[
    [Callable[[httpx.Request, list[dict[str, Any]]], httpx.Response]],
    tuple[httpx.MockTransport, list[dict[str, Any]]],
]:
    """Return a factory creating an offline httpx.MockTransport + request recorder."""

    def _factory(
        handler: Callable[[httpx.Request, list[dict[str, Any]]], httpx.Response],
    ) -> tuple[httpx.MockTransport, list[dict[str, Any]]]:
        recorded: list[dict[str, Any]] = []

        def _dispatch(request: httpx.Request) -> httpx.Response:
            body_text = request.content.decode("utf-8", errors="replace") if request.content else ""
            parsed_json: Any = None
            if body_text.strip():
                try:
                    parsed_json = json.loads(body_text)
                except json.JSONDecodeError:
                    parsed_json = None
            recorded.append(
                {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "body_text": body_text,
                    "json": parsed_json,
                }
            )
            return handler(request, recorded)

        return httpx.MockTransport(_dispatch), recorded

    return _factory


@pytest.fixture
def space_workspace(tmp_path: Path) -> Path:
    """Create a Windows space- and Unicode-containing workspace directory."""
    workspace = tmp_path / "Clinical Eval (Café v1)"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace
