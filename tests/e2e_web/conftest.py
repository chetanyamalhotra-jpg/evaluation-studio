"""Shared fixtures, environment isolation, and helpers for opaque-box E2E Web Studio tests."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import keyring
import keyring.errors
import pytest
from typer.testing import CliRunner

from eval_studio.gateway import secrets


def pytest_configure(config: pytest.Config) -> None:
    """Dynamically register the `e2e_web` marker without modifying pyproject.toml."""
    config.addinivalue_line(
        "markers",
        "e2e_web: End-to-end opaque-box Web Studio tests (offline CPU)",
    )


@dataclass
class IsolatedWebPaths:
    """Isolated per-test filesystem paths and fake in-memory keyring store."""

    workspace_dir: Path
    receipts_dir: Path
    tasks_dir: Path
    examples_dir: Path
    cache_dir: Path
    fake_keyring: dict[tuple[str, str], str]


@pytest.fixture(autouse=True)
def isolated_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> IsolatedWebPaths:
    """Isolate workspace, receipts, examples, cache, env vars, `.env`, and OS keyring per test."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    receipts_dir = workspace_dir / "receipts"
    src_receipts = repo_root / "receipts"
    if src_receipts.is_dir():
        shutil.copytree(src_receipts, receipts_dir)
    else:
        receipts_dir.mkdir(parents=True, exist_ok=True)

    examples_copy = workspace_dir / "examples"
    src_examples = repo_root / "examples"
    if src_examples.is_dir():
        shutil.copytree(src_examples, examples_copy)
    else:
        examples_copy.mkdir(parents=True, exist_ok=True)

    tasks_dir = workspace_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = tmp_path / "eval_studio_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(workspace_dir)
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("PYTHONUTF8", "1")

    for var in secrets.PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(secrets, "_LOADED_SECRETS", set())
    if hasattr(secrets, "_SESSION_KEYS"):
        monkeypatch.setattr(secrets, "_SESSION_KEYS", {})

    fake_keyring: dict[tuple[str, str], str] = {}

    def _fake_get(service: str, username: str) -> str | None:
        return fake_keyring.get((service, username))

    def _fake_set(service: str, username: str, password: str) -> None:
        fake_keyring[(service, username)] = password

    def _fake_delete(service: str, username: str) -> None:
        key = (service, username)
        if key not in fake_keyring:
            raise keyring.errors.PasswordDeleteError(f"Key not found: {username}")
        del fake_keyring[key]

    monkeypatch.setattr(keyring, "get_password", _fake_get)
    monkeypatch.setattr(keyring, "set_password", _fake_set)
    monkeypatch.setattr(keyring, "delete_password", _fake_delete)

    return IsolatedWebPaths(
        workspace_dir=workspace_dir,
        receipts_dir=receipts_dir,
        tasks_dir=tasks_dir,
        examples_dir=examples_copy,
        cache_dir=cache_dir,
        fake_keyring=fake_keyring,
    )


@pytest.fixture
def web_app(isolated_env: IsolatedWebPaths) -> Any:
    """Create a FastAPI app bound strictly to the per-test isolated directories."""
    pytest.importorskip("fastapi")
    server_mod = pytest.importorskip("eval_studio.web.server")
    create_app = server_mod.create_app
    sig = inspect.signature(create_app)
    kwargs: dict[str, Any] = {}
    if "workspace_dir" in sig.parameters:
        kwargs["workspace_dir"] = isolated_env.workspace_dir
    if "receipts_dir" in sig.parameters:
        kwargs["receipts_dir"] = isolated_env.receipts_dir
    if "tasks_dir" in sig.parameters:
        kwargs["tasks_dir"] = isolated_env.tasks_dir
    if "cache_dir" in sig.parameters:
        kwargs["cache_dir"] = isolated_env.cache_dir
    return create_app(**kwargs)


@pytest.fixture
def client(web_app: Any) -> Iterator[Any]:
    """Yield a FastAPI TestClient wrapped in a context manager to trigger lifespan hooks."""
    testclient_mod = pytest.importorskip("fastapi.testclient")
    with testclient_mod.TestClient(web_app) as tc:
        yield tc


@pytest.fixture
def make_async_client(web_app: Any) -> Callable[[], Any]:
    """Return an async context-manager factory for httpx.AsyncClient over ASGITransport."""

    @asynccontextmanager
    async def _factory() -> Any:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_app),
            base_url="http://testserver",
        ) as ac:
            yield ac

    return _factory


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
def bind_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[httpx.MockTransport], None]:
    """Patch httpx Client/AsyncClient when transport is None, preserving ASGITransport."""

    def _bind(transport: httpx.MockTransport) -> None:
        orig_async_init = httpx.AsyncClient.__init__
        orig_sync_init = httpx.Client.__init__

        def _patched_async_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            if kwargs.get("transport") is None:
                kwargs["transport"] = transport
            orig_async_init(self, *args, **kwargs)

        def _patched_sync_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
            if kwargs.get("transport") is None:
                kwargs["transport"] = transport
            orig_sync_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", _patched_async_init)
        monkeypatch.setattr(httpx.Client, "__init__", _patched_sync_init)

    return _bind


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Typer CliRunner compatible with typer>=0.29."""
    return CliRunner()


@pytest.fixture
def run_eval_studio_subprocess(
    repo_root: Path,
    isolated_env: IsolatedWebPaths,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run `python -m eval_studio ...` in an isolated environment with UTF-8 I/O."""
    src_dir = str(repo_root / "src")

    def _run(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        python_args: Sequence[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        proc_env = os.environ.copy()
        for var in secrets.PROVIDER_ENV_VARS:
            proc_env.pop(var, None)
        proc_env["EVAL_STUDIO_CACHE_DIR"] = str(isolated_env.cache_dir)
        proc_env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
        proc_env["PYTHONUTF8"] = "1"
        proc_env["PYTHONIOENCODING"] = "utf-8"
        existing_pp = proc_env.get("PYTHONPATH", "")
        parts = [p for p in existing_pp.split(os.pathsep) if p] if existing_pp else []
        if src_dir not in parts:
            proc_env["PYTHONPATH"] = os.pathsep.join([src_dir, *parts]) if parts else src_dir
        if env:
            proc_env.update(env)
        cmd: list[str] = [sys.executable]
        if python_args:
            cmd.extend(python_args)
        else:
            cmd.extend(["-m", "eval_studio", *args])
        return subprocess.run(
            cmd,
            cwd=str(cwd if cwd is not None else isolated_env.workspace_dir),
            env=proc_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    return _run


@pytest.fixture
def copied_live_receipts(isolated_env: IsolatedWebPaths) -> dict[str, Path]:
    """Return paths to isolated copies of `live_ds_vs_glm` and `live_ds_vs_glm_rescored`."""
    return {
        "live_ds_vs_glm": isolated_env.receipts_dir / "live_ds_vs_glm",
        "live_ds_vs_glm_rescored": isolated_env.receipts_dir / "live_ds_vs_glm_rescored",
    }


@pytest.fixture
def poll_run_until_done() -> Callable[[Any, str, float], dict[str, Any]]:
    """Poll `GET /api/runs/{run_id}` until status leaves queued/running or timeout expires."""

    def _poll(client: Any, run_id: str, timeout_s: float = 15.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            resp = client.get(f"/api/runs/{run_id}")
            assert resp.status_code == 200, resp.text
            last = resp.json()
            if last.get("status") not in ("queued", "running"):
                return last
            time.sleep(0.05)
        raise TimeoutError(f"Run {run_id} did not finish within {timeout_s}s; last={last}")

    return _poll


@pytest.fixture
def parse_sse_frames() -> Callable[[str], list[dict[str, Any]]]:
    """Parse raw `text/event-stream` response text into structured `{event, data}` dicts."""

    def _parse(raw_text: str) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        normalized = raw_text.replace("\r\n", "\n")
        for block in normalized.split("\n\n"):
            if not block.strip():
                continue
            event_name: str | None = None
            data_lines: list[str] = []
            for line in block.split("\n"):
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())
            if not data_lines:
                continue
            raw_data = "\n".join(data_lines)
            try:
                parsed: Any = json.loads(raw_data)
            except json.JSONDecodeError:
                parsed = raw_data
            if event_name is None and isinstance(parsed, dict) and "event" in parsed:
                event_name = str(parsed["event"])
            frames.append({"event": event_name or "message", "data": parsed})
        return frames

    return _parse
