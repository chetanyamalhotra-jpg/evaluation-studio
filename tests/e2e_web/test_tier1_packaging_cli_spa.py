"""Tier 1 E2E tests: Packaging (`[web]` extra), `eval-studio ui` CLI, `/api/health`, SPA & CORS."""

from __future__ import annotations

import builtins
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import tomllib
from typer.testing import CliRunner

from eval_studio.cli.app import app

pytestmark = pytest.mark.e2e_web

HEAVY_AND_WEB_PACKAGES: frozenset[str] = frozenset(
    {
        "fastapi",
        "uvicorn",
        "python-multipart",
        "torch",
        "transformers",
        "accelerate",
        "vllm",
        "sglang",
        "docker",
        "pypdf",
    }
)


def _extract_pkg_name(req: str) -> str:
    for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
        req = req.split(sep, 1)[0]
    return req.strip().lower()


# ---------------------------------------------------------------------------
# Area 1: Packaging (`[web]` extra, slim core) & CLI (`eval-studio ui`) — 6 Tests
# ---------------------------------------------------------------------------


def test_t1_pkg_01_pyproject_web_extra_and_slim_core(repo_root: Path) -> None:
    """T1-PKG-01: Base deps remain slim; `[web]` and `[dev]` expose web deps & static assets."""
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    base_deps = {_extract_pkg_name(r) for r in data["project"]["dependencies"]}
    assert base_deps.isdisjoint(HEAVY_AND_WEB_PACKAGES)

    extras = data["project"]["optional-dependencies"]
    assert "web" in extras
    web_deps = {_extract_pkg_name(r) for r in extras["web"]}
    assert {"fastapi", "uvicorn", "python-multipart"}.issubset(web_deps)

    dev_raw = [str(x).strip().lower() for x in extras["dev"]]
    dev_names = {_extract_pkg_name(r) for r in extras["dev"]}
    assert {"fastapi", "uvicorn", "python-multipart"} <= dev_names or any(
        d.startswith(("evaluation-studio[web]", "evaluation-studio[web]")) for d in dev_raw
    )

    pkg_data = data["tool"]["setuptools"]["package-data"]["eval_studio"]
    assert any("web/static" in entry for entry in pkg_data)

    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    ignore_rules = [
        line.strip()
        for line in gitignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert any("web/node_modules" in r for r in ignore_rules)
    assert any("web/dist" in r for r in ignore_rules)
    assert not any("src/eval_studio/web/static" in r for r in ignore_rules)


def test_t1_pkg_02_eval_studio_ui_help_lists_expected_flags(cli_runner: CliRunner) -> None:
    """T1-PKG-02: `eval-studio ui --help` documents --host, --port, --open/--no-open, --reload."""
    result = cli_runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "--host" in out
    assert "127.0.0.1" in out
    assert "--port" in out
    assert "8420" in out
    assert "--open" in out
    assert "--no-open" in out
    assert "--reload" in out


def test_t1_pkg_03_cli_help_and_ls_stay_lazy_without_web_imports(
    run_eval_studio_subprocess: Callable[..., Any],
) -> None:
    """T1-PKG-03: `eval-studio --help` and `eval-studio ls` never eagerly import web frameworks."""
    probe = (
        "import sys; "
        "from typer.testing import CliRunner; "
        "from eval_studio.cli.app import app; "
        "r1 = CliRunner().invoke(app, ['--help']); "
        "r2 = CliRunner().invoke(app, ['ls']); "
        "assert r1.exit_code == 0 and r2.exit_code == 0; "
        "prefixes = ('fastapi', 'uvicorn', 'eval_studio.web', 'python_multipart', 'multipart'); "
        "loaded = [m for m in sys.modules if m.startswith(prefixes)]; "
        "assert not loaded, f'Eagerly imported web modules: {loaded}'"
    )
    proc = run_eval_studio_subprocess([], python_args=["-c", probe])
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"


def test_t1_pkg_04_eval_studio_ui_missing_web_extra_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner: CliRunner,
) -> None:
    """T1-PKG-04: Missing `[web]` extra causes `eval-studio ui` to exit 1 with install hint."""
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.setitem(sys.modules, "eval_studio.web.server", None)
    real_import = builtins.__import__

    def _block_web(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name.startswith(("uvicorn", "fastapi", "eval_studio.web")):
            raise ImportError(f"Simulated missing [web] dependency: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_web)
    result = cli_runner.invoke(app, ["ui", "--no-open"])
    assert result.exit_code == 1
    out = result.output.lower()
    assert "web" in out
    assert "install" in out
    assert "traceback" not in out


def test_t1_pkg_05_eval_studio_ui_cli_invokes_uvicorn_with_configured_flags(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner: CliRunner,
) -> None:
    """T1-PKG-05: `eval-studio ui` forwards host/port/reload to uvicorn and respects --no-open."""
    uvicorn_mod = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    pytest.importorskip("eval_studio.web.server")

    recorded_calls: list[dict[str, Any]] = []
    browser_urls: list[str] = []

    def _fake_uvicorn_run(app_obj: Any, **kwargs: Any) -> None:
        recorded_calls.append({"app": app_obj, **kwargs})

    monkeypatch.setattr(uvicorn_mod, "run", _fake_uvicorn_run)
    monkeypatch.setattr("webbrowser.open", lambda url: browser_urls.append(url))

    result = cli_runner.invoke(
        app,
        ["ui", "--host", "127.0.0.1", "--port", "9123", "--no-open", "--reload"],
    )
    assert result.exit_code == 0, result.output
    assert len(recorded_calls) == 1
    call = recorded_calls[0]
    assert call.get("host") == "127.0.0.1"
    assert call.get("port") == 9123
    assert call.get("reload") is True
    assert browser_urls == []


def test_t1_pkg_06_static_spa_bundle_shipped_in_package(repo_root: Path) -> None:
    """T1-PKG-06: Production Vite SPA bundle is shipped under `src/eval_studio/web/static/`."""
    static_dir = repo_root / "src" / "eval_studio" / "web" / "static"
    index_html = static_dir / "index.html"
    assert index_html.is_file()
    html = index_html.read_text(encoding="utf-8")
    assert len(html) > 100
    assert 'id="root"' in html or "id='root'" in html
    assert "<script" in html
    assets_dir = static_dir / "assets"
    assert assets_dir.is_dir()
    js_files = list(assets_dir.glob("*.js"))
    assert len(js_files) >= 1


# ---------------------------------------------------------------------------
# Area 2: `/api/health` & Static SPA Serving + CORS — 5 Tests
# ---------------------------------------------------------------------------


def test_t1_hlt_01_health_endpoint_contract(client: Any) -> None:
    """T1-HLT-01: `GET /api/health` returns status, version, pdf_available, static_ready."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert isinstance(body["pdf_available"], bool)
    assert isinstance(body["static_ready"], bool)


def test_t1_hlt_02_static_spa_root_and_client_routes_return_html(client: Any) -> None:
    """T1-HLT-02: Root and all 6 SPA client routes return `text/html` index payload."""
    health = client.get("/api/health").json()
    if not health.get("static_ready"):
        pytest.skip("Static SPA bundle not yet built into src/eval_studio/web/static/")

    for route in ("/", "/compare", "/run", "/papers", "/tests", "/receipts", "/settings"):
        resp = client.get(route)
        assert resp.status_code == 200, f"Route {route} returned {resp.status_code}"
        assert resp.headers.get("content-type", "").startswith("text/html")
        assert "<html" in resp.text.lower()
        assert "root" in resp.text


def test_t1_hlt_03_unknown_api_route_returns_json_404_not_spa_html(client: Any) -> None:
    """T1-HLT-03: Unknown `/api/*` paths return JSON 404 instead of SPA `index.html`."""
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    assert resp.headers.get("content-type", "").startswith("application/json")
    payload = resp.json()
    assert "detail" in payload


def test_t1_hlt_04_cors_allows_vite_dev_origin_5173(client: Any) -> None:
    """T1-HLT-04: CORS middleware allows `http://localhost:5173` and `http://127.0.0.1:5173`."""
    for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
        pre = client.options(
            "/api/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert pre.status_code in (200, 204)
        assert pre.headers.get("access-control-allow-origin") == origin

        get_resp = client.get("/api/health", headers={"Origin": origin})
        assert get_resp.status_code == 200
        assert get_resp.headers.get("access-control-allow-origin") == origin


def test_t1_hlt_05_openapi_schema_and_docs_accessible(client: Any) -> None:
    """T1-HLT-05: `/openapi.json` exposes all Web Studio API route groups."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = json.loads(resp.text)
    paths = spec.get("paths", {})
    for expected in (
        "/api/health",
        "/api/byok/presets",
        "/api/byot/tasks",
        "/api/byop/papers",
        "/api/runs",
        "/api/receipts",
    ):
        assert expected in paths, f"Missing {expected} from OpenAPI paths"
