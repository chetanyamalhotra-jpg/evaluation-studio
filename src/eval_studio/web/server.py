"""FastAPI application factory, lifespan, redacted error handlers, and SPA static fallback."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import sys
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from eval_studio import __version__
from eval_studio.cache.fingerprint import resolve_cache_dir
from eval_studio.gateway.secrets import redact_secrets
from eval_studio.web.schemas import HealthResponse

_DEV_CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def check_pdf_available() -> bool:
    """Return True if optional `pypdf` dependency is importable without importing it."""
    if "pypdf" in sys.modules:
        return sys.modules["pypdf"] is not None
    try:
        return importlib.util.find_spec("pypdf") is not None
    except (ImportError, ValueError):
        return False


@dataclass(frozen=True)
class AppConfig:
    """Resolved workspace and storage paths for the Web Studio server."""

    workspace_dir: Path
    receipts_dir: Path
    tasks_dir: Path
    cache_dir: Path | None
    uploads_dir: Path
    examples_dir: Path
    static_dir: Path


@dataclass
class AppState:
    """Mutable runtime state attached to `app.state.studio`."""

    config: AppConfig
    runs_state: dict[str, Any] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def effective_cache_dir(self) -> Path:
        """Resolve active cache directory, honoring runtime EVAL_STUDIO_CACHE_DIR overrides."""
        if self.config.cache_dir is not None:
            return self.config.cache_dir
        return resolve_cache_dir()


def get_studio_state(request: Request) -> AppState:
    """FastAPI dependency returning the typed AppState from `app.state.studio`."""
    return cast(AppState, request.app.state.studio)


def _extract_input_secrets(raw_input: Any) -> list[str]:
    """Collect candidate secret strings from validation error inputs for extra redaction."""
    found: list[str] = []
    if isinstance(raw_input, dict):
        for key, val in raw_input.items():
            key_lower = str(key).lower()
            if isinstance(val, str) and (
                "key" in key_lower or "secret" in key_lower or "token" in key_lower
            ):
                cleaned = val.strip()
                if len(cleaned) >= 4:
                    found.append(cleaned)
            elif isinstance(val, (dict, list)):
                found.extend(_extract_input_secrets(val))
    elif isinstance(raw_input, list):
        for item in raw_input:
            found.extend(_extract_input_secrets(item))
    return found


def _mount_optional_router(app: FastAPI, module_path: str) -> None:
    """Mount an optional M3 router module without masking internal import errors."""
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name == module_path:
            return
        raise
    router = getattr(mod, "router", None)
    if router is not None:
        app.include_router(router)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage Web Studio startup/shutdown and cancel pending background run tasks."""
    yield
    studio = cast(AppState | None, getattr(app.state, "studio", None))
    if studio is None:
        return
    with studio.lock:
        entries = list(studio.runs_state.values())
    for entry in entries:
        task = entry.get("task") if isinstance(entry, dict) else getattr(entry, "task", None)
        if isinstance(task, asyncio.Task) and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


def _resolve_examples_dir(ws: Path, examples_dir: Path | str | None) -> Path:
    if examples_dir is not None:
        return Path(examples_dir).resolve()
    ws_examples = (ws / "examples").resolve()
    if ws_examples.is_dir():
        return ws_examples
    repo_examples = (Path(__file__).resolve().parents[3] / "examples").resolve()
    if repo_examples.is_dir():
        return repo_examples
    return ws_examples


def create_app(
    workspace_dir: Path | str | None = None,
    receipts_dir: Path | str | None = None,
    tasks_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    uploads_dir: Path | str | None = None,
    static_dir: Path | str | None = None,
    examples_dir: Path | str | None = None,
) -> FastAPI:
    """Create and configure the Evaluation Studio FastAPI application."""
    ws = Path(workspace_dir).resolve() if workspace_dir is not None else Path.cwd().resolve()
    rec = Path(receipts_dir).resolve() if receipts_dir is not None else (ws / "receipts").resolve()
    tsk = Path(tasks_dir).resolve() if tasks_dir is not None else (ws / "tasks").resolve()
    cdir = Path(cache_dir).resolve() if cache_dir is not None else None
    upl = (
        Path(uploads_dir).resolve()
        if uploads_dir is not None
        else (ws / "uploads" / "papers").resolve()
    )
    ex = _resolve_examples_dir(ws, examples_dir)
    st = (
        Path(static_dir).resolve()
        if static_dir is not None
        else (Path(__file__).resolve().parent / "static").resolve()
    )

    cfg = AppConfig(
        workspace_dir=ws,
        receipts_dir=rec,
        tasks_dir=tsk,
        cache_dir=cdir,
        uploads_dir=upl,
        examples_dir=ex,
        static_dir=st,
    )
    state = AppState(config=cfg)

    app = FastAPI(
        title="Evaluation Studio",
        version=__version__,
        lifespan=_lifespan,
    )

    # Expose structured state and top-level fields cleanly for M3 extensibility
    app.state.studio = state
    app.state.workspace_dir = cfg.workspace_dir
    app.state.receipts_dir = cfg.receipts_dir
    app.state.tasks_dir = cfg.tasks_dir
    app.state.cache_dir = cfg.cache_dir
    app.state.uploads_dir = cfg.uploads_dir
    app.state.examples_dir = cfg.examples_dir
    app.state.static_dir = cfg.static_dir
    app.state.runs_state = state.runs_state
    app.state.lock = state.lock

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_DEV_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        _request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": redact_secrets(str(exc.detail))},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        parts: list[str] = []
        extra: list[str] = []
        for err in exc.errors():
            extra.extend(_extract_input_secrets(err.get("input")))
        extra_tuple = tuple(extra)
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            msg = redact_secrets(str(err.get("msg", "Validation error")), extra_secrets=extra_tuple)
            parts.append(f"{loc}: {msg}" if loc else msg)
        detail = redact_secrets(
            "; ".join(parts) or "Invalid request payload.",
            extra_secrets=extra_tuple,
        )
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        detail = redact_secrets(str(exc) or "Internal server error")
        return JSONResponse(status_code=500, content={"detail": detail})

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            pdf_available=check_pdf_available(),
            static_ready=(cfg.static_dir / "index.html").is_file(),
        )

    # M2 routers
    _mount_optional_router(app, "eval_studio.web.routes_byok")
    _mount_optional_router(app, "eval_studio.web.routes_byot")
    _mount_optional_router(app, "eval_studio.web.routes_byop")
    # M3 routers (mounted automatically when implemented)
    _mount_optional_router(app, "eval_studio.web.routes_runs")
    _mount_optional_router(app, "eval_studio.web.routes_receipts")

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def spa_fallback(request: Request, full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/") or request.method != "GET":
            raise HTTPException(status_code=404, detail="API endpoint not found")

        index_file = (cfg.static_dir / "index.html").resolve()
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="Web Studio static bundle not built")

        static_root = cfg.static_dir.resolve()
        if full_path:
            candidate = (static_root / full_path).resolve()
            if not candidate.is_relative_to(static_root):
                raise HTTPException(status_code=404, detail="Not found")
            if candidate.is_file():
                return FileResponse(candidate)

        return FileResponse(index_file)

    return app
