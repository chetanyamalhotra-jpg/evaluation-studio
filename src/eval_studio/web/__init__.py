"""Local-first Web Studio backend for Evaluation Studio (`eval-studio ui`).

Keep package initialization lazy so importing `eval_studio` or running `eval-studio --help`
never eagerly imports `fastapi`, `uvicorn`, or `python-multipart`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from eval_studio.web.server import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
