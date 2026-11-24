"""Unit tests verifying F1 slim core dependencies and optional extras in pyproject.toml."""

from __future__ import annotations

from pathlib import Path

import tomllib

HEAVY_PACKAGES = {
    "torch",
    "transformers",
    "accelerate",
    "vllm",
    "sglang",
    "docker",
    "pypdf",
}


def _extract_pkg_name(req: str) -> str:
    for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
        req = req.split(sep, 1)[0]
    return req.strip().lower()


def test_pyproject_core_dependencies_are_slim(repo_root: Path) -> None:
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = {_extract_pkg_name(r) for r in data["project"]["dependencies"]}
    assert deps.isdisjoint(HEAVY_PACKAGES)


def test_pyproject_optional_extras_and_entry_point_groups(repo_root: Path) -> None:
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    for group in ("api", "hf", "vllm", "pdf", "sandbox", "dev"):
        assert group in extras

    eps = data["project"]["entry-points"]
    for group in (
        "eval_studio.models",
        "eval_studio.solvers",
        "eval_studio.filters",
        "eval_studio.metrics",
        "eval_studio.aggregations",
        "eval_studio.sandboxes",
    ):
        assert group in eps
