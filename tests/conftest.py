"""Shared offline fixtures for Evaluation Studio tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def examples_dir(repo_root: Path) -> Path:
    return repo_root / "examples"
