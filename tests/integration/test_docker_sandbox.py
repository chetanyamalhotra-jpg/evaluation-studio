"""Docker Tier 3 sandbox integration tests (skipped by default offline CPU pytest runs)."""

from __future__ import annotations

import pytest


@pytest.mark.docker
def test_docker_sandbox_isolation_placeholder() -> None:
    pytest.skip("Milestone 3: requires Docker daemon and optional [sandbox] extra.")
