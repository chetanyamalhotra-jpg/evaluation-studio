"""Live external BYOK API integration tests (skipped by default offline CPU pytest runs)."""

from __future__ import annotations

import pytest


@pytest.mark.live_api
def test_live_byok_http_gateway_placeholder() -> None:
    pytest.skip("Milestone 2: requires live TARGET_API_KEY and network access.")
