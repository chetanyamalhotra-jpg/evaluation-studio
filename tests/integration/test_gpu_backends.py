"""GPU backend integration tests (skipped by default offline CPU pytest runs)."""

from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_hf_and_vllm_gpu_parity_placeholder() -> None:
    pytest.skip("Milestone 2: requires CUDA GPU and optional [hf]/[vllm] extras.")
