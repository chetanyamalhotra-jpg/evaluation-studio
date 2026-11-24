"""Infrastructure noise floor measurement (restart, batch-size, engine drift)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseFloorReport:
    """Measured flip rate across repeated runs or backend/batch perturbations."""

    n_decisions: int
    n_flips: int
    flip_rate: float
    perturbation: str


def measure_noise_floor(
    baseline_decisions: Sequence[str],
    perturbed_decisions: Sequence[str],
    *,
    perturbation: str = "restart",
) -> NoiseFloorReport:
    """Compute item-level decision flip rate between two evaluation runs."""
    if len(baseline_decisions) != len(perturbed_decisions) or len(baseline_decisions) == 0:
        raise ValueError("Decision lists must be non-empty and equal length.")
    flips = sum(
        1 for a, b in zip(baseline_decisions, perturbed_decisions, strict=True) if a != b
    )
    n = len(baseline_decisions)
    return NoiseFloorReport(
        n_decisions=n,
        n_flips=flips,
        flip_rate=flips / n,
        perturbation=perturbation,
    )
