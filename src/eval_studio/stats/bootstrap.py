"""Exact bootstrap standard error estimation and paired bootstrap significance testing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap uncertainty summary."""

    stderr: float
    iters_executed: int
    seed: int


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired bootstrap comparison between two models evaluated on the same items."""

    mean_delta: float
    stderr: float
    ci_low: float
    ci_high: float
    p_value: float
    iters_executed: int
    seed: int
    significant: bool


def mean_stderr(values: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Compute classical sample standard error of the mean `s / sqrt(n)`."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite.")
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def bootstrap_stderr(
    values: Sequence[float] | np.ndarray[Any, Any],
    statistic: Callable[[np.ndarray[tuple[int], np.dtype[np.float64]]], float] | None = None,
    *,
    bootstrap_iters: int = 1000,
    seed: int = 1234,
) -> BootstrapResult:
    """Compute bootstrap standard error over the exact requested bootstrap_iters resamples."""
    if bootstrap_iters < 0:
        raise ValueError("bootstrap_iters must be >= 0.")

    arr = np.asarray(values, dtype=np.float64)
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite.")

    if bootstrap_iters == 0 or len(arr) < 2:
        return BootstrapResult(stderr=0.0, iters_executed=0, seed=seed)

    n = len(arr)
    rng = np.random.default_rng(seed)
    stat_fn = statistic if statistic is not None else lambda x: float(np.mean(x))

    estimates = np.empty(bootstrap_iters, dtype=np.float64)
    for i in range(bootstrap_iters):
        idx = rng.integers(0, n, size=n)
        estimates[i] = stat_fn(arr[idx])

    stderr = float(np.std(estimates, ddof=1)) if bootstrap_iters > 1 else 0.0
    return BootstrapResult(stderr=stderr, iters_executed=bootstrap_iters, seed=seed)


def paired_bootstrap_test(
    scores_a: Sequence[float] | np.ndarray[Any, Any],
    scores_b: Sequence[float] | np.ndarray[Any, Any],
    *,
    bootstrap_iters: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 1234,
) -> PairedBootstrapResult:
    """Run an exact paired bootstrap test comparing `scores_b` against `scores_a`."""
    if len(scores_a) != len(scores_b) or len(scores_a) == 0:
        raise ValueError("scores_a and scores_b must be non-empty and equal length.")
    if bootstrap_iters <= 0:
        raise ValueError("bootstrap_iters must be > 0.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must lie strictly between 0 and 1.")

    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
        raise ValueError("scores_a and scores_b must contain only finite values.")

    diffs = arr_b - arr_a
    n = len(diffs)
    mean_delta = float(np.mean(diffs))
    rng = np.random.default_rng(seed)

    boot_deltas = np.empty(bootstrap_iters, dtype=np.float64)
    for i in range(bootstrap_iters):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = float(np.mean(diffs[idx]))

    stderr = float(np.std(boot_deltas, ddof=1)) if bootstrap_iters > 1 else 0.0
    alpha = 1.0 - confidence_level
    q_low, q_high = np.quantile(boot_deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    ci_low = float(q_low)
    ci_high = float(q_high)

    if mean_delta == 0.0:
        p_value = 1.0
    else:
        null_deltas = boot_deltas - mean_delta
        extreme = int(np.sum(np.abs(null_deltas) >= abs(mean_delta)))
        p_value = float((extreme + 1) / (bootstrap_iters + 1))

    significant = bool(ci_low > 0.0 or ci_high < 0.0)
    return PairedBootstrapResult(
        mean_delta=mean_delta,
        stderr=stderr,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        iters_executed=bootstrap_iters,
        seed=seed,
        significant=significant,
    )
