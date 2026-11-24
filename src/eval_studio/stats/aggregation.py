"""Hierarchical group aggregation, clustered SEs, and paired McNemar significance testing."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from eval_studio.scoring.metrics import bits_per_byte, brier_score, bypass, weighted_perplexity


@dataclass(frozen=True)
class SubtaskStats:
    """Leaf subtask score, standard error, and sample size."""

    name: str
    score: float
    stderr: float
    n_samples: int
    metric: str = "score"


@dataclass(frozen=True)
class GroupAggregate:
    """Aggregated group score and properly paired standard error."""

    score: float
    stderr: float
    weight_by_size: bool
    stderr_method: str
    total_samples: int
    metric: str = "score"


@dataclass(frozen=True)
class ClusteredStdErrResult:
    """CR1 finite-sample cluster-robust standard error for clustered/passage items."""

    mean: float
    stderr: float
    n_samples: int
    n_clusters: int
    method: str = "cluster_robust_mean"


@dataclass(frozen=True)
class McNemarResult:
    """Paired binary significance test result comparing model B vs. model A."""

    n_items: int
    n_01: int
    n_10: int
    statistic: float
    p_value: float
    exact: bool
    delta: float


def aggregate_group(
    subtasks: Sequence[SubtaskStats],
    *,
    weight_by_size: bool = True,
) -> GroupAggregate:
    """Aggregate subtasks using micro-averaging (pooled_sample_stderr) or macro (unweighted)."""
    if not subtasks:
        raise ValueError("subtasks must be non-empty.")

    for subtask in subtasks:
        if subtask.n_samples <= 0:
            raise ValueError(
                f"Subtask '{subtask.name}' n_samples must be > 0 (got {subtask.n_samples})."
            )
        if subtask.stderr < 0.0 or not (
            math.isfinite(subtask.score) and math.isfinite(subtask.stderr)
        ):
            raise ValueError(
                f"Subtask '{subtask.name}' must have finite score and non-negative finite stderr."
            )

    metric_keys = sorted({s.metric for s in subtasks})
    if len(metric_keys) > 1:
        raise ValueError(
            f"Heterogeneous subtask metrics {metric_keys} cannot be collapsed into a single "
            "GroupAggregate; use aggregate_group_metrics() instead."
        )
    metric_name = metric_keys[0]

    total_n = sum(s.n_samples for s in subtasks)
    if total_n <= 0:
        raise ValueError("Total sample count across subtasks must be > 0.")

    if weight_by_size:
        score = sum(s.score * s.n_samples for s in subtasks) / total_n
        variance = sum(((s.n_samples / total_n) ** 2) * (s.stderr**2) for s in subtasks)
        return GroupAggregate(
            score=score,
            stderr=math.sqrt(variance),
            weight_by_size=True,
            stderr_method="pooled_sample_stderr",
            total_samples=total_n,
            metric=metric_name,
        )

    m = len(subtasks)
    score = sum(s.score for s in subtasks) / m
    variance = sum((s.stderr**2) for s in subtasks) / (m**2)
    return GroupAggregate(
        score=score,
        stderr=math.sqrt(variance),
        weight_by_size=False,
        stderr_method="unweighted_mean_stderr",
        total_samples=total_n,
        metric=metric_name,
    )


def aggregate_group_metrics(
    subtasks: Sequence[SubtaskStats],
    *,
    weight_by_size: bool = True,
) -> dict[str, GroupAggregate]:
    """Aggregate subtasks partitioned by metric key so heterogeneous metrics are preserved."""
    if not subtasks:
        raise ValueError("subtasks must be non-empty.")
    grouped: dict[str, list[SubtaskStats]] = {}
    for subtask in subtasks:
        grouped.setdefault(subtask.metric, []).append(subtask)
    return {
        metric: aggregate_group(group, weight_by_size=weight_by_size)
        for metric, group in grouped.items()
    }


def clustered_stderr(
    values: Sequence[float] | np.ndarray[Any, Any],
    cluster_ids: Sequence[str | int] | np.ndarray[Any, Any],
) -> ClusteredStdErrResult:
    """Compute CR1 cluster-robust standard error of the mean across correlated clusters."""
    if len(values) == 0 or len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must be non-empty and equal length.")
    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError("values must contain only finite numbers.")

    n_samples = len(values)
    mean_val = sum(float(v) for v in values) / n_samples

    cluster_sums: dict[str | int, float] = defaultdict(float)
    for val, cid in zip(values, cluster_ids, strict=True):
        cluster_sums[cid] += float(val) - mean_val

    n_clusters = len(cluster_sums)
    if n_clusters < 2:
        return ClusteredStdErrResult(
            mean=mean_val,
            stderr=0.0,
            n_samples=n_samples,
            n_clusters=n_clusters,
        )

    sum_sq = sum(s * s for s in cluster_sums.values())
    var_cr1 = (n_clusters / (n_clusters - 1)) * (sum_sq / (n_samples**2))
    return ClusteredStdErrResult(
        mean=mean_val,
        stderr=math.sqrt(var_cr1),
        n_samples=n_samples,
        n_clusters=n_clusters,
    )


def _to_binary_verdict(
    value: bool | int | float | np.bool_ | np.integer[Any] | np.floating[Any],
) -> int:
    if isinstance(value, (bool, np.bool_)):
        return 1 if value else 0
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        if math.isfinite(fval) and fval in (0.0, 1.0):
            return int(fval)
    raise ValueError(f"McNemar test requires binary 0/1 or bool outcomes; got {value!r}.")


def mcnemar_test(
    scores_a: Sequence[bool | int | float] | np.ndarray[Any, Any],
    scores_b: Sequence[bool | int | float] | np.ndarray[Any, Any],
    *,
    exact: bool | None = None,
    continuity_correction: bool = True,
) -> McNemarResult:
    """Run paired McNemar's test comparing binary outcomes of model B vs. model A."""
    if len(scores_a) == 0 or len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must be non-empty and equal length.")

    n_01 = 0
    n_10 = 0
    for raw_a, raw_b in zip(scores_a, scores_b, strict=True):
        a = _to_binary_verdict(raw_a)
        b = _to_binary_verdict(raw_b)
        if a == 0 and b == 1:
            n_01 += 1
        elif a == 1 and b == 0:
            n_10 += 1

    n_items = len(scores_a)
    n_d = n_01 + n_10
    use_exact = (n_d < 25) if exact is None else exact
    delta = (n_01 - n_10) / n_items

    if n_d == 0:
        return McNemarResult(
            n_items=n_items,
            n_01=0,
            n_10=0,
            statistic=0.0,
            p_value=1.0,
            exact=use_exact,
            delta=0.0,
        )

    diff = abs(n_01 - n_10)
    stat = (max(0.0, diff - 1.0) ** 2) / n_d if continuity_correction else (diff**2) / n_d

    if use_exact:
        k_min = min(n_01, n_10)
        if 2 * k_min == n_d:
            p_val = 1.0
        else:
            log_two = math.log(2.0)
            lgamma_n = math.lgamma(n_d + 1)
            tail_sum = sum(
                math.exp(lgamma_n - math.lgamma(i + 1) - math.lgamma(n_d - i + 1) - n_d * log_two)
                for i in range(k_min + 1)
            )
            p_val = min(1.0, float(2.0 * tail_sum))
    else:
        # Chi-square with df=1 survival function equals erfc(sqrt(stat / 2)).
        p_val = float(math.erfc(math.sqrt(stat / 2.0)))

    return McNemarResult(
        n_items=n_items,
        n_01=n_01,
        n_10=n_10,
        statistic=float(stat),
        p_value=p_val,
        exact=use_exact,
        delta=delta,
    )


# --- Registry-compatible aggregation wrappers (`eval_studio.aggregations`) ---


def agg_mean(values: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Compute arithmetic mean over finite sample scores."""
    if len(values) == 0:
        raise ValueError("values must be non-empty.")
    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError("values must be finite.")
    return sum(float(v) for v in values) / len(values)


def agg_pooled_sample_stderr(subtasks: Sequence[SubtaskStats]) -> GroupAggregate:
    """Micro-averaged group aggregation using pooled_sample_stderr."""
    return aggregate_group(subtasks, weight_by_size=True)


def agg_unweighted_mean_stderr(subtasks: Sequence[SubtaskStats]) -> GroupAggregate:
    """Macro-averaged group aggregation using unweighted_mean_stderr."""
    return aggregate_group(subtasks, weight_by_size=False)


def agg_weighted_perplexity(
    loglikelihoods: Sequence[float],
    weights: Sequence[int | float],
) -> float:
    """Corpus-level weighted perplexity aggregation."""
    return weighted_perplexity(loglikelihoods, weights)


def agg_bits_per_byte(loglikelihoods: Sequence[float], texts: Sequence[str]) -> float:
    """Corpus-level bits-per-byte aggregation."""
    return bits_per_byte(loglikelihoods, texts)


def agg_brier_score(
    scores: Sequence[float] | np.ndarray[Any, Any],
) -> float:
    """Mean Brier score across evaluated items (or call scoring.metrics.brier_score per item)."""
    if len(scores) == 0:
        raise ValueError("scores must be non-empty.")
    return agg_mean(scores)


def agg_bypass(*, log_samples: bool = True) -> float | None:
    """No-op aggregation for `bypass` metric enforcing `log_samples=True`."""
    return bypass(log_samples=log_samples)


# Keep direct reference to brier_score available for mypy/module parity
__all_brier__ = brier_score
