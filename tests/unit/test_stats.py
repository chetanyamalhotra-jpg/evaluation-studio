"""Unit tests for F10 statistical uncertainty, clustered SEs, and paired tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval_studio.stats.aggregation import (
    SubtaskStats,
    aggregate_group,
    aggregate_group_metrics,
    clustered_stderr,
    mcnemar_test,
)
from eval_studio.stats.bootstrap import bootstrap_stderr, mean_stderr, paired_bootstrap_test


def test_bootstrap_stderr_edge_cases_and_seed_isolation() -> None:
    assert bootstrap_stderr([1.0], bootstrap_iters=50).iters_executed == 0
    assert bootstrap_stderr([1.0, 2.0], bootstrap_iters=0).iters_executed == 0
    assert bootstrap_stderr([1.0, 2.0], bootstrap_iters=1).iters_executed == 1
    assert bootstrap_stderr([1.0, 2.0, 3.0], bootstrap_iters=37).iters_executed == 37

    np.random.seed(999)
    res_a = bootstrap_stderr([0.0, 1.0, 0.5, 1.0], bootstrap_iters=50, seed=123)
    np.random.seed(1)
    res_b = bootstrap_stderr([0.0, 1.0, 0.5, 1.0], bootstrap_iters=50, seed=123)
    assert res_a == res_b

    with pytest.raises(ValueError, match="finite"):
        bootstrap_stderr([1.0, float("nan")])


def test_paired_bootstrap_test() -> None:
    a = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    res = paired_bootstrap_test(a, b, bootstrap_iters=350, seed=7)
    assert res.iters_executed == 350
    assert res.mean_delta == pytest.approx(0.875)
    assert res.ci_low > 0.0
    assert res.significant is True
    assert res.p_value < 0.05


def test_aggregate_group_guards_and_metric_partitioning() -> None:
    with pytest.raises(ValueError, match="Heterogeneous"):
        aggregate_group(
            [
                SubtaskStats(name="t1", score=0.8, stderr=0.02, n_samples=50, metric="acc"),
                SubtaskStats(name="t2", score=0.5, stderr=0.03, n_samples=50, metric="f1"),
            ]
        )

    with pytest.raises(ValueError, match="n_samples"):
        aggregate_group([SubtaskStats(name="bad", score=0.5, stderr=0.1, n_samples=0)])

    grouped = aggregate_group_metrics(
        [
            SubtaskStats(name="t1", score=0.8, stderr=0.02, n_samples=100, metric="acc"),
            SubtaskStats(name="t2", score=0.6, stderr=0.04, n_samples=100, metric="acc"),
            SubtaskStats(name="t3", score=0.5, stderr=0.03, n_samples=50, metric="f1"),
        ],
        weight_by_size=True,
    )
    assert grouped["acc"].score == pytest.approx(0.7)
    assert grouped["f1"].score == pytest.approx(0.5)


def test_clustered_stderr_invariants() -> None:
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    # When G == N (1 item per cluster), CR1 SE matches classical mean_stderr
    res_unclustered = clustered_stderr(vals, list(range(len(vals))))
    assert math.isclose(res_unclustered.stderr, mean_stderr(vals), rel_tol=1e-12)

    # Correlated cluster outcomes yield larger SE than iid mean_stderr
    corr_vals = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    corr_clusters = ["p1", "p1", "p1", "p2", "p2", "p2"]
    res_clustered = clustered_stderr(corr_vals, corr_clusters)
    assert res_clustered.stderr > mean_stderr(corr_vals)

    # Single cluster is safe
    single = clustered_stderr([1.0, 0.0], ["only", "only"])
    assert single.stderr == 0.0
    assert single.n_clusters == 1


def test_mcnemar_exact_and_asymptotic_modes() -> None:
    # Exact binomial (n_d = 8 < 25)
    a = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    res_exact = mcnemar_test(a, b)
    assert res_exact.exact is True
    assert res_exact.n_01 == 8
    assert res_exact.n_10 == 0
    assert res_exact.p_value == pytest.approx(2.0 * (0.5**8))

    # Asymptotic chi-square (n_d >= 25)
    a_big = [0] * 30 + [1] * 5
    b_big = [1] * 30 + [0] * 5
    res_chi2 = mcnemar_test(a_big, b_big)
    assert res_chi2.exact is False
    assert res_chi2.p_value < 0.001

    # Zero discordant pairs
    tie = mcnemar_test([True, False], [True, False])
    assert tie.p_value == 1.0
    assert tie.statistic == 0.0
