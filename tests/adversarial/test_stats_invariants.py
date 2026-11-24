"""Adversarial tests for bootstrap_stderr, micro/macro stderr, clustered SE, and few-shot."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval_studio.core.schema import Sample
from eval_studio.stats.aggregation import (
    SubtaskStats,
    agg_brier_score,
    agg_mean,
    aggregate_group,
    clustered_stderr,
    mcnemar_test,
)
from eval_studio.stats.bootstrap import bootstrap_stderr, mean_stderr, paired_bootstrap_test
from eval_studio.stats.fewshot import (
    ComparisonStatus,
    assign_fewshot_sets,
    compute_required_k_sets,
    evaluate_fewshot_comparison,
    evaluate_fewshot_scores,
    format_fewshot_prompt,
)


def test_bootstrap_stderr_executes_exact_non_multiple_of_1000_iterations() -> None:
    res = bootstrap_stderr([0.0, 1.0, 1.0, 0.0, 1.0], bootstrap_iters=1500, seed=42)
    assert res.iters_executed == 1500
    assert res.stderr > 0.0


def test_micro_vs_macro_stderr_pairing() -> None:
    subtasks = [
        SubtaskStats(name="large", score=0.80, stderr=0.01, n_samples=1000),
        SubtaskStats(name="small", score=0.40, stderr=0.10, n_samples=10),
    ]
    micro = aggregate_group(subtasks, weight_by_size=True)
    macro = aggregate_group(subtasks, weight_by_size=False)

    assert micro.stderr_method == "pooled_sample_stderr"
    assert macro.stderr_method == "unweighted_mean_stderr"
    assert math.isclose(macro.score, 0.60, rel_tol=1e-9)
    # Macro stderr must NOT shrink via sample-size weighting
    assert macro.stderr > micro.stderr * 3.0


def test_heterogeneous_subtask_metrics_cannot_be_silently_dropped() -> None:
    with pytest.raises(ValueError, match="Heterogeneous"):
        aggregate_group(
            [
                SubtaskStats(name="s1", score=0.8, stderr=0.02, n_samples=50, metric="acc"),
                SubtaskStats(name="s2", score=0.4, stderr=0.05, n_samples=50, metric="exact_match"),
            ]
        )


def test_clustered_stderr_matches_mean_stderr_when_unclustered() -> None:
    vals = [1.0, 0.0, 1.0, 0.0, 1.0]
    res = clustered_stderr(vals, [0, 1, 2, 3, 4])
    assert math.isclose(res.stderr, mean_stderr(vals), rel_tol=1e-12)


def test_mcnemar_test_handles_small_discordant_exact_and_zero_discordant() -> None:
    res = mcnemar_test([0, 0, 1], [1, 1, 1])
    assert res.exact is True
    assert res.p_value == pytest.approx(0.5)
    assert mcnemar_test([1, 0], [1, 0]).p_value == 1.0


def test_fewshot_comparison_marks_unresolved_when_ci_spans_zero_or_within_noise_floor() -> None:
    cmp = evaluate_fewshot_comparison([-0.02, 0.01, -0.01, 0.02, 0.00])
    assert cmp.status == ComparisonStatus.UNRESOLVED

    cmp_nf = evaluate_fewshot_comparison([0.01, 0.012, 0.011], noise_floor=0.02)
    assert cmp_nf.status == ComparisonStatus.UNRESOLVED


def test_fewshot_excludes_active_doc_by_hash_and_isolates_gen_prefix() -> None:
    active = Sample(sample_id="active", input_text="Same prompt", target="A", doc_hash="h_same")
    pool = [
        Sample(
            sample_id="other_id_same_doc",
            input_text="Same prompt",
            target="A",
            doc_hash="h_same",
        ),
        Sample(sample_id="ex1", input_text="Clean 1", target="B", doc_hash="h1"),
        Sample(sample_id="ex2", input_text="Clean 2", target="C", doc_hash="h2"),
    ]
    assigned = assign_fewshot_sets([active], pool, num_fewshot=2, k_sets=1, seed=0)
    assert [ex.sample_id for ex in assigned[0].exemplars] == ["ex1", "ex2"] or {
        ex.sample_id for ex in assigned[0].exemplars
    } == {"ex1", "ex2"}

    prompt = format_fewshot_prompt(
        assigned[0].exemplars,
        active,
        render_doc=lambda s: f"Q: {s.input_text}",
        render_target=lambda s: f"A: {s.target}",
        gen_prefix=" <think>",
    )
    assert prompt.count("<think>") == 1
    assert prompt.endswith("Q: Same prompt <think>")
    assert compute_required_k_sets(0.04, 0.02) == 11


def test_stats_functions_accept_numpy_arrays_and_scalars() -> None:
    mc_int = mcnemar_test(np.array([0, 1, 0, 0]), np.array([1, 1, 1, 0]))
    assert mc_int.n_01 == 2
    assert mc_int.n_10 == 0

    mc_bool = mcnemar_test(np.array([False, True, False]), np.array([True, True, False]))
    assert mc_bool.n_01 == 1

    pb = paired_bootstrap_test(
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        bootstrap_iters=50,
        seed=5,
    )
    assert pb.mean_delta == pytest.approx(1.0)
    assert pb.significant is True

    cs = clustered_stderr(np.array([1.0, 1.0, 0.0, 0.0]), np.array(["c1", "c1", "c2", "c2"]))
    assert cs.n_clusters == 2
    assert cs.stderr > 0.0

    assert agg_mean(np.array([0.2, 0.4, 0.6])) == pytest.approx(0.4)
    assert agg_brier_score(np.array([0.1, 0.3])) == pytest.approx(0.2)

    fs_cmp = evaluate_fewshot_comparison(np.array([0.2, 0.3, 0.25]))
    assert fs_cmp.status == ComparisonStatus.RESOLVED_HIGHER
    fs_scores = evaluate_fewshot_scores(np.array([0.5, 0.5, 0.5]), np.array([0.8, 0.85, 0.82]))
    assert fs_scores.status == ComparisonStatus.RESOLVED_HIGHER

    with pytest.raises(ValueError):
        mcnemar_test(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        paired_bootstrap_test(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        evaluate_fewshot_comparison(np.array([]))


def test_evaluate_fewshot_comparison_rejects_invalid_z_value() -> None:
    for bad_z in (0.0, -1.96, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="z_value"):
            evaluate_fewshot_comparison([0.1, 0.2, 0.3], z_value=bad_z)
        with pytest.raises(ValueError, match="z_value"):
            evaluate_fewshot_scores([0.5, 0.5], [0.7, 0.8], z_value=bad_z)
