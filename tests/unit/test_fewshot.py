"""Unit tests for F11 few-shot hygiene, K-set rotation, and noise-floor comparison."""

from __future__ import annotations

import pytest

from eval_studio.core.schema import Sample
from eval_studio.stats.fewshot import (
    ComparisonStatus,
    assign_fewshot_sets,
    compute_required_k_sets,
    evaluate_fewshot_comparison,
    evaluate_fewshot_scores,
    format_fewshot_prompt,
)


def test_assign_fewshot_sets_excludes_active_id_and_doc_hash() -> None:
    pool = [
        Sample(sample_id="p1", input_text="Same text", target="A", doc_hash="dup_hash"),
        Sample(sample_id="p2", input_text="Other 1", target="B", doc_hash="h2"),
        Sample(sample_id="p3", input_text="Other 2", target="C", doc_hash="h3"),
        Sample(sample_id="p4", input_text="Other 3", target="D", doc_hash="h4"),
    ]
    test_items = [
        Sample(sample_id="test_1", input_text="Same text", target="A", doc_hash="dup_hash"),
        Sample(sample_id="test_2", input_text="Fresh Q", target="B", doc_hash="fresh_hash"),
    ]
    assigned = assign_fewshot_sets(test_items, pool, num_fewshot=2, k_sets=2, seed=42)
    assert len(assigned) == 2
    assert assigned[0].set_index == 0
    assert assigned[1].set_index == 1
    for item in assigned[0].exemplars:
        assert item.sample_id != "test_1"
        assert item.doc_hash != "dup_hash"

    with pytest.raises(ValueError, match="Insufficient non-leaking"):
        assign_fewshot_sets(test_items, pool[:1], num_fewshot=1, k_sets=1)


def test_format_fewshot_prompt_applies_gen_prefix_only_to_test_doc() -> None:
    ex = Sample(sample_id="ex1", input_text="2+2?", target="4")
    test_doc = Sample(sample_id="q1", input_text="3+3?", target="6")
    rendered = format_fewshot_prompt(
        [ex],
        test_doc,
        render_doc=lambda s: f"Q: {s.input_text}\nA:",
        render_target=lambda s: str(s.target),
        gen_prefix=" <think>",
    )
    assert rendered == "Q: 2+2?\nA: 4\n\nQ: 3+3?\nA: <think>"


def test_compute_required_k_sets_and_noise_floor_unresolved() -> None:
    # ceil((1.645 * 0.04 / 0.02)^2) = ceil(3.29^2) = ceil(10.8241) = 11
    assert compute_required_k_sets(0.04, 0.02) == 11
    assert compute_required_k_sets(0.0, 0.01) == 1

    # Positive CI above zero, yet within measured noise_floor -> UNRESOLVED
    cmp_nf = evaluate_fewshot_comparison([0.015, 0.016, 0.014, 0.015], noise_floor=0.02)
    assert cmp_nf.status == ComparisonStatus.UNRESOLVED

    cmp_scores = evaluate_fewshot_scores([0.60, 0.61, 0.59], [0.75, 0.76, 0.74])
    assert cmp_scores.status == ComparisonStatus.RESOLVED_HIGHER
