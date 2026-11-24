"""Unit tests for F9 FilterEnsemble pipelines and OLMES Cloze + MCF helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_studio.core.schema import EvalResponse, MultipleChoiceFormat
from eval_studio.scoring.filters import (
    MajorityVoteFilter,
    MultiChoiceRegexFilter,
    RegexFilter,
    TakeFirstFilter,
    TakeFirstKFilter,
    ThinkStripFilter,
    apply_filter_ensembles,
    build_olmes_dual_prompts,
    format_olmes_cloze_prompt,
    format_olmes_mcf_prompt,
    get_olmes_recommended_metric,
    select_olmes_best_format,
)
from eval_studio.tasks.loader import load_task_spec


def test_filters_individual_behaviors() -> None:
    # Empty choice guard in MultiChoiceRegexFilter
    mc_flt = MultiChoiceRegexFilter(["", "Guilty", "Guilty of Romance"])
    assert mc_flt.apply(["No match here"]) == ["[invalid]"]
    assert mc_flt.apply(["Guilty of Romance"]) == ["Guilty of Romance"]

    rx = RegexFilter(r"####\s*(-?\d+)", group=1)
    assert rx.apply(["Reasoning... #### 42", "missing"]) == ["42", "[invalid]"]

    assert TakeFirstFilter().apply(["a", "b", "c"]) == ["a"]
    assert TakeFirstFilter(fallback="[empty]").apply([]) == ["[empty]"]
    assert TakeFirstKFilter(k=2).apply(["a", "b", "c"]) == ["a", "b"]

    mv = MajorityVoteFilter()
    assert mv.apply(["[invalid]", "B", "A", "B", "A"]) == ["B"]
    assert mv.apply(["[invalid]", ""]) == ["[invalid]"]

    ts = ThinkStripFilter("</think>")
    assert ts.apply(["a </think> b </think>  C  "]) == ["C"]


def test_filter_ensemble_from_clinical_mcqa_yaml(examples_dir: Path) -> None:
    spec = load_task_spec(examples_dir / "tier2_declarative" / "clinical_mcqa.yaml")
    responses = [
        EvalResponse(request_id="r1", sample_id="s1", text="Answer is Hypertension."),
        EvalResponse(request_id="r2", sample_id="s1", text="Maybe Diabetes."),
    ]
    out = apply_filter_ensembles(
        responses,
        spec.filter_ensembles,
        choices=["Diabetes", "Hypertension"],
    )
    assert out["strict_choice"] == ["Hypertension", "Diabetes"]
    assert out["flexible_choice"] == ["Hypertension", "Diabetes"]

    with pytest.raises(ValueError, match="Duplicate"):
        apply_filter_ensembles(
            ["a"],
            [spec.filter_ensembles[0], spec.filter_ensembles[0]],
            choices=["a", "b"],
        )


def test_olmes_cloze_mcf_formatting_and_selection() -> None:
    mcf = format_olmes_mcf_prompt("Which vitamin prevents scurvy?", ["Vitamin A", "Vitamin C"])
    assert "\n A. Vitamin A\n B. Vitamin C\n" in mcf.context
    assert mcf.continuations == [" A", " B"]
    assert mcf.labels == ["A", "B"]

    cloze = format_olmes_cloze_prompt("Which vitamin prevents scurvy?", ["Vitamin A", "Vitamin C"])
    assert cloze.continuations == [" Vitamin A", " Vitamin C"]

    dual = build_olmes_dual_prompts("Q?", ["opt1", "opt2"])
    assert set(dual) == {"cloze", "mcf"}

    selected = select_olmes_best_format(cloze_score=0.62, mcf_score=0.71)
    assert selected.selected_format == MultipleChoiceFormat.MCF
    assert selected.best_score == 0.71
    assert selected.delta_mcf_minus_cloze == pytest.approx(0.09)

    assert get_olmes_recommended_metric("arc_challenge") == "acc_mutual_info"
    assert get_olmes_recommended_metric("mmlu_anatomy") == "acc_norm"
    assert get_olmes_recommended_metric("boolq") == "acc"
    assert get_olmes_recommended_metric("custom_med_task") == "acc_bytes"
