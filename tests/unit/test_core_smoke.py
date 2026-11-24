"""Core package smoke tests: imports, lazy registry, MockGateway, and golden prompt snapshot."""

from __future__ import annotations

from pathlib import Path

from eval_studio import __version__
from eval_studio.core.registry import ENTRY_POINT_GROUPS, get_registry
from eval_studio.core.schema import EvalRequest, OutputType
from eval_studio.gateway.mock import MockGateway
from eval_studio.tasks.loader import load_task_spec
from eval_studio.tasks.templates import render_sandboxed_template
from eval_studio.tasks.validator import validate_task_spec


def test_version_and_entry_point_groups() -> None:
    assert __version__ == "0.1.0"
    for group in ENTRY_POINT_GROUPS:
        reg = get_registry(group)
        assert isinstance(reg.list_names(), list)


def test_mock_gateway_deterministic_offline() -> None:
    gw = MockGateway()
    req = EvalRequest(
        request_id="req-1",
        sample_id="s-1",
        task_name="smoke",
        output_type=OutputType.LOGLIKELIHOOD,
        context="Question: 2+2?\nAnswer:",
        continuation=" 4",
    )
    first = gw.loglikelihood([req])[0]
    second = gw.loglikelihood([req])[0]
    assert first.loglikelihood == second.loglikelihood
    assert first.is_greedy == second.is_greedy


def test_tier1_task_loads_validates_and_matches_golden_snapshot(
    examples_dir: Path,
    repo_root: Path,
) -> None:
    spec = load_task_spec(examples_dir / "tier1_zero_code" / "task.yaml")
    result = validate_task_spec(spec)
    assert result.valid is True

    rendered = render_sandboxed_template(
        spec.doc_to_text,
        {"input_text": "Which vitamin deficiency causes scurvy?"},
    )
    snapshot = (repo_root / "tests" / "snapshots" / "tier1_sample.prompt.txt").read_text(
        encoding="utf-8"
    ).rstrip("\n")
    assert rendered == snapshot
