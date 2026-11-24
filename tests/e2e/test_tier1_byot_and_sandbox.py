"""Tier 1 E2E tests: BYOT 3-tier tasks, DockerSandbox offline fallback, validate & calibrate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from eval_studio.cli.app import app
from eval_studio.gateway.mock import MockGateway
from eval_studio.receipts.manifest import ReceiptManifest
from eval_studio.sandbox.base import ExecutionResult
from eval_studio.sandbox.docker import DockerSandbox
from eval_studio.sandbox.limits import SandboxLimits
from eval_studio.scoring.judge import JudgeRubric, LLMJudge
from eval_studio.tasks.loader import load_task_spec
from eval_studio.tasks.templates import render_sandboxed_template

pytestmark = pytest.mark.e2e


def test_t1_tier1_zero_code_csv_and_jsonl_end_to_end(
    repo_root: Path,
    examples_dir: Path,
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-BYOT-01: Tier 1 CSV (examples/) and JSONL zero-code tasks validate & run E2E."""
    monkeypatch.chdir(repo_root)
    csv_task = examples_dir / "tier1_zero_code" / "task.yaml"

    val_csv = cli_runner.invoke(app, ["validate", "--task", str(csv_task)])
    assert val_csv.exit_code == 0, val_csv.stdout
    assert "VALID" in val_csv.stdout

    csv_out = tmp_path / "csv_receipt"
    run_csv = cli_runner.invoke(
        app,
        ["run", "--task", str(csv_task), "--model", "mock", "--output", str(csv_out)],
    )
    assert run_csv.exit_code == 0, run_csv.stdout

    csv_samples = [
        json.loads(line)
        for line in (csv_out / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [s["sample_id"] for s in csv_samples] == ["med_001", "med_002", "med_003"]

    spec = load_task_spec(csv_task)
    rendered = render_sandboxed_template(
        spec.doc_to_text,
        {"input_text": "Which vitamin deficiency causes scurvy?"},
    )
    snapshot = (repo_root / "tests" / "snapshots" / "tier1_sample.prompt.txt").read_text(
        encoding="utf-8"
    ).rstrip("\n")
    assert rendered == snapshot

    for fname in ("manifest.json", "receipt.json", "samples.jsonl", "report.md", "report.html"):
        assert (csv_out / fname).is_file()

    # Tier 1 JSONL dataset
    jsonl_data = tmp_path / "tier1_items.jsonl"
    jsonl_data.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "j1",
                        "input_text": "First-line drug for anaphylaxis?",
                        "target": "A",
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "j2",
                        "input_text": "Primary hormone lowered in Type 1 diabetes?",
                        "target": "A",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    jsonl_task = tmp_path / "tier1_jsonl_task.yaml"
    jsonl_task.write_text(
        f"""name: t1_jsonl_zero_code
version: "0.1.0"
trust_tier: tier1_zero_code
output_type: generate_until
dataset_path: "{jsonl_data.as_posix()}"
doc_to_text: "Question: {{{{ input_text }}}}\\nAnswer:"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
""",
        encoding="utf-8",
    )
    val_jsonl = cli_runner.invoke(app, ["validate", "--task", str(jsonl_task)])
    assert val_jsonl.exit_code == 0, val_jsonl.stdout

    jsonl_out = tmp_path / "jsonl_receipt"
    run_jsonl = cli_runner.invoke(
        app,
        ["run", "--task", str(jsonl_task), "--model", "mock", "--output", str(jsonl_out)],
    )
    assert run_jsonl.exit_code == 0, run_jsonl.stdout
    jsonl_samples = [
        json.loads(line)
        for line in (jsonl_out / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jsonl_samples) == 2


def test_t1_tier2_declarative_yaml_filters_and_fewshot(
    examples_dir: Path,
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-BYOT-02: Tier 2 declarative clinical MCQA YAML with FilterEnsemble & few-shot."""
    mcqa_data = tmp_path / "clinical_mcqa.jsonl"
    rows = [
        {
            "sample_id": f"mcqa_{i}",
            "input_text": f"62yo patient #{i} with substernal chest pain.",
            "choices": ["ECG & troponin", "Discharge home", "Oral ibuprofen", "Knee X-ray"],
            "target": 0,
        }
        for i in range(10)
    ]
    mcqa_data.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    base_cfg = yaml.safe_load(
        (examples_dir / "tier2_declarative" / "clinical_mcqa.yaml").read_text(encoding="utf-8")
    )
    base_cfg["dataset_path"] = mcqa_data.as_posix()
    base_cfg["num_fewshot"] = 2
    base_cfg["fewshot_rotation_sets"] = 2

    tier2_yaml = tmp_path / "clinical_mcqa_local.yaml"
    tier2_yaml.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    val = cli_runner.invoke(app, ["validate", "--task", str(tier2_yaml)])
    assert val.exit_code == 0, val.stdout

    out_dir = tmp_path / "tier2_out"
    run_res = cli_runner.invoke(
        app,
        ["run", "--task", str(tier2_yaml), "--model", "mock", "--output", str(out_dir)],
    )
    assert run_res.exit_code == 0, run_res.stdout

    manifest = ReceiptManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert "acc_bytes" in manifest.metrics
    assert "acc_norm" in manifest.metrics
    samples = [
        json.loads(line)
        for line in (out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(samples) == 10
    assert "strict_choice" in samples[0]["filtered_responses"]
    assert "flexible_choice" in samples[0]["filtered_responses"]


def test_t1_tier3_plugin_with_trust_remote_code_and_sandbox(
    examples_dir: Path,
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-BYOT-03: Tier 3 plugin opt-in (EVAL_STUDIO_PLUGINS/register_plugins) & DockerSandbox."""
    base_plugin_src = (examples_dir / "tier3_plugin" / "custom_solver_scorer.py").read_text(
        encoding="utf-8"
    )
    plugin_copy = tmp_path / "custom_solver_scorer.py"
    plugin_copy.write_text(
        base_plugin_src
        + '''

class Tier3ClinicalScorer(CustomClinicalScorer):
    name: str = "custom_clinical_scorer"

    def score(self, sample: Sample, response: EvalResponse) -> dict[str, float]:
        base = super().score(sample, response)
        return {"custom_clinical_metric": base["exact_match"]}


def custom_clinical_metric(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip().casefold() == reference.strip().casefold() else 0.0


def register_plugins(get_registry_fn):
    get_registry_fn("eval_studio.solvers").register(
        "custom_clinical_solver", CustomClinicalSolver, builtin=False
    )


EVAL_STUDIO_PLUGINS = {
    "eval_studio.metrics": {
        "custom_clinical_scorer": Tier3ClinicalScorer,
        "custom_clinical_metric": custom_clinical_metric,
    }
}
''',
        encoding="utf-8",
    )

    data_jsonl = tmp_path / "tier3_data.jsonl"
    data_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "t3-1",
                "input_text": "Write function dose_mg(w) returning w * 10.",
                "target": "A",
                "test_harness": "assert dose_mg(5) == 50",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tier3_yaml = tmp_path / "tier3_task.yaml"
    tier3_yaml.write_text(
        f"""name: tier3_custom_clinical
version: "0.1.0"
trust_tier: tier3_code_plugin
output_type: generate_until
dataset_path: "{data_jsonl.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
unsafe_code: true
metrics: [custom_clinical_metric]
metadata:
  solver: custom_clinical_solver
  scorer: custom_clinical_scorer
""",
        encoding="utf-8",
    )

    captured_requests: list[str] = []
    orig_generate_until = MockGateway.generate_until

    def _spy_generate_until(self: MockGateway, requests: Any) -> Any:
        for req in requests:
            captured_requests.append(req.context)
        return orig_generate_until(self, requests)

    monkeypatch.setattr(MockGateway, "generate_until", _spy_generate_until)

    out_dir = tmp_path / "tier3_out"
    res = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(tier3_yaml),
            "--model",
            "mock",
            "--plugins",
            str(plugin_copy),
            "--trust-remote-code",
            "--output",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert len(captured_requests) == 1
    assert captured_requests[0].startswith("Clinical Case: ")

    manifest = ReceiptManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert "custom_clinical_metric" in manifest.metrics
    assert manifest.metrics["custom_clinical_metric"].value == pytest.approx(1.0)

    samples = [
        json.loads(line)
        for line in (out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(samples) == 1
    assert samples[0]["sample_id"] == "t3-1"
    assert "custom_clinical_metric" in samples[0]["metrics"] or any(
        "custom_clinical_metric" in m for m in samples[0]["metrics"].values()
    )

    limits = SandboxLimits(timeout_seconds=5.0, memory_mb=256, network_disabled=True)
    assert limits.network_disabled is True
    assert limits.read_only_rootfs is True

    sandbox = DockerSandbox()
    exec_result = sandbox.run_code(
        "def dose_mg(weight_kg):\n    return weight_kg * 10\n",
        test_harness="assert dose_mg(5) == 50\n",
        limits=limits,
    )
    assert isinstance(exec_result, ExecutionResult)
    assert exec_result.passed is True
    assert type(exec_result.passed) is bool
    assert exec_result.exit_code == 0


def test_t1_validate_canary_version_and_snapshot_check(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-BYOT-04: `eval-studio validate --task ... --snapshot ...` checks canary/version/parity."""
    csv_path = tmp_path / "val_items.csv"
    csv_path.write_text(
        'sample_id,input_text,target\nv1,"Which vitamin prevents scurvy?",Vitamin C\n',
        encoding="utf-8",
    )
    task_yaml = tmp_path / "validated_task.yaml"
    task_yaml.write_text(
        f"""name: t1_validated_pack
version: "1.0.0"
trust_tier: tier2_declarative
output_type: generate_until
dataset_path: "{csv_path.as_posix()}"
doc_to_text: "Question: {{{{ input_text }}}}\\nAnswer:"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
metadata:
  date_stamp: "2026-11-22"
  canary_guid: "eval_studio-canary-26b5c67b"
""",
        encoding="utf-8",
    )
    snapshot_path = tmp_path / "validated_task.prompt.txt"
    snapshot_path.write_text("Question: Which vitamin prevents scurvy?\nAnswer:", encoding="utf-8")

    result = cli_runner.invoke(
        app,
        ["validate", "--task", str(task_yaml), "--snapshot", str(snapshot_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert "VALID" in result.stdout


def test_t1_llm_judge_with_fallback_and_calibrate_aligned(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-BYOT-05: LLMJudge failover to fallback_judge + aligned `eval-studio calibrate`."""
    rubric = JudgeRubric(
        name="clinical_accuracy",
        criteria="Award 1.0 if diagnosis matches clinical gold standard, else 0.0.",
        scale_min=0.0,
        scale_max=1.0,
    )
    primary_gw = MockGateway(model_id="primary-judge", judge_mode="refusal")
    fallback_gw = MockGateway(
        model_id="fallback-judge",
        judge_mode="normal",
        default_reply='{"score": 1.0, "rationale": "Accurate clinical answer"}',
    )
    judge = LLMJudge(
        judge_model_id="primary-judge",
        rubric=rubric,
        gateway=primary_gw,
        fallback_judge=fallback_gw,
    )
    verdict = judge.grade(
        prompt="Which organ produces insulin?",
        response="Pancreas",
        reference="Pancreas",
    )
    assert verdict.score == pytest.approx(1.0)
    assert verdict.rationale

    # 9/10 exact balanced agreement -> kappa = 0.80 (>= 0.75)
    judge_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    human_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    cal_file = tmp_path / "calibration_aligned.jsonl"
    cal_file.write_text(
        "\n".join(
            json.dumps({"sample_id": f"c-{i}", "judge_label": j, "human_label": h})
            for i, (j, h) in enumerate(zip(judge_labels, human_labels, strict=True))
        )
        + "\n",
        encoding="utf-8",
    )

    res = cli_runner.invoke(app, ["calibrate", str(cal_file)])
    assert res.exit_code == 0, res.stdout
    out_lower = res.stdout.casefold()
    assert "0.8" in res.stdout or "kappa" in out_lower
    assert "unaligned" not in out_lower
