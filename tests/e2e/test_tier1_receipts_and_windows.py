"""Tier 1 E2E tests: Portable Eval Receipts, CLI tools, and Windows hygiene (F23–F24, F26)."""

from __future__ import annotations

import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from eval_studio.cli.app import app
from eval_studio.core.schema import EvalResponse, ScoredSample, TokenUsage
from eval_studio.gateway.mock import MockGateway
from eval_studio.receipts.manifest import MetricSummary, ReceiptManifest, compute_manifest_digest
from eval_studio.receipts.verify import verify_receipt

pytestmark = pytest.mark.e2e


def _build_synthetic_receipt(
    receipt_dir: Path,
    *,
    run_id: str,
    flipped_indices: set[int],
) -> None:
    """Create a cryptographically valid 10-sample receipt dir for noise-floor tests."""
    receipt_dir.mkdir(parents=True, exist_ok=True)
    samples_lines: list[str] = []
    n_correct = 0
    for i in range(10):
        pred = "B" if i in flipped_indices else "A"
        score = 0.0 if i in flipped_indices else 1.0
        n_correct += int(score)
        rec = ScoredSample(
            sample_id=f"nf-{i}",
            task_name="clinical_triage",
            target="A",
            raw_responses=[
                EvalResponse(
                    request_id=f"req-{i}",
                    sample_id=f"nf-{i}",
                    text=pred,
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=1),
                )
            ],
            filtered_responses={"default": [pred]},
            metrics={"default": {"exact_match": score}},
        )
        samples_lines.append(rec.model_dump_json())

    samples_bytes = ("\n".join(samples_lines) + "\n").encode("utf-8")
    samples_path = receipt_dir / "samples.jsonl"
    samples_path.write_bytes(samples_bytes)

    import hashlib

    samples_hash = hashlib.sha256(samples_bytes).hexdigest()
    manifest = ReceiptManifest(
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        backend="mock",
        model_id="mock-deterministic",
        task_name="clinical_triage",
        task_sha256="a" * 64,
        samples_sha256=samples_hash,
        seeds={"python_seed": 0, "numpy_seed": 1234, "fewshot_seed": 1234, "bootstrap_seed": 1234},
        redacted_config={"model_backend": "mock"},
        metrics={
            "exact_match": MetricSummary(
                value=n_correct / 10.0,
                stderr=0.1,
                n_samples=10,
                bootstrap_iters=100,
            )
        },
    )
    digest = compute_manifest_digest(manifest)
    signed = manifest.model_copy(update={"manifest_sha256": digest})
    canonical = signed.model_dump_json(indent=2)
    (receipt_dir / "manifest.json").write_text(canonical, encoding="utf-8")
    (receipt_dir / "receipt.json").write_text(canonical, encoding="utf-8")


def test_t1_run_emits_complete_portable_receipt_bundle(
    repo_root: Path,
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-RCP-01: `eval-studio run` emits complete, canonical Portable Eval Receipt bundle."""
    monkeypatch.chdir(repo_root)
    receipt_dir = tmp_path / "receipt_bundle"

    result = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            "examples/tier1_zero_code/task.yaml",
            "--model",
            "mock",
            "--output",
            str(receipt_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout

    expected_files = ("manifest.json", "receipt.json", "samples.jsonl", "report.md", "report.html")
    for fname in expected_files:
        assert (receipt_dir / fname).is_file(), f"Missing receipt file: {fname}"

    manifest_raw = (receipt_dir / "manifest.json").read_text(encoding="utf-8")
    receipt_raw = (receipt_dir / "receipt.json").read_text(encoding="utf-8")
    assert json.loads(manifest_raw) == json.loads(receipt_raw)

    manifest_data: dict[str, Any] = json.loads(manifest_raw)
    parsed = ReceiptManifest.model_validate(manifest_data)
    assert parsed.harness_version
    assert parsed.run_id
    assert parsed.created_at_utc
    assert parsed.backend == "mock"
    assert parsed.model_id
    assert len(parsed.task_sha256) == 64
    assert len(parsed.samples_sha256) == 64
    assert parsed.seeds
    assert isinstance(parsed.redacted_config, dict)
    assert "exact_match" in parsed.metrics
    assert parsed.metrics["exact_match"].stderr is not None

    for key in ("dtype", "quantization", "batch_size", "reproduce_command"):
        assert key in manifest_data or key in parsed.redacted_config
    assert "cost" in manifest_raw.casefold() or "total_cost_usd" in manifest_data


def test_t1_verify_receipt_passes_valid_bundle(
    repo_root: Path,
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-RCP-02: `eval-studio verify-receipt` validates a clean run bundle and exits 0."""
    monkeypatch.chdir(repo_root)
    receipt_dir = tmp_path / "valid_receipt"

    run_res = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            "examples/tier1_zero_code/task.yaml",
            "--model",
            "mock",
            "--output",
            str(receipt_dir),
        ],
    )
    assert run_res.exit_code == 0, run_res.stdout

    ver_res = cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)])
    assert ver_res.exit_code == 0, ver_res.stdout
    assert "VERIFIED" in ver_res.stdout


def test_t1_score_offline_rescoring_parity(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-RCP-03: `eval-studio score` re-scores saved raw responses 100% offline with parity."""
    csv_path = tmp_path / "score_items.csv"
    csv_path.write_text(
        'sample_id,input_text,target\ns1,"Reply a",a\ns2,"Reply a",A\n',
        encoding="utf-8",
    )
    task_path = tmp_path / "score_task.yaml"
    task_path.write_text(
        f"""name: t1_score_parity
version: "0.1.0"
trust_tier: tier1_zero_code
output_type: generate_until
dataset_path: "{csv_path.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
""",
        encoding="utf-8",
    )
    receipt_dir = tmp_path / "score_receipt"
    run_res = cli_runner.invoke(
        app,
        ["run", "--task", str(task_path), "--model", "mock", "--output", str(receipt_dir)],
    )
    assert run_res.exit_code == 0, run_res.stdout

    orig_manifest = ReceiptManifest.model_validate_json(
        (receipt_dir / "manifest.json").read_text(encoding="utf-8")
    )

    def _forbid_inference(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Offline `eval-studio score` must not call inference gateway")

    monkeypatch.setattr(MockGateway, "generate_until", _forbid_inference)
    monkeypatch.setattr(MockGateway, "loglikelihood", _forbid_inference)

    rescored_dir = tmp_path / "rescored_default"
    score_res = cli_runner.invoke(
        app,
        ["score", "--receipt", str(receipt_dir), "--output", str(rescored_dir)],
    )
    assert score_res.exit_code == 0, score_res.stdout

    assert (rescored_dir / "manifest.json").is_file(), "Rescored manifest.json must be written"
    new_manifest = ReceiptManifest.model_validate_json(
        (rescored_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert new_manifest.metrics["exact_match"].value == pytest.approx(
        orig_manifest.metrics["exact_match"].value, abs=1e-9
    )
    if orig_manifest.metrics["exact_match"].stderr is not None:
        assert new_manifest.metrics["exact_match"].stderr == pytest.approx(
            orig_manifest.metrics["exact_match"].stderr, abs=1e-9
        )

    # Second offline re-scoring pass with updated task/metric configuration (ignore_case=False)
    alt_task_path = tmp_path / "score_task_strict.yaml"
    alt_task_path.write_text(
        f"""name: t1_score_parity_strict
version: "0.1.0"
trust_tier: tier2_declarative
output_type: generate_until
dataset_path: "{csv_path.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
metadata:
  ignore_case: false
""",
        encoding="utf-8",
    )
    rescored_strict = tmp_path / "rescored_strict"
    score_strict = cli_runner.invoke(
        app,
        [
            "score",
            "--receipt",
            str(receipt_dir),
            "--task",
            str(alt_task_path),
            "--output",
            str(rescored_strict),
        ],
    )
    assert score_strict.exit_code == 0, score_strict.stdout
    assert (rescored_strict / "manifest.json").is_file(), "Strict rescored manifest.json missing"
    strict_manifest = ReceiptManifest.model_validate_json(
        (rescored_strict / "manifest.json").read_text(encoding="utf-8")
    )
    assert strict_manifest.metrics["exact_match"].value == pytest.approx(0.5)


def test_t1_noise_floor_flip_rate_diagnostics(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-RCP-04: `eval-studio noise-floor` reports n_decisions=10, n_flips=2, flip_rate=0.20."""
    baseline_dir = tmp_path / "baseline"
    perturbed_dir = tmp_path / "perturbed"

    _build_synthetic_receipt(baseline_dir, run_id="base-1", flipped_indices=set())
    _build_synthetic_receipt(perturbed_dir, run_id="pert-1", flipped_indices={2, 7})

    assert verify_receipt(baseline_dir).valid is True
    assert verify_receipt(perturbed_dir).valid is True

    result = cli_runner.invoke(
        app,
        ["noise-floor", "--baseline", str(baseline_dir), "--perturbed", str(perturbed_dir)],
    )
    assert result.exit_code == 0, result.stdout
    out_lower = result.stdout.casefold()
    assert "10" in result.stdout
    assert "2" in result.stdout
    assert "0.2" in result.stdout or "20.0%" in result.stdout or "20%" in out_lower


def test_t1_windows_utf8_and_space_path_hygiene(
    space_workspace: Path,
    run_eval_studio_subprocess: Any,
) -> None:
    """T1-RCP-05: Windows UTF-8 I/O, space-quoted reproduce_command, and handle cleanup."""
    data_path = space_workspace / "clinical data α.csv"
    data_path.write_text(
        "sample_id,input_text,target\n"
        'w1,"Diagnose patient in İstanbul with Straße clinic dose 50 µg/dL α-blocker.",α-blocker\n',
        encoding="utf-8",
    )
    task_path = space_workspace / "task v1.yaml"
    task_path.write_text(
        f"""name: windows_utf8_space_task
version: "0.1.0"
trust_tier: tier1_zero_code
output_type: generate_until
dataset_path: "{data_path.as_posix()}"
doc_to_text: "Case: {{{{ input_text }}}}\\nAnswer:"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
""",
        encoding="utf-8",
    )
    out_dir = space_workspace / "out dir (v1)"

    val_proc = run_eval_studio_subprocess(["validate", "--task", str(task_path)])
    assert val_proc.returncode == 0, val_proc.stdout + val_proc.stderr

    run_proc = run_eval_studio_subprocess(
        ["run", "--task", str(task_path), "--model", "mock", "--output", str(out_dir)]
    )
    assert run_proc.returncode == 0, run_proc.stdout + run_proc.stderr

    score_proc = run_eval_studio_subprocess(["score", "--receipt", str(out_dir)])
    assert score_proc.returncode == 0, score_proc.stdout + score_proc.stderr

    ver_proc = run_eval_studio_subprocess(["verify-receipt", str(out_dir)])
    assert ver_proc.returncode == 0, ver_proc.stdout + ver_proc.stderr

    manifest_data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    repro_cmd = manifest_data.get("reproduce_command") or manifest_data.get(
        "redacted_config", {}
    ).get("reproduce_command", "")
    assert repro_cmd
    tokens = [tok.strip("\"'") for tok in shlex.split(repro_cmd, posix=False)]
    assert any(tok in (str(task_path), task_path.as_posix()) for tok in tokens)
    assert any(tok in (str(out_dir), out_dir.as_posix()) for tok in tokens)

    # Verify UTF-8 text survived round-trip without cp1252 corruption
    samples_text = (out_dir / "samples.jsonl").read_text(encoding="utf-8")
    sample_rows = [json.loads(line) for line in samples_text.splitlines() if line.strip()]
    assert len(sample_rows) == 1
    assert sample_rows[0]["target"] == "α-blocker"
    assert "α-blocker" in samples_text

    shutil.rmtree(space_workspace)
    assert not space_workspace.exists()
