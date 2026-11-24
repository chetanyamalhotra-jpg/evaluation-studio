"""Tier 3 cross-feature pairwise E2E tests (8 tests)."""

from __future__ import annotations

import json
import math
import re
import shlex
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from eval_studio.cli.app import app
from eval_studio.gateway.base import transfer_trailing_whitespace
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import SecretRole, contains_secret
from eval_studio.paper.compiler import sha256_file
from eval_studio.sandbox.docker import DockerSandbox
from eval_studio.scoring.calibrate import cohens_kappa
from eval_studio.scoring.judge import JudgeRubric, LLMJudge
from eval_studio.stats.aggregation import SubtaskStats, aggregate_group
from eval_studio.stats.fewshot import (
    ComparisonStatus,
    evaluate_fewshot_scores,
)

pytestmark = pytest.mark.e2e


def _bind_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
) -> None:
    orig_async_init = httpx.AsyncClient.__init__
    orig_sync_init = httpx.Client.__init__

    def _patched_async_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("transport") is None:
            kwargs["transport"] = transport
        orig_async_init(self, *args, **kwargs)

    def _patched_sync_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("transport") is None:
            kwargs["transport"] = transport
        orig_sync_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _patched_async_init)
    monkeypatch.setattr(httpx.Client, "__init__", _patched_sync_init)


def test_t3_byop_validate_costguard_mock_score_verify_chain(
    tmp_path: Path,
    cli_runner: CliRunner,
    examples_dir: Path,
) -> None:
    """T3-PW-01: compile-paper -> validate -> dry-run -> run -> score -> verify-receipt."""
    paper_copy = tmp_path / "sample_paper.md"
    shutil.copyfile(examples_dir / "papers" / "sample_paper.md", paper_copy)
    suite_yaml = tmp_path / "compiled_suite.yaml"

    res_comp = cli_runner.invoke(
        app,
        ["compile-paper", str(paper_copy), "--out", str(suite_yaml)],
    )
    assert res_comp.exit_code == 0, res_comp.stdout

    res_val = cli_runner.invoke(app, ["validate", "--task", str(suite_yaml)])
    assert res_val.exit_code == 0, res_val.stdout

    res_dry = cli_runner.invoke(
        app,
        ["run", "--task", str(suite_yaml), "--model", "mock", "--dry-run", "--estimate-cost"],
    )
    dry_out = res_dry.stdout + getattr(res_dry, "stderr", "")
    assert res_dry.exit_code == 0, dry_out
    assert "no such option" not in dry_out.lower()
    assert "usd" in dry_out.lower() or "cost" in dry_out.lower()

    receipt_dir = tmp_path / "pw1_receipt"
    res_run = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(suite_yaml),
            "--model",
            "mock",
            "--max-cost-usd",
            "1.00",
            "--output",
            str(receipt_dir),
        ],
    )
    assert res_run.exit_code == 0, res_run.stdout

    manifest = json.loads((receipt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_paper_sha256"] == sha256_file(paper_copy)

    res_score = cli_runner.invoke(app, ["score", "--receipt", str(receipt_dir)])
    assert res_score.exit_code == 0, res_score.stdout
    for metric_name, metric_info in manifest["metrics"].items():
        assert metric_name in res_score.stdout or f"{metric_info['value']:.4f}" in res_score.stdout

    res_verify = cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)])
    assert res_verify.exit_code == 0, res_verify.stdout


def test_t3_tier2_filter_ensemble_kset_thinking_http_asgi_mock(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T3-PW-02: Tier 2 FilterEnsemble + K-set rotation + enable_thinking + HttpBYOKGateway."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-pw2-1234567890abcd")

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"resp-{len(recorded)}",
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>ECG shows ST elevation</think>\n"
                                "Acute Coronary Syndrome"
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    _bind_mock_transport(monkeypatch, transport)

    data_path = tmp_path / "mcqa_acs.jsonl"
    items = [
        {
            "sample_id": f"acs_{i}",
            "input_text": f"Unique Vignette Token VIGNETTE_{i}: 58yo chest pain with ST elevation.",
            "choices": ["Acute Coronary", "Acute Coronary Syndrome"],
            "target": "Acute Coronary Syndrome",
        }
        for i in range(6)
    ]
    data_path.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n",
        encoding="utf-8",
    )

    task_yaml = tmp_path / "tier2_acs.yaml"
    task_yaml.write_text(
        "name: tier2_acs_thinking\n"
        'version: "1.0.0"\n'
        "trust_tier: tier2_declarative\n"
        "output_type: generate_until\n"
        f"dataset_path: {data_path}\n"
        'doc_to_text: "Case: {{ input_text }}\\nChoices: {{ choices }}\\nAnswer:"\n'
        'doc_to_target: "{{ target }}"\n'
        "num_fewshot: 2\n"
        "fewshot_rotation_sets: 4\n"
        "metrics:\n"
        "  - exact_match\n"
        "filter_ensembles:\n"
        "  - name: strict_choice\n"
        "    steps:\n"
        "      - name: multi_choice_regex\n"
        "        params:\n"
        '          fallback: "[invalid]"\n'
        "  - name: flexible_choice\n"
        "    steps:\n"
        "      - name: multi_choice_regex\n"
        "        params:\n"
        '          fallback: "[invalid]"\n',
        encoding="utf-8",
    )

    receipt_dir = tmp_path / "pw2_receipt"
    res = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "http-byok",
            "--enable-thinking",
            "--think-end-token",
            "</think>",
            "--output",
            str(receipt_dir),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert len(recorded) >= 6

    # Active test sample must never appear in its own few-shot demonstrations,
    # num_fewshot=2 exemplars must actually be injected, and 4-set rotation must vary sets.
    exemplar_signatures: set[tuple[str, ...]] = set()
    for req_rec in recorded:
        payload_str = req_rec["body_text"]
        positions = [
            (payload_str.index(f"VIGNETTE_{i}:"), f"VIGNETTE_{i}:")
            for i in range(6)
            if f"VIGNETTE_{i}:" in payload_str
        ]
        positions.sort()
        assert len(positions) == 3, f"Expected 1 active + 2 few-shot exemplars, got {positions}"
        for _, tok in positions:
            assert payload_str.count(tok) == 1
        exemplar_signatures.add(tuple(tok for _, tok in positions[:-1]))
    assert len(exemplar_signatures) >= 2

    sample_rows = [
        json.loads(line)
        for line in (receipt_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(sample_rows) == 6
    for row in sample_rows:
        raw_resp = row["raw_responses"][0]
        assert "ECG shows ST elevation" in (raw_resp.get("reasoning_trace") or "")
        assert "<think>" not in (raw_resp.get("text") or "")
        assert row["filtered_responses"]["strict_choice"][0] == "Acute Coronary Syndrome"
        assert row["filtered_responses"]["flexible_choice"][0] == "Acute Coronary Syndrome"

    # Derive per-set scores from the actual 4-set rotated samples and verify close comparison
    set_buckets: dict[int, list[float]] = {k: [] for k in range(4)}
    for idx, row in enumerate(sample_rows):
        first_pipe_metrics = next(iter(row["metrics"].values()))
        set_buckets[idx % 4].append(float(first_pipe_metrics["exact_match"]))
    actual_by_set = [sum(set_buckets[k]) / len(set_buckets[k]) for k in range(4)]
    baseline_by_set = [1.02, 0.99, 1.03, 0.98]
    comp = evaluate_fewshot_scores(baseline_by_set, actual_by_set)
    assert comp.status == ComparisonStatus.UNRESOLVED
    assert comp.n_sets == 4


def test_t3_calibrate_llm_judge_fallback_and_role_keys(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3-PW-03: LLMJudge failover with role-separated keys feeds eval-studio calibrate cleanly."""
    from eval_studio.core.schema import EvalRequest, EvalResponse, TokenUsage

    primary_key = "sk-judge-primary-1111111111"
    fallback_key = "sk-judge-fallback-2222222222"
    monkeypatch.setenv(SecretRole.JUDGE.value, primary_key)
    monkeypatch.setenv("FALLBACK_JUDGE_API_KEY", fallback_key)

    class RubricFallbackMockGateway(MockGateway):
        def generate_until(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
            out: list[EvalResponse] = []
            for req in requests:
                positive = "administer aspirin" in req.context.casefold()
                payload = {
                    "score": 1.0 if positive else 0.0,
                    "rationale": (
                        "Clinically accurate aspirin dose."
                        if positive
                        else "Missing aspirin therapy."
                    ),
                }
                out.append(
                    EvalResponse(
                        request_id=req.request_id,
                        sample_id=req.sample_id,
                        text=json.dumps(payload),
                        backend=self.backend_name,
                        model_id=self.model_id,
                        usage=TokenUsage(prompt_tokens=10, completion_tokens=8),
                    )
                )
            return out

    rubric = JudgeRubric(
        name="clinical_accuracy",
        criteria="Score 1 if clinically accurate, 0 otherwise.",
        scale_min=0.0,
        scale_max=1.0,
    )
    primary_gw = MockGateway(model_id="mock-judge-primary", judge_mode="refusal")
    fallback_gw = RubricFallbackMockGateway(model_id="mock-judge-fallback", judge_mode="normal")
    judge = LLMJudge(
        judge_model_id="mock-judge-primary",
        rubric=rubric,
        gateway=primary_gw,
        fallback_judge=fallback_gw,
    )

    human_labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    judge_labels: list[int] = []
    calib_rows: list[str] = []
    for idx, h_lbl in enumerate(human_labels):
        candidate_resp = (
            "Administer aspirin 325 mg." if h_lbl == 1 else "Discharge home without therapy."
        )
        verdict = judge.grade(
            prompt=f"Clinical case {idx}",
            response=candidate_resp,
            reference="Aspirin 325 mg",
        )
        assert verdict.score in (0.0, 1.0)
        assert not contains_secret(verdict.model_dump_json())
        assert primary_key not in verdict.model_dump_json()
        assert fallback_key not in verdict.model_dump_json()
        j_lbl = int(round(verdict.score))
        judge_labels.append(j_lbl)
        calib_rows.append(
            json.dumps(
                {
                    "sample_id": f"j{idx}",
                    "judge_label": j_lbl,
                    "human_label": h_lbl,
                }
            )
        )

    calib_file = tmp_path / "judge_calib.jsonl"
    calib_file.write_text("\n".join(calib_rows) + "\n", encoding="utf-8")

    res = cli_runner.invoke(app, ["calibrate", str(calib_file)])
    out_text = res.stdout + getattr(res, "stderr", "")
    assert res.exit_code == 0, out_text
    assert "stub:" not in out_text.lower()
    assert primary_key not in out_text
    assert fallback_key not in out_text
    expected_k = cohens_kappa(judge_labels, human_labels)
    assert expected_k == pytest.approx(1.0)
    assert "1.00" in out_text


def test_t3_dosample_cache_bypass_vs_deterministic_noise_floor(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T3-PW-04: Deterministic runs hit cache (0% flips); do_sample=True bypasses cache."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-pw4-1234567890abcd")
    call_counter = {"n": 0}

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        call_counter["n"] += 1
        body = json.loads(request.content.decode("utf-8"))
        temp = float(body.get("temperature", 0.0))
        reply = (
            "Pancreas"
            if temp == 0.0 or call_counter["n"] % 2 != 0
            else "Liver"
        )
        return httpx.Response(
            200,
            json={
                "id": f"nf-{call_counter['n']}",
                "choices": [{"message": {"content": reply}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    _bind_mock_transport(monkeypatch, transport)

    csv_path = tmp_path / "nf10.csv"
    rows = ["sample_id,input_text,target"] + [
        f's{i},"Which organ produces insulin ({i})?",Pancreas' for i in range(10)
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    det_yaml = tmp_path / "det.yaml"
    det_yaml.write_text(
        "name: nf_det\n"
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_path}\n"
        'doc_to_text: "Question: {{ input_text }}\\nAnswer:"\n'
        'doc_to_target: "{{ target }}"\n'
        "generation_kwargs:\n"
        "  temperature: 0.0\n",
        encoding="utf-8",
    )

    samp_yaml = tmp_path / "samp.yaml"
    samp_yaml.write_text(
        "name: nf_det\n"
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_path}\n"
        'doc_to_text: "Question: {{ input_text }}\\nAnswer:"\n'
        'doc_to_target: "{{ target }}"\n'
        "generation_kwargs:\n"
        "  do_sample: true\n"
        "  temperature: 0.8\n",
        encoding="utf-8",
    )

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run3 = tmp_path / "run3"

    assert cli_runner.invoke(
        app, ["run", "--task", str(det_yaml), "--model", "http-byok", "--output", str(run1)]
    ).exit_code == 0
    assert len(recorded) == 10

    assert cli_runner.invoke(
        app, ["run", "--task", str(det_yaml), "--model", "http-byok", "--output", str(run2)]
    ).exit_code == 0
    assert len(recorded) == 10  # 100% cache hit

    res_nf_det = cli_runner.invoke(
        app, ["noise-floor", "--baseline", str(run1), "--perturbed", str(run2)]
    )
    assert res_nf_det.exit_code == 0, res_nf_det.stdout
    assert "0.00" in res_nf_det.stdout or "0.0%" in res_nf_det.stdout

    assert cli_runner.invoke(
        app, ["run", "--task", str(samp_yaml), "--model", "http-byok", "--output", str(run3)]
    ).exit_code == 0
    assert len(recorded) == 20  # Cache bypassed on do_sample=True

    res_nf_samp = cli_runner.invoke(
        app, ["noise-floor", "--baseline", str(run1), "--perturbed", str(run3)]
    )
    assert res_nf_samp.exit_code == 0, res_nf_samp.stdout
    assert not re.search(r"(?<!\d)0\.0+%?", res_nf_samp.stdout)
    assert "50.0%" in res_nf_samp.stdout or "0.5" in res_nf_samp.stdout


def test_t3_tier3_plugins_sandbox_eq_spoof_and_receipt(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T3-PW-05: Tier 3 --plugins + DockerSandbox fallback blocks __eq__ spoof and verifies."""
    plugin_path = tmp_path / "code_eval_plugin.py"
    plugin_path.write_text(
        "from collections.abc import Sequence\n"
        "from eval_studio.core.schema import (\n"
        "    EvalRequest,\n"
        "    EvalResponse,\n"
        "    OutputType,\n"
        "    Sample,\n"
        "    TokenUsage,\n"
        ")\n"
        "from eval_studio.gateway.mock import MockGateway\n"
        "from eval_studio.sandbox.docker import DockerSandbox\n"
        "from eval_studio.sandbox.limits import SandboxLimits\n\n"
        "class EchoCodeGateway(MockGateway):\n"
        "    backend_name = 'pw5_echo_code'\n"
        "    def generate_until(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:\n"
        "        return [\n"
        "            EvalResponse(\n"
        "                request_id=req.request_id,\n"
        "                sample_id=req.sample_id,\n"
        "                text=req.context,\n"
        "                backend=self.backend_name,\n"
        "                model_id=self.model_id,\n"
        "                usage=TokenUsage(\n"
        "                    prompt_tokens=max(1, len(req.context.split())),\n"
        "                    completion_tokens=max(1, len(req.context.split())),\n"
        "                ),\n"
        "            )\n"
        "            for req in requests\n"
        "        ]\n\n"
        "class CodeEvalSolver:\n"
        "    name = 'pw5_solver'\n"
        "    def build_request(self, sample: Sample) -> EvalRequest:\n"
        "        return EvalRequest(\n"
        "            request_id=f'req-{sample.sample_id}',\n"
        "            sample_id=sample.sample_id,\n"
        "            task_name='pw5_code_eval',\n"
        "            output_type=OutputType.GENERATE_UNTIL,\n"
        "            context=sample.input_text,\n"
        "        )\n\n"
        "class CodeEvalScorer:\n"
        "    name = 'pw5_scorer'\n"
        "    def score(self, sample: Sample, response: EvalResponse) -> dict[str, float]:\n"
        "        verdict = DockerSandbox().run_code(\n"
        "            response.text or '',\n"
        "            test_harness=str(sample.target),\n"
        "            limits=SandboxLimits(timeout_seconds=2.0, network_disabled=True),\n"
        "        )\n"
        "        return {'pass@1': 1.0 if verdict.passed else 0.0}\n\n"
        "def pw5_pass_at_1(prediction: str, reference: str) -> float:\n"
        "    verdict = DockerSandbox().run_code(\n"
        "        prediction,\n"
        "        test_harness=reference,\n"
        "        limits=SandboxLimits(timeout_seconds=2.0, network_disabled=True),\n"
        "    )\n"
        "    return 1.0 if verdict.passed else 0.0\n\n"
        "EVAL_STUDIO_PLUGINS = {\n"
        "    'eval_studio.models': {'pw5_echo_code': EchoCodeGateway},\n"
        "    'eval_studio.solvers': {'pw5_solver': CodeEvalSolver},\n"
        "    'eval_studio.metrics': {'pw5_scorer': CodeEvalScorer, 'pass@1': pw5_pass_at_1},\n"
        "}\n",
        encoding="utf-8",
    )

    data_path = tmp_path / "code_samples.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "sample_id": "c1",
                "input_text": "def calc(x):\n    return x * 2\nresult = calc(21)",
                "target": "assert type(result) is int and result == 42",
            }
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": "c2",
                "input_text": (
                    "class AlwaysEqual:\n    def __eq__(self, o): return True\n"
                    "result = AlwaysEqual()"
                ),
                "target": "assert type(result) is int and result == 42",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    task_yaml = tmp_path / "tier3_code.yaml"
    task_yaml.write_text(
        "name: pw5_code_eval\n"
        "trust_tier: tier3_code_plugin\n"
        "output_type: generate_until\n"
        f"dataset_path: {data_path.as_posix()}\n"
        "unsafe_code: true\n"
        "metrics:\n"
        "  - pass@1\n"
        "metadata:\n"
        "  solver: pw5_solver\n"
        "  scorer: pw5_scorer\n",
        encoding="utf-8",
    )

    # Confirm sandbox directly too
    sb = DockerSandbox()
    ok_verdict = sb.run_code(
        "result = 42",
        test_harness="assert type(result) is int and result == 42",
    )
    assert ok_verdict.passed is True
    assert sb.run_code(
        "class AlwaysEqual:\n    def __eq__(self, o): return True\nresult = AlwaysEqual()",
        test_harness="assert type(result) is int and result == 42",
    ).passed is False

    receipt_dir = tmp_path / "pw5_receipt"
    res = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "pw5_echo_code",
            "--plugins",
            str(plugin_path),
            "--trust-remote-code",
            "--output",
            str(receipt_dir),
        ],
    )
    assert res.exit_code == 0, res.stdout
    manifest = json.loads((receipt_dir / "manifest.json").read_text(encoding="utf-8"))
    score_val = manifest["metrics"].get("pass@1", manifest["metrics"].get("pw5_scorer"))["value"]
    assert score_val == pytest.approx(0.5)
    assert cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)]).exit_code == 0


def test_t3_whitespace_transfer_olmes_cloze_mcf_and_byte_metrics(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T3-PW-06: _encode_pair whitespace shift + OLMES Cloze/MCF + byte/char/PMI metrics."""
    ctx, cont = transfer_trailing_whitespace(
        "Patient presents with chest pain. Answer:  ",
        "Aspirin",
    )
    assert ctx == "Patient presents with chest pain. Answer:"
    assert cont == "  Aspirin"

    data_path = tmp_path / "olmes_mc.jsonl"
    data_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": f"om{i}",
                    "input_text": f"Chest pain case {i}. Answer:  ",
                    "choices": ["\n A", "\n B"],
                    "target": 0,
                }
            )
            for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )

    task_yaml = tmp_path / "olmes_task.yaml"
    task_yaml.write_text(
        "name: pw6_olmes_mc\n"
        "trust_tier: tier2_declarative\n"
        "output_type: multiple_choice\n"
        f"dataset_path: {data_path.as_posix()}\n"
        'doc_to_text: "{{ input_text }}"\n'
        'doc_to_target: "{{ target }}"\n'
        "target_delimiter: ' '\n"
        "mc_format: dual\n"
        "olmes_option_labels:\n"
        '  - "\\n A"\n'
        '  - "\\n B"\n'
        "olmes:\n"
        "  enabled: true\n"
        "  formats:\n"
        "    - cloze\n"
        "    - mcf\n"
        '  choice_prefix: "\\n "\n'
        "  metric_preference: acc_norm\n"
        "metrics:\n"
        "  - acc_bytes\n"
        "  - acc_norm\n"
        "  - acc_mutual_info\n",
        encoding="utf-8",
    )

    assert cli_runner.invoke(app, ["validate", "--task", str(task_yaml)]).exit_code == 0

    receipt_dir = tmp_path / "pw6_receipt"
    res = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "mock",
            "--apply-chat-template",
            "--output",
            str(receipt_dir),
        ],
    )
    assert res.exit_code == 0, res.stdout
    manifest = json.loads((receipt_dir / "manifest.json").read_text(encoding="utf-8"))
    for metric_key in ("acc_bytes", "acc_norm", "acc_mutual_info"):
        assert any(metric_key in k for k in manifest["metrics"])
    manifest_lower = json.dumps(manifest).casefold()
    assert "cloze" in manifest_lower
    assert "mcf" in manifest_lower


def test_t3_unbalanced_group_micro_macro_bootstrap1500_report(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T3-PW-07: Unbalanced subtasks + bootstrap_iters=1500 + micro/macro report rendering."""
    cardio_csv = tmp_path / "cardio.csv"
    cardio_csv.write_text(
        "sample_id,input_text,target\n"
        + "\n".join(f'c{i},"Cardio Q{i}","A"' for i in range(20))
        + "\n",
        encoding="utf-8",
    )
    renal_csv = tmp_path / "renal.csv"
    renal_csv.write_text(
        "sample_id,input_text,target\n"
        + "\n".join(f'r{i},"Renal Q{i}","B"' for i in range(5))
        + "\n",
        encoding="utf-8",
    )

    cardio_yaml = tmp_path / "cardio.yaml"
    cardio_yaml.write_text(
        "name: cardio\n"
        "trust_tier: tier2_declarative\n"
        "output_type: generate_until\n"
        f"dataset_path: {cardio_csv.as_posix()}\n"
        'doc_to_text: "{{ input_text }}"\n'
        'doc_to_target: "{{ target }}"\n'
        "metrics: [exact_match]\n",
        encoding="utf-8",
    )
    renal_yaml = tmp_path / "renal.yaml"
    renal_yaml.write_text(
        "name: renal\n"
        "trust_tier: tier2_declarative\n"
        "output_type: generate_until\n"
        f"dataset_path: {renal_csv.as_posix()}\n"
        'doc_to_text: "{{ input_text }}"\n'
        'doc_to_target: "{{ target }}"\n'
        "metrics: [exact_match]\n",
        encoding="utf-8",
    )

    group_yaml = tmp_path / "group_cardio_renal.yaml"
    group_yaml.write_text(
        "name: pw7_clinical_group\n"
        "trust_tier: tier2_declarative\n"
        "output_type: generate_until\n"
        "subtasks:\n"
        f"  - {cardio_yaml.as_posix()}\n"
        f"  - {renal_yaml.as_posix()}\n"
        "metrics:\n"
        "  - exact_match\n",
        encoding="utf-8",
    )

    micro = aggregate_group(
        [
            SubtaskStats("cardio", score=1.0, stderr=0.01, n_samples=20),
            SubtaskStats("renal", score=0.0, stderr=0.20, n_samples=5),
        ],
        weight_by_size=True,
    )
    macro = aggregate_group(
        [
            SubtaskStats("cardio", score=1.0, stderr=0.01, n_samples=20),
            SubtaskStats("renal", score=0.0, stderr=0.20, n_samples=5),
        ],
        weight_by_size=False,
    )
    assert micro.stderr_method == "pooled_sample_stderr"
    assert macro.stderr_method == "unweighted_mean_stderr"
    assert micro.score == pytest.approx(0.80)
    assert macro.score == pytest.approx(0.50)

    receipt_dir = tmp_path / "pw7_receipt"
    res = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(group_yaml),
            "--model",
            "mock",
            "--bootstrap-iters",
            "1500",
            "--output",
            str(receipt_dir),
        ],
    )
    assert res.exit_code == 0, res.stdout
    manifest = json.loads((receipt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert any(m["bootstrap_iters"] == 1500 for m in manifest["metrics"].values())
    assert (receipt_dir / "report.md").exists()
    assert (receipt_dir / "report.html").exists()

    report_md = (receipt_dir / "report.md").read_text(encoding="utf-8")
    report_html = (receipt_dir / "report.html").read_text(encoding="utf-8")
    combined_report = f"{json.dumps(manifest['metrics'])}\n{report_md}\n{report_html}"
    assert any(
        math.isclose(m["value"], micro.score, abs_tol=1e-6)
        for m in manifest["metrics"].values()
    )
    assert re.search(r"(?<!\d)(?:0\.80*|80(?:\.0+)?%)", combined_report)
    assert re.search(r"(?<!\d)(?:0\.50*|50(?:\.0+)?%)", combined_report)


def test_t3_windows_space_paths_wal_halt_resume_and_reproduce_cmd(
    space_workspace: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T3-PW-08: Space/UTF-8 workspace + mid-run WAL halt/resume + shlex reproduce_command."""
    cache_dir = space_workspace / "cache store (Café)"
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-pw8-1234567890abcd")

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"pw8-{len(recorded)}",
                "choices": [{"message": {"content": "Pancreas"}}],
                "usage": {"prompt_tokens": 5000, "completion_tokens": 500},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    _bind_mock_transport(monkeypatch, transport)

    csv_path = space_workspace / "triage samples (v1).csv"
    csv_path.write_text(
        "sample_id,input_text,target\n"
        + "\n".join(f's{i},"Which organ makes insulin ({i})?",Pancreas' for i in range(4))
        + "\n",
        encoding="utf-8",
    )
    task_yaml = space_workspace / "triage task (v1).yaml"
    task_yaml.write_text(
        "name: pw8_space_resume\n"
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_path}\n"
        'doc_to_text: "Question: {{ input_text }}\\nAnswer:"\n'
        'doc_to_target: "{{ target }}"\n'
        "metrics: [exact_match]\n"
        "generation_kwargs:\n"
        "  max_gen_toks: 500\n"
        "metadata:\n"
        "  pricing:\n"
        "    input_usd_per_1m: 1.0\n"
        "    output_usd_per_1m: 10.0\n",
        encoding="utf-8",
    )

    out1 = space_workspace / "receipt halted"
    res1 = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "http-byok",
            "--max-cost-usd",
            "0.025",
            "--output",
            str(out1),
        ],
    )
    halt_out = res1.stdout + getattr(res1, "stderr", "") + (
        f" {res1.exception}" if res1.exception else ""
    )
    assert res1.exit_code != 0
    assert "no such option" not in halt_out.lower()
    assert "stub:" not in halt_out.lower()
    assert 1 <= len(recorded) < 4

    wal_files = list(cache_dir.rglob("*.jsonl"))
    assert wal_files, "Halted run must preserve completed samples in WAL"
    wal_entries = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in wal_files
    )
    assert wal_entries >= len(recorded)

    out2 = space_workspace / "receipt resumed"
    res2 = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "http-byok",
            "--max-cost-usd",
            "0.10",
            "--output",
            str(out2),
        ],
    )
    assert res2.exit_code == 0, res2.stdout
    assert len(recorded) == 4
    assert cli_runner.invoke(app, ["verify-receipt", str(out2)]).exit_code == 0

    manifest = json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))
    repro_cmd = manifest.get("reproduce_command", "")
    assert repro_cmd
    tokens = [tok.strip("\"'") for tok in shlex.split(repro_cmd, posix=False)]
    assert any(str(task_yaml) in t or task_yaml.name in t for t in tokens)

    shutil.rmtree(space_workspace)
    assert not space_workspace.exists()
