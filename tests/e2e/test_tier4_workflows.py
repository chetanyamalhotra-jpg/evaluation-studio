"""Tier 4 real-world clinical and biomedical workflow E2E tests (6 workflows)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from eval_studio.cli.app import app
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import SecretRole, contains_secret
from eval_studio.paper.compiler import compile_paper, sha256_file
from eval_studio.paper.schema import Citation, ExtractedItem
from eval_studio.scoring.calibrate import cohens_kappa
from eval_studio.scoring.judge import JudgeRubric, LLMJudge
from eval_studio.stats.fewshot import (
    ComparisonStatus,
    evaluate_fewshot_comparison,
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


def test_t4_workflow1_clinical_mcqa_olmes_kset_rotation(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T4-WF-01: Clinical MCQA dual Cloze+MCF eval with OLMES & 4-set few-shot rotation."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-wf1-1234567890abcd")

    data_path = tmp_path / "cardio_pharm_12.jsonl"
    items = [
        {
            "sample_id": f"cp_{i:02d}",
            "input_text": (
                f"CARDIO_VIGNETTE_{i:02d}: Patient with atrial fibrillation and renal impairment "
                f"(eGFR {35 + i} mL/min). Best anticoagulant choice?"
            ),
            "choices": ["\n A", "\n B", "\n C", "\n D"],
            "target": i % 4,
        }
        for i in range(12)
    ]
    data_path.write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")

    task_yaml = tmp_path / "cardio_olmes.yaml"
    task_yaml.write_text(
        "name: wf1_cardio_pharm_olmes\n"
        'version: "1.0.0"\n'
        "trust_tier: tier2_declarative\n"
        "output_type: multiple_choice\n"
        f"dataset_path: {data_path.as_posix()}\n"
        'doc_to_text: "Vignette: {{ input_text }}\\nOptions: A, B, C, D\\nAnswer:"\n'
        'doc_to_target: "{{ target }}"\n'
        "num_fewshot: 3\n"
        "fewshot_rotation_sets: 4\n"
        "mc_format: dual\n"
        "olmes_option_labels:\n"
        '  - "\\n A"\n'
        '  - "\\n B"\n'
        '  - "\\n C"\n'
        '  - "\\n D"\n'
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
        "  - acc_mutual_info\n"
        "  - brier_score\n",
        encoding="utf-8",
    )

    assert cli_runner.invoke(app, ["validate", "--task", str(task_yaml)]).exit_code == 0

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"mc-{len(recorded)}",
                "choices": [{"logprobs": {"token_logprobs": [-0.4, -0.2]}}],
                "usage": {"prompt_tokens": 64, "completion_tokens": 2},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    _bind_mock_transport(monkeypatch, transport)

    receipt_a = tmp_path / "ckpt_a"
    receipt_b = tmp_path / "ckpt_b"
    res_a = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "http-byok",
            "--model-id",
            "ckpt-a",
            "--output",
            str(receipt_a),
        ],
    )
    res_b = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "http-byok",
            "--model-id",
            "ckpt-b",
            "--output",
            str(receipt_b),
        ],
    )
    assert res_a.exit_code == 0, res_a.stdout
    assert res_b.exit_code == 0, res_b.stdout
    assert len(recorded) >= 24

    # Zero active-doc few-shot leakage, actual num_fewshot=3 injection, and 4-set rotation
    conditional_reqs = [r for r in recorded if "CARDIO_VIGNETTE_" in r["body_text"]]
    unconditional_reqs = [r for r in recorded if "CARDIO_VIGNETTE_" not in r["body_text"]]
    assert len(conditional_reqs) >= 24
    assert len(unconditional_reqs) >= 4
    exemplar_signatures: set[tuple[str, ...]] = set()
    for req_rec in conditional_reqs:
        body = req_rec["body_text"]
        positions = [
            (body.index(f"CARDIO_VIGNETTE_{i:02d}:"), f"CARDIO_VIGNETTE_{i:02d}:")
            for i in range(12)
            if f"CARDIO_VIGNETTE_{i:02d}:" in body
        ]
        positions.sort()
        assert len(positions) == 4, f"Expected 1 active + 3 few-shot exemplars, got {positions}"
        for _, tok in positions:
            assert body.count(tok) == 1
        exemplar_signatures.add(tuple(tok for _, tok in positions[:-1]))
    assert len(exemplar_signatures) >= 2

    manifest_a = json.loads((receipt_a / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((receipt_b / "manifest.json").read_text(encoding="utf-8"))
    for metric_name in ("acc_bytes", "acc_norm", "acc_mutual_info", "brier_score"):
        assert any(metric_name in k for k in manifest_a["metrics"])
        assert any(metric_name in k for k in manifest_b["metrics"])
    manifest_a_lower = json.dumps(manifest_a).casefold()
    assert "cloze" in manifest_a_lower
    assert "mcf" in manifest_a_lower

    # Compute actual per-set scores from both checkpoint receipts and verify unresolved CI
    def _set_means(receipt_dir: Path) -> list[float]:
        rows = [
            json.loads(line)
            for line in (receipt_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 12
        buckets: dict[int, list[float]] = {k: [] for k in range(4)}
        for idx, row in enumerate(rows):
            pipe_scores = next(iter(row["metrics"].values()))
            score_val = pipe_scores.get("acc_norm", next(iter(pipe_scores.values())))
            buckets[idx % 4].append(float(score_val))
        return [sum(buckets[k]) / len(buckets[k]) for k in range(4)]

    comp = evaluate_fewshot_scores(_set_means(receipt_a), _set_means(receipt_b))
    assert comp.status == ComparisonStatus.UNRESOLVED
    assert comp.n_sets == 4

    assert cli_runner.invoke(app, ["verify-receipt", str(receipt_a)]).exit_code == 0
    assert cli_runner.invoke(app, ["verify-receipt", str(receipt_b)]).exit_code == 0


def test_t4_workflow2_biomedical_paper_to_eval_and_tamper_audit(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T4-WF-02: Biomedical paper compilation, Stage 2 gate, run, and 1-char tamper audit."""
    monkeypatch.setenv("EXTRACTOR_API_KEY", "sk-extract-wf2-1234567890abcd")

    paper_path = tmp_path / "biomarker_study.md"
    paper_path.write_text(
        "# High-Sensitivity Troponin Triage Study\n\n"
        "## Section 1: Background\n"
        "Elevated serum troponin I (> 0.04 ng/mL) indicates acute myocardial injury.\n\n"
        "## Section 2: Cohort Findings\n"
        "Across 240 emergency department chest-pain patients, the 0/1-hour protocol achieved "
        "a negative predictive value of 99.6% and sensitivity of 98.4%.\n"
        "| Subgroup | N | NPV |\n|---|---|---|\n| Early (<3h) | 96 | 99.0% |\n\n"
        "## Section 3: Limitations\n"
        "Single-center observational cohort excluding chronic kidney disease stage 5.\n",
        encoding="utf-8",
    )

    # Verify two-stage gating over 4 valid cited + 1 uncited/blank-ish + 1 leaky candidate
    c1 = Citation(page=1, section="Section 1: Background", quote="> 0.04 ng/mL")
    c2 = Citation(
        page=1,
        section="Section 2: Cohort Findings",
        quote="negative predictive value of 99.6%",
    )
    c3 = Citation(page=1, section="Section 2: Cohort Findings", quote="sensitivity of 98.4%")
    c4 = Citation(
        page=1,
        section="Section 3: Limitations",
        quote="excluding chronic kidney disease stage 5",
    )
    candidates = [
        ExtractedItem(
            item_id="i1",
            question="What troponin I threshold indicates acute myocardial injury?",
            reference_answer="> 0.04 ng/mL",
            citations=[c1],
        ),
        ExtractedItem(
            item_id="i2",
            question="What NPV did the 0/1-hour protocol achieve?",
            reference_answer="99.6%",
            citations=[c2],
        ),
        ExtractedItem(
            item_id="i3",
            question="What diagnostic sensitivity was observed in the cohort?",
            reference_answer="98.4%",
            citations=[c3],
        ),
        ExtractedItem(
            item_id="i4",
            question="Which renal population was excluded from the cohort?",
            reference_answer="chronic kidney disease stage 5",
            citations=[c4],
        ),
        ExtractedItem(
            item_id="i_leaky",
            question="Why did the protocol achieve 99.6% NPV?",
            reference_answer="99.6%",
            citations=[c2],
        ),
        ExtractedItem(
            item_id="i_dangling",
            question="What did Table 2 above report for late presenters?",
            reference_answer="97.1%",
            citations=[c2],
        ),
    ]
    gated_suite = compile_paper(paper_path, items=candidates, suite_name="biomarker_gated")
    assert len(gated_suite.items) == 4
    assert all(item.citations and item.citations[0].page >= 1 for item in gated_suite.items)

    suite_yaml = tmp_path / "biomarker_suite.yaml"
    res_comp = cli_runner.invoke(
        app,
        ["compile-paper", str(paper_path), "--out", str(suite_yaml)],
    )
    assert res_comp.exit_code == 0, res_comp.stdout

    assert cli_runner.invoke(app, ["validate", "--task", str(suite_yaml)]).exit_code == 0

    receipt_dir = tmp_path / "wf2_receipt"
    res_run = cli_runner.invoke(
        app,
        ["run", "--task", str(suite_yaml), "--model", "mock", "--output", str(receipt_dir)],
    )
    assert res_run.exit_code == 0, res_run.stdout

    manifest = json.loads((receipt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_paper_sha256"] == sha256_file(paper_path)
    assert not contains_secret(json.dumps(manifest))

    assert cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)]).exit_code == 0

    # 1-char tamper in source paper breaks verification
    orig = paper_path.read_text(encoding="utf-8")
    paper_path.write_text(orig.replace("99.6%", "99.5%"), encoding="utf-8")
    assert cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)]).exit_code == 1


def test_t4_workflow3_budget_capped_api_interrupt_and_resume(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T4-WF-03: Budget-capped HTTP triage run halts at 4/8, saves WAL, resumes with 0 dups."""
    secret_key = "sk-triage-live-secret-9876543210"
    monkeypatch.setenv("TARGET_API_KEY", secret_key)
    cache_dir = tmp_path / "wf3_cache"
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(cache_dir))

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        assert secret_key in request.headers.get("authorization", "")
        return httpx.Response(
            200,
            json={
                "id": f"triage-{len(recorded)}",
                "choices": [{"message": {"content": "Urgent ECG"}}],
                "usage": {"prompt_tokens": 5000, "completion_tokens": 500},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    _bind_mock_transport(monkeypatch, transport)

    csv_path = tmp_path / "triage_8.csv"
    csv_path.write_text(
        "sample_id,input_text,target\n"
        + "\n".join(
            f'tr_{i},"62yo chest tightness triage step {i}?","Urgent ECG"' for i in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    task_yaml = tmp_path / "triage_task.yaml"
    task_yaml.write_text(
        "name: wf3_clinical_triage\n"
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_path.as_posix()}\n"
        'doc_to_text: "Triage: {{ input_text }}\\nAction:"\n'
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

    # Pre-flight dry-run estimate dispatches 0 HTTP requests
    res_dry = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "http-byok",
            "--dry-run",
            "--estimate-cost",
        ],
    )
    assert res_dry.exit_code == 0, res_dry.stdout
    assert len(recorded) == 0

    # Half-budget run halts mid-run and preserves completed samples in WAL
    out_halted = tmp_path / "wf3_halted"
    res_halt = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "http-byok",
            "--max-cost-usd",
            "0.045",
            "--output",
            str(out_halted),
        ],
    )
    halt_out = res_halt.stdout + getattr(res_halt, "stderr", "") + (
        f" {res_halt.exception}" if res_halt.exception else ""
    )
    assert res_halt.exit_code != 0
    assert "no such option" not in halt_out.lower()
    assert "stub:" not in halt_out.lower()
    assert 1 <= len(recorded) < 8
    assert secret_key not in halt_out
    assert not contains_secret(halt_out)

    wal_files = list(cache_dir.rglob("*.jsonl"))
    assert wal_files, "Mid-run cost halt must preserve completed samples in WAL"
    wal_entries = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in wal_files
    )
    assert wal_entries >= len(recorded)

    # Raise budget & resume -> dispatches only remaining 4 items (total == 8)
    out_full = tmp_path / "wf3_resumed"
    res_resume = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "http-byok",
            "--max-cost-usd",
            "0.15",
            "--output",
            str(out_full),
        ],
    )
    assert res_resume.exit_code == 0, res_resume.stdout
    assert len(recorded) == 8

    for artifact in (
        *(cache_dir.rglob("*.jsonl")),
        out_full / "samples.jsonl",
        out_full / "manifest.json",
    ):
        text = artifact.read_text(encoding="utf-8")
        assert secret_key not in text
        assert not contains_secret(text)

    assert cli_runner.invoke(app, ["verify-receipt", str(out_full)]).exit_code == 0


def test_t4_workflow4_multi_judge_rubric_calibration_and_rescore(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T4-WF-04: CoT strip + rubric calibration (<0.40 -> >=0.70) + offline fallback_judge."""
    target_key = "sk-target-wf4-111111111111"
    judge_key = "sk-judge-wf4-222222222222"
    fallback_key = "sk-fallback-wf4-333333333333"
    monkeypatch.setenv("TARGET_API_KEY", target_key)
    monkeypatch.setenv(SecretRole.JUDGE.value, judge_key)
    monkeypatch.setenv("FALLBACK_JUDGE_API_KEY", fallback_key)

    target_calls = {"count": 0}

    def _target_handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        target_calls["count"] += 1
        return httpx.Response(
            200,
            json={
                "id": f"dc-{target_calls['count']}",
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>Check discharge meds and follow-up</think>"
                                "Continue lisinopril 10mg daily; follow up in 7 days."
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 18},
            },
        )

    transport, _ = make_mock_transport(_target_handler)
    _bind_mock_transport(monkeypatch, transport)

    csv_path = tmp_path / "discharge.csv"
    plan = "Continue lisinopril 10mg daily; follow up in 7 days."
    csv_path.write_text(
        "sample_id,input_text,target\n"
        f'd1,"Summarize discharge plan for CHF patient.","{plan}"\n'
        f'd2,"Summarize discharge plan for HTN patient.","{plan}"\n',
        encoding="utf-8",
    )
    task_yaml = tmp_path / "discharge_task.yaml"
    task_yaml.write_text(
        "name: wf4_discharge_summary\n"
        "trust_tier: tier2_declarative\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_path}\n"
        'doc_to_text: "Discharge Case: {{ input_text }}\\nSummary:"\n'
        'doc_to_target: "{{ target }}"\n'
        "metrics: [exact_match]\n",
        encoding="utf-8",
    )

    receipt_dir = tmp_path / "wf4_receipt"
    res_run = cli_runner.invoke(
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
    assert res_run.exit_code == 0, res_run.stdout
    assert target_calls["count"] == 2

    # Verify CoT stripping in emitted wf4_receipt/samples.jsonl
    wf4_samples = [
        json.loads(line)
        for line in (receipt_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(wf4_samples) == 2
    for sample_rec in wf4_samples:
        raw0 = sample_rec["raw_responses"][0]
        assert raw0["text"] == plan
        assert "<think>" not in (raw0["text"] or "")
        assert raw0.get("reasoning_trace") is not None
        assert "Check discharge meds and follow-up" in raw0["reasoning_trace"]

    # Step 2: Unaligned rubric (< 0.40) rejected, then aligned rubric (>= 0.70) passes
    human_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    unaligned_judge = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]  # kappa = 0.20
    aligned_judge = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]  # kappa = 0.80

    assert cohens_kappa(unaligned_judge, human_labels) < 0.40
    assert cohens_kappa(aligned_judge, human_labels) >= 0.70

    unaligned_file = tmp_path / "unaligned.jsonl"
    unaligned_file.write_text(
        "\n".join(
            json.dumps({"sample_id": f"u{i}", "judge_label": j, "human_label": h})
            for i, (j, h) in enumerate(zip(unaligned_judge, human_labels, strict=True))
        )
        + "\n",
        encoding="utf-8",
    )
    res_unaligned = cli_runner.invoke(app, ["calibrate", str(unaligned_file)])
    unaligned_out = res_unaligned.stdout + getattr(res_unaligned, "stderr", "")
    assert "stub:" not in unaligned_out.lower()
    assert "unaligned" in unaligned_out.lower()

    aligned_file = tmp_path / "aligned.jsonl"
    aligned_file.write_text(
        "\n".join(
            json.dumps({"sample_id": f"a{i}", "judge_label": j, "human_label": h})
            for i, (j, h) in enumerate(zip(aligned_judge, human_labels, strict=True))
        )
        + "\n",
        encoding="utf-8",
    )
    res_aligned = cli_runner.invoke(app, ["calibrate", str(aligned_file)])
    aligned_out = res_aligned.stdout + getattr(res_aligned, "stderr", "")
    assert res_aligned.exit_code == 0, aligned_out
    assert "stub:" not in aligned_out.lower()
    assert "unaligned" not in aligned_out.lower()

    # Step 3: Offline re-scoring + fallback_judge failover grading actual wf4_receipt outputs
    rubric = JudgeRubric(name="discharge_completeness", criteria="Verify dose and follow-up.")
    judge = LLMJudge(
        judge_model_id="mock-judge-primary",
        rubric=rubric,
        gateway=MockGateway(model_id="mock-judge-primary", judge_mode="refusal"),
        fallback_judge=MockGateway(
            model_id="mock-judge-fallback",
            judge_mode="normal",
            default_reply=json.dumps({"score": 1.0, "rationale": "Complete discharge plan."}),
        ),
    )
    for sample_rec in wf4_samples:
        verdict = judge.grade(
            prompt=f"Discharge summary for {sample_rec['sample_id']}",
            response=sample_rec["raw_responses"][0]["text"],
            reference=str(sample_rec["target"]),
        )
        assert verdict.score == 1.0
        assert verdict.rationale

    res_score = cli_runner.invoke(app, ["score", "--receipt", str(receipt_dir)])
    assert res_score.exit_code == 0, res_score.stdout
    assert target_calls["count"] == 2  # Zero target calls during offline score

    for secret in (target_key, judge_key, fallback_key):
        assert secret not in res_score.stdout
        assert secret not in (receipt_dir / "manifest.json").read_text(encoding="utf-8")

    assert cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)]).exit_code == 0


def test_t4_workflow5_tier3_clinical_dosing_plugin_and_eq_defense(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T4-WF-05: Tier 3 pediatric dosing plugin with --trust-remote-code & __eq__ defense."""
    plugin_path = tmp_path / "pediatric_dosing_plugin.py"
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
        "class EchoDosingGateway(MockGateway):\n"
        "    backend_name = 'wf5_echo_code'\n"
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
        "class DosingSolver:\n"
        "    name = 'peds_dosing_solver'\n"
        "    def build_request(self, sample: Sample) -> EvalRequest:\n"
        "        return EvalRequest(\n"
        "            request_id=f'req-{sample.sample_id}',\n"
        "            sample_id=sample.sample_id,\n"
        "            task_name='wf5_peds_dosing',\n"
        "            output_type=OutputType.GENERATE_UNTIL,\n"
        "            context=sample.input_text,\n"
        "        )\n\n"
        "class DosingScorer:\n"
        "    name = 'peds_dosing_scorer'\n"
        "    def score(self, sample: Sample, response: EvalResponse) -> dict[str, float]:\n"
        "        verdict = DockerSandbox().run_code(\n"
        "            response.text or '',\n"
        "            test_harness=str(sample.target),\n"
        "            limits=SandboxLimits(timeout_seconds=2.0, network_disabled=True),\n"
        "        )\n"
        "        return {'pass@1': 1.0 if verdict.passed else 0.0}\n\n"
        "def wf5_pass_at_1(prediction: str, reference: str) -> float:\n"
        "    verdict = DockerSandbox().run_code(\n"
        "        prediction,\n"
        "        test_harness=reference,\n"
        "        limits=SandboxLimits(timeout_seconds=2.0, network_disabled=True),\n"
        "    )\n"
        "    return 1.0 if verdict.passed else 0.0\n\n"
        "EVAL_STUDIO_PLUGINS = {\n"
        "    'eval_studio.models': {'wf5_echo_code': EchoDosingGateway},\n"
        "    'eval_studio.solvers': {'peds_dosing_solver': DosingSolver},\n"
        "    'eval_studio.metrics': {\n"
        "        'peds_dosing_scorer': DosingScorer,\n"
        "        'pass@1': wf5_pass_at_1,\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )

    data_path = tmp_path / "dosing_4.jsonl"
    samples = [
        {
            "sample_id": "dose_ok_1",
            "input_text": (
                "def pediatric_dose_mg(kg):\n    return kg * 15\n"
                "result = pediatric_dose_mg(20)"
            ),
            "target": "assert type(result) is int and result == 300",
        },
        {
            "sample_id": "dose_ok_2",
            "input_text": (
                "def pediatric_dose_mg(kg):\n    return kg * 10\n"
                "result = pediatric_dose_mg(12)"
            ),
            "target": "assert type(result) is int and result == 120",
        },
        {
            "sample_id": "dose_wrong",
            "input_text": (
                "def pediatric_dose_mg(kg):\n    return kg * 5\n"
                "result = pediatric_dose_mg(20)"
            ),
            "target": "assert type(result) is int and result == 300",
        },
        {
            "sample_id": "dose_spoof",
            "input_text": (
                "class AlwaysEqual:\n    def __eq__(self, o): return True\n"
                "result = AlwaysEqual()"
            ),
            "target": "assert type(result) is int and result == 300",
        },
    ]
    data_path.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")

    task_yaml = tmp_path / "wf5_dosing.yaml"
    task_yaml.write_text(
        "name: wf5_peds_dosing\n"
        "trust_tier: tier3_code_plugin\n"
        "output_type: generate_until\n"
        f"dataset_path: {data_path.as_posix()}\n"
        "unsafe_code: true\n"
        "metrics:\n"
        "  - pass@1\n"
        "metadata:\n"
        "  solver: peds_dosing_solver\n"
        "  scorer: peds_dosing_scorer\n",
        encoding="utf-8",
    )

    # Blocked without --trust-remote-code
    res_blocked = cli_runner.invoke(app, ["validate", "--task", str(task_yaml)])
    blocked_out = res_blocked.stdout + getattr(res_blocked, "stderr", "") + (
        f" {res_blocked.exception}" if res_blocked.exception else ""
    )
    assert res_blocked.exit_code != 0
    assert "no such option" not in blocked_out.lower()
    assert "stub:" not in blocked_out.lower()
    assert "trust" in blocked_out.lower()

    # Allowed with --trust-remote-code
    res_valid = cli_runner.invoke(
        app,
        [
            "validate",
            "--task",
            str(task_yaml),
            "--plugins",
            str(plugin_path),
            "--trust-remote-code",
        ],
    )
    assert res_valid.exit_code == 0, res_valid.stdout

    receipt_dir = tmp_path / "wf5_receipt"
    res_run = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            "wf5_echo_code",
            "--plugins",
            str(plugin_path),
            "--trust-remote-code",
            "--output",
            str(receipt_dir),
        ],
    )
    assert res_run.exit_code == 0, res_run.stdout
    manifest = json.loads((receipt_dir / "manifest.json").read_text(encoding="utf-8"))
    pass_at_1 = manifest["metrics"].get("pass@1", manifest["metrics"].get("peds_dosing_scorer"))[
        "value"
    ]
    assert pass_at_1 == pytest.approx(0.5)
    assert cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)]).exit_code == 0


def test_t4_workflow6_cross_backend_noise_floor_audit(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T4-WF-06: Cross-backend & batch-size noise-floor audit prior to delta reporting."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-wf6-1234567890abcd")

    csv_path = tmp_path / "decisions_10.csv"
    csv_path.write_text(
        "sample_id,input_text,target\n"
        + "\n".join(f'dec_{i},"Sepsis alert decision {i}?","A"' for i in range(10))
        + "\n",
        encoding="utf-8",
    )
    task_yaml = tmp_path / "sepsis_decisions.yaml"
    task_yaml.write_text(
        "name: wf6_sepsis_audit\n"
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_path}\n"
        'doc_to_text: "Alert: {{ input_text }}\\nDecision:"\n'
        'doc_to_target: "{{ target }}"\n'
        "metrics: [exact_match]\n",
        encoding="utf-8",
    )

    baseline_dir = tmp_path / "baseline"
    restart_dir = tmp_path / "restart"
    perturbed_dir = tmp_path / "perturbed"

    assert (
        cli_runner.invoke(
            app,
            [
                "run",
                "--task",
                str(task_yaml),
                "--model",
                "mock",
                "--batch-size",
                "1",
                "--output",
                str(baseline_dir),
            ],
        ).exit_code
        == 0
    )

    # Fresh cache restart check
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(tmp_path / "fresh_cache_restart"))
    assert (
        cli_runner.invoke(
            app,
            [
                "run",
                "--task",
                str(task_yaml),
                "--model",
                "mock",
                "--batch-size",
                "1",
                "--output",
                str(restart_dir),
            ],
        ).exit_code
        == 0
    )

    res_nf_restart = cli_runner.invoke(
        app,
        ["noise-floor", "--baseline", str(baseline_dir), "--perturbed", str(restart_dir)],
    )
    assert res_nf_restart.exit_code == 0, res_nf_restart.stdout
    assert "0.0" in res_nf_restart.stdout

    # Cross-backend + batch_size=4 perturbation flipping 2 of 10 decisions (20%)
    idx_counter = {"i": 0}

    def _perturbed_handler(
        request: httpx.Request,
        recorded: list[dict[str, Any]],
    ) -> httpx.Response:
        cur = idx_counter["i"]
        idx_counter["i"] += 1
        decision = "B" if cur in (2, 7) else "A"
        return httpx.Response(
            200,
            json={
                "id": f"pert-{cur}",
                "choices": [{"message": {"content": decision}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
            },
        )

    transport, _ = make_mock_transport(_perturbed_handler)
    _bind_mock_transport(monkeypatch, transport)

    assert (
        cli_runner.invoke(
            app,
            [
                "run",
                "--task",
                str(task_yaml),
                "--model",
                "http-byok",
                "--batch-size",
                "4",
                "--output",
                str(perturbed_dir),
            ],
        ).exit_code
        == 0
    )

    res_nf_pert = cli_runner.invoke(
        app,
        ["noise-floor", "--baseline", str(baseline_dir), "--perturbed", str(perturbed_dir)],
    )
    assert res_nf_pert.exit_code == 0, res_nf_pert.stdout
    assert "0.20" in res_nf_pert.stdout or "20.0%" in res_nf_pert.stdout

    m_rate = re.search(r"(\d+(?:\.\d+)?)\s*%", res_nf_pert.stdout)
    if m_rate:
        measured_noise_floor = float(m_rate.group(1)) / 100.0
    else:
        m_dec = re.search(r"(0\.\d+)", res_nf_pert.stdout)
        assert m_dec is not None, res_nf_pert.stdout
        measured_noise_floor = float(m_dec.group(1))
    assert measured_noise_floor == pytest.approx(0.20, abs=1e-6)

    # A +0.10 checkpoint delta with positive CI is unresolved when <= measured 0.20 noise floor
    comp = evaluate_fewshot_comparison(
        [0.10, 0.11, 0.09, 0.10],
        noise_floor=measured_noise_floor,
    )
    assert comp.ci_low > 0.0
    assert comp.status == ComparisonStatus.UNRESOLVED
