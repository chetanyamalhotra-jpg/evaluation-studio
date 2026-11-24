"""Tier 2 boundary & negative E2E tests for Core, Stats, Cache, and BYOK (12 tests)."""

from __future__ import annotations

import json
import math
import random
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from eval_studio.cache.fingerprint import resolve_cache_dir
from eval_studio.cache.wal import JsonlWriteAheadLog
from eval_studio.cli.app import app
from eval_studio.core.config import EvalRunConfig, RuntimeLimits
from eval_studio.core.registry import get_registry
from eval_studio.core.schema import EvalRequest, OutputType, TaskSpec
from eval_studio.gateway.http import HttpBYOKGateway
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import (
    REDACTED_PLACEHOLDER,
    SecretRole,
    contains_secret,
    load_api_key,
)
from eval_studio.scoring.filters import MultiChoiceRegexFilter
from eval_studio.scoring.metrics import exact_match, numeric_exact_match
from eval_studio.stats.aggregation import SubtaskStats, aggregate_group
from eval_studio.stats.bootstrap import bootstrap_stderr
from eval_studio.tasks.validator import validate_task_spec

pytestmark = pytest.mark.e2e


def _bind_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
) -> None:
    """Route all httpx Client/AsyncClient requests in-process to an offline MockTransport."""
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


def _write_csv_task(
    base_dir: Path,
    *,
    name: str,
    rows: list[tuple[str, str, str]],
    metrics: list[str] | None = None,
    extra_yaml: str = "",
) -> Path:
    data_path = base_dir / f"{name}.csv"
    lines = ["sample_id,input_text,target"]
    for sid, inp, tgt in rows:
        lines.append(f'{sid},"{inp}","{tgt}"')
    data_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics_yaml = "\n".join(f"  - {m}" for m in (metrics or ["exact_match"]))
    task_path = base_dir / f"{name}.yaml"
    task_path.write_text(
        f"name: {name}\n"
        'version: "1.0.0"\n'
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {data_path}\n"
        'doc_to_text: "Question: {{ input_text }}\\nAnswer:"\n'
        'doc_to_target: "{{ target }}"\n'
        "metrics:\n"
        f"{metrics_yaml}\n"
        f"{extra_yaml}",
        encoding="utf-8",
    )
    return task_path


# ---------------------------------------------------------------------------
# Area 2A: Core, Stats & Cache Boundaries (T2-CORE-01 .. T2-CORE-07)
# ---------------------------------------------------------------------------


def test_t2_bos_plus_chat_template_conflict_rejected(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-CORE-01: Reject apply_chat_template=True combined with add_bos_token=True."""
    default_cfg = EvalRunConfig()
    assert default_cfg.add_bos_token is None

    with pytest.raises(ValidationError, match="apply_chat_template"):
        EvalRunConfig(apply_chat_template=True, add_bos_token=True)

    task_path = _write_csv_task(
        tmp_path,
        name="bos_conflict_task",
        rows=[("s1", "Which organ makes insulin?", "Pancreas")],
    )
    out_dir = tmp_path / "receipt_out"
    res = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_path),
            "--model",
            "mock",
            "--apply-chat-template",
            "--add-bos-token",
            "--output",
            str(out_dir),
        ],
    )
    combined = res.stdout + getattr(res, "stderr", "")
    assert res.exit_code != 0
    assert "no such option" not in combined.lower()
    assert "bos" in combined.lower() or "duplicate" in combined.lower()
    assert not (out_dir / "manifest.json").exists()


def test_t2_enable_thinking_on_mc_and_missing_delimiter_rejected(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T2-CORE-02: Thinking flag rejected on MC/without delimiter; truncated CoT handled."""
    mc_yaml = tmp_path / "mc_thinking.yaml"
    mc_yaml.write_text(
        "name: mc_thinking\n"
        "trust_tier: tier2_declarative\n"
        "output_type: multiple_choice\n"
        "metrics:\n"
        "  - acc_bytes\n",
        encoding="utf-8",
    )
    mc_spec = TaskSpec(name="mc_thinking", output_type=OutputType.MULTIPLE_CHOICE)
    assert validate_task_spec(mc_spec, enable_thinking=True).valid is False

    res_val = cli_runner.invoke(
        app,
        ["validate", "--task", str(mc_yaml), "--enable-thinking"],
    )
    val_out = res_val.stdout + getattr(res_val, "stderr", "")
    assert res_val.exit_code != 0
    assert "no such option" not in val_out.lower()
    assert "enable_thinking" in val_out.lower() or "log-likelihood" in val_out.lower()

    with pytest.raises(ValidationError, match="think_end_token"):
        EvalRunConfig(enable_thinking=True, think_end_token=None)

    # Mid-thought truncation must not crash the batch; emits thinking_truncated=True & "" answer
    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp-trunc",
                "choices": [{"message": {"content": "<think>still reasoning without closing"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            },
        )

    transport, _ = make_mock_transport(_handler)
    _bind_mock_transport(monkeypatch, transport)
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-thinking-1234567890")

    gen_task = _write_csv_task(
        tmp_path,
        name="truncated_thinking",
        rows=[("t1", "What vitamin prevents scurvy?", "Vitamin C")],
    )
    out_dir = tmp_path / "trunc_receipt"
    res_run = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(gen_task),
            "--model",
            "http-byok",
            "--enable-thinking",
            "--think-end-token",
            "</think>",
            "--output",
            str(out_dir),
        ],
    )
    assert res_run.exit_code == 0, res_run.stdout
    sample_lines = (out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(sample_lines) == 1
    sample_rec = json.loads(sample_lines[0])
    raw_resp = sample_rec["raw_responses"][0]
    assert raw_resp.get("thinking_truncated") is True
    assert (raw_resp.get("text") or "") == ""
    assert "still reasoning" in (raw_resp.get("reasoning_trace") or "")


def test_t2_bootstrap_iters_exact_boundary_values(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-CORE-03: bootstrap_stderr executes exact requested iterations without truncation."""
    values = [0.0, 1.0, 1.0, 0.0, 1.0]

    r0 = bootstrap_stderr(values, bootstrap_iters=0, seed=1234)
    assert r0.iters_executed == 0
    assert r0.stderr == 0.0

    r37 = bootstrap_stderr(values, bootstrap_iters=37, seed=1234)
    assert r37.iters_executed == 37
    assert r37.stderr > 0.0

    r1500 = bootstrap_stderr(values, bootstrap_iters=1500, seed=1234)
    assert r1500.iters_executed == 1500
    assert r1500.stderr > 0.0

    with pytest.raises(ValueError):
        bootstrap_stderr(values, bootstrap_iters=-1)

    task_path = _write_csv_task(
        tmp_path,
        name="boot_task",
        rows=[
            ("b1", "Q1?", "A"),
            ("b2", "Q2?", "B"),
            ("b3", "Q3?", "A"),
            ("b4", "Q4?", "A"),
        ],
    )

    out_1500 = tmp_path / "boot_1500"
    res_1500 = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_path),
            "--model",
            "mock",
            "--bootstrap-iters",
            "1500",
            "--output",
            str(out_1500),
        ],
    )
    assert res_1500.exit_code == 0, res_1500.stdout
    manifest = json.loads((out_1500 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics"]["exact_match"]["bootstrap_iters"] == 1500

    out_neg = tmp_path / "boot_neg"
    res_neg = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_path),
            "--model",
            "mock",
            "--bootstrap-iters",
            "-1",
            "--output",
            str(out_neg),
        ],
    )
    neg_out = res_neg.stdout + getattr(res_neg, "stderr", "") + (
        f" {res_neg.exception}" if res_neg.exception else ""
    )
    assert res_neg.exit_code != 0
    assert "no such option" not in neg_out.lower()
    assert "stub:" not in neg_out.lower()
    assert any(
        kw in neg_out.lower()
        for kw in ("bootstrap", "negative", "non-negative", "invalid", ">= 0")
    )
    assert not (out_neg / "manifest.json").exists()


def test_t2_macro_stderr_unbalanced_subtasks_and_heterogeneous_keys(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-CORE-04: Macro vs micro stderr pairing and heterogeneous metric-key guard."""
    large = SubtaskStats(name="large_subtask", score=0.80, stderr=0.01, n_samples=1000)
    small = SubtaskStats(name="small_subtask", score=0.40, stderr=0.10, n_samples=10)

    micro = aggregate_group([large, small], weight_by_size=True)
    macro = aggregate_group([large, small], weight_by_size=False)

    assert micro.stderr_method == "pooled_sample_stderr"
    assert macro.stderr_method == "unweighted_mean_stderr"
    expected_macro_se = math.sqrt(0.01**2 + 0.10**2) / 2.0
    assert macro.score == pytest.approx(0.60, abs=1e-9)
    assert macro.stderr == pytest.approx(expected_macro_se, abs=1e-9)
    assert macro.stderr > 3.0 * micro.stderr

    # Heterogeneous metric keys across tasks must never be silently dropped in group runs
    task_a = _write_csv_task(
        tmp_path,
        name="subtask_em",
        rows=[("a1", "Q1?", "A"), ("a2", "Q2?", "A")],
        metrics=["exact_match"],
    )
    task_b = _write_csv_task(
        tmp_path,
        name="subtask_ww",
        rows=[("b1", "Q3?", "A"), ("b2", "Q4?", "B")],
        metrics=["whole_word_match"],
    )
    group_yaml = tmp_path / "hetero_group.yaml"
    group_yaml.write_text(
        "name: hetero_group\n"
        "trust_tier: tier2_declarative\n"
        "output_type: generate_until\n"
        "subtasks:\n"
        f"  - {task_a}\n"
        f"  - {task_b}\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "hetero_out"
    res = cli_runner.invoke(
        app,
        ["run", "--task", str(group_yaml), "--model", "mock", "--output", str(out_dir)],
    )
    if res.exit_code == 0:
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = manifest["metrics"]
        assert any("exact_match" in k for k in metrics)
        assert any("whole_word_match" in k for k in metrics)
    else:
        combined = res.stdout + getattr(res, "stderr", "")
        assert "no such option" not in combined.lower()
        assert "stub:" not in combined.lower()
        assert "metric" in combined.lower() or "heterogeneous" in combined.lower()


def test_t2_regex_prefix_shadowing_and_unicode_casefold_boundary(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-CORE-05: Longest-first regex alternation and Unicode casefold without truncation."""
    filt = MultiChoiceRegexFilter(["Guilty", "Guilty of Romance"])
    assert filt.apply(["Answer: Guilty of Romance"]) == ["Guilty of Romance"]

    assert exact_match("İstanbul", "i\u0307stanbul") == 1.0
    assert exact_match("Straße", "STRASSE") == 1.0
    long_sharp_s = "ß" * 200
    assert exact_match(long_sharp_s, "SS" * 200) == 1.0

    # End-to-end check via Tier 1 Unicode dataset + mock reply
    task_path = _write_csv_task(
        tmp_path,
        name="unicode_task",
        rows=[("u1", "Capital?", "Straße")],
    )
    out_dir = tmp_path / "unicode_out"
    res = cli_runner.invoke(
        app,
        ["run", "--task", str(task_path), "--model", "mock", "--output", str(out_dir)],
    )
    assert res.exit_code == 0, res.stdout
    samples = (out_dir / "samples.jsonl").read_text(encoding="utf-8")
    assert "Straße" in samples


def test_t2_exact_large_integers_and_unseeded_checker_determinism(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-CORE-06: Large ints/Decimals avoid float64 loss; scoring is immune to global random."""
    assert numeric_exact_match("9007199254740992", "9007199254740993") == 0.0
    assert numeric_exact_match("9007199254740993", "9007199254740993") == 1.0
    d1 = Decimal("1.00000000000000001")
    d2 = Decimal("1.00000000000000002")
    assert numeric_exact_match(str(d1), str(d2)) == 0.0

    task_path = _write_csv_task(
        tmp_path,
        name="large_int_task",
        rows=[("li1", "Exact patient cohort ID?", "9007199254740993")],
        metrics=["numeric_exact_match"],
    )
    verdicts: list[float] = []
    for seed_val in range(5):
        random.seed(seed_val * 97 + 1)
        out_dir = tmp_path / f"run_{seed_val}"
        res = cli_runner.invoke(
            app,
            ["run", "--task", str(task_path), "--model", "mock", "--output", str(out_dir)],
        )
        assert res.exit_code == 0, res.stdout
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        verdicts.append(manifest["metrics"]["numeric_exact_match"]["value"])
    assert len(set(verdicts)) == 1


def test_t2_do_sample_bypasses_cache_and_plugin_no_shadow(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T2-CORE-07: do_sample=True bypasses cache; external plugins never shadow built-ins."""
    dummy = object()
    models_reg = get_registry("eval_studio.models")
    metrics_reg = get_registry("eval_studio.metrics")
    models_reg.register("mock", dummy, builtin=False)
    metrics_reg.register("exact_match", dummy, builtin=False)
    assert models_reg.get("mock") is MockGateway
    assert metrics_reg.get("exact_match") is exact_match

    monkeypatch.delenv("EVAL_STUDIO_CACHE_DIR", raising=False)
    default_cache = resolve_cache_dir()
    assert "site-packages" not in str(default_cache).lower()

    cache_dir = tmp_path / "dosample_cache"
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-dosample-1234567890")

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"resp-{len(recorded)}",
                "choices": [{"message": {"content": "Pancreas"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    _bind_mock_transport(monkeypatch, transport)

    task_path = _write_csv_task(
        tmp_path,
        name="dosample_task",
        rows=[("d1", "Which organ makes insulin?", "Pancreas")],
        extra_yaml="generation_kwargs:\n  do_sample: true\n  temperature: 0.7\n",
    )

    for idx in (1, 2):
        res = cli_runner.invoke(
            app,
            [
                "run",
                "--task",
                str(task_path),
                "--model",
                "http-byok",
                "--output",
                str(tmp_path / f"out_{idx}"),
            ],
        )
        assert res.exit_code == 0, res.stdout

    assert len(recorded) == 2


# ---------------------------------------------------------------------------
# Area 2B: BYOK Gateways, Cost Caps & Secret Hygiene (T2-BYOK-01 .. T2-BYOK-05)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret_literal",
    [
        "sk-proj-1234567890abcdef12345678",
        "sk-ant-api03-abcdef1234567890",
        "sk-or-v1-1234567890abcdef",
        "AIzaSyD1234567890abcdefghij",
        "Bearer tok_1234567890abcdef",
    ],
)
def test_t2_hardcoded_api_keys_in_task_yaml_rejected(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
    secret_literal: str,
) -> None:
    """T2-BYOK-01: Hardcoded API keys or bearer tokens in task YAML are rejected."""
    transport, recorded = make_mock_transport(
        lambda req, rec: httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})
    )
    _bind_mock_transport(monkeypatch, transport)

    bad_yaml = _write_csv_task(
        tmp_path,
        name="secret_leak_task",
        rows=[("s1", "Test?", "A")],
        extra_yaml=f'description: "Do not embed {secret_literal} here"\n',
    )

    res_val = cli_runner.invoke(app, ["validate", "--task", str(bad_yaml)])
    val_out = res_val.stdout + getattr(res_val, "stderr", "")
    assert res_val.exit_code == 1
    assert "no such option" not in val_out.lower()
    assert "stub:" not in val_out.lower()
    assert "secret" in val_out.lower() or "api key" in val_out.lower()

    out_dir = tmp_path / "secret_run_out"
    res_run = cli_runner.invoke(
        app,
        ["run", "--task", str(bad_yaml), "--model", "http-byok", "--output", str(out_dir)],
    )
    run_out = res_run.stdout + getattr(res_run, "stderr", "") + (
        f" {res_run.exception}" if res_run.exception else ""
    )
    assert res_run.exit_code != 0
    assert "no such option" not in run_out.lower()
    assert "stub:" not in run_out.lower()
    assert "secret" in run_out.lower() or "api key" in run_out.lower()
    assert len(recorded) == 0
    assert not (out_dir / "manifest.json").exists()


def test_t2_secret_redaction_in_crash_traceback_logs_wal_and_receipt(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T2-BYOK-02: Secrets are scrubbed from HTTP error tracebacks, WAL entries, and receipts."""
    raw_secret = "sk-live-secret99887766554433"
    monkeypatch.setenv("TARGET_API_KEY", raw_secret)

    def _error_handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": f"Invalid key Authorization: Bearer {raw_secret}"}},
        )

    transport, recorded = make_mock_transport(_error_handler)
    _bind_mock_transport(monkeypatch, transport)

    gw = HttpBYOKGateway(
        "gpt-4o-mini",
        base_url="https://mock.local/v1",
        role=SecretRole.TARGET,
        max_retries=0,
    )
    req = EvalRequest(
        request_id="req-err-1",
        sample_id="s1",
        task_name="redact_test",
        output_type=OutputType.GENERATE_UNTIL,
        context="Question: Test\nAnswer:",
    )
    with pytest.raises(Exception) as exc_info:
        gw.generate_until([req])

    assert not isinstance(exc_info.value, NotImplementedError)
    assert len(recorded) == 1
    err_msg = str(exc_info.value)
    assert raw_secret not in err_msg
    assert contains_secret(err_msg) is False

    wal_path = tmp_path / "wal" / "events.jsonl"
    wal = JsonlWriteAheadLog(wal_path)
    wal.append({"request_id": "req-err-1", "error": f"Bearer {raw_secret}", "status": "error"})
    wal_text = wal_path.read_text(encoding="utf-8")
    assert raw_secret not in wal_text
    assert REDACTED_PLACEHOLDER in wal_text
    assert contains_secret(wal_text) is False

    # Verify Eval Receipt + cache WAL emitted by CLI run redact TARGET_API_KEY completely
    cache_dir = tmp_path / "receipt_cache"
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(cache_dir))
    ok_transport, _ = make_mock_transport(
        lambda r, rec: httpx.Response(
            200,
            json={
                "id": "resp-ok",
                "choices": [{"message": {"content": "Pancreas"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2},
            },
        )
    )
    _bind_mock_transport(monkeypatch, ok_transport)

    task_path = _write_csv_task(
        tmp_path,
        name="redact_receipt_task",
        rows=[("r1", "Which organ produces insulin?", "Pancreas")],
    )
    receipt_dir = tmp_path / "redact_receipt_out"
    res_run = cli_runner.invoke(
        app,
        ["run", "--task", str(task_path), "--model", "http-byok", "--output", str(receipt_dir)],
    )
    assert res_run.exit_code == 0, res_run.stdout
    for fname in ("samples.jsonl", "manifest.json", "receipt.json", "report.md", "report.html"):
        content = (receipt_dir / fname).read_text(encoding="utf-8")
        assert raw_secret not in content
        assert contains_secret(content) is False
    for wal_file in cache_dir.rglob("*.jsonl"):
        w_text = wal_file.read_text(encoding="utf-8")
        assert raw_secret not in w_text
        assert contains_secret(w_text) is False


def test_t2_max_cost_usd_midrun_halt_wal_checkpoint_and_resume(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T2-BYOK-03: --max-cost-usd halts mid-run, preserves WAL, and resumes with 0 duplicates."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-costcap-1234567890")
    cache_dir = tmp_path / "byok03_cache"
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(cache_dir))

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"resp-{len(recorded)}",
                "choices": [{"message": {"content": "Pancreas"}}],
                "usage": {"prompt_tokens": 5000, "completion_tokens": 500},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    _bind_mock_transport(monkeypatch, transport)

    # Configured pricing ($1.00 / 1M input + $10.00 / 1M output -> $0.005 + $0.005 = $0.01/sample)
    task_path = _write_csv_task(
        tmp_path,
        name="cost_cap_resume",
        rows=[(f"s{i}", f"Question {i}?", "Pancreas") for i in range(1, 7)],
        extra_yaml=(
            "generation_kwargs:\n"
            "  max_gen_toks: 500\n"
            "metadata:\n"
            "  pricing:\n"
            "    input_usd_per_1m: 1.0\n"
            "    output_usd_per_1m: 10.0\n"
        ),
    )

    out1 = tmp_path / "run1_halted"
    res1 = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_path),
            "--model",
            "http-byok",
            "--max-cost-usd",
            "0.035",
            "--output",
            str(out1),
        ],
    )
    out1_text = res1.stdout + getattr(res1, "stderr", "") + (
        f" {res1.exception}" if res1.exception else ""
    )
    assert res1.exit_code != 0
    assert "no such option" not in out1_text.lower()
    assert "stub:" not in out1_text.lower()
    assert "cost" in out1_text.lower() or "budget" in out1_text.lower()
    assert 1 <= len(recorded) < 6

    wal_files = list(cache_dir.rglob("*.jsonl"))
    assert wal_files, "Mid-run cost halt must preserve completed samples in WAL"
    wal_entries = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in wal_files
    )
    assert wal_entries >= len(recorded)

    out2 = tmp_path / "run2_resumed"
    res2 = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_path),
            "--model",
            "http-byok",
            "--max-cost-usd",
            "0.10",
            "--output",
            str(out2),
        ],
    )
    assert res2.exit_code == 0, res2.stdout
    assert len(recorded) == 6
    resumed_samples = (out2 / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(resumed_samples) == 6


def test_t2_zero_and_negative_max_cost_usd_boundaries(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T2-BYOK-04: --max-cost-usd 0.0 blocks non-zero batch; negative fails validation."""
    with pytest.raises(ValidationError):
        EvalRunConfig(limits=RuntimeLimits(max_cost_usd=-0.01))

    monkeypatch.setenv("TARGET_API_KEY", "sk-target-zerocap-1234567890")
    transport, recorded = make_mock_transport(
        lambda req, rec: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "A"}}],
                "usage": {"prompt_tokens": 5000, "completion_tokens": 500},
            },
        )
    )
    _bind_mock_transport(monkeypatch, transport)

    task_path = _write_csv_task(
        tmp_path,
        name="zero_cap_task",
        rows=[("z1", "Question?", "A")],
        extra_yaml=(
            "generation_kwargs:\n"
            "  max_gen_toks: 500\n"
            "metadata:\n"
            "  pricing:\n"
            "    input_usd_per_1m: 1.0\n"
            "    output_usd_per_1m: 10.0\n"
        ),
    )

    out_zero = tmp_path / "zero_out"
    res_zero = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_path),
            "--model",
            "http-byok",
            "--max-cost-usd",
            "0.0",
            "--output",
            str(out_zero),
        ],
    )
    zero_text = res_zero.stdout + getattr(res_zero, "stderr", "") + (
        f" {res_zero.exception}" if res_zero.exception else ""
    )
    assert res_zero.exit_code != 0
    assert "no such option" not in zero_text.lower()
    assert "stub:" not in zero_text.lower()
    assert "cost" in zero_text.lower() or "budget" in zero_text.lower()
    assert len(recorded) == 0

    out_neg = tmp_path / "neg_out"
    res_neg = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_path),
            "--model",
            "http-byok",
            "--max-cost-usd",
            "-0.01",
            "--output",
            str(out_neg),
        ],
    )
    neg_text = res_neg.stdout + getattr(res_neg, "stderr", "") + (
        f" {res_neg.exception}" if res_neg.exception else ""
    )
    assert res_neg.exit_code != 0
    assert "no such option" not in neg_text.lower()
    assert "stub:" not in neg_text.lower()
    assert any(
        kw in neg_text.lower()
        for kw in ("cost", "budget", "negative", "invalid", ">= 0", "validation")
    )
    assert len(recorded) == 0


def test_t2_http_gateway_retries_exhausted_and_role_key_isolation(
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
) -> None:
    """T2-BYOK-05: Exhausted 503 retries sanitize keys; role keys remain strictly isolated."""
    secret = "sk-target-retrysecret123456"
    monkeypatch.setenv("TARGET_API_KEY", secret)

    assert load_api_key(SecretRole.TARGET) == secret
    assert load_api_key(SecretRole.JUDGE) is None
    assert load_api_key(SecretRole.EXTRACTOR) is None

    transport, recorded = make_mock_transport(
        lambda req, rec: httpx.Response(503, json={"error": f"Unavailable for {secret}"})
    )
    _bind_mock_transport(monkeypatch, transport)

    gw = HttpBYOKGateway(
        "gpt-4o-mini",
        base_url="https://mock.local/v1",
        role=SecretRole.TARGET,
        max_retries=2,
    )
    req = EvalRequest(
        request_id="req-503",
        sample_id="s1",
        task_name="retry_test",
        output_type=OutputType.GENERATE_UNTIL,
        context="Question: Test\nAnswer:",
    )
    with pytest.raises(Exception) as exc_info:
        gw.generate_until([req])

    assert len(recorded) == 3
    assert secret not in str(exc_info.value)
    assert contains_secret(str(exc_info.value)) is False
