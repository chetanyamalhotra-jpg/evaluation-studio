"""Tier 2 E2E tests: Boundary, security, guardrail, tamper, and path-safety edge cases."""

from __future__ import annotations

import builtins
import importlib
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

pytestmark = pytest.mark.e2e_web


# ---------------------------------------------------------------------------
# B1: Secret Leakage & Redaction on HTTP Errors & `/api/byok/*` (4 Tests)
# ---------------------------------------------------------------------------


def test_t2_sec_01_byok_save_and_status_never_return_raw_keys_for_any_prefix(
    client: Any,
) -> None:
    """T2-SEC-01: All 4 secret prefixes are masked and never echoed in save/status responses."""
    cases = [
        ("TARGET_API_KEY", "sk-or-v1-abcdefgh123456", "sk-...3456"),
        ("JUDGE_API_KEY", "sk-ant-abcdefgh123456", "sk-...3456"),
        ("EXTRACTOR_API_KEY", "sk_sarvam_abcdefgh123456", "sk_...3456"),
        ("FALLBACK_JUDGE_API_KEY", "AIzaSyAbcdefgh1234567890", "AIz...7890"),
    ]
    for role, raw_key, expected_hint in cases:
        resp = client.post(
            "/api/byok/keys",
            json={"role": role, "api_key": raw_key, "storage": "session"},
        )
        assert resp.status_code == 200
        assert raw_key not in resp.text
        assert resp.json()["masked_hint"] == expected_hint

    status = client.get("/api/byok/status")
    assert status.status_code == 200
    for _, raw_key, _ in cases:
        assert raw_key not in status.text


def test_t2_sec_02_byok_test_connection_redacts_upstream_error_echoing_secret(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
) -> None:
    """T2-SEC-02: Upstream error bodies echoing bearer or Sarvam keys are redacted."""
    bearer_secret = "sk-secret-leak-99887766"
    sarvam_secret = "sk_sarvam_leak_112233"
    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": bearer_secret, "storage": "session"},
    )

    def _failing_handler(
        request: httpx.Request, recorded: list[dict[str, Any]]
    ) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": (
                    f"Unauthorized Bearer {bearer_secret} and "
                    f"api-subscription-key: {sarvam_secret}"
                )
            },
        )

    transport, _ = make_mock_transport(_failing_handler)
    bind_mock_transport(transport)

    resp = client.post(
        "/api/byok/test",
        json={
            "backend": "http-byok",
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "role": "TARGET_API_KEY",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert bearer_secret not in resp.text
    assert sarvam_secret not in resp.text
    assert "[REDACTED_SECRET]" in (body.get("error") or "")


def test_t2_sec_03_global_exception_handler_redacts_secrets_on_unhandled_500(
    web_app: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2-SEC-03: Unhandled server exceptions pass through `redact_secrets()` before HTTP 500."""
    testclient_mod = pytest.importorskip("fastapi.testclient")
    leaked_token = "sk-unhandled-500-secret-abcdef"

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"Unexpected upstream failure using token {leaked_token}")

    monkeypatch.setattr("eval_studio.tasks.validator.validate_task_spec", _boom)
    routes_byot = sys.modules.get("eval_studio.web.routes_byot")
    if routes_byot is None:
        try:
            routes_byot = importlib.import_module("eval_studio.web.routes_byot")
        except ImportError:
            routes_byot = None
    if routes_byot is not None and hasattr(routes_byot, "validate_task_spec"):
        monkeypatch.setattr(routes_byot, "validate_task_spec", _boom)

    with testclient_mod.TestClient(web_app, raise_server_exceptions=False) as tc:
        resp = tc.post(
            "/api/byot/validate",
            json={
                "yaml_text": (
                    "name: boom\ntrust_tier: tier1_zero_code\n"
                    "output_type: generate_until\n"
                    "doc_to_text: '{{ input_text }}'\ndoc_to_target: '{{ target }}'\n"
                )
            },
        )
    assert resp.status_code == 500
    assert leaked_token not in resp.text
    assert "[REDACTED_SECRET]" in resp.json()["detail"]


def test_t2_sec_04_byot_save_and_validate_reject_embedded_secrets_in_yaml(
    client: Any,
    isolated_env: Any,
) -> None:
    """T2-SEC-04: Task YAMLs containing API keys fail validation and are blocked from save."""
    evil_yaml = yaml.safe_dump(
        {
            "name": "leaked_yaml_task",
            "version": "1.0.0",
            "trust_tier": "tier1_zero_code",
            "output_type": "generate_until",
            "description": "Use key sk-ant-embedded-secret-123456",
            "doc_to_text": "Q: {{ input_text }}",
            "doc_to_target": "{{ target }}",
            "metrics": ["exact_match"],
            "metadata": {"api_key": "AIzaSyEmbeddedSecret123456789"},
        },
        sort_keys=False,
    )

    val = client.post("/api/byot/validate", json={"yaml_text": evil_yaml})
    assert val.status_code == 200
    val_body = val.json()
    assert val_body["valid"] is False
    assert val_body["secret_scan_clean"] is False

    save = client.post(
        "/api/byot/save",
        json={"filename": "leaked_yaml.yaml", "yaml_text": evil_yaml},
    )
    assert save.status_code in (400, 422)
    assert "sk-ant-embedded-secret-123456" not in save.text
    assert "AIzaSyEmbeddedSecret123456789" not in save.text
    assert not (isolated_env.tasks_dir / "leaked_yaml.yaml").exists()


# ---------------------------------------------------------------------------
# B2: SSTI & Untrusted YAML Rejection in BYOT (3 Tests)
# ---------------------------------------------------------------------------


def test_t2_ssti_01_byot_validate_blocks_jinja2_dunder_ssti(client: Any) -> None:
    """T2-SSTI-01: Jinja2 dunder and `cycler.__init__.__globals__` SSTI payloads are blocked."""
    ssti_yaml = yaml.safe_dump(
        {
            "name": "ssti_attempt",
            "version": "1.0.0",
            "trust_tier": "tier2_declarative",
            "output_type": "generate_until",
            "doc_to_text": "{{ ''.__class__.__mro__[1].__subclasses__() }}",
            "doc_to_target": "{{ cycler.__init__.__globals__.os }}",
            "metrics": ["exact_match"],
            "metadata": {"date_stamp": "2026-11-23"},
        },
        sort_keys=False,
    )
    resp = client.post("/api/byot/validate", json={"yaml_text": ssti_yaml})
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert len(body["errors"]) >= 1
    assert body["preview_prompt"] is None


def test_t2_ssti_02_byot_save_blocks_ssti_and_yaml_function_tags(
    client: Any,
    isolated_env: Any,
) -> None:
    """T2-SSTI-02: `!function` YAML tags and `eval()` expressions are rejected on save."""
    func_yaml = (
        "name: unsafe_tag\n"
        "version: '1.0.0'\n"
        "trust_tier: tier2_declarative\n"
        "output_type: generate_until\n"
        "doc_to_text: !function os:system\n"
        "doc_to_target: '{{ target }}'\n"
        "metrics:\n  - exact_match\n"
    )
    r1 = client.post(
        "/api/byot/save",
        json={"filename": "unsafe_tag.yaml", "yaml_text": func_yaml},
    )
    assert r1.status_code in (400, 422)
    assert not (isolated_env.tasks_dir / "unsafe_tag.yaml").exists()

    eval_yaml = yaml.safe_dump(
        {
            "name": "unsafe_eval",
            "version": "1.0.0",
            "trust_tier": "tier2_declarative",
            "output_type": "generate_until",
            "doc_to_text": "{{ eval('1+1') }}",
            "doc_to_target": "{{ target }}",
            "metrics": ["exact_match"],
        },
        sort_keys=False,
    )
    r2 = client.post(
        "/api/byot/save",
        json={"filename": "unsafe_eval.yaml", "yaml_text": eval_yaml},
    )
    assert r2.status_code in (400, 422)
    assert not (isolated_env.tasks_dir / "unsafe_eval.yaml").exists()


def test_t2_ssti_03_byot_save_and_run_reject_tier3_without_trust_remote_code(
    client: Any,
    isolated_env: Any,
) -> None:
    """T2-SSTI-03: Tier 3 tasks cannot be saved or run without `trust_remote_code=True`."""
    t3_yaml = yaml.safe_dump(
        {
            "name": "tier3_plugin_test",
            "version": "1.0.0",
            "trust_tier": "tier3_code_plugin",
            "output_type": "generate_until",
            "doc_to_text": "{{ input_text }}",
            "doc_to_target": "{{ target }}",
            "metrics": ["exact_match"],
            "unsafe_code": True,
        },
        sort_keys=False,
    )
    save = client.post(
        "/api/byot/save",
        json={"filename": "tier3.yaml", "yaml_text": t3_yaml},
    )
    assert save.status_code in (400, 422)
    assert not (isolated_env.tasks_dir / "tier3.yaml").exists()

    t3_file = isolated_env.tasks_dir / "manual_t3.yaml"
    t3_file.write_text(t3_yaml, encoding="utf-8")
    run_resp = client.post(
        "/api/runs",
        json={
            "task_path": str(t3_file),
            "model_backend": "mock",
            "model_id": "mock-deterministic",
            "trust_remote_code": False,
        },
    )
    assert run_resp.status_code in (400, 422)
    detail = run_resp.json()["detail"].lower()
    assert "trust_remote_code" in detail or "trust-remote-code" in detail


# ---------------------------------------------------------------------------
# B3: BOS + Chat Template Conflict & `enable_thinking` on Loglikelihood (3 Tests)
# ---------------------------------------------------------------------------


def test_t2_fmt_01_validate_rejects_apply_chat_template_with_add_bos_token_true(
    client: Any,
) -> None:
    """T2-FMT-01: Combining `apply_chat_template=True` and `add_bos_token=True` fails."""
    task_yaml = yaml.safe_dump(
        {
            "name": "bos_conflict",
            "version": "1.0.0",
            "trust_tier": "tier1_zero_code",
            "output_type": "generate_until",
            "dataset_path": "examples/tier1_zero_code/sample.csv",
            "doc_to_text": "{{ input_text }}",
            "doc_to_target": "{{ target }}",
            "metrics": ["exact_match"],
            "metadata": {"date_stamp": "2026-11-23"},
        },
        sort_keys=False,
    )
    resp = client.post(
        "/api/byot/validate",
        json={
            "yaml_text": task_yaml,
            "apply_chat_template": True,
            "add_bos_token": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    joined = " ".join(body["errors"]).lower()
    assert "bos" in joined or "apply_chat_template" in joined


def test_t2_fmt_02_validate_and_run_reject_enable_thinking_on_mc_and_loglikelihood(
    client: Any,
    isolated_env: Any,
) -> None:
    """T2-FMT-02: `enable_thinking=True` is rejected on `multiple_choice` and `loglikelihood`."""
    for otype in ("multiple_choice", "loglikelihood"):
        task_dict = {
            "name": f"think_conflict_{otype}",
            "version": "1.0.0",
            "trust_tier": "tier2_declarative",
            "output_type": otype,
            "doc_to_text": "{{ input_text }}",
            "doc_to_target": "{{ target }}",
            "metrics": ["acc"],
            "metadata": {"date_stamp": "2026-11-23"},
        }
        yaml_text = yaml.safe_dump(task_dict, sort_keys=False)
        val = client.post(
            "/api/byot/validate",
            json={"yaml_text": yaml_text, "enable_thinking": True},
        )
        assert val.status_code == 200
        assert val.json()["valid"] is False
        assert "enable_thinking" in " ".join(val.json()["errors"])

    run_resp = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier2_declarative/clinical_mcqa.yaml",
            "model_backend": "mock",
            "model_id": "mock-deterministic",
            "enable_thinking": True,
            "think_end_token": "</think>",
        },
    )
    assert run_resp.status_code in (400, 422)


def test_t2_fmt_03_run_rejects_enable_thinking_without_think_end_token(
    client: Any,
) -> None:
    """T2-FMT-03: Launching a run with `enable_thinking=True` and null `think_end_token` fails."""
    resp = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "mock",
            "model_id": "mock-deterministic",
            "enable_thinking": True,
            "think_end_token": None,
        },
    )
    assert resp.status_code in (400, 422)
    assert "think_end_token" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# B4: Missing `[pdf]` Extra & Corrupt Paper Uploads (3 Tests)
# ---------------------------------------------------------------------------


def test_t2_pdf_01_byop_upload_pdf_without_pdf_extra_returns_422_with_install_hint(
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2-PDF-01: Uploading `.pdf` without `pypdf` returns 422 with install hint."""
    monkeypatch.setitem(sys.modules, "pypdf", None)
    real_import = builtins.__import__

    def _block_pypdf(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError("No module named 'pypdf'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_pypdf)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["pdf_available"] is False

    resp = client.post(
        "/api/byop/upload",
        files={"file": ("paper.pdf", io.BytesIO(b"%PDF-1.4\n%%EOF\n"), "application/pdf")},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"].lower()
    assert "pdf" in detail or "pypdf" in detail


def test_t2_pdf_02_byop_upload_rejects_corrupt_binary_and_empty_paper(client: Any) -> None:
    """T2-PDF-02: NUL-byte corrupt Markdown and empty `.txt` uploads are rejected cleanly."""
    r_bin = client.post(
        "/api/byop/upload",
        files={"file": ("corrupt.md", io.BytesIO(b"\x00\xff\xfe"), "text/markdown")},
    )
    assert r_bin.status_code in (400, 422)

    r_empty = client.post(
        "/api/byop/upload",
        files={"file": ("empty.txt", io.BytesIO(b"   \n"), "text/plain")},
    )
    assert r_empty.status_code in (400, 422)


def test_t2_pdf_03_byop_compile_all_rejected_and_empty_lock_rejected(
    client: Any,
    isolated_env: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
) -> None:
    """T2-PDF-03: All-rejected compile leaves `approved_items=[]`; locking empty list fails."""
    paper_path = isolated_env.examples_dir / "papers" / "sample_paper.md"
    client.post(
        "/api/byok/keys",
        json={"role": "EXTRACTOR_API_KEY", "api_key": "sk-ext-empty-1234", "storage": "session"},
    )
    client.post(
        "/api/byok/keys",
        json={"role": "JUDGE_API_KEY", "api_key": "sk-ver-empty-1234", "storage": "session"},
    )

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        if len(recorded) == 1:
            content = json.dumps(
                {
                    "items": [
                        {
                            "item_id": "bad-1",
                            "question": "As shown in Table 99, what was the result?",
                            "reference_answer": "10%",
                            "rubric": "Unverifiable",
                            "category": "limitation",
                            "citations": [],
                        }
                    ]
                }
            )
        else:
            content = json.dumps({"approved": False, "reasons": ["Missing citation"]})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    transport, _ = make_mock_transport(_handler)
    bind_mock_transport(transport)

    comp = client.post(
        "/api/byop/compile",
        json={
            "paper_path": str(paper_path),
            "suite_name": "all_rejected",
            "mode": "http-byok",
            "provider": "openai",
            "extractor_model": "gpt-4o-mini",
            "verifier_model": "gpt-4o-mini",
        },
    )
    assert comp.status_code == 200
    c_body = comp.json()
    assert c_body["approved_items"] == []
    assert len(c_body["rejected_items"]) >= 1

    lock = client.post(
        "/api/byop/lock",
        json={
            "paper_path": str(paper_path),
            "suite_name": "all_rejected",
            "extractor_model": "gpt-4o-mini",
            "verifier_model": "gpt-4o-mini",
            "items": [],
        },
    )
    assert lock.status_code in (400, 422)


# ---------------------------------------------------------------------------
# B5: Cohen's Kappa `< 0.40` `UNALIGNED` Calibration (3 Tests)
# ---------------------------------------------------------------------------


def test_t2_cal_01_calibrate_kappa_below_threshold_returns_unaligned(client: Any) -> None:
    """T2-CAL-01: Low judge-vs-human agreement (`kappa < 0.40`) returns `UNALIGNED`."""
    rows = [
        {"judge_label": 1, "human_label": 0},
        {"judge_label": 0, "human_label": 1},
        {"judge_label": 1, "human_label": 0},
        {"judge_label": 0, "human_label": 1},
        {"judge_label": 1, "human_label": 1},
        {"judge_label": 0, "human_label": 0},
    ]
    resp = client.post("/api/byot/calibrate", json={"rows": rows})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cohens_kappa"] < 0.40
    assert body["passed_threshold"] is False
    assert body["status"] == "UNALIGNED"


def test_t2_cal_02_calibrate_exact_threshold_boundary_and_custom_threshold(
    client: Any,
) -> None:
    """T2-CAL-02: Exact kappa boundary (`0.50`) passes `threshold=0.50` and fails `0.75`."""
    rows = [
        {"judge_label": 1, "human_label": 1},
        {"judge_label": 1, "human_label": 1},
        {"judge_label": 1, "human_label": 0},
        {"judge_label": 0, "human_label": 0},
    ]
    r_pass = client.post("/api/byot/calibrate", json={"rows": rows, "threshold": 0.50})
    assert r_pass.status_code == 200
    assert r_pass.json()["cohens_kappa"] == pytest.approx(0.50)
    assert r_pass.json()["passed_threshold"] is True
    assert r_pass.json()["status"] == "ALIGNED"

    r_fail = client.post("/api/byot/calibrate", json={"rows": rows, "threshold": 0.75})
    assert r_fail.status_code == 200
    assert r_fail.json()["passed_threshold"] is False
    assert r_fail.json()["status"] == "UNALIGNED"


def test_t2_cal_03_calibrate_empty_or_malformed_jsonl_returns_400_or_422(
    client: Any,
) -> None:
    """T2-CAL-03: Empty or missing-field calibration payloads return 400/422 cleanly."""
    r_empty = client.post("/api/byot/calibrate", json={"jsonl_text": ""})
    assert r_empty.status_code in (400, 422)

    r_bad = client.post(
        "/api/byot/calibrate",
        json={"jsonl_text": '{"judge_label": 1}\n'},
    )
    assert r_bad.status_code in (400, 422)


# ---------------------------------------------------------------------------
# B6: `--max-cost-usd` Budget Cap & SSE `budget_exceeded` (3 Tests)
# ---------------------------------------------------------------------------


def test_t2_cost_01_estimate_flags_exceeds_max_cost_true_when_cap_too_low(
    client: Any,
) -> None:
    """T2-COST-01: Pre-flight estimate flags `exceeds_max_cost=True` when above cap."""
    resp = client.post(
        "/api/runs/estimate",
        json={
            "task_path": "examples/tier1_zero_code/live_smoke.yaml",
            "model_backend": "mock",
            "model_id": "mock-deterministic",
            "max_cost_usd": 0.000001,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_usd"] > 0.000001
    assert body["exceeds_max_cost"] is True


def test_t2_cost_02_run_halts_on_max_cost_usd_and_emits_sse_budget_exceeded(
    client: Any,
    isolated_env: Any,
    parse_sse_frames: Callable[[str], list[dict[str, Any]]],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T2-COST-02: Hard `--max-cost-usd` cap halts run and emits SSE `budget_exceeded`."""
    csv_path = isolated_env.tasks_dir / "expensive.csv"
    rows = ["sample_id,input_text,target"] + [
        f"e{i},Clinical question number {i} with extra prompt tokens?,A" for i in range(1, 6)
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    task_path = isolated_env.tasks_dir / "expensive.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "name": "expensive_budget_test",
                "version": "1.0.0",
                "trust_tier": "tier1_zero_code",
                "output_type": "generate_until",
                "dataset_path": str(csv_path),
                "doc_to_text": "Question: {{ input_text }}\nAnswer:",
                "doc_to_target": "{{ target }}",
                "metrics": ["exact_match"],
                "metadata": {
                    "date_stamp": "2026-11-23",
                    "pricing": {"input_usd_per_1m": 1000.0, "output_usd_per_1m": 1000.0},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    launch = client.post(
        "/api/runs",
        json={
            "task_path": str(task_path),
            "model_backend": "mock",
            "model_id": "mock-deterministic",
            "max_cost_usd": 0.001,
        },
    )
    assert launch.status_code in (200, 202)
    run_id = launch.json()["run_id"]

    chunks: list[str] = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream_resp:
        assert stream_resp.status_code == 200
        for chunk in stream_resp.iter_text():
            chunks.append(chunk)

    frames = parse_sse_frames("".join(chunks))
    events = [f["event"] for f in frames]
    assert "budget_exceeded" in events or "error" in events

    status = poll_run_until_done(client, run_id, 15.0)
    assert status["status"] != "completed"
    assert status["spent_usd"] <= 0.001 + 1e-9


def test_t2_cost_03_run_rejects_negative_max_cost_usd(client: Any) -> None:
    """T2-COST-03: Negative `max_cost_usd` is rejected with HTTP 422."""
    for endpoint in ("/api/runs/estimate", "/api/runs"):
        resp = client.post(
            endpoint,
            json={
                "task_path": "examples/tier1_zero_code/task.yaml",
                "model_backend": "mock",
                "model_id": "mock-deterministic",
                "max_cost_usd": -1.0,
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# B7: `verify-receipt` Tamper Detection (3 Tests)
# ---------------------------------------------------------------------------


def test_t2_tamper_01_verify_detects_modified_samples_jsonl(
    client: Any,
    copied_live_receipts: dict[str, Path],
) -> None:
    """T2-TAMPER-01: Mutating `samples.jsonl` invalidates `samples_sha256` verification."""
    samples_path = copied_live_receipts["live_ds_vs_glm"] / "samples.jsonl"
    samples_path.write_bytes(samples_path.read_bytes() + b" ")

    resp = client.post("/api/receipts/live_ds_vs_glm/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert any("samples sha-256 mismatch" in e.lower() for e in body["errors"])


def test_t2_tamper_02_verify_detects_modified_manifest_score_and_receipt_mismatch(
    client: Any,
    copied_live_receipts: dict[str, Path],
) -> None:
    """T2-TAMPER-02: Editing a score in `manifest.json` triggers manifest & receipt errors."""
    manifest_path = copied_live_receipts["live_ds_vs_glm"] / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["metrics"]["exact_match"]["value"] = 0.99
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    resp = client.post("/api/receipts/live_ds_vs_glm/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    joined = " ".join(body["errors"]).lower()
    assert "manifest sha-256 mismatch" in joined
    assert "receipt.json differs" in joined


def test_t2_tamper_03_verify_detects_secret_leak_in_receipt_artifacts(
    client: Any,
    copied_live_receipts: dict[str, Path],
) -> None:
    """T2-TAMPER-03: Secret token injected into `report.md` fails receipt verification."""
    report_path = copied_live_receipts["live_ds_vs_glm"] / "report.md"
    report_path.write_text(
        report_path.read_text(encoding="utf-8") + "\nLeaked: sk-leak-inside-report-1234567890\n",
        encoding="utf-8",
    )

    resp = client.post("/api/receipts/live_ds_vs_glm/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert any("secret leak" in e.lower() for e in body["errors"])


# ---------------------------------------------------------------------------
# B8: Invalid Receipt IDs & Non-Binary McNemar & Mismatched Counts (3 Tests)
# ---------------------------------------------------------------------------


def test_t2_rcp_01_missing_receipt_id_returns_404_across_all_receipt_endpoints(
    client: Any,
) -> None:
    """T2-RCP-01: Unknown `receipt_id` returns 404 JSON on detail, samples, verify, rescore."""
    rid = "nonexistent_run_999"
    for method, url, kwargs in (
        ("GET", f"/api/receipts/{rid}", {}),
        ("GET", f"/api/receipts/{rid}/samples", {}),
        ("POST", f"/api/receipts/{rid}/verify", {}),
        ("POST", f"/api/receipts/{rid}/rescore", {"json": {"metrics": ["exact_match"]}}),
    ):
        resp = client.request(method, url, **kwargs)
        assert resp.status_code == 404, f"{method} {url} -> {resp.status_code}"
        assert "detail" in resp.json()


def test_t2_rcp_02_compare_non_binary_scores_handles_mcnemar_gracefully(
    client: Any,
    isolated_env: Any,
) -> None:
    """T2-RCP-02: Comparing fractional `token_f1` receipts avoids McNemar 500 crash."""
    r1 = client.post(
        "/api/receipts/live_ds_vs_glm/rescore",
        json={"metrics": ["token_f1"], "output_name": "f1_run_a"},
    )
    r2 = client.post(
        "/api/receipts/live_ds_vs_glm_rescored/rescore",
        json={"metrics": ["token_f1"], "output_name": "f1_run_b"},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    id1 = Path(r1.json()["path"]).name
    id2 = Path(r2.json()["path"]).name

    cmp_resp = client.post(
        "/api/receipts/compare",
        json={"receipt_ids": [id1, id2], "baseline_id": id1, "bootstrap_iters": 100},
    )
    assert cmp_resp.status_code == 200
    body = cmp_resp.json()
    assert len(body["pairwise"]) >= 1
    for pair in body["pairwise"]:
        assert "paired_bootstrap" in pair
        assert pair.get("mcnemar") is None


def test_t2_rcp_03_noise_floor_and_compare_mismatched_sample_counts_return_400_or_422(
    client: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T2-RCP-03: Comparing a 3-sample receipt with a 2-sample receipt returns 400/422."""
    launch = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "mock",
            "model_id": "mock-two",
            "limit": 2,
        },
    )
    assert launch.status_code in (200, 202)
    two_done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    two_id = Path(two_done["receipt_dir"]).name

    nf = client.post(
        "/api/receipts/noise-floor",
        json={"baseline_id": "live_ds_vs_glm", "perturbed_id": two_id},
    )
    assert nf.status_code in (400, 422)

    cmp_resp = client.post(
        "/api/receipts/compare",
        json={"receipt_ids": ["live_ds_vs_glm", two_id], "baseline_id": "live_ds_vs_glm"},
    )
    assert cmp_resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# B9: Path Traversal & Unsafe Filename Rejection (3 Tests)
# ---------------------------------------------------------------------------


def test_t2_path_01_byot_save_and_detail_reject_path_traversal(
    client: Any,
    isolated_env: Any,
    repo_root: Path,
) -> None:
    """T2-PATH-01: Path traversal on BYOT save and task detail never escapes sandbox."""
    save = client.post(
        "/api/byot/save",
        json={
            "filename": "../../evil.yaml",
            "yaml_text": "name: evil\noutput_type: generate_until\n",
        },
    )
    assert save.status_code in (400, 403, 404, 422)
    assert not (repo_root / "evil.yaml").exists()
    assert not (isolated_env.workspace_dir.parent / "evil.yaml").exists()
    assert not (isolated_env.workspace_dir / "evil.yaml").exists()

    detail = client.get("/api/byot/tasks/detail", params={"path": "../../pyproject.toml"})
    assert detail.status_code in (400, 403, 404, 422)


def test_t2_path_02_byop_upload_and_compile_reject_path_traversal(
    client: Any,
    isolated_env: Any,
) -> None:
    """T2-PATH-02: BYOP upload/compile sanitizes filenames and blocks parent-dir traversal."""
    up = client.post(
        "/api/byop/upload",
        files={"file": ("../outside.md", io.BytesIO(b"# Safe Title\nBody text."), "text/markdown")},
    )
    assert up.status_code in (200, 400, 403, 422)
    assert not (isolated_env.workspace_dir.parent / "outside.md").exists()
    if up.status_code == 200:
        resolved = Path(up.json()["paper_path"]).resolve()
        assert isolated_env.workspace_dir.resolve() in resolved.parents

    comp = client.post(
        "/api/byop/compile",
        json={
            "paper_path": "../../README.md",
            "suite_name": "escape_suite",
            "mode": "heuristic",
        },
    )
    assert comp.status_code in (400, 403, 404, 422)


def test_t2_path_03_receipt_endpoints_reject_path_traversal_id(client: Any) -> None:
    """T2-PATH-03: Traversal receipt IDs (`..%2F..%2Fsrc`) are rejected on GET and POST."""
    r_get = client.get("/api/receipts/..%2F..%2Fsrc")
    assert r_get.status_code in (400, 403, 404, 422)

    r_post = client.post("/api/receipts/..%2F..%2Fsrc/verify")
    assert r_post.status_code in (400, 403, 404, 422)
