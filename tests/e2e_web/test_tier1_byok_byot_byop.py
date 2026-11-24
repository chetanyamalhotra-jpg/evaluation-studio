"""Tier 1 E2E tests: `/api/byok/*`, `/api/byot/*`, and `/api/byop/*` endpoints."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

pytestmark = pytest.mark.e2e_web


# ---------------------------------------------------------------------------
# Area 3: `/api/byok/*` — 7 Tests
# ---------------------------------------------------------------------------


def test_t1_byok_01_presets_include_deepseek_sarvam_and_all_dialects(client: Any) -> None:
    """T1-BYOK-01: `/api/byok/presets` exposes all 8 provider dialects and model pricing."""
    resp = client.get("/api/byok/presets")
    assert resp.status_code == 200
    presets = resp.json()["presets"]
    by_id = {p["id"]: p for p in presets}
    assert {
        "deepseek",
        "sarvam",
        "openai",
        "anthropic",
        "gemini",
        "openrouter",
        "litellm",
        "ollama",
    }.issubset(by_id.keys())

    ds = by_id["deepseek"]
    assert ds["default_base_url"] == "https://api.deepseek.com"
    assert ds["env_var"] == "DEEPSEEK_API_KEY"
    ds_models = {m["id"]: m for m in ds["models"]}
    assert "deepseek-flash" in ds_models
    assert isinstance(ds_models["deepseek-flash"]["supports_thinking"], bool)
    assert isinstance(ds_models["deepseek-flash"]["input_usd_per_1m"], (int, float))
    assert isinstance(ds_models["deepseek-flash"]["output_usd_per_1m"], (int, float))

    sarvam = by_id["sarvam"]
    assert sarvam["default_base_url"] == "https://api.sarvam.ai/v2"
    assert sarvam["env_var"] == "SARVAM_API_KEY"
    sarvam_models = {m["id"]: m for m in sarvam["models"]}
    assert "glm5.3-flash" in sarvam_models


def test_t1_byok_02_status_reports_unset_and_cache_stats(
    client: Any,
    isolated_env: Any,
) -> None:
    """T1-BYOK-02: `/api/byok/status` reports 4 unset roles and isolated cache telemetry."""
    resp = client.get("/api/byok/status")
    assert resp.status_code == 200
    body = resp.json()
    roles = {r["role"]: r for r in body["roles"]}
    assert set(roles.keys()) == {
        "TARGET_API_KEY",
        "JUDGE_API_KEY",
        "EXTRACTOR_API_KEY",
        "FALLBACK_JUDGE_API_KEY",
    }
    for role_info in roles.values():
        assert role_info["configured"] is False
        assert role_info["source"] == "unset"
        assert role_info["masked_hint"] is None

    assert Path(body["cache_dir"]) == isolated_env.cache_dir
    stats = body["cache_stats"]
    assert isinstance(stats["exists"], bool)
    assert isinstance(stats["db_bytes"], int)
    assert isinstance(stats["wal_entries"], int)


def test_t1_byok_03_status_masks_env_key_without_leaking_raw(
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-BYOK-03: Env API key is masked (`sk-...a1b2`) and never echoed raw."""
    secret = "sk-test-env-secret-9988a1b2"
    monkeypatch.setenv("TARGET_API_KEY", secret)

    resp = client.get("/api/byok/status")
    assert resp.status_code == 200
    assert secret not in resp.text
    roles = {r["role"]: r for r in resp.json()["roles"]}
    target = roles["TARGET_API_KEY"]
    assert target["configured"] is True
    assert target["source"] == "env"
    assert target["masked_hint"] == "sk-...a1b2"


def test_t1_byok_04_session_key_save_status_and_delete_roundtrip(client: Any) -> None:
    """T1-BYOK-04: Session-scoped key save -> status -> delete roundtrip."""
    secret = "sk-sess-secret-1234wxyz"
    save = client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": secret, "storage": "session"},
    )
    assert save.status_code == 200
    assert secret not in save.text
    save_body = save.json()
    assert save_body["role"] == "TARGET_API_KEY"
    assert save_body["configured"] is True
    assert save_body["source"] == "session"
    assert save_body["masked_hint"] == "sk-...wxyz"

    status = client.get("/api/byok/status")
    assert status.status_code == 200
    assert secret not in status.text
    roles = {r["role"]: r for r in status.json()["roles"]}
    assert roles["TARGET_API_KEY"]["source"] == "session"
    assert roles["TARGET_API_KEY"]["masked_hint"] == "sk-...wxyz"

    deleted = client.delete("/api/byok/keys/TARGET_API_KEY?storage=session")
    assert deleted.status_code == 200
    del_body = deleted.json()
    assert del_body["configured"] is False
    assert del_body["source"] == "unset"


def test_t1_byok_05_keyring_key_save_status_and_delete_roundtrip(
    client: Any,
    isolated_env: Any,
) -> None:
    """T1-BYOK-05: OS-keyring key save -> status -> delete roundtrip uses isolated keyring."""
    secret = "sk-kr-secret-5566mnop"
    save = client.post(
        "/api/byok/keys",
        json={"role": "JUDGE_API_KEY", "api_key": secret, "storage": "keyring"},
    )
    assert save.status_code == 200
    assert secret not in save.text
    assert save.json()["source"] == "keyring"
    assert save.json()["masked_hint"] == "sk-...mnop"
    assert isolated_env.fake_keyring[("eval-studio", "JUDGE_API_KEY")] == secret

    status = client.get("/api/byok/status")
    assert secret not in status.text
    roles = {r["role"]: r for r in status.json()["roles"]}
    assert roles["JUDGE_API_KEY"]["configured"] is True
    assert roles["JUDGE_API_KEY"]["source"] == "keyring"

    deleted = client.delete("/api/byok/keys/JUDGE_API_KEY?storage=keyring")
    assert deleted.status_code == 200
    assert deleted.json()["configured"] is False
    assert deleted.json()["source"] == "unset"
    assert ("eval-studio", "JUDGE_API_KEY") not in isolated_env.fake_keyring


def test_t1_byok_06_connection_mock_gateway_succeeds_offline(client: Any) -> None:
    """T1-BYOK-06: `POST /api/byok/test` with `mock` backend succeeds offline."""
    resp = client.post(
        "/api/byok/test",
        json={"backend": "mock", "model_id": "mock-deterministic"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["latency_ms"] >= 0.0
    assert body["model_id"] == "mock-deterministic"
    assert len(body["reply_preview"]) > 0


def test_t1_byok_07_connection_http_byok_via_mock_transport(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
) -> None:
    """T1-BYOK-07: `POST /api/byok/test` exercises `HttpBYOKGateway` via `MockTransport`."""
    secret = "sk-deepseek-ping-1234567890"
    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": secret, "storage": "session"},
    )

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "PING-DEEPSEEK-OK"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    bind_mock_transport(transport)

    resp = client.post(
        "/api/byok/test",
        json={
            "backend": "http-byok",
            "provider": "deepseek",
            "model_id": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "role": "TARGET_API_KEY",
        },
    )
    assert resp.status_code == 200
    assert secret not in resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["model_id"] == "deepseek-flash"
    assert "PING-DEEPSEEK-OK" in body["reply_preview"]
    assert len(recorded) == 1
    assert recorded[0]["headers"].get("authorization") == f"Bearer {secret}"


# ---------------------------------------------------------------------------
# Area 4: `/api/byot/*` — 7 Tests
# ---------------------------------------------------------------------------


def test_t1_byot_01_list_tasks_discovers_builtin_and_user_tasks(
    client: Any,
    isolated_env: Any,
) -> None:
    """T1-BYOT-01: `/api/byot/tasks` returns built-in example tasks and user tasks."""
    user_yaml = isolated_env.tasks_dir / "user_custom.yaml"
    user_yaml.write_text(
        yaml.safe_dump(
            {
                "name": "user_custom_qa",
                "version": "1.0.0",
                "trust_tier": "tier1_zero_code",
                "output_type": "generate_until",
                "dataset_path": "examples/tier1_zero_code/sample.csv",
                "doc_to_text": "Q: {{ input_text }}\nA:",
                "doc_to_target": "{{ target }}",
                "metrics": ["exact_match"],
                "metadata": {"date_stamp": "2026-11-23"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    resp = client.get("/api/byot/tasks")
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    by_name = {t["name"]: t for t in tasks}
    assert "tier1_med_qa_sample" in by_name
    assert by_name["tier1_med_qa_sample"]["source"] == "builtin"
    assert "tier2_clinical_mcqa" in by_name
    assert by_name["tier2_clinical_mcqa"]["source"] == "builtin"
    assert "user_custom_qa" in by_name
    assert by_name["user_custom_qa"]["source"] == "user"
    for item in (by_name["tier1_med_qa_sample"], by_name["user_custom_qa"]):
        assert {"id", "name", "path", "source", "trust_tier", "output_type", "metrics"}.issubset(
            item.keys()
        )


def test_t1_byot_02_task_detail_returns_yaml_spec_and_sample_rows(client: Any) -> None:
    """T1-BYOT-02: `/api/byot/tasks/detail` parses YAML, TaskSpec, and sample rows."""
    resp = client.get(
        "/api/byot/tasks/detail",
        params={"path": "examples/tier1_zero_code/task.yaml"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "tier1_med_qa_sample" in body["yaml_text"]
    assert body["spec"]["name"] == "tier1_med_qa_sample"
    assert len(body["sample_rows"]) == 3
    assert body["sample_rows"][0]["sample_id"] == "med_001"


def test_t1_byot_03_upload_dataset_csv_and_jsonl_returns_columns_and_preview(
    client: Any,
) -> None:
    """T1-BYOT-03: Multipart upload for CSV and JSONL persists dataset and returns preview."""
    csv_bytes = (
        b"sample_id,input_text,target\n"
        b"c1,Normal potassium?,3.5-5.0\n"
        b"c2,Normal sodium?,135-145\n"
        b"c3,Normal calcium?,8.5-10.5\n"
    )
    csv_resp = client.post(
        "/api/byot/upload-dataset",
        files={"file": ("electrolytes.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert csv_resp.status_code == 200
    csv_body = csv_resp.json()
    assert csv_body["columns"] == ["sample_id", "input_text", "target"]
    assert csv_body["total_rows"] == 3
    assert len(csv_body["preview_rows"]) == 3
    assert Path(csv_body["dataset_path"]).is_file()

    jsonl_bytes = (
        b'{"sample_id": "j1", "input_text": "First?", "target": "One"}\n'
        b'{"sample_id": "j2", "input_text": "Second?", "target": "Two"}\n'
    )
    jsonl_resp = client.post(
        "/api/byot/upload-dataset",
        files={"file": ("mini.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl")},
    )
    assert jsonl_resp.status_code == 200
    jsonl_body = jsonl_resp.json()
    assert jsonl_body["total_rows"] == 2
    assert jsonl_body["preview_rows"][0]["sample_id"] == "j1"


def test_t1_byot_04_save_and_validate_tier1_tier2_yaml_with_live_preview(
    client: Any,
) -> None:
    """T1-BYOT-04: Save Tier 2 YAML referencing uploaded dataset and live-render sample 0."""
    csv_bytes = b"sample_id,input_text,target\nrow1,Which vitamin prevents scurvy?,Vitamin C\n"
    up = client.post(
        "/api/byot/upload-dataset",
        files={"file": ("vitamins.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert up.status_code == 200
    dataset_path = up.json()["dataset_path"]

    yaml_text = yaml.safe_dump(
        {
            "name": "custom_t2_vitamins",
            "version": "1.0.0",
            "trust_tier": "tier2_declarative",
            "output_type": "generate_until",
            "dataset_path": dataset_path,
            "doc_to_text": "Clinical Q: {{ input_text }}\nAnswer:",
            "doc_to_target": "{{ target }}",
            "metrics": ["exact_match"],
            "filter_ensembles": [
                {"name": "default", "steps": [{"name": "take_first", "params": {}}]}
            ],
            "metadata": {"date_stamp": "2026-11-23"},
        },
        sort_keys=False,
    )

    save = client.post(
        "/api/byot/save",
        json={"filename": "custom_t2.yaml", "yaml_text": yaml_text},
    )
    assert save.status_code == 200
    save_body = save.json()
    assert save_body["task_name"] == "custom_t2_vitamins"
    assert Path(save_body["path"]).is_file()

    val = client.post("/api/byot/validate", json={"yaml_text": yaml_text})
    assert val.status_code == 200
    val_body = val.json()
    assert val_body["valid"] is True
    assert val_body["errors"] == []
    assert val_body["secret_scan_clean"] is True
    assert "Which vitamin prevents scurvy?" in (val_body["preview_prompt"] or "")
    assert val_body["preview_target"] == "Vitamin C"


def test_t1_byot_05_registries_lists_all_builtin_filters_metrics_aggregations(
    client: Any,
) -> None:
    """T1-BYOT-05: `/api/byot/registries` exposes built-in filters, metrics, aggregations."""
    resp = client.get("/api/byot/registries")
    assert resp.status_code == 200
    body = resp.json()
    assert {
        "multi_choice_regex",
        "regex",
        "take_first",
        "take_first_k",
        "majority_vote",
        "think_strip",
    }.issubset(set(body["filters"]))
    assert {
        "exact_match",
        "acc",
        "acc_norm",
        "acc_bytes",
        "acc_mutual_info",
        "brier_score",
        "bits_per_byte",
        "classification_f1",
        "token_f1",
    }.issubset(set(body["metrics"]))
    assert {
        "mean",
        "bootstrap_stderr",
        "pooled_sample_stderr",
        "unweighted_mean_stderr",
        "clustered_stderr",
    }.issubset(set(body["aggregations"]))


def test_t1_byot_06_calibrate_aligned_judge_rows_and_jsonl(client: Any) -> None:
    """T1-BYOT-06: `/api/byot/calibrate` accepts both `rows` and `jsonl_text`."""
    rows = [
        {"judge_label": 1, "human_label": 1},
        {"judge_label": 0, "human_label": 0},
        {"judge_label": 1, "human_label": 1},
        {"judge_label": 0, "human_label": 0},
        {"judge_label": 1, "human_label": 1},
    ]
    r_rows = client.post("/api/byot/calibrate", json={"rows": rows})
    assert r_rows.status_code == 200
    b1 = r_rows.json()
    assert b1["n_samples"] == 5
    assert b1["cohens_kappa"] == pytest.approx(1.0)
    assert b1["raw_agreement"] == pytest.approx(1.0)
    assert b1["passed_threshold"] is True
    assert b1["status"] == "ALIGNED"
    assert b1["threshold"] == pytest.approx(0.40)

    jsonl_text = "\n".join(json.dumps(r) for r in rows)
    r_jsonl = client.post("/api/byot/calibrate", json={"jsonl_text": jsonl_text})
    assert r_jsonl.status_code == 200
    b2 = r_jsonl.json()
    assert b2["status"] == "ALIGNED"
    assert b2["cohens_kappa"] == pytest.approx(1.0)


def test_t1_byot_07_upload_parquet_handled_gracefully_with_or_without_pyarrow(
    client: Any,
) -> None:
    """T1-BYOT-07: Parquet upload either parses via pyarrow or returns clean 400/422 hint."""
    has_pyarrow = importlib.util.find_spec("pyarrow") is not None
    if not has_pyarrow:
        resp = client.post(
            "/api/byot/upload-dataset",
            files={"file": ("data.parquet", io.BytesIO(b"PAR1fake"), "application/octet-stream")},
        )
        assert resp.status_code in (400, 422)
        assert "pyarrow" in resp.json()["detail"].lower()
        return

    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    buf = io.BytesIO()
    table = pa.table({"sample_id": ["p1"], "input_text": ["Q?"], "target": ["A"]})
    pq.write_table(table, buf)
    resp = client.post(
        "/api/byot/upload-dataset",
        files={"file": ("data.parquet", io.BytesIO(buf.getvalue()), "application/octet-stream")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_rows"] == 1
    assert set(body["columns"]) == {"sample_id", "input_text", "target"}


# ---------------------------------------------------------------------------
# Area 5: `/api/byop/*` — 6 Tests
# ---------------------------------------------------------------------------


def test_t1_byop_01_list_papers_discovers_sample_paper(client: Any) -> None:
    """T1-BYOP-01: `/api/byop/papers` discovers `sample_paper.md` and `compiled_suites`."""
    resp = client.get("/api/byop/papers")
    assert resp.status_code == 200
    body = resp.json()
    names = [s["name"] for s in body["samples"]]
    assert any("sample_paper.md" in n for n in names)
    assert isinstance(body["compiled_suites"], list)


def test_t1_byop_02_upload_markdown_and_txt_paper_returns_sha256_and_preview(
    client: Any,
) -> None:
    """T1-BYOP-02: Uploading `.md` and `.txt` papers computes exact SHA-256 and char counts."""
    md_text = "# Acute Stroke Study\n\n## Section 1\nIV alteplase within 4.5 hours improves mRS.\n"
    md_bytes = md_text.encode("utf-8")
    r_md = client.post(
        "/api/byop/upload",
        files={"file": ("stroke.md", io.BytesIO(md_bytes), "text/markdown")},
    )
    assert r_md.status_code == 200
    b_md = r_md.json()
    assert b_md["filename"] == "stroke.md"
    assert Path(b_md["paper_path"]).is_file()
    assert b_md["sha256"] == hashlib.sha256(md_bytes).hexdigest()
    assert b_md["char_count"] in (len(md_text.strip()), len(md_text))
    assert b_md["page_count"] >= 1
    assert "Acute Stroke Study" in b_md["preview_text"]

    txt_text = "Plain clinical trial summary: 120 patients enrolled with 85% survival."
    txt_bytes = txt_text.encode("utf-8")
    r_txt = client.post(
        "/api/byop/upload",
        files={"file": ("trial.txt", io.BytesIO(txt_bytes), "text/plain")},
    )
    assert r_txt.status_code == 200
    b_txt = r_txt.json()
    assert b_txt["sha256"] == hashlib.sha256(txt_bytes).hexdigest()
    assert b_txt["char_count"] in (len(txt_text.strip()), len(txt_text))


def test_t1_byop_03_compile_heuristic_produces_grounded_approved_items(
    client: Any,
    isolated_env: Any,
) -> None:
    """T1-BYOP-03: Heuristic `compile-paper` extracts cited, grounded `approved_items`."""
    paper_path = isolated_env.examples_dir / "papers" / "sample_paper.md"
    paper_text = paper_path.read_text(encoding="utf-8")

    resp = client.post(
        "/api/byop/compile",
        json={
            "paper_path": str(paper_path),
            "suite_name": "trop_heuristic",
            "mode": "heuristic",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["source_sha256"]) == 64
    assert len(body["approved_items"]) >= 2
    assert isinstance(body["rejected_items"], list)
    for item in body["approved_items"]:
        assert item["citations"]
        for cite in item["citations"]:
            assert cite["page"] >= 1
            assert cite["section"].strip()
            assert cite["quote"] in paper_text


def test_t1_byop_04_compile_mock_mode_separates_approved_and_rejected_with_reasons(
    client: Any,
    isolated_env: Any,
) -> None:
    """T1-BYOP-04: Mock mixed mode separates grounded APPROVED items from REJECTED items."""
    paper_path = isolated_env.examples_dir / "papers" / "sample_paper.md"
    resp = client.post(
        "/api/byop/compile",
        json={
            "paper_path": str(paper_path),
            "suite_name": "trop_mock_mixed",
            "mode": "mock",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    approved_answers = {item["reference_answer"] for item in body["approved_items"]}
    assert "99.6%" in approved_answers
    assert "> 0.04 ng/mL" in approved_answers

    rejected = body["rejected_items"]
    assert len(rejected) >= 2
    for entry in rejected:
        assert "item" in entry
        assert isinstance(entry["reasons"], list) and len(entry["reasons"]) >= 1


def test_t1_byop_05_lock_edited_items_writes_yaml_manifest_and_appears_in_lists(
    client: Any,
    isolated_env: Any,
) -> None:
    """T1-BYOP-05: Locking edited approved items writes YAML + manifest and registers suite."""
    paper_path = isolated_env.examples_dir / "papers" / "sample_paper.md"
    comp = client.post(
        "/api/byop/compile",
        json={"paper_path": str(paper_path), "suite_name": "locked_trop", "mode": "mock"},
    )
    assert comp.status_code == 200
    approved = comp.json()["approved_items"]
    edited = list(approved)
    edited[0] = {**edited[0], "rubric": "Edited clinical rubric: state 99.6% NPV exactly."}

    lock = client.post(
        "/api/byop/lock",
        json={
            "paper_path": str(paper_path),
            "suite_name": "locked_trop",
            "extractor_model": "mock-extractor",
            "verifier_model": "mock-verifier",
            "items": edited,
        },
    )
    assert lock.status_code == 200
    lock_body = lock.json()
    assert Path(lock_body["yaml_path"]).is_file()
    assert Path(lock_body["manifest_path"]).is_file()
    assert len(lock_body["suite_sha256"]) == 64
    assert len(lock_body["source_sha256"]) == 64
    assert lock_body["item_count"] == len(edited)

    parsed_yaml = yaml.safe_load(lock_body["yaml_text"])
    assert parsed_yaml["trust_tier"] == "tier2_declarative"
    assert parsed_yaml["metadata"]["source_paper_sha256"] == lock_body["source_sha256"]
    assert parsed_yaml["metadata"]["suite_sha256"] == lock_body["suite_sha256"]

    papers_list = client.get("/api/byop/papers").json()["compiled_suites"]
    assert any(s["name"] == "locked_trop" for s in papers_list)

    tasks_list = client.get("/api/byot/tasks").json()["tasks"]
    assert any(t["name"] == "locked_trop" for t in tasks_list)


def test_t1_byop_06_upload_valid_pdf_with_pypdf_extracts_page_markers(
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-BYOP-06: Uploading `.pdf` with pypdf available extracts pages & matches raw SHA-256."""
    fake_pypdf = types.ModuleType("pypdf")

    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakePdfReader:
        def __init__(self, path: str) -> None:
            self.pages = [
                _FakePage("Section 1: Troponin I threshold > 0.04 ng/mL."),
                _FakePage("Section 2: Negative predictive value was 99.6%."),
            ]

    fake_pypdf.PdfReader = _FakePdfReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    pdf_bytes = b"%PDF-1.4\n%fake-clinical-paper\n%%EOF\n"
    resp = client.post(
        "/api/byop/upload",
        files={"file": ("clinical_study.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sha256"] == hashlib.sha256(pdf_bytes).hexdigest()
    assert body["page_count"] == 2
    assert "Troponin I" in body["preview_text"]
