"""Tier 1 E2E tests: `/api/runs/*` (pre-flight, async, SSE) and `/api/receipts/*`."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

pytestmark = pytest.mark.e2e_web


def _write_two_sample_mcqa_task(tasks_dir: Path) -> Path:
    csv_path = tasks_dir / "mcqa_samples.csv"
    csv_path.write_text(
        "sample_id,input_text,choices,target\n"
        'q1,"Which vitamin deficiency causes scurvy?","Vitamin A|Vitamin C|Vitamin D|Vitamin K",1\n'
        'q2,"Which organ produces insulin?","Liver|Spleen|Pancreas|Kidney",2\n',
        encoding="utf-8",
    )
    yaml_path = tasks_dir / "mcqa_olmes.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "name": "e2e_mcqa_olmes",
                "version": "1.0.0",
                "trust_tier": "tier2_declarative",
                "output_type": "multiple_choice",
                "dataset_path": str(csv_path),
                "doc_to_text": "Question: {{ input_text }}\nAnswer:",
                "doc_to_target": "{{ target }}",
                "metrics": ["acc_bytes", "acc_norm"],
                "metadata": {"date_stamp": "2026-11-23"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


def _create_mock_receipt_via_api(
    client: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
    *,
    task_path: str,
    model_backend: str = "mock",
    model_id: str = "mock-deterministic",
    base_url: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_path": task_path,
        "model_backend": model_backend,
        "model_id": model_id,
    }
    if base_url is not None:
        payload["base_url"] = base_url
    launch = client.post("/api/runs", json=payload)
    assert launch.status_code in (200, 202), launch.text
    run_id = launch.json()["run_id"]
    return poll_run_until_done(client, run_id, 15.0)


# ---------------------------------------------------------------------------
# Area 6: `/api/runs/*` — 6 Tests
# ---------------------------------------------------------------------------


def test_t1_run_01_estimate_returns_target_judge_and_total_usd_breakdown(
    client: Any,
) -> None:
    """T1-RUN-01: Pre-flight estimate computes target + judge + total USD breakdown."""
    resp = client.post(
        "/api/runs/estimate",
        json={
            "task_path": "examples/tier1_zero_code/live_smoke.yaml",
            "model_backend": "mock",
            "model_id": "mock-deterministic",
            "repeats": 2,
            "max_cost_usd": 1.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_samples"] == 3
    assert body["repeats"] == 2
    assert body["target_prompt_tokens"] > 0
    assert body["target_completion_tokens"] > 0
    assert body["target_usd"] > 0.0
    assert body["judge_enabled"] is True
    assert body["judge_usd"] > 0.0
    assert body["total_usd"] == pytest.approx(body["target_usd"] + body["judge_usd"], abs=1e-12)
    assert body["exceeds_max_cost"] is False


def test_t1_run_02_launch_mock_run_and_poll_status_until_completed(
    client: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T1-RUN-02: Launching mock run returns run_id and completes with valid receipt bundle."""
    launch = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "mock",
            "model_id": "mock-deterministic",
        },
    )
    assert launch.status_code in (200, 202)
    launch_body = launch.json()
    run_id = launch_body["run_id"]
    assert launch_body["status"] in ("queued", "running", "completed")
    assert launch_body["output_dir"]

    done = poll_run_until_done(client, run_id, 15.0)
    assert done["status"] == "completed"
    assert done["total_samples"] == 3
    assert done["completed_samples"] == 3
    receipt_dir = Path(done["receipt_dir"])
    assert (receipt_dir / "manifest.json").is_file()
    assert (receipt_dir / "samples.jsonl").is_file()
    assert (receipt_dir / "report.md").is_file()
    assert isinstance(done["manifest"], dict)


def test_t1_run_03_sse_events_stream_lifecycle_and_sample_scored(
    client: Any,
    parse_sse_frames: Callable[[str], list[dict[str, Any]]],
) -> None:
    """T1-RUN-03: SSE `/events` streams `run_started`, 3 `sample_scored`, and `completed`."""
    launch = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "mock",
            "model_id": "mock-deterministic",
        },
    )
    assert launch.status_code in (200, 202)
    run_id = launch.json()["run_id"]

    chunks: list[str] = []
    with client.stream(
        "GET",
        f"/api/runs/{run_id}/events",
        headers={"Accept": "text/event-stream"},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        assert stream_resp.headers.get("content-type", "").startswith("text/event-stream")
        for chunk in stream_resp.iter_text():
            chunks.append(chunk)

    frames = parse_sse_frames("".join(chunks))
    event_names = [f["event"] for f in frames]
    assert "run_started" in event_names
    scored_frames = [f for f in frames if f["event"] == "sample_scored"]
    assert len(scored_frames) == 3
    for frame in scored_frames:
        data = frame["data"]
        assert data["total"] == 3
        assert "index" in data
        assert "sample" in data
        assert "spent_usd" in data
    assert event_names[-1] == "completed"
    assert frames[-1]["data"]["run_id"]
    assert "spent_usd" in frames[-1]["data"]


def test_t1_run_04_run_with_mock_judge_and_thinking_records_judge_metric_and_traces(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T1-RUN-04: Run with `judge_model='mock'`, thinking strip, and exact `bootstrap_iters`."""
    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": "sk-think-run-123456", "storage": "session"},
    )

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "<think>clinical reasoning trace</think>Vitamin C",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            },
        )

    transport, _ = make_mock_transport(_handler)
    bind_mock_transport(transport)

    launch = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "http-byok",
            "model_id": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "judge_model": "mock",
            "enable_thinking": True,
            "think_end_token": "</think>",
            "bootstrap_iters": 250,
        },
    )
    assert launch.status_code in (200, 202), launch.text
    done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    assert done["status"] == "completed"
    manifest = done["manifest"]
    assert "exact_match" in manifest["metrics"]
    assert "llm_judge" in manifest["metrics"]
    assert manifest["metrics"]["exact_match"]["bootstrap_iters"] == 250
    assert manifest["metrics"]["llm_judge"]["bootstrap_iters"] == 250

    samples_lines = (Path(done["receipt_dir"]) / "samples.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    first_sample = json.loads(samples_lines[0])
    assert "clinical reasoning trace" in (first_sample["raw_responses"][0]["reasoning_trace"] or "")


def test_t1_run_05_run_with_olmes_dual_cloze_mcf_on_mcqa_task(
    client: Any,
    isolated_env: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T1-RUN-05: `olmes_enabled=True` executes dual Cloze+MCF evaluation on MCQA task."""
    mcqa_path = _write_two_sample_mcqa_task(isolated_env.tasks_dir)
    launch = client.post(
        "/api/runs",
        json={
            "task_path": str(mcqa_path),
            "model_backend": "mock",
            "model_id": "mock-deterministic",
            "olmes_enabled": True,
        },
    )
    assert launch.status_code in (200, 202), launch.text
    done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    assert done["status"] == "completed"
    metrics = done["manifest"]["metrics"]
    assert "acc_bytes" in metrics or "acc_norm" in metrics
    assert any(k.endswith(("_cloze", "_mcf")) for k in metrics)
    redacted_cfg = done["manifest"]["redacted_config"]
    assert "olmes_selection" in redacted_cfg
    assert redacted_cfg["olmes_selection"]["selected_format"] in ("cloze", "mcf")


def test_t1_run_06_run_with_limit_and_repeats_respects_sample_cap(
    client: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T1-RUN-06: `limit=2` and `repeats=3` caps samples at 2 and records 3 responses each."""
    launch = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "mock",
            "model_id": "mock-deterministic",
            "limit": 2,
            "repeats": 3,
        },
    )
    assert launch.status_code in (200, 202)
    done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    assert done["status"] == "completed"
    assert done["manifest"]["metrics"]["exact_match"]["n_samples"] == 2
    lines = [
        json.loads(line)
        for line in (Path(done["receipt_dir"]) / "samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    for row in lines:
        assert len(row["raw_responses"]) == 3


# ---------------------------------------------------------------------------
# Area 7: `/api/receipts/*` — 7 Tests
# ---------------------------------------------------------------------------


def test_t1_rcp_01_list_receipts_ignores_internal_subtask_dirs(
    client: Any,
    isolated_env: Any,
) -> None:
    """T1-RCP-01: `/api/receipts` lists valid receipts and excludes `_subtask_*` folders."""
    subtask_dir = isolated_env.receipts_dir / "_subtask_alpha"
    subtask_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        isolated_env.receipts_dir / "live_ds_vs_glm" / "manifest.json",
        subtask_dir / "manifest.json",
    )

    resp = client.get("/api/receipts")
    assert resp.status_code == 200
    receipts = resp.json()["receipts"]
    dir_names = {r["dir_name"] for r in receipts}
    assert "live_ds_vs_glm" in dir_names
    assert "_subtask_alpha" not in dir_names

    ds = next(r for r in receipts if r["dir_name"] == "live_ds_vs_glm")
    assert ds["run_id"] == "live_smoke_ds_glm-1cc97597"
    assert ds["model_id"] == "deepseek-flash"
    assert "exact_match" in ds["metrics"]
    assert len(ds["task_sha256"]) == 64
    assert len(ds["manifest_sha256"]) == 64


def test_t1_rcp_02_receipt_detail_returns_manifest_report_md_and_total_samples(
    client: Any,
) -> None:
    """T1-RCP-02: `/api/receipts/{id}` returns manifest, report_md, and total_samples."""
    resp = client.get("/api/receipts/live_ds_vs_glm")
    assert resp.status_code == 200
    body = resp.json()
    assert body["manifest"]["run_id"] == "live_smoke_ds_glm-1cc97597"
    assert "deepseek-flash" in body["report_md"]
    assert body["total_samples"] == 3


def test_t1_rcp_03_receipt_samples_pagination_and_search_query(client: Any) -> None:
    """T1-RCP-03: `/api/receipts/{id}/samples` supports `offset`, `limit`, and `q` search."""
    page = client.get("/api/receipts/live_ds_vs_glm/samples?offset=1&limit=1")
    assert page.status_code == 200
    p_body = page.json()
    assert p_body["total"] == 3
    assert p_body["offset"] == 1
    assert p_body["limit"] == 1
    assert len(p_body["samples"]) == 1
    assert p_body["samples"][0]["sample_id"] == "med_002"

    search = client.get("/api/receipts/live_ds_vs_glm/samples?q=insulin")
    assert search.status_code == 200
    s_body = search.json()
    assert s_body["total"] == 1
    assert s_body["samples"][0]["sample_id"] == "med_003"


def test_t1_rcp_04_verify_valid_receipt_returns_true(client: Any) -> None:
    """T1-RCP-04: `/api/receipts/live_ds_vs_glm/verify` confirms SHA-256 integrity."""
    resp = client.post("/api/receipts/live_ds_vs_glm/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["errors"] == []


def test_t1_rcp_05_rescore_offline_with_custom_metrics_creates_new_receipt(
    client: Any,
    isolated_env: Any,
) -> None:
    """T1-RCP-05: Offline `rescore` creates new receipt while leaving original untouched."""
    orig_manifest_bytes = (
        isolated_env.receipts_dir / "live_ds_vs_glm" / "manifest.json"
    ).read_bytes()

    resp = client.post(
        "/api/receipts/live_ds_vs_glm/rescore",
        json={
            "metrics": ["exact_match", "token_f1"],
            "output_name": "rescored_custom",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert Path(body["path"]).is_dir()
    assert "exact_match" in body["manifest"]["metrics"]
    assert "token_f1" in body["manifest"]["metrics"]
    assert (
        isolated_env.receipts_dir / "live_ds_vs_glm" / "manifest.json"
    ).read_bytes() == orig_manifest_bytes


def test_t1_rcp_06_noise_floor_between_two_receipts(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T1-RCP-06: `/api/receipts/noise-floor` reports exact `n_flips` and `flip_rate`."""
    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": "sk-nf-1234567890", "storage": "session"},
    )
    mode = {"flipped": False}

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        msg = request.content.decode("utf-8")
        if "scurvy" in msg.lower():
            ans = "Vitamin D" if mode["flipped"] else "Vitamin C"
        elif "heart rate" in msg.lower():
            ans = "60-100"
        else:
            ans = "Pancreas"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ans}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    transport, _ = make_mock_transport(_handler)
    bind_mock_transport(transport)

    base = _create_mock_receipt_via_api(
        client,
        poll_run_until_done,
        task_path="examples/tier1_zero_code/task.yaml",
        model_backend="http-byok",
        model_id="gpt-4o-mini",
    )
    mode["flipped"] = True
    pert = _create_mock_receipt_via_api(
        client,
        poll_run_until_done,
        task_path="examples/tier1_zero_code/task.yaml",
        model_backend="http-byok",
        model_id="gpt-4o-mini-pert",
    )

    base_id = Path(base["receipt_dir"]).name
    pert_id = Path(pert["receipt_dir"]).name
    nf = client.post(
        "/api/receipts/noise-floor",
        json={
            "baseline_id": base_id,
            "perturbed_id": pert_id,
            "perturbation": "batch_size_4",
        },
    )
    assert nf.status_code == 200
    body = nf.json()
    assert body["n_decisions"] == 3
    assert body["n_flips"] == 1
    assert body["flip_rate"] == pytest.approx(1 / 3, rel=1e-6)
    assert body["perturbation"] == "batch_size_4"


def test_t1_rcp_07_compare_returns_runs_pairwise_mcnemar_bootstrap_and_item_diffs(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T1-RCP-07: `/api/receipts/compare` produces runs CIs, McNemar, bootstrap, item diffs."""
    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": "sk-cmp-1234567890", "storage": "session"},
    )
    phase = {"model": "a"}

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        msg = request.content.decode("utf-8").lower()
        if "scurvy" in msg:
            ans = "Vitamin C" if phase["model"] == "a" else "Iron"
        elif "heart rate" in msg:
            ans = "60-100"
        else:
            ans = "Liver" if phase["model"] == "a" else "Pancreas"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ans}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    transport, _ = make_mock_transport(_handler)
    bind_mock_transport(transport)

    run_a = _create_mock_receipt_via_api(
        client,
        poll_run_until_done,
        task_path="examples/tier1_zero_code/task.yaml",
        model_backend="http-byok",
        model_id="model-a",
    )
    phase["model"] = "b"
    run_b = _create_mock_receipt_via_api(
        client,
        poll_run_until_done,
        task_path="examples/tier1_zero_code/task.yaml",
        model_backend="http-byok",
        model_id="model-b",
    )

    id_a = Path(run_a["receipt_dir"]).name
    id_b = Path(run_b["receipt_dir"]).name
    cmp_resp = client.post(
        "/api/receipts/compare",
        json={"receipt_ids": [id_a, id_b], "baseline_id": id_a, "bootstrap_iters": 200},
    )
    assert cmp_resp.status_code == 200
    body = cmp_resp.json()
    assert len(body["runs"]) == 2
    for run_entry in body["runs"]:
        em = run_entry["metrics"]["exact_match"]
        assert {"value", "stderr", "ci_low", "ci_high", "unresolved"}.issubset(em.keys())

    assert len(body["pairwise"]) >= 1
    pair = body["pairwise"][0]
    assert "paired_bootstrap" in pair
    boot = pair["paired_bootstrap"]
    assert {"mean_delta", "ci_low", "ci_high", "p_value", "significant"}.issubset(boot.keys())
    assert pair["mcnemar"] is not None
    assert pair["mcnemar"]["n_items"] == 3
    assert "noise_floor" in pair
    diff_ids = {d["sample_id"] for d in pair["item_diffs"]}
    assert diff_ids == {"med_001", "med_003"}
