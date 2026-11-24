"""Tier 4 E2E tests: Realistic end-to-end clinical evaluation workflows (6 scenarios)."""

from __future__ import annotations

import inspect
import io
import json
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

pytestmark = pytest.mark.e2e_web


def test_t4_01_e2e_multi_model_comparison_with_live_ds_vs_glm_fixtures(
    client: Any,
    copied_live_receipts: dict[str, Path],
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T4-01: Compare copied live DeepSeek receipts against a new simulated glm5.3-flash run."""
    assert copied_live_receipts["live_ds_vs_glm"].is_dir()
    assert copied_live_receipts["live_ds_vs_glm_rescored"].is_dir()

    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": "sk-glm-challenger-1234", "storage": "session"},
    )

    def _glm_handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        msg = request.content.decode("utf-8").lower()
        if "scurvy" in msg:
            ans = "Vitamin C"
        elif "heart rate" in msg:
            ans = "60-100"
        else:
            ans = "Liver"  # 2/3 exact_match vs live_ds_vs_glm's 1/3
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ans}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    transport, _ = make_mock_transport(_glm_handler)
    bind_mock_transport(transport)

    launch = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/live_smoke.yaml",
            "model_backend": "http-byok",
            "model_id": "glm5.3-flash",
            "base_url": "https://api.sarvam.ai/v2",
        },
    )
    assert launch.status_code in (200, 202)
    glm_done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    glm_id = Path(glm_done["receipt_dir"]).name

    all_rcps = client.get("/api/receipts").json()["receipts"]
    dir_names = {r["dir_name"] for r in all_rcps}
    assert {"live_ds_vs_glm", "live_ds_vs_glm_rescored", glm_id}.issubset(dir_names)

    for rid in ("live_ds_vs_glm", "live_ds_vs_glm_rescored", glm_id):
        ver = client.post(f"/api/receipts/{rid}/verify")
        assert ver.status_code == 200
        assert ver.json()["valid"] is True

    insulin_samples = client.get("/api/receipts/live_ds_vs_glm/samples?q=insulin").json()
    assert insulin_samples["total"] == 1
    assert insulin_samples["samples"][0]["raw_responses"][0]["reasoning_trace"]

    cmp_resp = client.post(
        "/api/receipts/compare",
        json={
            "receipt_ids": ["live_ds_vs_glm", "live_ds_vs_glm_rescored", glm_id],
            "baseline_id": "live_ds_vs_glm",
            "bootstrap_iters": 300,
        },
    )
    assert cmp_resp.status_code == 200
    cmp_body = cmp_resp.json()
    assert len(cmp_body["runs"]) == 3
    for run_summary in cmp_body["runs"]:
        em = run_summary["metrics"]["exact_match"]
        if em["stderr"] is not None:
            assert em["ci_high"] >= em["value"] >= em["ci_low"]

    glm_pair = next(
        p
        for p in cmp_body["pairwise"]
        if p.get("challenger_id", p.get("receipt_id")) == glm_id
    )
    assert glm_pair["mcnemar"] is not None
    assert glm_pair["paired_bootstrap"]["ci_low"] <= 0.0 <= glm_pair["paired_bootstrap"]["ci_high"]
    assert glm_pair["unresolved"] is True
    assert len(glm_pair["item_diffs"]) == 3


def test_t4_02_e2e_clinical_paper_compilation_to_multi_model_comparison(
    client: Any,
    isolated_env: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    parse_sse_frames: Callable[[str], list[dict[str, Any]]],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T4-02: Clinical paper upload -> 2-stage compile -> edit/lock -> 2-model SSE -> compare."""
    paper_bytes = (isolated_env.examples_dir / "papers" / "sample_paper.md").read_bytes()
    up = client.post(
        "/api/byop/upload",
        files={"file": ("troponin_study.md", io.BytesIO(paper_bytes), "text/markdown")},
    )
    assert up.status_code == 200
    paper_path = up.json()["paper_path"]

    comp = client.post(
        "/api/byop/compile",
        json={"paper_path": paper_path, "suite_name": "troponin_eval", "mode": "mock"},
    )
    assert comp.status_code == 200
    c_body = comp.json()
    assert len(c_body["approved_items"]) == 2
    assert len(c_body["rejected_items"]) >= 2

    approved = list(c_body["approved_items"])
    approved[0] = {**approved[0], "rubric": "Updated rubric: state 99.6% NPV exactly."}
    lock = client.post(
        "/api/byop/lock",
        json={
            "paper_path": paper_path,
            "suite_name": "troponin_eval",
            "extractor_model": "mock-extractor",
            "verifier_model": "mock-verifier",
            "items": approved,
        },
    )
    assert lock.status_code == 200
    locked_yaml = lock.json()["yaml_path"]

    est = client.post(
        "/api/runs/estimate",
        json={
            "task_path": locked_yaml,
            "model_backend": "http-byok",
            "model_id": "model-alpha",
        },
    )
    assert est.status_code == 200
    assert est.json()["n_samples"] == 2

    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": "sk-t4-paper-123456", "storage": "session"},
    )
    phase = {"model": "alpha"}

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        msg = request.content.decode("utf-8").lower()
        if "negative predictive value" in msg:
            ans = "99.6%"
        else:
            ans = "> 0.04 ng/mL" if phase["model"] == "alpha" else "> 0.08 ng/mL"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ans}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    transport, _ = make_mock_transport(_handler)
    bind_mock_transport(transport)

    r1 = client.post(
        "/api/runs",
        json={"task_path": locked_yaml, "model_backend": "http-byok", "model_id": "model-alpha"},
    )
    with client.stream("GET", f"/api/runs/{r1.json()['run_id']}/events") as s1:
        f1 = parse_sse_frames("".join(s1.iter_text()))
    assert [f["event"] for f in f1][-1] == "completed"
    d1 = poll_run_until_done(client, r1.json()["run_id"], 15.0)

    phase["model"] = "beta"
    r2 = client.post(
        "/api/runs",
        json={"task_path": locked_yaml, "model_backend": "http-byok", "model_id": "model-beta"},
    )
    d2 = poll_run_until_done(client, r2.json()["run_id"], 15.0)

    id1 = Path(d1["receipt_dir"]).name
    id2 = Path(d2["receipt_dir"]).name
    assert client.post(f"/api/receipts/{id1}/verify").json()["valid"] is True
    assert client.post(f"/api/receipts/{id2}/verify").json()["valid"] is True

    cmp_body = client.post(
        "/api/receipts/compare",
        json={"receipt_ids": [id1, id2], "baseline_id": id1, "bootstrap_iters": 150},
    ).json()
    cats_text = json.dumps(cmp_body["category_breakdown"])
    assert "quantitative_finding" in cats_text
    assert "claim" in cats_text
    pair = cmp_body["pairwise"][0]
    assert pair["mcnemar"] is not None
    assert "paired_bootstrap" in pair
    assert len(pair["item_diffs"]) == 1


def test_t4_03_e2e_judge_calibration_cross_family_rescore_and_noise_floor(
    client: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T4-03: Unaligned->Aligned calibration, judge run, fallback rescore, noise-floor, verify."""
    unaligned = client.post(
        "/api/byot/calibrate",
        json={
            "rows": [
                {"judge_label": 1, "human_label": 0},
                {"judge_label": 0, "human_label": 1},
                {"judge_label": 1, "human_label": 0},
                {"judge_label": 0, "human_label": 1},
            ]
        },
    )
    assert unaligned.status_code == 200
    assert unaligned.json()["status"] == "UNALIGNED"

    aligned = client.post(
        "/api/byot/calibrate",
        json={
            "rows": [
                {"judge_label": 1, "human_label": 1},
                {"judge_label": 0, "human_label": 0},
                {"judge_label": 1, "human_label": 1},
                {"judge_label": 0, "human_label": 0},
                {"judge_label": 1, "human_label": 1},
            ]
        },
    )
    assert aligned.status_code == 200
    assert aligned.json()["status"] == "ALIGNED"

    launch = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "mock",
            "model_id": "mock-deterministic",
            "judge_model": "mock",
        },
    )
    done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    orig_id = Path(done["receipt_dir"]).name

    resc = client.post(
        f"/api/receipts/{orig_id}/rescore",
        json={
            "judge_model": "mock-refusal",
            "fallback_judge_model": "mock",
            "output_name": f"{orig_id}_cross_family",
        },
    )
    assert resc.status_code == 200
    resc_id = Path(resc.json()["path"]).name
    assert resc.json()["manifest"]["metrics"]["llm_judge"]["value"] == pytest.approx(1.0)

    nf = client.post(
        "/api/receipts/noise-floor",
        json={"baseline_id": orig_id, "perturbed_id": resc_id, "perturbation": "judge_rescore"},
    )
    assert nf.status_code == 200
    assert nf.json()["flip_rate"] == pytest.approx(0.0)

    for rid in (orig_id, resc_id):
        ver = client.post(f"/api/receipts/{rid}/verify")
        assert ver.status_code == 200
        assert ver.json()["valid"] is True


def test_t4_04_e2e_byok_provider_lifecycle_with_budget_cap_and_redaction(
    client: Any,
    isolated_env: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    parse_sse_frames: Callable[[str], list[dict[str, Any]]],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T4-04: DeepSeek + Sarvam BYOK lifecycle, budget cap SSE halt, zero secret persistence."""
    ds_secret = "sk-ds-lifecycle-99887766"
    sarvam_secret = "sk_sarvam_lifecycle_11223344"

    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": ds_secret, "storage": "session"},
    )
    client.post(
        "/api/byok/keys",
        json={"role": "JUDGE_API_KEY", "api_key": sarvam_secret, "storage": "session"},
    )

    presets = client.get("/api/byok/presets").json()["presets"]
    preset_ids = {p["id"]: p for p in presets}
    assert any(m["id"] == "deepseek-flash" for m in preset_ids["deepseek"]["models"])
    assert any(m["id"] == "glm5.3-flash" for m in preset_ids["sarvam"]["models"])

    status = client.get("/api/byok/status")
    assert ds_secret not in status.text and sarvam_secret not in status.text

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 500},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    bind_mock_transport(transport)

    p1 = client.post(
        "/api/byok/test",
        json={
            "backend": "http-byok",
            "provider": "deepseek",
            "model_id": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "role": "TARGET_API_KEY",
        },
    )
    p2 = client.post(
        "/api/byok/test",
        json={
            "backend": "http-byok",
            "provider": "sarvam",
            "model_id": "glm5.3-flash",
            "base_url": "https://api.sarvam.ai/v2",
            "role": "JUDGE_API_KEY",
        },
    )
    assert p1.json()["ok"] is True
    assert p2.json()["ok"] is True
    assert recorded[0]["headers"].get("authorization") == f"Bearer {ds_secret}"
    assert recorded[1]["headers"].get("api-subscription-key") == sarvam_secret

    csv_path = isolated_env.tasks_dir / "t4_budget.csv"
    csv_path.write_text(
        "sample_id,input_text,target\n"
        + "\n".join(f"b{i},Detailed clinical case {i}?,A" for i in range(1, 6))
        + "\n",
        encoding="utf-8",
    )
    task_path = isolated_env.tasks_dir / "t4_budget.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "name": "t4_budget_cap",
                "version": "1.0.0",
                "trust_tier": "tier1_zero_code",
                "output_type": "generate_until",
                "dataset_path": str(csv_path),
                "doc_to_text": "Q: {{ input_text }}\nA:",
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

    est = client.post(
        "/api/runs/estimate",
        json={
            "task_path": str(task_path),
            "model_backend": "http-byok",
            "model_id": "deepseek-flash",
            "max_cost_usd": 0.001,
        },
    )
    assert est.json()["exceeds_max_cost"] is True

    launch = client.post(
        "/api/runs",
        json={
            "task_path": str(task_path),
            "model_backend": "http-byok",
            "model_id": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "max_cost_usd": 0.001,
        },
    )
    run_id = launch.json()["run_id"]
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream_resp:
        raw_sse = "".join(stream_resp.iter_text())
    assert ds_secret not in raw_sse and sarvam_secret not in raw_sse
    frames = parse_sse_frames(raw_sse)
    assert any(f["event"] in ("budget_exceeded", "error") for f in frames)

    poll_run_until_done(client, run_id, 15.0)
    for cache_file in isolated_env.cache_dir.rglob("*.jsonl"):
        text = cache_file.read_text(encoding="utf-8", errors="replace")
        assert ds_secret not in text and sarvam_secret not in text

    for role in ("TARGET_API_KEY", "JUDGE_API_KEY"):
        d = client.delete(f"/api/byok/keys/{role}?storage=all")
        assert d.status_code == 200
        assert d.json()["source"] == "unset"


def test_t4_05_e2e_byot_workbench_authoring_ssti_guard_and_thinking_eval(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    parse_sse_frames: Callable[[str], list[dict[str, Any]]],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T4-05: Author 4-vignette CSV, block SSTI, save Tier 2 YAML, run thinking eval & inspect."""
    csv_bytes = (
        b"sample_id,input_text,target\n"
        b"v1,Dx for Koplik spots?,Measles\n"
        b"v2,Dx for Kayser-Fleischer rings?,Wilson disease\n"
        b"v3,Dx for Auer rods?,AML\n"
        b"v4,Dx for Reed-Sternberg cells?,Hodgkin lymphoma\n"
    )
    up = client.post(
        "/api/byot/upload-dataset",
        files={"file": ("vignettes.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert up.status_code == 200
    dataset_path = up.json()["dataset_path"]

    bad_yaml = yaml.safe_dump(
        {
            "name": "vignette_ssti",
            "version": "1.0.0",
            "trust_tier": "tier2_declarative",
            "output_type": "generate_until",
            "dataset_path": dataset_path,
            "doc_to_text": "{{ ''.__class__.__mro__[1].__subclasses__() }}",
            "doc_to_target": "{{ target }}",
            "metrics": ["exact_match"],
        },
        sort_keys=False,
    )
    assert client.post("/api/byot/validate", json={"yaml_text": bad_yaml}).json()["valid"] is False
    assert (
        client.post(
            "/api/byot/save", json={"filename": "vignette_ssti.yaml", "yaml_text": bad_yaml}
        ).status_code
        in (400, 422)
    )

    good_yaml = yaml.safe_dump(
        {
            "name": "clinical_vignettes_t2",
            "version": "1.0.0",
            "trust_tier": "tier2_declarative",
            "output_type": "generate_until",
            "dataset_path": dataset_path,
            "doc_to_text": "Vignette: {{ input_text }}\nDiagnosis:",
            "doc_to_target": "{{ target }}",
            "metrics": ["exact_match"],
            "filter_ensembles": [
                {
                    "name": "default",
                    "steps": [{"name": "think_strip", "params": {"think_end_token": "</think>"}}],
                }
            ],
            "metadata": {"date_stamp": "2026-11-23"},
        },
        sort_keys=False,
    )
    val = client.post("/api/byot/validate", json={"yaml_text": good_yaml}).json()
    assert val["valid"] is True
    assert "Koplik spots" in (val["preview_prompt"] or "")

    save = client.post(
        "/api/byot/save",
        json={"filename": "clinical_vignettes_t2.yaml", "yaml_text": good_yaml},
    )
    task_path = save.json()["path"]

    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": "sk-t4-think-123456", "storage": "session"},
    )
    answers = {
        "koplik": "Measles",
        "kayser": "Wilson disease",
        "auer": "AML",
        "reed": "Hodgkin lymphoma",
    }

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        msg = request.content.decode("utf-8").lower()
        ans = next(val for key, val in answers.items() if key in msg)
        content = f"<think>differential diagnosis</think>{ans}"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 6},
            },
        )

    transport, _ = make_mock_transport(_handler)
    bind_mock_transport(transport)

    launch = client.post(
        "/api/runs",
        json={
            "task_path": task_path,
            "model_backend": "http-byok",
            "model_id": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "enable_thinking": True,
            "think_end_token": "</think>",
        },
    )
    run_id = launch.json()["run_id"]
    with client.stream("GET", f"/api/runs/{run_id}/events") as sse:
        frames = parse_sse_frames("".join(sse.iter_text()))
    assert [f["event"] for f in frames][-1] == "completed"

    done = poll_run_until_done(client, run_id, 15.0)
    assert done["manifest"]["metrics"]["exact_match"]["value"] == pytest.approx(1.0)
    rid = Path(done["receipt_dir"]).name

    samples = client.get(f"/api/receipts/{rid}/samples").json()["samples"]
    assert len(samples) == 4
    for s in samples:
        assert "differential diagnosis" in (s["raw_responses"][0]["reasoning_trace"] or "")
        assert "</think>" not in s["filtered_responses"]["default"][0]

    assert client.post(f"/api/receipts/{rid}/verify").json()["valid"] is True


def test_t4_06_e2e_receipt_vault_tamper_audit_and_reproduce_command_parity(
    tmp_path: Path,
    isolated_env: Any,
    run_eval_studio_subprocess: Callable[..., Any],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T4-06: Space/Unicode workspace receipt audit, `reproduce_command` parity, tamper check."""
    pytest.importorskip("fastapi")
    server_mod = pytest.importorskip("eval_studio.web.server")
    testclient_mod = pytest.importorskip("fastapi.testclient")

    space_ws = tmp_path / "Clinical Eval (Café v1)"
    space_receipts = space_ws / "receipts"
    space_tasks = space_ws / "tasks"
    space_receipts.mkdir(parents=True, exist_ok=True)
    space_tasks.mkdir(parents=True, exist_ok=True)

    csv_path = space_ws / "café_samples.csv"
    csv_path.write_text(
        (isolated_env.examples_dir / "tier1_zero_code" / "sample.csv").read_text("utf-8"),
        encoding="utf-8",
    )
    task_path = space_tasks / "café task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "name": "cafe_parity_task",
                "version": "1.0.0",
                "trust_tier": "tier1_zero_code",
                "output_type": "generate_until",
                "dataset_path": str(csv_path),
                "doc_to_text": "Q: {{ input_text }}\nA:",
                "doc_to_target": "{{ target }}",
                "metrics": ["exact_match"],
                "metadata": {"date_stamp": "2026-11-23"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    sig = inspect.signature(server_mod.create_app)
    kwargs: dict[str, Any] = {}
    if "workspace_dir" in sig.parameters:
        kwargs["workspace_dir"] = space_ws
    if "receipts_dir" in sig.parameters:
        kwargs["receipts_dir"] = space_receipts
    if "tasks_dir" in sig.parameters:
        kwargs["tasks_dir"] = space_tasks
    if "cache_dir" in sig.parameters:
        kwargs["cache_dir"] = isolated_env.cache_dir
    app = server_mod.create_app(**kwargs)

    with testclient_mod.TestClient(app) as sc:
        launch = sc.post(
            "/api/runs",
            json={
                "task_path": str(task_path),
                "model_backend": "mock",
                "model_id": "mock-deterministic",
            },
        )
        assert launch.status_code in (200, 202)
        done = poll_run_until_done(sc, launch.json()["run_id"], 15.0)
        rid1 = Path(done["receipt_dir"]).name

        assert sc.post(f"/api/receipts/{rid1}/verify").json()["valid"] is True
        detail = sc.get(f"/api/receipts/{rid1}").json()
        repro_cmd = detail["manifest"]["reproduce_command"]
        tokens = shlex.split(repro_cmd)
        assert tokens[:2] == ["eval-studio", "run"]

        # Re-execute reproduce_command to a second receipt directory inside space_receipts
        repro_dir = space_receipts / "repro_copy"
        out_idx = tokens.index("--output")
        tokens[out_idx + 1] = str(repro_dir)
        proc = run_eval_studio_subprocess(tokens[1:], cwd=space_ws)
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

        cmp_resp = sc.post(
            "/api/receipts/compare",
            json={"receipt_ids": [rid1, "repro_copy"], "baseline_id": rid1},
        )
        assert cmp_resp.status_code == 200
        pair = cmp_resp.json()["pairwise"][0]
        assert pair["noise_floor"]["flip_rate"] == pytest.approx(0.0)
        assert pair["item_diffs"] == []

        # Tamper copy 1 samples.jsonl and verify failure
        samples_file = Path(done["receipt_dir"]) / "samples.jsonl"
        samples_file.write_bytes(samples_file.read_bytes() + b"\n")
        tamper_ver = sc.post(f"/api/receipts/{rid1}/verify").json()
        assert tamper_ver["valid"] is False
