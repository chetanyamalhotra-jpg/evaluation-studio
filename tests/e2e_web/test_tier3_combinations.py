"""Tier 3 E2E tests: Cross-feature pairwise integration flows (13 scenarios)."""

from __future__ import annotations

import io
import json
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

pytestmark = pytest.mark.e2e_web


def test_t3_01_byot_upload_save_validate_estimate_run_verify_rescore(
    client: Any,
    parse_sse_frames: Callable[[str], list[dict[str, Any]]],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T3-01: BYOT Upload -> Save -> Validate -> Estimate -> Run + SSE -> Verify -> Rescore."""
    csv_bytes = (
        b"sample_id,input_text,target\n"
        b"s1,First-line treatment for anaphylaxis?,A\n"
        b"s2,Drug causing gingival hyperplasia?,A\n"
        b"s3,Vitamin for scurvy?,A\n"
        b"s4,Organ secreting insulin?,A\n"
    )
    up = client.post(
        "/api/byot/upload-dataset",
        files={"file": ("t3_four.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert up.status_code == 200
    dataset_path = up.json()["dataset_path"]

    yaml_text = yaml.safe_dump(
        {
            "name": "t3_byot_flow",
            "version": "1.0.0",
            "trust_tier": "tier2_declarative",
            "output_type": "generate_until",
            "dataset_path": dataset_path,
            "doc_to_text": "Q: {{ input_text }}\nAnswer (A/B):",
            "doc_to_target": "{{ target }}",
            "metrics": ["exact_match"],
            "filter_ensembles": [
                {
                    "name": "default",
                    "steps": [
                        {"name": "multi_choice_regex", "params": {"choices": ["A", "B"]}},
                        {"name": "take_first", "params": {}},
                    ],
                }
            ],
            "metadata": {"date_stamp": "2026-11-23"},
        },
        sort_keys=False,
    )

    save = client.post(
        "/api/byot/save",
        json={"filename": "t3_byot_flow.yaml", "yaml_text": yaml_text},
    )
    assert save.status_code == 200
    saved_path = save.json()["path"]

    val = client.post("/api/byot/validate", json={"yaml_text": yaml_text})
    assert val.status_code == 200
    assert val.json()["valid"] is True

    est = client.post(
        "/api/runs/estimate",
        json={
            "task_path": saved_path,
            "model_backend": "mock",
            "model_id": "mock-deterministic",
        },
    )
    assert est.status_code == 200
    assert est.json()["n_samples"] == 4

    launch = client.post(
        "/api/runs",
        json={
            "task_path": saved_path,
            "model_backend": "mock",
            "model_id": "mock-deterministic",
        },
    )
    assert launch.status_code in (200, 202)
    run_id = launch.json()["run_id"]

    with client.stream("GET", f"/api/runs/{run_id}/events") as stream_resp:
        frames = parse_sse_frames("".join(stream_resp.iter_text()))
    assert [f["event"] for f in frames][-1] == "completed"

    done = poll_run_until_done(client, run_id, 15.0)
    receipt_id = Path(done["receipt_dir"]).name

    ver1 = client.post(f"/api/receipts/{receipt_id}/verify")
    assert ver1.status_code == 200
    assert ver1.json()["valid"] is True

    resc = client.post(
        f"/api/receipts/{receipt_id}/rescore",
        json={"metrics": ["exact_match", "token_f1"], "output_name": f"{receipt_id}_rescored"},
    )
    assert resc.status_code == 200
    resc_id = Path(resc.json()["path"]).name
    ver2 = client.post(f"/api/receipts/{resc_id}/verify")
    assert ver2.status_code == 200
    assert ver2.json()["valid"] is True


def test_t3_02_byop_upload_compile_lock_run_two_models_and_compare_categories(
    client: Any,
    isolated_env: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T3-02: BYOP Upload -> Compile -> Lock -> 2-Model Run -> Compare `category_breakdown`."""
    paper_bytes = (isolated_env.examples_dir / "papers" / "sample_paper.md").read_bytes()
    up = client.post(
        "/api/byop/upload",
        files={"file": ("sample_copy.md", io.BytesIO(paper_bytes), "text/markdown")},
    )
    assert up.status_code == 200
    paper_path = up.json()["paper_path"]

    comp = client.post(
        "/api/byop/compile",
        json={"paper_path": paper_path, "suite_name": "t3_paper_cmp", "mode": "mock"},
    )
    assert comp.status_code == 200
    items = comp.json()["approved_items"]
    assert len(items) == 2
    items[0] = {**items[0], "rubric": "Edited rubric for 99.6% NPV"}

    lock = client.post(
        "/api/byop/lock",
        json={
            "paper_path": paper_path,
            "suite_name": "t3_paper_cmp",
            "extractor_model": "mock-extractor",
            "verifier_model": "mock-verifier",
            "items": items,
        },
    )
    assert lock.status_code == 200
    locked_yaml = lock.json()["yaml_path"]

    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": "sk-t3-byop-123456", "storage": "session"},
    )
    phase = {"model": "alpha"}

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        msg = request.content.decode("utf-8")
        if "99.6%" in msg or "negative predictive value" in msg.lower():
            ans = "99.6%"
        else:
            ans = "> 0.04 ng/mL" if phase["model"] == "alpha" else "> 0.10 ng/mL"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ans}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            },
        )

    transport, _ = make_mock_transport(_handler)
    bind_mock_transport(transport)

    r1 = client.post(
        "/api/runs",
        json={"task_path": locked_yaml, "model_backend": "http-byok", "model_id": "mock-a"},
    )
    d1 = poll_run_until_done(client, r1.json()["run_id"], 15.0)
    phase["model"] = "beta"
    r2 = client.post(
        "/api/runs",
        json={"task_path": locked_yaml, "model_backend": "http-byok", "model_id": "mock-b"},
    )
    d2 = poll_run_until_done(client, r2.json()["run_id"], 15.0)

    id1 = Path(d1["receipt_dir"]).name
    id2 = Path(d2["receipt_dir"]).name
    cmp_resp = client.post(
        "/api/receipts/compare",
        json={"receipt_ids": [id1, id2], "baseline_id": id1, "bootstrap_iters": 100},
    )
    assert cmp_resp.status_code == 200
    cmp_body = cmp_resp.json()
    cats_text = json.dumps(cmp_body["category_breakdown"])
    assert "quantitative_finding" in cats_text
    assert "claim" in cats_text
    assert len(cmp_body["pairwise"][0]["item_diffs"]) == 1


def test_t3_03_byok_save_test_redacted_error_and_delete(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
) -> None:
    """T3-03: Save session key -> masked status -> failing test redacts token -> delete key."""
    secret = "sk-t3-redacted-9876543210"
    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": secret, "storage": "session"},
    )
    st = client.get("/api/byok/status").json()
    target_row = next(r for r in st["roles"] if r["role"] == "TARGET_API_KEY")
    assert target_row["masked_hint"] == "sk-...3210"

    def _fail(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(403, json={"error": f"Invalid key Bearer {secret}"})

    transport, _ = make_mock_transport(_fail)
    bind_mock_transport(transport)

    ping = client.post(
        "/api/byok/test",
        json={
            "backend": "http-byok",
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "role": "TARGET_API_KEY",
        },
    )
    assert ping.status_code == 200
    assert ping.json()["ok"] is False
    assert secret not in ping.text
    assert "[REDACTED_SECRET]" in (ping.json().get("error") or "")

    deleted = client.delete("/api/byok/keys/TARGET_API_KEY?storage=session")
    assert deleted.status_code == 200
    assert deleted.json()["source"] == "unset"


def test_t3_04_byok_to_byop_http_byok_compile_uses_separate_extractor_and_judge_keys(
    client: Any,
    isolated_env: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
) -> None:
    """T3-04: BYOP `http-byok` compile routes `EXTRACTOR_API_KEY` & `JUDGE_API_KEY` cleanly."""
    ext_key = "sk-ext-role-111122223333"
    ver_key = "sk-ver-role-444455556666"
    client.post(
        "/api/byok/keys",
        json={"role": "EXTRACTOR_API_KEY", "api_key": ext_key, "storage": "session"},
    )
    client.post(
        "/api/byok/keys",
        json={"role": "JUDGE_API_KEY", "api_key": ver_key, "storage": "session"},
    )
    paper_path = isolated_env.examples_dir / "papers" / "sample_paper.md"

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        if len(recorded) == 1:
            content = json.dumps(
                {
                    "items": [
                        {
                            "item_id": "ext-1",
                            "question": "What NPV did serial troponin testing achieve?",
                            "reference_answer": "99.6%",
                            "rubric": "State 99.6% NPV.",
                            "category": "quantitative_finding",
                            "citations": [
                                {
                                    "page": 1,
                                    "section": "Section 2: Cohort Findings",
                                    "quote": "achieved a negative predictive value of 99.6%",
                                }
                            ],
                        }
                    ]
                }
            )
        else:
            content = json.dumps({"approved": True, "reasons": []})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 10},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    bind_mock_transport(transport)

    comp = client.post(
        "/api/byop/compile",
        json={
            "paper_path": str(paper_path),
            "suite_name": "t3_http_byop",
            "mode": "http-byok",
            "provider": "openai",
            "extractor_model": "gpt-4o-mini",
            "verifier_model": "gpt-4o-mini",
        },
    )
    assert comp.status_code == 200
    assert len(comp.json()["approved_items"]) == 1
    assert ext_key not in comp.text and ver_key not in comp.text
    assert recorded[0]["headers"].get("authorization") == f"Bearer {ext_key}"
    assert recorded[1]["headers"].get("authorization") == f"Bearer {ver_key}"

    lock = client.post(
        "/api/byop/lock",
        json={
            "paper_path": str(paper_path),
            "suite_name": "t3_http_byop",
            "extractor_model": "gpt-4o-mini",
            "verifier_model": "gpt-4o-mini",
            "items": comp.json()["approved_items"],
        },
    )
    assert lock.status_code == 200
    assert ext_key not in lock.text and ver_key not in lock.text


def test_t3_05_byok_to_runs_http_byok_target_and_judge_redacted_receipt(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T3-05: Role-separated target + judge keys execute `http-byok` run with zero leakage."""
    t_key = "sk-ds-target-1234567890"
    j_key = "sk-oai-judge-0987654321"
    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": t_key, "storage": "session"},
    )
    client.post(
        "/api/byok/keys",
        json={"role": "JUDGE_API_KEY", "api_key": j_key, "storage": "session"},
    )

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        if "api.deepseek.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "Vitamin C",
                                "reasoning_content": "Scurvy results from ascorbate deficiency.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 6},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"score": 1.0, "rationale": "Exact match."}'}}
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 8},
            },
        )

    transport, recorded = make_mock_transport(_handler)
    bind_mock_transport(transport)

    launch = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "http-byok",
            "model_id": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "judge_model": "gpt-4o-mini",
            "judge_base_url": "https://api.openai.com/v1",
            "limit": 1,
        },
    )
    assert launch.status_code in (200, 202)
    done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    assert done["status"] == "completed"
    assert "llm_judge" in done["manifest"]["metrics"]

    rdir = Path(done["receipt_dir"])
    for fname in ("manifest.json", "samples.jsonl", "report.md"):
        content = (rdir / fname).read_text(encoding="utf-8")
        assert t_key not in content
        assert j_key not in content
    first = json.loads((rdir / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["raw_responses"][0]["reasoning_trace"]
    assert any(r["headers"].get("authorization") == f"Bearer {t_key}" for r in recorded)
    assert any(r["headers"].get("authorization") == f"Bearer {j_key}" for r in recorded)


def test_t3_06_byot_calibrate_run_and_rescore_with_fallback_judge(
    client: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T3-06: Calibrate aligned judge -> run `judge_model='mock'` -> rescore w/ fallback."""
    cal = client.post(
        "/api/byot/calibrate",
        json={
            "rows": [
                {"judge_label": 1, "human_label": 1},
                {"judge_label": 0, "human_label": 0},
                {"judge_label": 1, "human_label": 1},
                {"judge_label": 0, "human_label": 0},
            ]
        },
    )
    assert cal.status_code == 200
    assert cal.json()["status"] == "ALIGNED"

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
    rid = Path(done["receipt_dir"]).name

    resc = client.post(
        f"/api/receipts/{rid}/rescore",
        json={
            "judge_model": "mock-refusal",
            "fallback_judge_model": "mock",
            "output_name": f"{rid}_fb_judge",
        },
    )
    assert resc.status_code == 200
    assert resc.json()["manifest"]["metrics"]["llm_judge"]["value"] == pytest.approx(1.0)


def test_t3_07_byot_registry_and_custom_think_strip_filter_rescore(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T3-07: Registry lists `think_strip`; rescoring with `think_strip` fixes exact_match."""
    regs = client.get("/api/byot/registries").json()
    assert "think_strip" in regs["filters"]
    assert "exact_match" in regs["metrics"]

    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": "sk-t3-strip-123456", "storage": "session"},
    )

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        msg = request.content.decode("utf-8").lower()
        ans = "Vitamin C" if "scurvy" in msg else ("60-100" if "heart" in msg else "Pancreas")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": f"<think>scratchpad</think>{ans}"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
        )

    transport, _ = make_mock_transport(_handler)
    bind_mock_transport(transport)

    launch = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "http-byok",
            "model_id": "gpt-4o-mini",
            "enable_thinking": False,
        },
    )
    done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    assert done["manifest"]["metrics"]["exact_match"]["value"] == pytest.approx(0.0)
    rid = Path(done["receipt_dir"]).name

    resc = client.post(
        f"/api/receipts/{rid}/rescore",
        json={
            "filter_ensembles": [
                {
                    "name": "stripped",
                    "steps": [{"name": "think_strip", "params": {"think_end_token": "</think>"}}],
                }
            ],
            "metrics": ["exact_match"],
            "output_name": f"{rid}_stripped",
        },
    )
    assert resc.status_code == 200
    assert resc.json()["manifest"]["metrics"]["exact_match"]["value"] == pytest.approx(1.0)


def test_t3_08_cache_stats_and_prompt_fingerprint_invalidation(
    client: Any,
    isolated_env: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T3-08: Repeated run populates WAL/SQLite cache; editing prompt invalidates cache hit."""
    base_yaml = {
        "name": "cache_invalidation_task",
        "version": "1.0.0",
        "trust_tier": "tier1_zero_code",
        "output_type": "generate_until",
        "dataset_path": "examples/tier1_zero_code/sample.csv",
        "doc_to_text": "Prompt v1: {{ input_text }}\nAnswer:",
        "doc_to_target": "{{ target }}",
        "metrics": ["exact_match"],
        "metadata": {"date_stamp": "2026-11-23"},
    }
    s1 = client.post(
        "/api/byot/save",
        json={"filename": "cache_task.yaml", "yaml_text": yaml.safe_dump(base_yaml)},
    )
    task_path = s1.json()["path"]

    r1 = client.post(
        "/api/runs",
        json={"task_path": task_path, "model_backend": "mock", "model_id": "mock-deterministic"},
    )
    d1 = poll_run_until_done(client, r1.json()["run_id"], 15.0)
    st1 = client.get("/api/byok/status").json()["cache_stats"]
    assert st1["exists"] is True
    assert st1["wal_entries"] >= 3

    # Repeat identical run: cache hit keeps wal_entries unchanged
    r2 = client.post(
        "/api/runs",
        json={"task_path": task_path, "model_backend": "mock", "model_id": "mock-deterministic"},
    )
    poll_run_until_done(client, r2.json()["run_id"], 15.0)
    st2 = client.get("/api/byok/status").json()["cache_stats"]
    assert st2["wal_entries"] == st1["wal_entries"]

    # Edit prompt template: fingerprint changes -> cache misses -> wal_entries increases
    base_yaml["doc_to_text"] = "Prompt v2 revised: {{ input_text }}\nAnswer:"
    client.post(
        "/api/byot/save",
        json={"filename": "cache_task.yaml", "yaml_text": yaml.safe_dump(base_yaml)},
    )
    r3 = client.post(
        "/api/runs",
        json={"task_path": task_path, "model_backend": "mock", "model_id": "mock-deterministic"},
    )
    d3 = poll_run_until_done(client, r3.json()["run_id"], 15.0)
    st3 = client.get("/api/byok/status").json()["cache_stats"]
    assert st3["wal_entries"] > st2["wal_entries"]

    p1 = json.loads((Path(d1["receipt_dir"]) / "samples.jsonl").read_text("utf-8").splitlines()[0])
    p3 = json.loads((Path(d3["receipt_dir"]) / "samples.jsonl").read_text("utf-8").splitlines()[0])
    assert p1["prompt_hash"] != p3["prompt_hash"]


def test_t3_09_noise_floor_and_compare_pairwise_flip_rate_parity(
    client: Any,
    make_mock_transport: Callable[..., tuple[httpx.MockTransport, list[dict[str, Any]]]],
    bind_mock_transport: Callable[[httpx.MockTransport], None],
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T3-09: Standalone `/noise-floor` flip rate matches `/compare` pairwise noise_floor."""
    client.post(
        "/api/byok/keys",
        json={"role": "TARGET_API_KEY", "api_key": "sk-t3-nf-123456", "storage": "session"},
    )
    state = {"pert": False}

    def _handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
        msg = request.content.decode("utf-8").lower()
        if "scurvy" in msg:
            ans = "Vitamin A" if state["pert"] else "Vitamin C"
        elif "heart" in msg:
            ans = "60-100"
        else:
            ans = "Pancreas"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ans}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    transport, _ = make_mock_transport(_handler)
    bind_mock_transport(transport)

    r1 = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "http-byok",
            "model_id": "m-base",
            "repeats": 1,
        },
    )
    d1 = poll_run_until_done(client, r1.json()["run_id"], 15.0)
    state["pert"] = True
    r2 = client.post(
        "/api/runs",
        json={
            "task_path": "examples/tier1_zero_code/task.yaml",
            "model_backend": "http-byok",
            "model_id": "m-pert",
            "repeats": 1,
        },
    )
    d2 = poll_run_until_done(client, r2.json()["run_id"], 15.0)

    id1 = Path(d1["receipt_dir"]).name
    id2 = Path(d2["receipt_dir"]).name
    nf = client.post(
        "/api/receipts/noise-floor",
        json={"baseline_id": id1, "perturbed_id": id2, "perturbation": "seed_change"},
    ).json()
    cmp_body = client.post(
        "/api/receipts/compare",
        json={"receipt_ids": [id1, id2], "baseline_id": id1, "bootstrap_iters": 100},
    ).json()
    assert cmp_body["pairwise"][0]["noise_floor"]["flip_rate"] == pytest.approx(nf["flip_rate"])


def test_t3_10_byop_lock_run_and_verify_detects_tampered_source_paper(
    client: Any,
    isolated_env: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T3-10: Mutating the source paper on disk after BYOP lock+run fails `verify-receipt`."""
    paper_copy = isolated_env.workspace_dir / "mutable_paper.md"
    paper_copy.write_text(
        (isolated_env.examples_dir / "papers" / "sample_paper.md").read_text("utf-8"),
        encoding="utf-8",
    )
    comp = client.post(
        "/api/byop/compile",
        json={"paper_path": str(paper_copy), "suite_name": "tamper_paper", "mode": "heuristic"},
    )
    lock = client.post(
        "/api/byop/lock",
        json={
            "paper_path": str(paper_copy),
            "suite_name": "tamper_paper",
            "extractor_model": "heuristic",
            "verifier_model": "heuristic",
            "items": comp.json()["approved_items"],
        },
    )
    yaml_path = lock.json()["yaml_path"]

    launch = client.post(
        "/api/runs",
        json={"task_path": yaml_path, "model_backend": "mock", "model_id": "mock-deterministic"},
    )
    done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    rid = Path(done["receipt_dir"]).name

    assert client.post(f"/api/receipts/{rid}/verify").json()["valid"] is True

    paper_copy.write_text(
        paper_copy.read_text("utf-8") + "\nTampered sentence.\n",
        encoding="utf-8",
    )
    ver_after = client.post(f"/api/receipts/{rid}/verify").json()
    assert ver_after["valid"] is False
    assert any("source paper sha-256 mismatch" in e.lower() for e in ver_after["errors"])


def test_t3_11_ephemeral_uvicorn_live_http_spa_and_sse_smoke(web_app: Any) -> None:
    """T3-11: Live TCP Uvicorn server handles ASGI lifespan, SPA, REST, and SSE framing."""
    uvicorn_mod = pytest.importorskip("uvicorn")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(5)

    config = uvicorn_mod.Config(web_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn_mod.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base_url, timeout=5.0) as hc:
            ready = False
            for _ in range(50):
                try:
                    if hc.get("/api/health").status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    time.sleep(0.05)
            assert ready, "Ephemeral uvicorn server failed to become ready"

            presets = hc.get("/api/byok/presets")
            assert presets.status_code == 200
            assert len(presets.json()["presets"]) >= 8

            health = hc.get("/api/health").json()
            if health.get("static_ready"):
                spa = hc.get("/compare")
                assert spa.status_code == 200
                assert spa.headers.get("content-type", "").startswith("text/html")

            launch = hc.post(
                "/api/runs",
                json={
                    "task_path": "examples/tier1_zero_code/task.yaml",
                    "model_backend": "mock",
                    "model_id": "mock-deterministic",
                },
            )
            assert launch.status_code in (200, 202)
            run_id = launch.json()["run_id"]

            with hc.stream("GET", f"/api/runs/{run_id}/events") as sse:
                assert sse.status_code == 200
                body = "".join(sse.iter_text())
            assert "run_started" in body
            assert "sample_scored" in body
            assert "completed" in body
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        sock.close()


def test_t3_12_byot_validation_warnings_run_and_space_safe_reproduce_command(
    client: Any,
    isolated_env: Any,
    poll_run_until_done: Callable[[Any, str, float], dict[str, Any]],
) -> None:
    """T3-12: Missing `date_stamp` produces warning (`valid=True`), run succeeds, cmd valid."""
    warn_yaml = yaml.safe_dump(
        {
            "name": "warn_only_task",
            "version": "1.0.0",
            "trust_tier": "tier1_zero_code",
            "output_type": "generate_until",
            "dataset_path": "examples/tier1_zero_code/sample.csv",
            "doc_to_text": "Q: {{ input_text }}\nA:",
            "doc_to_target": "{{ target }}",
            "metrics": ["exact_match"],
        },
        sort_keys=False,
    )
    val = client.post("/api/byot/validate", json={"yaml_text": warn_yaml})
    assert val.status_code == 200
    assert val.json()["valid"] is True
    assert len(val.json()["warnings"]) >= 1

    save = client.post(
        "/api/byot/save",
        json={"filename": "warn task.yaml", "yaml_text": warn_yaml},
    )
    launch = client.post(
        "/api/runs",
        json={
            "task_path": save.json()["path"],
            "model_backend": "mock",
            "model_id": "mock-deterministic",
        },
    )
    done = poll_run_until_done(client, launch.json()["run_id"], 15.0)
    rid = Path(done["receipt_dir"]).name
    detail = client.get(f"/api/receipts/{rid}").json()
    repro = detail["manifest"]["reproduce_command"]
    assert "eval-studio run" in repro
    assert '"tasks' in repro or "warn task.yaml" in repro


def test_t3_13_byop_upload_txt_edit_answer_lock_and_inspect_task_detail(
    client: Any,
) -> None:
    """T3-13: Upload `.txt` paper -> compile heuristic -> edit target -> lock -> task detail."""
    txt = (
        "# Sepsis Protocol\n\n"
        "## Section 1: Timing\n"
        "Administer broad-spectrum antibiotics within 1 hour of sepsis recognition.\n"
    )
    up = client.post(
        "/api/byop/upload",
        files={"file": ("sepsis.txt", io.BytesIO(txt.encode("utf-8")), "text/plain")},
    )
    assert up.status_code == 200
    paper_path = up.json()["paper_path"]

    comp = client.post(
        "/api/byop/compile",
        json={"paper_path": paper_path, "suite_name": "sepsis_suite", "mode": "heuristic"},
    )
    assert comp.status_code == 200
    items = comp.json()["approved_items"]
    assert len(items) >= 1
    items[0] = {**items[0], "reference_answer": "Within 1 hour"}

    lock = client.post(
        "/api/byop/lock",
        json={
            "paper_path": paper_path,
            "suite_name": "sepsis_suite",
            "extractor_model": "heuristic",
            "verifier_model": "heuristic",
            "items": items,
        },
    )
    assert lock.status_code == 200
    yaml_path = lock.json()["yaml_path"]

    detail = client.get("/api/byot/tasks/detail", params={"path": yaml_path})
    assert detail.status_code == 200
    assert detail.json()["sample_rows"][0]["target"] == "Within 1 hour"
