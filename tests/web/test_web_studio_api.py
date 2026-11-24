"""Focused backend API & security tests for Evaluation Studio (`src/eval_studio/web/`)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from eval_studio.gateway.secrets import SecretRole, clear_session_key
from eval_studio.web.server import create_app


def _make_client(tmp_path: Path) -> TestClient:
    app = create_app(
        workspace_dir=tmp_path,
        receipts_dir=tmp_path / "receipts",
        tasks_dir=tmp_path / "tasks",
        cache_dir=tmp_path / "cache",
        uploads_dir=tmp_path / "uploads" / "papers",
    )
    return TestClient(app)


def test_health_and_byok_key_masking_never_echoes_secret(tmp_path: Path) -> None:
    """Health check works and BYOK endpoints never echo raw API keys."""
    raw_secret = "sk-studio-unit-test-secret-9876543210"
    try:
        with _make_client(tmp_path) as client:
            health = client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            save = client.post(
                "/api/byok/keys",
                json={"role": SecretRole.TARGET.value, "api_key": raw_secret, "storage": "session"},
            )
            assert save.status_code == 200
            assert raw_secret not in save.text
            assert save.json()["configured"] is True
            assert save.json()["source"] == "session"

            status = client.get("/api/byok/status")
            assert status.status_code == 200
            assert raw_secret not in status.text
    finally:
        clear_session_key(SecretRole.TARGET)


def test_byop_compile_lock_and_offline_run_verify_rescore_flow(tmp_path: Path) -> None:
    """Smoke-test BYOP compile -> lock -> run -> verify -> offline rescore -> compare."""
    with _make_client(tmp_path) as client:
        papers = client.get("/api/byop/papers")
        assert papers.status_code == 200
        samples = papers.json()["samples"]
        sample_paper = next(
            (s for s in samples if s["name"] == "sample_paper.md"),
            samples[0],
        )

        comp = client.post(
            "/api/byop/compile",
            json={
                "paper_path": sample_paper["path"],
                "suite_name": "unit_paper_suite",
                "mode": "mock",
            },
        )
        assert comp.status_code == 200
        comp_body = comp.json()
        assert len(comp_body["approved_items"]) == 2
        assert len(comp_body["rejected_items"]) >= 2

        lock = client.post(
            "/api/byop/lock",
            json={
                "paper_path": sample_paper["path"],
                "suite_name": "unit_paper_suite",
                "items": comp_body["approved_items"],
            },
        )
        assert lock.status_code == 200
        yaml_path = lock.json()["yaml_path"]

        est = client.post(
            "/api/runs/estimate",
            json={
                "task_path": yaml_path,
                "model_backend": "mock",
                "model_id": "mock-deterministic",
            },
        )
        assert est.status_code == 200
        assert est.json()["n_samples"] == 2

        launch = client.post(
            "/api/runs",
            json={
                "task_path": yaml_path,
                "model_backend": "mock",
                "model_id": "mock-deterministic",
            },
        )
        assert launch.status_code == 200
        run_id = launch.json()["run_id"]
        out_dir_name = launch.json()["output_dir"]

        # Stream SSE until completion
        events_text = client.get(f"/api/runs/{run_id}/events").text
        assert "event: completed" in events_text

        verify = client.post(f"/api/receipts/{out_dir_name}/verify")
        assert verify.status_code == 200
        assert verify.json()["valid"] is True

        rescore = client.post(
            f"/api/receipts/{out_dir_name}/rescore",
            json={"output_name": f"{out_dir_name}-rescored"},
        )
        assert rescore.status_code == 200
        assert rescore.json()["receipt_id"] == f"{out_dir_name}-rescored"

        cmp_resp = client.post(
            "/api/receipts/compare",
            json={"receipt_ids": [out_dir_name, f"{out_dir_name}-rescored"]},
        )
        assert cmp_resp.status_code == 200
        assert len(cmp_resp.json()["runs"]) == 2
        assert len(cmp_resp.json()["pairwise"]) == 1
