"""Unit tests for F12 WAL recovery, per-rank SQLite caching, and SHA-256 fingerprinting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from eval_studio.cache.fingerprint import (
    compute_request_fingerprint,
    compute_task_hash,
    fingerprint_eval_request,
    resolve_cache_dir,
)
from eval_studio.cache.sqlite_store import SQLiteCacheStore
from eval_studio.cache.wal import JsonlWriteAheadLog
from eval_studio.core.schema import EvalRequest, OutputType, RuntimeLimits


def test_resolve_cache_dir_respects_env_and_avoids_site_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custom = tmp_path / "custom_cache"
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(custom))
    assert resolve_cache_dir() == custom

    monkeypatch.delenv("EVAL_STUDIO_CACHE_DIR", raising=False)
    default_dir = resolve_cache_dir()
    assert "site-packages" not in str(default_dir).lower()


def test_no_pickle_or_dill_in_cache_package() -> None:
    for mod_name in ("fingerprint", "sqlite_store", "wal"):
        mod = sys.modules[f"eval_studio.cache.{mod_name}"]
        assert "pickle" not in vars(mod)
        assert "dill" not in vars(mod)


def test_fingerprint_invalidates_on_any_config_change() -> None:
    task_hash_a = compute_task_hash("name: t1\ndoc_to_text: Q {{ input_text }}")
    task_hash_b = compute_task_hash("name: t1\ndoc_to_text: Q: {{ input_text }}")
    assert task_hash_a != task_hash_b

    req = EvalRequest(
        request_id="r1",
        sample_id="s1",
        task_name="t1",
        output_type=OutputType.GENERATE_UNTIL,
        context="Hello",
        generation_kwargs={"temperature": 0.0},
    )
    fp_base = fingerprint_eval_request(
        req,
        task_hash=task_hash_a,
        model_id="m1",
        backend="mock",
        quantization="fp8",
        dtype="bfloat16",
        prompt_template="{{ input_text }}",
    )
    fp_quant = fingerprint_eval_request(
        req,
        task_hash=task_hash_a,
        model_id="m1",
        backend="mock",
        quantization="int4",
        dtype="bfloat16",
        prompt_template="{{ input_text }}",
    )
    fp_prompt = compute_request_fingerprint(
        task_hash=task_hash_a,
        model_id="m1",
        backend="mock",
        request_payload={"context": "Hello"},
        prompt_template="Question: {{ input_text }}",
    )
    assert fp_base != fp_quant
    assert fp_base != fp_prompt

    req_budget_a = EvalRequest(
        request_id="r1",
        sample_id="s1",
        task_name="t1",
        output_type=OutputType.GENERATE_UNTIL,
        context="Hello",
        limits=RuntimeLimits(max_cost_usd=1.0, wall_clock_seconds=30.0, max_tokens=128),
    )
    req_budget_b = EvalRequest(
        request_id="r2",
        sample_id="s2",
        task_name="t1",
        output_type=OutputType.GENERATE_UNTIL,
        context="Hello",
        limits=RuntimeLimits(max_cost_usd=10.0, wall_clock_seconds=120.0, max_tokens=128),
    )
    req_tokens_changed = EvalRequest(
        request_id="r3",
        sample_id="s1",
        task_name="t1",
        output_type=OutputType.GENERATE_UNTIL,
        context="Hello",
        limits=RuntimeLimits(max_cost_usd=10.0, wall_clock_seconds=120.0, max_tokens=256),
    )
    fp_budget_a = fingerprint_eval_request(
        req_budget_a, task_hash=task_hash_a, model_id="m1", backend="mock"
    )
    fp_budget_b = fingerprint_eval_request(
        req_budget_b, task_hash=task_hash_a, model_id="m1", backend="mock"
    )
    fp_tokens = fingerprint_eval_request(
        req_tokens_changed, task_hash=task_hash_a, model_id="m1", backend="mock"
    )
    assert fp_budget_a == fp_budget_b
    assert fp_budget_a != fp_tokens


def test_sqlite_store_do_sample_bypass_and_per_rank(tmp_path: Path) -> None:
    store0 = SQLiteCacheStore.for_rank(tmp_path, rank=0)
    store1 = SQLiteCacheStore.for_rank(tmp_path, rank=1)
    assert store0.db_path.name == "responses_rank_0.sqlite3"
    assert store1.db_path.name == "responses_rank_1.sqlite3"

    store0.put("fp1", {"text": "deterministic"}, do_sample=False)
    assert store0.get("fp1", do_sample=False) == {"text": "deterministic"}
    assert store0.get("fp1", do_sample=True) is None
    assert store1.get("fp1", do_sample=False) is None

    store0.put("fp2", {"text": "stochastic"}, do_sample=True)
    assert store0.get("fp2", do_sample=False) is None


def test_wal_append_redacts_and_recovers_partial_trailing_line(tmp_path: Path) -> None:
    wal = JsonlWriteAheadLog(tmp_path / "run.wal.jsonl")
    wal.append({"sample_id": "s1", "text": "ok sk-abcdefghijklmnopqrstuv", "note": "café"})
    wal.append({"sample_id": "s2", "text": "second"})

    # Simulate partial crash write on the last line
    with wal.path.open("a", encoding="utf-8") as fh:
        fh.write('{"sample_id": "s3", "truncated": ')

    with pytest.warns(RuntimeWarning, match="truncated trailing WAL line"):
        records = wal.read_all()

    assert len(records) == 2
    assert "sk-abcdefghijklmnopqrstuv" not in json.dumps(records)
    assert records[0]["note"] == "café"
