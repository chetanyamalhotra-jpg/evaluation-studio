"""Adversarial tests for F12/F26 Windows SQLite handle closure, WAL recovery, and cache safety."""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
from pathlib import Path

from eval_studio.cache.sqlite_store import SQLiteCacheStore
from eval_studio.cache.wal import JsonlWriteAheadLog


def test_windows_sqlite_cache_closes_handles_for_immediate_unlink(tmp_path: Path) -> None:
    cache_dir = tmp_path / "win_sqlite"
    store = SQLiteCacheStore.for_rank(cache_dir, rank=0)
    store.put("fp-win-1", {"text": "Unicode: café / 日本語"})
    assert store.get("fp-win-1") == {"text": "Unicode: café / 日本語"}

    # Must succeed immediately on Windows without gc.collect() or PermissionError
    store.db_path.unlink()
    shutil.rmtree(cache_dir)
    assert not cache_dir.exists()


def test_windows_wal_utf8_and_immediate_cleanup(tmp_path: Path) -> None:
    wal_dir = tmp_path / "win_wal"
    wal = JsonlWriteAheadLog(wal_dir / "events.wal.jsonl")
    wal.append({"id": 1, "text": "İstanbul — Straße — café"})
    assert wal.read_all() == [{"id": 1, "text": "İstanbul — Straße — café"}]
    wal.path.unlink()
    shutil.rmtree(wal_dir)
    assert not wal_dir.exists()


def test_sqlite_cache_redacts_secrets_before_disk_persistence(tmp_path: Path) -> None:
    cache_dir = tmp_path / "secret_sqlite"
    store = SQLiteCacheStore.for_rank(cache_dir, rank=0)
    secret_key = "sk-abcdefghijklmnopqrstuvwx"
    bearer_tok = "Bearer my_super_secret_token_123456"
    store.put(
        "fp-secret-1",
        {"error": f"Failed auth with {secret_key} and {bearer_tok}", "text": "ok"},
    )

    with contextlib.closing(sqlite3.connect(store.db_path)) as conn:
        raw_rows = conn.execute("SELECT payload_json FROM responses").fetchall()

    assert len(raw_rows) == 1
    raw_json = raw_rows[0][0]
    assert secret_key not in raw_json
    assert "my_super_secret_token_123456" not in raw_json
    assert "[REDACTED_SECRET]" in raw_json
