"""SQLite key-value cache store (JSON values only, never pickle/dill)."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from eval_studio.gateway.secrets import redact_secrets


class SQLiteCacheStore:
    """User-dir SQLite response cache keyed by SHA-256 request fingerprints."""

    def __init__(self, db_path: Path, *, rank: int = 0) -> None:
        if rank < 0:
            raise ValueError("rank must be >= 0.")
        self.rank = rank
        if rank > 0 and f"_rank_{rank}" not in db_path.stem:
            self.db_path = db_path.with_name(f"{db_path.stem}_rank_{rank}{db_path.suffix}")
        else:
            self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @classmethod
    def for_rank(cls, cache_dir: Path, *, rank: int = 0) -> SQLiteCacheStore:
        """Construct a per-rank SQLiteCacheStore inside the resolved cache directory."""
        if rank < 0:
            raise ValueError("rank must be >= 0.")
        return cls(cache_dir / f"responses_rank_{rank}.sqlite3", rank=rank)

    def _init_schema(self) -> None:
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "fingerprint TEXT PRIMARY KEY, "
                "payload_json TEXT NOT NULL)"
            )

    def get(self, fingerprint: str, *, do_sample: bool = False) -> dict[str, Any] | None:
        """Look up a cached response payload; always bypasses when do_sample=True."""
        if do_sample:
            return None
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
            row = conn.execute(
                "SELECT payload_json FROM responses WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
        if row is None:
            return None
        parsed: dict[str, Any] = json.loads(row[0])
        return parsed

    def put(
        self,
        fingerprint: str,
        payload: dict[str, Any],
        *,
        do_sample: bool = False,
    ) -> None:
        """Persist a cached response payload; always bypasses when do_sample=True."""
        if do_sample:
            return
        serialized = redact_secrets(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses (fingerprint, payload_json) VALUES (?, ?)",
                (fingerprint, serialized),
            )
