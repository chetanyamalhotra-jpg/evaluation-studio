"""Crash-safe fsynced JSONL write-ahead log (no pickle/dill)."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

from eval_studio.gateway.secrets import redact_secrets


class JsonlWriteAheadLog:
    """Append-only, fsynced JSONL WAL with automatic secret redaction."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        """Serialize, redact, append, and fsync a single JSON record."""
        line = redact_secrets(json.dumps(record, sort_keys=True, ensure_ascii=False))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read_all(self) -> list[dict[str, Any]]:
        """Recover all complete JSON records from the WAL, skipping a partial trailing line."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            raw_lines = [line.strip() for line in fh if line.strip()]
        records: list[dict[str, Any]] = []
        last_idx = len(raw_lines) - 1
        for idx, raw_line in enumerate(raw_lines):
            try:
                parsed: Any = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                if idx == last_idx:
                    warnings.warn(
                        f"Skipping truncated trailing WAL line in '{self.path}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break
                raise ValueError(
                    f"Corrupted WAL record at line {idx + 1} in '{self.path}': {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"WAL record at line {idx + 1} in '{self.path}' must be a JSON object."
                )
            records.append(parsed)
        return records
