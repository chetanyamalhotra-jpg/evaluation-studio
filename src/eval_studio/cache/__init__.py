"""Crash-safe user-dir WAL + SQLite caching and SHA-256 request fingerprinting."""

from eval_studio.cache.fingerprint import (
    compute_request_fingerprint,
    compute_task_hash,
    fingerprint_eval_request,
    resolve_cache_dir,
)
from eval_studio.cache.sqlite_store import SQLiteCacheStore
from eval_studio.cache.wal import JsonlWriteAheadLog

__all__ = [
    "JsonlWriteAheadLog",
    "SQLiteCacheStore",
    "compute_request_fingerprint",
    "compute_task_hash",
    "fingerprint_eval_request",
    "resolve_cache_dir",
]
