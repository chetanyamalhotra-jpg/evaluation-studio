"""SHA-256 request fingerprinting and user-owned cache directory resolution."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

from eval_studio.core.schema import EvalRequest, TaskSpec


def resolve_cache_dir(override: Path | None = None) -> Path:
    """Resolve user-owned cache directory (never inside site-packages)."""
    if override is not None:
        return override
    if env_dir := os.environ.get("EVAL_STUDIO_CACHE_DIR"):
        return Path(env_dir)
    return Path(user_cache_dir("eval-studio", "evaluation-studio"))


def compute_task_hash(task_source: str | Mapping[str, Any] | TaskSpec) -> str:
    """Compute a deterministic SHA-256 hash for a task YAML string, mapping, or TaskSpec."""
    if isinstance(task_source, str):
        payload_bytes = task_source.encode("utf-8")
        return hashlib.sha256(payload_bytes).hexdigest()
    if isinstance(task_source, TaskSpec):
        raw: Any = task_source.model_dump(mode="json")
    else:
        raw = dict(task_source)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_request_fingerprint(
    *,
    task_hash: str,
    model_id: str,
    backend: str,
    request_payload: dict[str, Any],
    quantization: str | None = None,
    dtype: str | None = None,
    gen_kwargs: Mapping[str, Any] | None = None,
    prompt_template: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Compute a deterministic SHA-256 cache key that invalidates on any prompt/config edit."""
    canonical_payload: dict[str, Any] = {
        "task_hash": task_hash,
        "model_id": model_id,
        "backend": backend,
        "quantization": quantization,
        "dtype": dtype,
        "prompt_template": prompt_template,
        "gen_kwargs": dict(gen_kwargs) if gen_kwargs is not None else {},
        "request": request_payload,
    }
    if extra:
        canonical_payload["extra"] = dict(extra)
    canonical = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_eval_request(
    request: EvalRequest,
    *,
    task_hash: str,
    model_id: str,
    backend: str,
    quantization: str | None = None,
    dtype: str | None = None,
    prompt_template: str | None = None,
) -> str:
    """Compute a deterministic SHA-256 cache key directly from an EvalRequest."""
    payload = request.model_dump(
        mode="json",
        exclude={
            "request_id": True,
            "sample_id": True,
            "limits": {"max_cost_usd", "wall_clock_seconds"},
        },
    )
    return compute_request_fingerprint(
        task_hash=task_hash,
        model_id=model_id,
        backend=backend,
        request_payload=payload,
        quantization=quantization,
        dtype=dtype,
        gen_kwargs=request.generation_kwargs,
        prompt_template=prompt_template,
    )
