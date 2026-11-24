"""BYOT 3-tier task loader (Tier 1 CSV/JSONL/Parquet, Tier 2 declarative YAML, Tier 3 plugins)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from eval_studio.core.schema import Sample, TaskSpec, TrustTier


class TaskLoadError(RuntimeError):
    """Raised when a task specification fails to load or violates trust boundaries."""


def _compute_doc_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_task_spec(path: Path, *, trust_remote_code: bool = False) -> TaskSpec:
    """Load a declarative TaskSpec YAML while enforcing Tier 3 opt-in and safe YAML rules."""
    if not path.is_file():
        raise TaskLoadError(f"Task file does not exist: {path}")
    raw_text = path.read_text(encoding="utf-8")
    try:
        data: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise TaskLoadError(
            f"Unsafe or invalid YAML (custom tags like !function are blocked in Tiers 1-2): {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise TaskLoadError(f"Task YAML must be a mapping: {path}")
    spec = TaskSpec.model_validate(data)
    if (
        spec.trust_tier == TrustTier.TIER3_CODE_PLUGIN or spec.unsafe_code
    ) and not trust_remote_code:
        raise TaskLoadError(
            f"Task '{spec.name}' is Tier 3 / unsafe_code and requires --trust-remote-code."
        )
    return spec


def _parse_choices(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                pass
        return [c.strip() for c in stripped.split("|") if c.strip()]
    return [str(raw)]


def _row_to_sample(row: dict[str, Any], index: int) -> Sample:
    sample_id = str(row.get("sample_id") or row.get("id") or f"sample_{index:04d}")
    input_text = str(
        row.get("input_text")
        or row.get("question")
        or row.get("prompt")
        or row.get("text")
        or ""
    )
    target: str | int | list[str]
    raw_target = (
        row["target"]
        if "target" in row
        else row.get("reference_answer", row.get("answer", row.get("label", "")))
    )
    target = raw_target if isinstance(raw_target, (int, list)) else str(raw_target)

    choices = _parse_choices(row.get("choices"))
    metadata = {
        k: v
        for k, v in row.items()
        if k not in {"sample_id", "input_text", "target", "choices"}
    }
    doc_hash = _compute_doc_hash(
        {
            "sample_id": sample_id,
            "input_text": input_text,
            "target": target,
            "choices": choices,
        }
    )
    return Sample(
        sample_id=sample_id,
        input_text=input_text,
        target=target,
        choices=choices,
        metadata=metadata,
        doc_hash=doc_hash,
    )


def _resolve_dataset_path(dataset_path: str, base_dir: Path | None) -> Path:
    candidate = Path(dataset_path)
    if candidate.is_absolute() or candidate.is_file():
        return candidate
    if base_dir is not None:
        rel = base_dir / candidate
        if rel.is_file():
            return rel
    return candidate


def load_samples(spec: TaskSpec, *, base_dir: Path | None = None) -> list[Sample]:
    """Load Sample records from CSV, JSONL, JSON, Parquet, or inline BYOP metadata items."""
    inline_items = spec.metadata.get("items")
    if spec.dataset_path is None and isinstance(inline_items, list):
        return [_row_to_sample(dict(item), idx) for idx, item in enumerate(inline_items)]

    if not spec.dataset_path:
        return []

    path = _resolve_dataset_path(spec.dataset_path, base_dir)
    if not path.is_file():
        raise TaskLoadError(f"Dataset file not found for task '{spec.name}': {path}")

    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
    elif suffix in (".jsonl", ".ndjson"):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
    elif suffix == ".json":
        parsed_json = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed_json, list):
            rows = [dict(r) for r in parsed_json]
        elif isinstance(parsed_json, dict) and isinstance(parsed_json.get("items"), list):
            rows = [dict(r) for r in parsed_json["items"]]
        else:
            raise TaskLoadError(f"Unsupported JSON dataset structure in {path}")
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Reading .parquet datasets requires pyarrow. Install pyarrow to load Parquet."
            ) from exc
        table = pq.read_table(path)
        rows = [dict(r) for r in table.to_pylist()]
    else:
        raise TaskLoadError(f"Unsupported dataset format '{suffix}' for {path}")

    return [_row_to_sample(row, idx) for idx, row in enumerate(rows)]
