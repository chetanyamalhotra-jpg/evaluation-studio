"""BYOT task discovery, dataset upload, YAML save/validate, registries, and calibration."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import ValidationError

from eval_studio.core.registry import get_registry
from eval_studio.core.schema import Sample, TaskSpec, TrustTier
from eval_studio.gateway.secrets import contains_secret, redact_object, redact_secrets
from eval_studio.scoring.calibrate import cohens_kappa
from eval_studio.tasks.loader import TaskLoadError, load_samples
from eval_studio.tasks.templates import (
    _FORBIDDEN_TEMPLATE_PATTERNS,
    _SANDBOX_ENV,
    TemplateSecurityError,
    render_sandboxed_template,
)
from eval_studio.tasks.validator import validate_task_spec
from eval_studio.web.schemas import (
    ByotRegistriesResponse,
    ByotSaveRequest,
    ByotSaveResponse,
    ByotTaskDetailResponse,
    ByotTaskListResponse,
    ByotTaskSummary,
    ByotValidateRequest,
    ByotValidateResponse,
    CalibrateRequest,
    CalibrateResponse,
    DatasetUploadResponse,
    TaskSource,
)
from eval_studio.web.server import AppState, get_studio_state

router = APIRouter(prefix="/api/byot", tags=["byot"])

StateDep = Annotated[AppState, Depends(get_studio_state)]

_ALLOWED_DATASET_SUFFIXES: frozenset[str] = frozenset(
    {".csv", ".jsonl", ".ndjson", ".json", ".parquet"}
)
_ALLOWED_TASK_SUFFIXES: frozenset[str] = frozenset({".yaml", ".yml"})


def _deep_contains_secret(value: Any) -> bool:
    """Recursively inspect parsed YAML/JSON structures for embedded secrets."""
    if isinstance(value, str):
        return contains_secret(value)
    if isinstance(value, Mapping):
        if any(
            _deep_contains_secret(k) or _deep_contains_secret(v) for k, v in value.items()
        ):
            return True
    elif isinstance(value, (list, tuple, set)):
        if any(_deep_contains_secret(item) for item in value):
            return True
    else:
        return contains_secret(str(value))
    return contains_secret(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def _format_task_path(resolved: Path, workspace_dir: Path, examples_dir: Path) -> str:
    ws_resolved = workspace_dir.resolve()
    if resolved.is_relative_to(ws_resolved):
        return resolved.relative_to(ws_resolved).as_posix()
    ex_resolved = examples_dir.resolve()
    if resolved.is_relative_to(ex_resolved):
        rel = resolved.relative_to(ex_resolved).as_posix()
        return f"examples/{rel}" if ex_resolved.name == "examples" else resolved.as_posix()
    return resolved.as_posix()


def _infer_source(resolved: Path, spec: TaskSpec, examples_dir: Path) -> TaskSource:
    if "suite_sha256" in spec.metadata or "source_paper_sha256" in spec.metadata:
        return "paper"
    if examples_dir.is_dir() and resolved.is_relative_to(examples_dir.resolve()):
        return "builtin"
    return "user"


def _resolve_task_base_dir(
    spec: TaskSpec,
    *,
    explicit_base_dir: Path | None,
    task_dir: Path | None,
    tasks_dir: Path,
    workspace_dir: Path,
    examples_dir: Path,
) -> Path | None:
    """Choose the first candidate directory where `spec.dataset_path` resolves on disk."""
    candidates: list[Path] = []
    for cand in (
        explicit_base_dir,
        task_dir,
        tasks_dir,
        workspace_dir,
        examples_dir.parent if examples_dir.is_dir() else None,
    ):
        if cand is not None:
            resolved_cand = cand.resolve()
            if resolved_cand not in candidates:
                candidates.append(resolved_cand)

    if spec.dataset_path:
        ds_path = Path(spec.dataset_path)
        if ds_path.is_absolute():
            return ds_path.parent
        for cand in candidates:
            if (cand / ds_path).is_file():
                return cand
    return candidates[0] if candidates else None


def _load_samples_safe(
    spec: TaskSpec,
    *,
    candidate_dirs: Sequence[Path | None],
) -> list[Sample] | None:
    """Load samples without crashing task discovery if datasets/optional deps are absent."""
    if not spec.dataset_path and not isinstance(spec.metadata.get("items"), list):
        return None
    non_null = [d.resolve() for d in candidate_dirs if d is not None]
    base_dir = non_null[0] if non_null else None
    if spec.dataset_path:
        ds_path = Path(spec.dataset_path)
        if not ds_path.is_absolute():
            for d in non_null:
                if (d / ds_path).is_file():
                    base_dir = d
                    break
    try:
        return load_samples(spec, base_dir=base_dir)
    except (TaskLoadError, ImportError, OSError, ValueError, json.JSONDecodeError):
        return None


def _resolve_allowed_task_path(
    raw_path: str,
    *,
    workspace_dir: Path,
    tasks_dir: Path,
    examples_dir: Path,
) -> Path:
    cleaned = raw_path.strip()
    if not cleaned or "\x00" in cleaned:
        raise HTTPException(status_code=400, detail="Task path must be non-empty.")

    ws_root = workspace_dir.resolve()
    tasks_root = tasks_dir.resolve()
    examples_root = examples_dir.resolve()
    allowed_roots = [tasks_root, examples_root]

    raw_p = Path(cleaned)
    if raw_p.is_absolute():
        candidates = [raw_p.resolve()]
    else:
        candidates = [
            (ws_root / raw_p).resolve(),
            (tasks_root / raw_p).resolve(),
            (examples_root / raw_p).resolve(),
            (examples_root.parent / raw_p).resolve(),
        ]

    in_bounds = [c for c in candidates if any(c.is_relative_to(r) for r in allowed_roots)]
    if not in_bounds:
        raise HTTPException(
            status_code=403,
            detail=(
                "Access denied: path must reside within workspace examples/ or tasks/ directory."
            ),
        )

    existing = next((c for c in in_bounds if c.is_file()), None)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Task file not found: {cleaned}")

    if existing.suffix.lower() not in _ALLOWED_TASK_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Task detail path must point to a .yaml or .yml file.",
        )
    return existing


def _iter_template_strings(spec: TaskSpec) -> list[tuple[str, str]]:
    templates: list[tuple[str, str]] = [
        ("doc_to_text", spec.doc_to_text),
        ("doc_to_target", spec.doc_to_target),
    ]
    if isinstance(spec.doc_to_choice, str):
        templates.append(("doc_to_choice", spec.doc_to_choice))
    elif isinstance(spec.doc_to_choice, list):
        for idx, item in enumerate(spec.doc_to_choice):
            templates.append((f"doc_to_choice[{idx}]", item))

    if isinstance(spec.doc_to_messages, str):
        templates.append(("doc_to_messages", spec.doc_to_messages))
    elif isinstance(spec.doc_to_messages, list):
        for idx, msg in enumerate(spec.doc_to_messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                templates.append((f"doc_to_messages[{idx}].content", content))
            elif isinstance(content, list):
                for part_idx, part in enumerate(content):
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        templates.append(
                            (f"doc_to_messages[{idx}].content[{part_idx}].text", part["text"])
                        )
    return templates


def _check_template_static_safety(field_name: str, template_str: str) -> str | None:
    for pattern in _FORBIDDEN_TEMPLATE_PATTERNS:
        if pattern.search(template_str):
            return (
                f"Blocked unsafe {field_name} template matching forbidden pattern: "
                f"{pattern.pattern}"
            )
    try:
        _SANDBOX_ENV.from_string(template_str)
    except Exception as exc:
        return f"Invalid {field_name} Jinja2 template syntax: {exc}"
    return None


def _build_sample_context(
    spec: TaskSpec,
    *,
    base_dir: Path | None,
) -> tuple[dict[str, Any], str | None]:
    sample_ctx: dict[str, Any] = {
        "sample_id": "dryrun_0",
        "input_text": "Sample clinical question?",
        "target": "A",
        "choices": ["Option A", "Option B"],
    }
    if not spec.dataset_path and not isinstance(spec.metadata.get("items"), list):
        return sample_ctx, None

    try:
        samples = load_samples(spec, base_dir=base_dir)
    except (TaskLoadError, ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        return sample_ctx, redact_secrets(str(exc))

    if not samples:
        return sample_ctx, None

    s0 = samples[0]
    ctx: dict[str, Any] = {
        "sample_id": s0.sample_id,
        "input_text": s0.input_text,
        "target": s0.target,
        "choices": s0.choices or ["Option A", "Option B"],
        **s0.metadata,
    }
    return ctx, None


def _coerce_calibration_label(raw: Any) -> str | int:
    if isinstance(raw, (str, int)):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else str(raw)
    raise HTTPException(
        status_code=422,
        detail="Calibration labels must be strings, integers, floats, or booleans.",
    )


def _parse_calibration_jsonl(text: str) -> tuple[list[str | int], list[str | int]]:
    judge_labels: list[str | int] = []
    human_labels: list[str | int] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid JSON on calibration line {line_no}: {exc}",
            ) from None
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Calibration line {line_no} must be a JSON object.",
            )
        if "judge_label" not in parsed or "human_label" not in parsed:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Calibration line {line_no} must contain 'judge_label' and 'human_label'."
                ),
            )
        judge_labels.append(_coerce_calibration_label(parsed["judge_label"]))
        human_labels.append(_coerce_calibration_label(parsed["human_label"]))
    return judge_labels, human_labels


@router.get("/tasks", response_model=ByotTaskListResponse)
def list_tasks(state: StateDep) -> ByotTaskListResponse:
    """List built-in example tasks, user tasks, and compiled BYOP paper suites."""
    cfg = state.config
    seen: set[Path] = set()
    candidates: list[Path] = []

    for root in (cfg.examples_dir, cfg.tasks_dir):
        if not root.is_dir():
            continue
        discovered = sorted(
            [*root.rglob("*.yaml"), *root.rglob("*.yml")],
            key=lambda p: p.as_posix(),
        )
        for path in discovered:
            resolved = path.resolve()
            if resolved not in seen and resolved.is_file():
                seen.add(resolved)
                candidates.append(resolved)

    summaries: list[ByotTaskSummary] = []
    for path in candidates:
        try:
            raw_text = path.read_text(encoding="utf-8")
            data: Any = yaml.safe_load(raw_text)
            if not isinstance(data, dict):
                continue
            spec = TaskSpec.model_validate(data)
        except Exception:
            continue

        samples = _load_samples_safe(
            spec,
            candidate_dirs=[
                path.parent,
                cfg.tasks_dir,
                cfg.workspace_dir,
                cfg.examples_dir.parent if cfg.examples_dir.is_dir() else None,
            ],
        )
        rel_path = _format_task_path(path, cfg.workspace_dir, cfg.examples_dir)
        summaries.append(
            ByotTaskSummary(
                id=rel_path,
                name=spec.name,
                path=rel_path,
                source=_infer_source(path, spec, cfg.examples_dir),
                trust_tier=spec.trust_tier,
                output_type=spec.output_type,
                metrics=list(spec.metrics),
                description=spec.description,
                num_samples=len(samples) if samples is not None else None,
            )
        )

    return ByotTaskListResponse(tasks=summaries)


@router.get("/tasks/detail", response_model=ByotTaskDetailResponse)
def get_task_detail(
    state: StateDep,
    path: Annotated[str, Query(min_length=1)],
) -> ByotTaskDetailResponse:
    """Return full YAML, parsed TaskSpec, and preview sample rows for an allowed task path."""
    cfg = state.config
    resolved = _resolve_allowed_task_path(
        path,
        workspace_dir=cfg.workspace_dir,
        tasks_dir=cfg.tasks_dir,
        examples_dir=cfg.examples_dir,
    )
    yaml_text = resolved.read_text(encoding="utf-8")
    try:
        data: Any = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(f"Invalid or unsafe task YAML: {exc}"),
        ) from None
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Task YAML must be a mapping.")
    try:
        spec = TaskSpec.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=redact_secrets(str(exc))) from None

    samples = _load_samples_safe(
        spec,
        candidate_dirs=[
            resolved.parent,
            cfg.tasks_dir,
            cfg.workspace_dir,
            cfg.examples_dir.parent if cfg.examples_dir.is_dir() else None,
        ],
    )
    preview_rows: list[dict[str, Any]] = []
    for sample in (samples or [])[:20]:
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "input_text": sample.input_text,
            "target": sample.target,
            "choices": sample.choices,
            **sample.metadata,
        }
        preview_rows.append(redact_object(row))

    return ByotTaskDetailResponse(
        path=_format_task_path(resolved, cfg.workspace_dir, cfg.examples_dir),
        yaml_text=redact_secrets(yaml_text),
        spec=redact_object(spec.model_dump(mode="json")),
        sample_rows=preview_rows,
    )


@router.post("/upload-dataset", response_model=DatasetUploadResponse)
async def upload_dataset(
    state: StateDep,
    file: Annotated[UploadFile, File(...)],
) -> DatasetUploadResponse:
    """Parse and validate a CSV/JSONL/JSON/Parquet dataset, then save under tasks/datasets/."""
    raw_name = (file.filename or "").strip()
    if (
        not raw_name
        or "\x00" in raw_name
        or "/" in raw_name
        or "\\" in raw_name
        or raw_name in {".", ".."}
    ):
        raise HTTPException(
            status_code=422,
            detail="Dataset filename must be a plain filename without path separators.",
        )

    suffix = Path(raw_name).suffix.lower()
    if suffix not in _ALLOWED_DATASET_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported dataset format '{suffix or '(none)'}'. "
                "Upload .csv, .jsonl, .ndjson, .json, or .parquet."
            ),
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=422, detail="Uploaded dataset file is empty.")

    rows: list[dict[str, Any]] = []
    columns: list[str] = []

    if suffix == ".csv":
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid UTF-8 CSV: {exc}") from None
        reader = csv.DictReader(io.StringIO(text, newline=""))
        columns = [str(c) for c in (reader.fieldnames or []) if c is not None]
        if not columns:
            raise HTTPException(status_code=422, detail="CSV dataset contains no header columns.")
        rows = [dict(r) for r in reader]
    elif suffix in {".jsonl", ".ndjson"}:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid UTF-8 JSONL: {exc}") from None
        seen_cols: dict[str, None] = {}
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid JSON on line {line_no}: {exc}",
                ) from None
            if not isinstance(parsed, dict):
                raise HTTPException(
                    status_code=422,
                    detail=f"JSONL line {line_no} must be a JSON object.",
                )
            rows.append(parsed)
            for key in parsed:
                seen_cols.setdefault(str(key), None)
        columns = list(seen_cols.keys())
        if not rows:
            raise HTTPException(status_code=422, detail="JSONL dataset contains no rows.")
    elif suffix == ".json":
        try:
            parsed_json: Any = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON dataset: {exc}") from None
        if isinstance(parsed_json, list):
            raw_items = parsed_json
        elif isinstance(parsed_json, dict) and isinstance(parsed_json.get("items"), list):
            raw_items = parsed_json["items"]
        else:
            raise HTTPException(
                status_code=422,
                detail="JSON dataset must be a list of objects or {'items': [...]}.",
            )
        if not raw_items:
            raise HTTPException(status_code=422, detail="JSON dataset contains no rows.")
        seen_json_cols: dict[str, None] = {}
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=422,
                    detail=f"JSON dataset item {idx} must be an object.",
                )
            rows.append(item)
            for key in item:
                seen_json_cols.setdefault(str(key), None)
        columns = list(seen_json_cols.keys())
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Reading .parquet datasets requires pyarrow. "
                    "Install pyarrow to load Parquet."
                ),
            ) from None
        try:
            table = pq.read_table(io.BytesIO(raw_bytes))
            columns = [str(c) for c in table.column_names]
            rows = [dict(r) for r in table.to_pylist()]
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=redact_secrets(f"Invalid Parquet dataset: {exc}"),
            ) from None

    datasets_dir = (state.config.tasks_dir / "datasets").resolve()
    dest_path = (datasets_dir / raw_name).resolve()
    if not dest_path.is_relative_to(datasets_dir):
        raise HTTPException(status_code=403, detail="Dataset path escapes tasks/datasets/.")

    datasets_dir.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(raw_bytes)

    return DatasetUploadResponse(
        dataset_path=str(dest_path),
        columns=columns,
        preview_rows=redact_object(rows[:20]),
        total_rows=len(rows),
    )


@router.post("/save", response_model=ByotSaveResponse)
def save_task(state: StateDep, body: ByotSaveRequest) -> ByotSaveResponse:
    """Validate secret hygiene, trust tier, and SSTI safety, then save YAML under tasks_dir."""
    raw_filename = body.filename.strip()
    if not raw_filename or "\x00" in raw_filename:
        raise HTTPException(status_code=422, detail="Task filename must be non-empty.")

    tasks_root = state.config.tasks_dir.resolve()
    if Path(raw_filename).is_absolute() or "/" in raw_filename or "\\" in raw_filename:
        raise HTTPException(
            status_code=403,
            detail="Task filename must be a plain filename inside tasks/.",
        )
    if raw_filename in {".", ".."} or ".." in Path(raw_filename).parts:
        raise HTTPException(status_code=403, detail="Path traversal is not allowed.")

    candidate_name = raw_filename
    if Path(candidate_name).suffix == "":
        candidate_name = f"{candidate_name}.yaml"
    elif Path(candidate_name).suffix.lower() not in _ALLOWED_TASK_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail="Task filename must end with .yaml or .yml.",
        )

    dest_path = (tasks_root / candidate_name).resolve()
    if not dest_path.is_relative_to(tasks_root):
        raise HTTPException(status_code=403, detail="Task destination escapes tasks_dir.")

    if contains_secret(body.yaml_text):
        raise HTTPException(
            status_code=422,
            detail="Task YAML appears to contain an embedded API key or secret.",
        )

    try:
        data: Any = yaml.safe_load(body.yaml_text)
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(
                f"Unsafe or invalid YAML (custom tags like !function are blocked): {exc}"
            ),
        ) from None
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Task YAML must be a mapping.")

    if _deep_contains_secret(data):
        raise HTTPException(
            status_code=422,
            detail="Task YAML appears to contain an embedded API key or secret.",
        )

    try:
        spec = TaskSpec.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=redact_secrets(str(exc))) from None

    if spec.trust_tier == TrustTier.TIER3_CODE_PLUGIN or spec.unsafe_code:
        raise HTTPException(
            status_code=422,
            detail="Saving Tier 3 / unsafe_code tasks via the Web Studio is not allowed.",
        )

    for field_name, tmpl in _iter_template_strings(spec):
        err = _check_template_static_safety(field_name, tmpl)
        if err is not None:
            raise HTTPException(status_code=422, detail=redact_secrets(err))

    tasks_root.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(body.yaml_text, encoding="utf-8")

    return ByotSaveResponse(
        path=_format_task_path(dest_path, state.config.workspace_dir, state.config.examples_dir),
        task_name=spec.name,
    )


@router.post("/validate", response_model=ByotValidateResponse)
def validate_task(state: StateDep, body: ByotValidateRequest) -> ByotValidateResponse:
    """Validate a BYOT YAML task and return live sandboxed sample #0 preview."""
    cfg = state.config
    explicit_base: Path | None = None
    if body.base_dir is not None and body.base_dir.strip():
        raw_base = Path(body.base_dir.strip())
        resolved_base = (
            raw_base.resolve()
            if raw_base.is_absolute()
            else (cfg.workspace_dir / raw_base).resolve()
        )
        allowed_bases = [
            cfg.workspace_dir.resolve(),
            cfg.tasks_dir.resolve(),
            cfg.examples_dir.resolve(),
        ]
        if not any(resolved_base.is_relative_to(root) for root in allowed_bases):
            raise HTTPException(
                status_code=403,
                detail="base_dir must reside within workspace or tasks directory.",
            )
        explicit_base = resolved_base

    raw_has_secret = contains_secret(body.yaml_text)
    try:
        data: Any = yaml.safe_load(body.yaml_text)
    except yaml.YAMLError as exc:
        err_msg = redact_secrets(
            f"Unsafe or invalid YAML (custom tags like !function are blocked): {exc}"
        )
        return ByotValidateResponse(
            valid=False,
            task_name="",
            errors=[err_msg],
            warnings=[],
            secret_scan_clean=not raw_has_secret,
            preview_prompt=None,
            preview_target=None,
        )

    if not isinstance(data, dict):
        return ByotValidateResponse(
            valid=False,
            task_name="",
            errors=["Task YAML must be a mapping."],
            warnings=[],
            secret_scan_clean=not raw_has_secret,
            preview_prompt=None,
            preview_target=None,
        )

    parsed_has_secret = _deep_contains_secret(data)
    secret_clean = not (raw_has_secret or parsed_has_secret)

    try:
        spec = TaskSpec.model_validate(data)
    except ValidationError as exc:
        errors = [redact_secrets(str(exc))]
        if not secret_clean:
            errors.append("Task specification appears to contain an embedded API key or secret.")
        return ByotValidateResponse(
            valid=False,
            task_name=str(data.get("name", "")),
            errors=errors,
            warnings=[],
            secret_scan_clean=secret_clean,
            preview_prompt=None,
            preview_target=None,
        )

    effective_base_dir = _resolve_task_base_dir(
        spec,
        explicit_base_dir=explicit_base,
        task_dir=None,
        tasks_dir=cfg.tasks_dir,
        workspace_dir=cfg.workspace_dir,
        examples_dir=cfg.examples_dir,
    )

    try:
        val_result = validate_task_spec(
            spec,
            enable_thinking=body.enable_thinking,
            apply_chat_template=body.apply_chat_template,
            add_bos_token=body.add_bos_token,
            base_dir=effective_base_dir,
        )
        errors = list(val_result.errors)
        warnings = list(val_result.warnings)
    except (ValueError, ValidationError, TaskLoadError, TemplateSecurityError) as exc:
        errors = [redact_secrets(str(exc))]
        warnings = []

    if (spec.trust_tier == TrustTier.TIER3_CODE_PLUGIN or spec.unsafe_code) and not any(
        "tier 3" in e.lower() or "unsafe_code" in e.lower() for e in errors
    ):
        errors.append(
            "Task requires Tier 3 / unsafe_code execution and cannot run in Tier 1-2 sandbox."
        )

    if not secret_clean and not any(
        "secret" in e.lower() or "api key" in e.lower() for e in errors
    ):
        errors.append("Task specification appears to contain an embedded API key or secret.")

    sample_ctx, load_err = _build_sample_context(spec, base_dir=effective_base_dir)
    if load_err and load_err not in errors:
        errors.append(load_err)

    preview_prompt: str | None = None
    preview_target: str | None = None
    if spec.trust_tier in (TrustTier.TIER1_ZERO_CODE, TrustTier.TIER2_DECLARATIVE):
        for field_name, tmpl in _iter_template_strings(spec):
            if field_name.startswith("doc_to_messages"):
                try:
                    render_sandboxed_template(tmpl, sample_ctx)
                except TemplateSecurityError as exc:
                    errors.append(f"Blocked unsafe {field_name} template: {exc}")

        try:
            rendered_prompt = render_sandboxed_template(spec.doc_to_text, sample_ctx)
            if spec.description:
                rendered_prompt = f"{spec.description}{spec.fewshot_delimiter}{rendered_prompt}"
            preview_prompt = redact_secrets(rendered_prompt)
        except TemplateSecurityError:
            preview_prompt = None

        try:
            preview_target = redact_secrets(
                render_sandboxed_template(spec.doc_to_target, sample_ctx)
            )
        except TemplateSecurityError:
            preview_target = None

    return ByotValidateResponse(
        valid=len(errors) == 0,
        task_name=spec.name,
        errors=[redact_secrets(e) for e in errors],
        warnings=[redact_secrets(w) for w in warnings],
        secret_scan_clean=secret_clean,
        preview_prompt=preview_prompt,
        preview_target=preview_target,
    )


@router.get("/registries", response_model=ByotRegistriesResponse)
def get_registries() -> ByotRegistriesResponse:
    """List registered filters, metrics, and aggregation plugin names."""
    return ByotRegistriesResponse(
        filters=get_registry("eval_studio.filters").list_names(),
        metrics=get_registry("eval_studio.metrics").list_names(),
        aggregations=get_registry("eval_studio.aggregations").list_names(),
    )


@router.post("/calibrate", response_model=CalibrateResponse)
async def calibrate_judge(request: Request) -> CalibrateResponse:
    """Compute Cohen's kappa alignment from JSON (`jsonl_text`/`rows`) or multipart JSONL."""
    content_type = request.headers.get("content-type", "").lower()
    judge_labels: list[str | int] = []
    human_labels: list[str | int] = []
    threshold = 0.40

    if "multipart/form-data" in content_type:
        form = await request.form()
        raw_threshold = form.get("threshold")
        if raw_threshold is not None and str(raw_threshold).strip() != "":
            try:
                threshold = float(str(raw_threshold))
            except ValueError:
                raise HTTPException(
                    status_code=422, detail="threshold must be a valid number."
                ) from None
            if not -1.0 <= threshold <= 1.0:
                raise HTTPException(
                    status_code=422, detail="threshold must be between -1.0 and 1.0."
                )

        uploaded = form.get("file")
        raw_jsonl = form.get("jsonl_text")
        if isinstance(uploaded, UploadFile) or hasattr(uploaded, "read"):
            raw_bytes = await uploaded.read()  # type: ignore[union-attr]
            try:
                jsonl_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(
                    status_code=422, detail=f"Invalid UTF-8 calibration file: {exc}"
                ) from None
            judge_labels, human_labels = _parse_calibration_jsonl(jsonl_text)
        elif isinstance(raw_jsonl, str) and raw_jsonl.strip():
            judge_labels, human_labels = _parse_calibration_jsonl(raw_jsonl)
        else:
            raise HTTPException(
                status_code=422,
                detail="Multipart calibration requires a 'file' or 'jsonl_text' field.",
            )
    else:
        try:
            raw_body: Any = await request.json()
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid JSON request body.") from None
        try:
            body = CalibrateRequest.model_validate(raw_body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=redact_secrets(str(exc))) from None
        threshold = body.threshold
        if body.rows is not None:
            for row in body.rows:
                judge_labels.append(_coerce_calibration_label(row.judge_label))
                human_labels.append(_coerce_calibration_label(row.human_label))
        elif body.jsonl_text is not None:
            judge_labels, human_labels = _parse_calibration_jsonl(body.jsonl_text)
        else:
            raise HTTPException(
                status_code=422,
                detail="Provide either 'rows' or 'jsonl_text' for calibration.",
            )

    if not judge_labels:
        raise HTTPException(
            status_code=422,
            detail="Calibration input contains no labeled rows.",
        )

    kappa = cohens_kappa(judge_labels, human_labels)
    raw_agreement = sum(
        1 for j, h in zip(judge_labels, human_labels, strict=True) if j == h
    ) / len(judge_labels)
    passed = (kappa + 1e-12) >= threshold
    status: Literal["ALIGNED", "UNALIGNED"] = "ALIGNED" if passed else "UNALIGNED"

    return CalibrateResponse(
        n_samples=len(judge_labels),
        cohens_kappa=kappa,
        raw_agreement=raw_agreement,
        passed_threshold=passed,
        threshold=threshold,
        status=status,
    )
