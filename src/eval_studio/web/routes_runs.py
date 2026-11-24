"""Run Launcher, Pre-Flight Cost Estimation, and SSE Live Telemetry endpoints."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from eval_studio.core.config import EvalRunConfig, RuntimeLimits
from eval_studio.core.runner import (
    build_judge_from_args,
    run_evaluation,
)
from eval_studio.core.schema import OlmesConfig, validate_thinking_output_type
from eval_studio.gateway.cost import CostBudgetExceededError, CostGuard
from eval_studio.gateway.secrets import redact_object, redact_secrets
from eval_studio.tasks.loader import load_samples, load_task_spec
from eval_studio.web.routes_byok import _PROVIDER_PRESETS
from eval_studio.web.schemas import (
    CostEstimateResponse,
    RunLaunchRequest,
    RunLaunchResponse,
    RunStatusLiteral,
    RunStatusResponse,
)
from eval_studio.web.server import AppState, get_studio_state

router = APIRouter(prefix="/api/runs", tags=["runs"])

StateDep = Annotated[AppState, Depends(get_studio_state)]

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ActiveRunRecord:
    """In-memory telemetry state and SSE event log for a launched run."""

    run_id: str
    output_dir: Path
    status: RunStatusLiteral = "queued"
    total_samples: int = 0
    completed_samples: int = 0
    spent_usd: float = 0.0
    max_cost_usd: float | None = None
    manifest: dict[str, Any] | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    task: asyncio.Task[None] | None = None
    notify: asyncio.Event = field(default_factory=asyncio.Event)

    def push_event(self, event_name: str, payload: dict[str, Any]) -> None:
        safe_payload = redact_object({"event": event_name, **payload})
        self.events.append(safe_payload)
        self.notify.set()


def _resolve_task_path(raw_path: str, state: AppState) -> Path:
    cleaned = raw_path.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="task_path must be non-empty.")

    candidate = Path(cleaned)
    ws = state.config.workspace_dir.resolve()
    ex = state.config.examples_dir.resolve()
    tsk = state.config.tasks_dir.resolve()
    allowed_roots = [ws, ex, tsk]

    resolved_candidates: list[Path] = []
    if candidate.is_absolute():
        resolved_candidates.append(candidate.resolve())
    else:
        resolved_candidates.append((ws / candidate).resolve())
        resolved_candidates.append((tsk / candidate).resolve())
        if cleaned.startswith("examples/"):
            rel = cleaned[len("examples/") :]
            resolved_candidates.append((ex / rel).resolve())
        else:
            resolved_candidates.append((ex / candidate).resolve())

    for resolved in resolved_candidates:
        if any(resolved.is_relative_to(root) for root in allowed_roots) and resolved.is_file():
            return resolved

    for resolved in resolved_candidates:
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(
                status_code=400,
                detail="Access denied: task_path must reside inside the workspace.",
            )
    raise HTTPException(status_code=404, detail=f"Task YAML not found: {raw_path}")


def _lookup_preset_pricing(model_id: str) -> tuple[float, float]:
    lowered = model_id.strip().lower()
    for preset in _PROVIDER_PRESETS:
        for model in preset.models:
            if model.id.lower() == lowered:
                return model.input_usd_per_1m, model.output_usd_per_1m
    return 0.0, 0.0


def _compute_cost_estimate(req: RunLaunchRequest, state: AppState) -> CostEstimateResponse:
    task_path = _resolve_task_path(req.task_path, state)
    try:
        spec = load_task_spec(task_path, trust_remote_code=req.trust_remote_code)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(f"Failed to load task spec: {exc}"),
        ) from exc

    if spec.subtasks:
        all_samples_count = 0
        word_counts: list[int] = []
        for sub_ref in spec.subtasks:
            sub_path = (task_path.parent / sub_ref).resolve()
            if sub_path.is_file():
                sub_spec = load_task_spec(sub_path, trust_remote_code=req.trust_remote_code)
                sub_samples = load_samples(sub_spec, base_dir=sub_path.parent)
                if req.limit is not None:
                    sub_samples = sub_samples[: req.limit]
                all_samples_count += len(sub_samples)
                word_counts.extend(len(s.input_text.split()) for s in sub_samples)
        n_samples = all_samples_count
        avg_prompt_tokens = (
            max(1, sum(word_counts) // max(1, len(word_counts))) if word_counts else 32
        )
    else:
        samples = load_samples(spec, base_dir=task_path.parent)
        if req.limit is not None:
            samples = samples[: req.limit]
        n_samples = len(samples)
        avg_prompt_tokens = (
            max(1, sum(len(s.input_text.split()) for s in samples) // max(1, len(samples)))
            if samples
            else 32
        )

    raw_pricing = spec.metadata.get("pricing")
    spec_pricing: dict[str, float] = {}
    if isinstance(raw_pricing, dict):
        for k, v in raw_pricing.items():
            if isinstance(v, (int, float)):
                spec_pricing[str(k)] = float(v)

    preset_in, preset_out = _lookup_preset_pricing(req.model_id)
    in_price = spec_pricing.get("input_usd_per_1m", preset_in)
    out_price = spec_pricing.get("output_usd_per_1m", preset_out)
    judge_in_price = spec_pricing.get("judge_input_usd_per_1m", 0.0)
    judge_out_price = spec_pricing.get("judge_output_usd_per_1m", 0.0)
    if req.judge_model:
        j_in, j_out = _lookup_preset_pricing(req.judge_model)
        if judge_in_price == 0.0:
            judge_in_price = j_in
        if judge_out_price == 0.0:
            judge_out_price = j_out

    if req.pricing_override is not None:
        if req.pricing_override.input_usd_per_1m is not None:
            in_price = req.pricing_override.input_usd_per_1m
        if req.pricing_override.output_usd_per_1m is not None:
            out_price = req.pricing_override.output_usd_per_1m
        if req.pricing_override.judge_input_usd_per_1m is not None:
            judge_in_price = req.pricing_override.judge_input_usd_per_1m
        if req.pricing_override.judge_output_usd_per_1m is not None:
            judge_out_price = req.pricing_override.judge_output_usd_per_1m

    repeats = max(req.repeats, int(spec.generation_kwargs.get("repeats", 1)))
    max_gen_toks = int(spec.generation_kwargs.get("max_gen_toks", 128))
    judge_enabled = bool(req.judge_model) or bool(spec.metadata.get("judge_enabled", False))

    guard = CostGuard(max_cost_usd=req.max_cost_usd)
    est = guard.estimate_run(
        n_samples=n_samples,
        avg_prompt_tokens=avg_prompt_tokens,
        max_gen_toks=max_gen_toks,
        repeats=repeats,
        input_usd_per_1m=in_price,
        output_usd_per_1m=out_price,
        judge_enabled=judge_enabled,
        judge_input_usd_per_1m=judge_in_price,
        judge_output_usd_per_1m=judge_out_price,
    )
    exceeds = req.max_cost_usd is not None and est.total_usd > req.max_cost_usd
    return CostEstimateResponse(
        n_samples=est.n_samples,
        repeats=est.repeats,
        target_prompt_tokens=est.target_prompt_tokens,
        target_completion_tokens=est.target_completion_tokens,
        target_usd=est.target_usd,
        judge_enabled=judge_enabled,
        judge_prompt_tokens=est.judge_prompt_tokens,
        judge_completion_tokens=est.judge_completion_tokens,
        judge_usd=est.judge_usd,
        total_usd=est.total_usd,
        exceeds_max_cost=exceeds,
    )


@router.post("/estimate", response_model=CostEstimateResponse)
def estimate_run_endpoint(
    payload: RunLaunchRequest,
    state: StateDep,
) -> CostEstimateResponse:
    """Compute pre-flight USD cost breakdown without calling any inference gateway."""
    return _compute_cost_estimate(payload, state)


@router.post("", response_model=RunLaunchResponse)
async def launch_run_endpoint(
    payload: RunLaunchRequest,
    state: StateDep,
) -> RunLaunchResponse:
    """Launch an evaluation run asynchronously and stream progress via SSE."""
    task_path = _resolve_task_path(payload.task_path, state)

    # Validate task spec synchronously so bad configs return 422 immediately
    try:
        spec = load_task_spec(task_path, trust_remote_code=payload.trust_remote_code)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(f"Invalid task spec: {exc}"),
        ) from exc

    # Validate EvalRunConfig & formatting invariants synchronously before launching
    try:
        validate_thinking_output_type(spec.output_type, payload.enable_thinking)
        config = EvalRunConfig(
            model_backend=payload.model_backend,
            model_id=payload.model_id,
            apply_chat_template=payload.apply_chat_template,
            add_bos_token=payload.add_bos_token,
            enable_thinking=payload.enable_thinking,
            think_end_token=payload.think_end_token,
            trust_remote_code=payload.trust_remote_code,
            bootstrap_iters=payload.bootstrap_iters,
            limit=payload.limit,
            cache_dir=state.effective_cache_dir(),
            limits=RuntimeLimits(max_cost_usd=payload.max_cost_usd),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(f"Invalid run configuration: {exc}"),
        ) from exc

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    safe_task = _SAFE_ID_RE.sub("_", spec.name).strip("._") or "task"
    safe_model = _SAFE_ID_RE.sub("_", payload.model_id).strip("._") or "model"
    dir_name = f"{safe_task}__{safe_model}__{run_id[-6:]}"
    output_dir = (state.config.receipts_dir / dir_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    record = ActiveRunRecord(
        run_id=run_id,
        output_dir=output_dir,
        status="queued",
        max_cost_usd=payload.max_cost_usd,
    )
    with state.lock:
        state.runs_state[run_id] = record

    loop = asyncio.get_running_loop()

    def _on_progress(event: dict[str, Any]) -> None:
        ev_type = str(event.get("event", ""))
        def _apply() -> None:
            if ev_type == "run_started":
                record.status = "running"
                record.total_samples = int(event.get("total_samples", 0))
                record.push_event(
                    "run_started",
                    {
                        "run_id": run_id,
                        "task_name": str(event.get("task_name", spec.name)),
                        "total_samples": record.total_samples,
                    },
                )
            elif ev_type == "subtask_started":
                record.push_event("subtask_started", {"run_id": run_id, **event})
            elif ev_type == "sample_scored":
                raw_idx = int(event.get("index", record.completed_samples))
                record.completed_samples = max(record.completed_samples + 1, raw_idx + 1)
                record.total_samples = max(
                    record.total_samples,
                    int(event.get("total", record.total_samples)),
                    record.completed_samples,
                )
                record.spent_usd = float(event.get("spent_usd", record.spent_usd))
                record.push_event(
                    "sample_scored",
                    {
                        "run_id": run_id,
                        "index": raw_idx,
                        "total": record.total_samples,
                        "sample": event.get("sample", {}),
                        "spent_usd": record.spent_usd,
                        "prompt_tokens": event.get("prompt_tokens"),
                        "completion_tokens": event.get("completion_tokens"),
                    },
                )
            elif ev_type == "completed":
                record.spent_usd = float(event.get("spent_usd", record.spent_usd))

        loop.call_soon_threadsafe(_apply)

    async def _execute() -> None:
        record.status = "running"
        try:
            judge = (
                build_judge_from_args(
                    payload.judge_model,
                    judge_base_url=payload.judge_base_url,
                    fallback_judge_model=payload.fallback_judge_model,
                )
                if payload.judge_model
                else None
            )

            # Apply optional run-level overrides (repeats / olmes_enabled) cleanly via temp copy
            effective_task_path = task_path
            temp_override_path: Path | None = None
            if payload.repeats > 1 or payload.olmes_enabled:
                spec_copy = spec.model_copy(deep=True)
                if payload.repeats > 1:
                    spec_copy.generation_kwargs["repeats"] = payload.repeats
                if payload.olmes_enabled:
                    spec_copy.olmes = (
                        spec_copy.olmes.model_copy(update={"enabled": True})
                        if spec_copy.olmes is not None
                        else OlmesConfig(enabled=True)
                    )
                temp_override_path = output_dir / "_effective_task.yaml"
                import yaml

                temp_override_path.write_text(
                    yaml.safe_dump(spec_copy.model_dump(mode="json"), sort_keys=False),
                    encoding="utf-8",
                )
                effective_task_path = temp_override_path

            try:
                manifest = await asyncio.to_thread(
                    run_evaluation,
                    effective_task_path,
                    config=config,
                    output_dir=output_dir,
                    base_url=payload.base_url,
                    judge=judge,
                    on_progress=_on_progress,
                )
            finally:
                if temp_override_path is not None:
                    temp_override_path.unlink(missing_ok=True)

            manifest_dict = manifest.model_dump(mode="json")
            record.manifest = manifest_dict
            record.spent_usd = float(manifest.total_cost_usd)
            record.status = "completed"
            record.done = True
            record.push_event(
                "completed",
                {
                    "run_id": run_id,
                    "task_name": manifest.task_name,
                    "spent_usd": record.spent_usd,
                    "total_samples": record.total_samples or record.completed_samples,
                    "manifest_sha256": manifest.manifest_sha256,
                    "receipt_dir": dir_name,
                },
            )
        except CostBudgetExceededError as exc:
            redacted = redact_secrets(str(exc))
            record.status = "budget_exceeded"
            record.error = redacted
            record.done = True
            record.push_event(
                "budget_exceeded",
                {
                    "run_id": run_id,
                    "task_name": spec.name,
                    "index": record.completed_samples,
                    "total": record.total_samples,
                    "spent_usd": record.spent_usd,
                    "max_cost_usd": record.max_cost_usd,
                    "error": redacted,
                },
            )
        except Exception as exc:
            redacted = redact_secrets(str(exc))
            record.status = "failed"
            record.error = redacted
            record.done = True
            record.push_event(
                "error",
                {
                    "run_id": run_id,
                    "error": redacted,
                },
            )

    record.task = asyncio.create_task(_execute())

    return RunLaunchResponse(
        run_id=run_id,
        status=record.status,
        output_dir=dir_name,
    )


@router.get("/{run_id}", response_model=RunStatusResponse)
def get_run_status(run_id: str, state: StateDep) -> RunStatusResponse:
    """Return current status, progress counters, spend, and manifest for a run."""
    with state.lock:
        record = state.runs_state.get(run_id)
    if not isinstance(record, ActiveRunRecord):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return RunStatusResponse(
        run_id=record.run_id,
        status=record.status,
        total_samples=record.total_samples,
        completed_samples=record.completed_samples,
        spent_usd=record.spent_usd,
        max_cost_usd=record.max_cost_usd,
        receipt_dir=str(record.output_dir) if record.status == "completed" else None,
        manifest=record.manifest,
        error=record.error,
    )


@router.get("/{run_id}/events")
async def stream_run_events(run_id: str, state: StateDep) -> StreamingResponse:
    """Stream Server-Sent Events (SSE) for a launched evaluation run."""
    with state.lock:
        record = state.runs_state.get(run_id)
    if not isinstance(record, ActiveRunRecord):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    async def _event_generator() -> AsyncIterator[str]:
        cursor = 0
        while True:
            while cursor < len(record.events):
                ev = record.events[cursor]
                cursor += 1
                ev_name = str(ev.get("event", "message"))
                data = json.dumps(ev, ensure_ascii=False)
                yield f"event: {ev_name}\ndata: {data}\n\n"
                if ev_name in ("completed", "budget_exceeded", "error"):
                    return

            if record.done and cursor >= len(record.events):
                return

            record.notify.clear()
            try:
                await asyncio.wait_for(record.notify.wait(), timeout=15.0)
            except TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
