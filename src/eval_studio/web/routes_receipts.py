"""Receipts Vault, Verification, Offline Re-Scoring, Noise Floor, and Multi-Run Compare."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query

from eval_studio.core.runner import build_judge_from_args, rescore_receipt
from eval_studio.core.schema import FilterEnsembleSpec, ScoredSample, TaskSpec
from eval_studio.gateway.secrets import redact_secrets
from eval_studio.receipts.manifest import ReceiptManifest
from eval_studio.receipts.verify import verify_receipt
from eval_studio.stats.aggregation import mcnemar_test
from eval_studio.stats.bootstrap import paired_bootstrap_test
from eval_studio.stats.noise_floor import measure_noise_floor
from eval_studio.tasks.loader import load_task_spec
from eval_studio.web.schemas import (
    CategoryBreakdownEntry,
    CompareMetricEntry,
    CompareRequest,
    CompareResponse,
    CompareRunSummary,
    ItemDiff,
    McNemarSummary,
    NoiseFloorRequest,
    NoiseFloorResponse,
    PairedBootstrapSummary,
    PairwiseComparison,
    ReceiptDetailResponse,
    ReceiptListItem,
    ReceiptSamplesResponse,
    ReceiptsListResponse,
    RescoreRequest,
    RescoreResponse,
    VerifyReceiptResponse,
)
from eval_studio.web.server import AppState, get_studio_state

router = APIRouter(prefix="/api/receipts", tags=["receipts"])

StateDep = Annotated[AppState, Depends(get_studio_state)]

_SAFE_DIR_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _discover_receipt_dirs(receipts_dir: Path) -> list[Path]:
    """Find receipt directories containing manifest.json, skipping _subtask_* folders."""
    if not receipts_dir.is_dir():
        return []
    found: list[Path] = []
    for entry in sorted(receipts_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_subtask_"):
            continue
        if (entry / "manifest.json").is_file():
            found.append(entry)
    return found


def _resolve_receipt_dir(receipt_id: str, state: AppState) -> Path:
    """Resolve a receipt directory by dir_name or manifest.run_id safely inside receipts_dir."""
    cleaned = receipt_id.strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise HTTPException(status_code=400, detail=f"Invalid receipt_id: {receipt_id}")

    root = state.config.receipts_dir.resolve()
    direct = (root / cleaned).resolve()
    if direct.is_relative_to(root) and (direct / "manifest.json").is_file():
        return direct

    for candidate in _discover_receipt_dirs(root):
        try:
            manifest = ReceiptManifest.model_validate_json(
                (candidate / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.run_id == cleaned or candidate.name == cleaned:
                return candidate
        except Exception:
            continue

    raise HTTPException(status_code=404, detail=f"Receipt not found: {receipt_id}")


def _load_manifest_and_samples(receipt_dir: Path) -> tuple[ReceiptManifest, list[ScoredSample]]:
    manifest_path = receipt_dir / "manifest.json"
    samples_path = receipt_dir / "samples.jsonl"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail=f"Missing manifest.json in {receipt_dir.name}")
    manifest = ReceiptManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    samples: list[ScoredSample] = []
    if samples_path.is_file():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                samples.append(ScoredSample.model_validate_json(stripped))
    return manifest, samples


def _extract_sample_decision(sample: ScoredSample) -> str:
    """Extract primary string decision from filtered responses or raw responses."""
    for outputs in sample.filtered_responses.values():
        if outputs:
            return str(outputs[0]).strip()
    if sample.raw_responses:
        return (sample.raw_responses[0].text or "").strip()
    return ""


def _flatten_sample_scores(sample: ScoredSample) -> dict[str, float]:
    """Flatten ScoredSample.metrics (filter -> metric -> score) matching manifest metric keys."""
    flat: dict[str, float] = {}
    for filter_name, m_map in sample.metrics.items():
        for m_name, val in m_map.items():
            fval = float(val)
            flat[f"{filter_name}:{m_name}"] = fval
            if m_name not in flat:
                flat[m_name] = fval
    return flat


def _primary_metric_name(manifest: ReceiptManifest) -> str | None:
    if not manifest.metrics:
        return None
    for preferred in ("exact_match", "acc", "acc_norm", "acc_bytes", "llm_judge", "f1"):
        if preferred in manifest.metrics:
            return preferred
    return next(iter(manifest.metrics.keys()))


def _load_sample_category_map(manifest: ReceiptManifest) -> dict[str, str]:
    """Load sample_id -> category mapping from task YAML metadata.items when available."""
    if not manifest.task_path:
        return {}
    task_file = Path(manifest.task_path)
    if not task_file.is_file():
        return {}
    try:
        spec = load_task_spec(task_file)
    except Exception:
        return {}
    raw_items = spec.metadata.get("items")
    if not isinstance(raw_items, list):
        return {}
    mapping: dict[str, str] = {}
    for entry in raw_items:
        if isinstance(entry, dict):
            sid = str(entry.get("sample_id") or entry.get("item_id") or "").strip()
            cat = str(entry.get("category") or "").strip()
            if sid and cat:
                mapping[sid] = cat
    return mapping


def _is_binary_scores(values: list[float]) -> bool:
    return bool(values) and all(v in (0.0, 1.0) for v in values)


@router.get("", response_model=ReceiptsListResponse)
def list_receipts(state: StateDep) -> ReceiptsListResponse:
    """List all Portable Eval Receipts sorted newest-first."""
    items: list[ReceiptListItem] = []
    ws = state.config.workspace_dir.resolve()

    for receipt_dir in _discover_receipt_dirs(state.config.receipts_dir):
        try:
            manifest = ReceiptManifest.model_validate_json(
                (receipt_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except Exception:
            continue

        rel_path = (
            receipt_dir.resolve().relative_to(ws).as_posix()
            if receipt_dir.resolve().is_relative_to(ws)
            else receipt_dir.resolve().as_posix()
        )
        metrics_dict = {k: v.model_dump(mode="json") for k, v in manifest.metrics.items()}
        items.append(
            ReceiptListItem(
                run_id=manifest.run_id,
                dir_name=receipt_dir.name,
                path=rel_path,
                created_at_utc=manifest.created_at_utc,
                backend=manifest.backend,
                model_id=manifest.model_id,
                task_name=manifest.task_name,
                total_cost_usd=manifest.total_cost_usd,
                total_prompt_tokens=manifest.total_prompt_tokens,
                total_completion_tokens=manifest.total_completion_tokens,
                metrics=metrics_dict,
                task_sha256=manifest.task_sha256,
                manifest_sha256=manifest.manifest_sha256,
                source_paper_path=manifest.source_paper_path,
                source_paper_sha256=manifest.source_paper_sha256,
            )
        )

    items.sort(key=lambda r: r.created_at_utc, reverse=True)
    return ReceiptsListResponse(receipts=items)


# Static POST routes MUST be registered before /{receipt_id} to avoid path shadowing.


@router.post("/noise-floor", response_model=NoiseFloorResponse)
def compute_noise_floor_endpoint(
    payload: NoiseFloorRequest,
    state: StateDep,
) -> NoiseFloorResponse:
    """Compute item-level decision flip rate between two receipts."""
    base_dir = _resolve_receipt_dir(payload.baseline_id, state)
    pert_dir = _resolve_receipt_dir(payload.perturbed_id, state)
    _, base_samples = _load_manifest_and_samples(base_dir)
    _, pert_samples = _load_manifest_and_samples(pert_dir)

    if len(base_samples) != len(pert_samples):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Sample count mismatch: baseline has {len(base_samples)} samples, "
                f"perturbed has {len(pert_samples)}."
            ),
        )

    pert_map = {s.sample_id: _extract_sample_decision(s) for s in pert_samples}
    base_decisions: list[str] = []
    pert_decisions: list[str] = []
    for sample in base_samples:
        if sample.sample_id in pert_map:
            base_decisions.append(_extract_sample_decision(sample))
            pert_decisions.append(pert_map[sample.sample_id])

    if len(base_decisions) != len(base_samples):
        raise HTTPException(
            status_code=422,
            detail="Baseline and perturbed receipts do not share identical sample_ids.",
        )

    report = measure_noise_floor(
        base_decisions,
        pert_decisions,
        perturbation=payload.perturbation,
    )
    return NoiseFloorResponse(
        n_decisions=report.n_decisions,
        n_flips=report.n_flips,
        flip_rate=report.flip_rate,
        perturbation=report.perturbation,
    )


@router.post("/compare", response_model=CompareResponse)
def compare_receipts_endpoint(
    payload: CompareRequest,
    state: StateDep,
) -> CompareResponse:
    """Compare multiple receipts with CIs, paired bootstrap, McNemar, and category breakdown."""
    loaded: list[tuple[str, Path, ReceiptManifest, list[ScoredSample]]] = []
    for rid in payload.receipt_ids:
        rdir = _resolve_receipt_dir(rid, state)
        manifest, samples = _load_manifest_and_samples(rdir)
        loaded.append((rid, rdir, manifest, samples))

    runs_out: list[CompareRunSummary] = []
    # Track category / task breakdown: (group, category) -> {receipt_id: [scores]}
    cat_scores: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rid, rdir, manifest, samples in loaded:
        metric_entries: dict[str, CompareMetricEntry] = {}
        for m_name, m_sum in manifest.metrics.items():
            se = m_sum.stderr if m_sum.stderr is not None else 0.0
            ci_low = m_sum.value - 1.96 * se
            ci_high = m_sum.value + 1.96 * se
            metric_entries[m_name] = CompareMetricEntry(
                value=m_sum.value,
                stderr=m_sum.stderr,
                ci_low=ci_low,
                ci_high=ci_high,
                n_samples=m_sum.n_samples,
                unresolved=m_sum.unresolved,
            )

        latencies = [
            r.latency_ms
            for s in samples
            for r in s.raw_responses
            if r.latency_ms is not None and r.latency_ms >= 0.0
        ]
        mean_latency = (sum(latencies) / len(latencies)) if latencies else None

        runs_out.append(
            CompareRunSummary(
                receipt_id=rid,
                run_id=manifest.run_id,
                dir_name=rdir.name,
                model_id=manifest.model_id,
                backend=manifest.backend,
                task_name=manifest.task_name,
                total_cost_usd=manifest.total_cost_usd,
                total_prompt_tokens=manifest.total_prompt_tokens,
                total_completion_tokens=manifest.total_completion_tokens,
                mean_latency_ms=mean_latency,
                source_paper_path=manifest.source_paper_path,
                metrics=metric_entries,
            )
        )

        # Populate category breakdown from sample scores (using task item categories when present)
        prim = _primary_metric_name(manifest)
        group_label = manifest.task_name
        cat_map = _load_sample_category_map(manifest)
        default_cat = (
            Path(manifest.source_paper_path).stem if manifest.source_paper_path else "overall"
        )
        for s in samples:
            s_scores = _flatten_sample_scores(s)
            if prim is None or prim not in s_scores:
                continue
            cat = cat_map.get(s.sample_id, default_cat)
            cat_scores[(group_label, cat)][rid].append(s_scores[prim])

    # Determine baseline
    baseline_idx = 0
    if payload.baseline_id:
        for i, (rid, rdir, manifest, _) in enumerate(loaded):
            if payload.baseline_id in (rid, rdir.name, manifest.run_id):
                baseline_idx = i
                break

    base_rid, _, base_manifest, base_samples = loaded[baseline_idx]
    base_by_id = {s.sample_id: s for s in base_samples}

    pairwise_out: list[PairwiseComparison] = []
    for idx, (chal_rid, _, chal_manifest, chal_samples) in enumerate(loaded):
        if idx == baseline_idx:
            continue

        shared_metrics = [m for m in base_manifest.metrics if m in chal_manifest.metrics]
        if not shared_metrics:
            continue
        metric_name = shared_metrics[0]

        chal_by_id = {s.sample_id: s for s in chal_samples}
        common_ids = [sid for sid in base_by_id if sid in chal_by_id]
        if not common_ids:
            continue
        if len(base_samples) != len(chal_samples) or len(common_ids) != len(base_samples):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Cannot run paired comparison between '{base_rid}' ({len(base_samples)} "
                    f"samples) and '{chal_rid}' ({len(chal_samples)} samples): sample sets differ."
                ),
            )

        scores_a: list[float] = []
        scores_b: list[float] = []
        decisions_a: list[str] = []
        decisions_b: list[str] = []
        item_diffs: list[ItemDiff] = []

        for sid in common_ids:
            sa = base_by_id[sid]
            sb = chal_by_id[sid]
            scores_map_a = _flatten_sample_scores(sa)
            scores_map_b = _flatten_sample_scores(sb)
            if metric_name not in scores_map_a or metric_name not in scores_map_b:
                continue
            va = scores_map_a[metric_name]
            vb = scores_map_b[metric_name]
            scores_a.append(va)
            scores_b.append(vb)
            da = _extract_sample_decision(sa)
            db = _extract_sample_decision(sb)
            decisions_a.append(da)
            decisions_b.append(db)
            if va != vb:
                item_diffs.append(
                    ItemDiff(
                        sample_id=sid,
                        input_text=sa.input_text or sb.input_text,
                        target=sa.target,
                        baseline_score=va,
                        challenger_score=vb,
                        baseline_output=da,
                        challenger_output=db,
                    )
                )

        if not scores_a:
            continue

        pb = paired_bootstrap_test(
            scores_a,
            scores_b,
            bootstrap_iters=payload.bootstrap_iters,
        )
        pb_summary = PairedBootstrapSummary(
            mean_delta=pb.mean_delta,
            ci_low=pb.ci_low,
            ci_high=pb.ci_high,
            p_value=pb.p_value,
            significant=pb.significant,
            stderr=pb.stderr,
            iters_executed=pb.iters_executed,
            seed=pb.seed,
        )

        mcnemar_summary: McNemarSummary | None = None
        if _is_binary_scores(scores_a) and _is_binary_scores(scores_b):
            mc = mcnemar_test(scores_a, scores_b)
            mcnemar_summary = McNemarSummary(
                n_items=mc.n_items,
                n_01=mc.n_01,
                n_10=mc.n_10,
                statistic=mc.statistic,
                p_value=mc.p_value,
                exact=mc.exact,
                delta=mc.delta,
            )

        nf = measure_noise_floor(decisions_a, decisions_b, perturbation="model_comparison")
        nf_summary = NoiseFloorResponse(
            n_decisions=nf.n_decisions,
            n_flips=nf.n_flips,
            flip_rate=nf.flip_rate,
            perturbation=nf.perturbation,
        )

        unresolved = not pb.significant
        pairwise_out.append(
            PairwiseComparison(
                baseline_id=base_rid,
                challenger_id=chal_rid,
                metric=metric_name,
                paired_bootstrap=pb_summary,
                mcnemar=mcnemar_summary,
                noise_floor=nf_summary,
                unresolved=unresolved,
                item_diffs=item_diffs,
            )
        )

    category_breakdown: list[CategoryBreakdownEntry] = []
    for (group, category), by_run in sorted(cat_scores.items()):
        avg_scores = {
            rid: (sum(vals) / len(vals)) for rid, vals in by_run.items() if vals
        }
        counts = {rid: len(vals) for rid, vals in by_run.items()}
        category_breakdown.append(
            CategoryBreakdownEntry(
                group=group,
                category=category,
                scores=avg_scores,
                counts=counts,
            )
        )

    return CompareResponse(
        runs=runs_out,
        pairwise=pairwise_out,
        category_breakdown=category_breakdown,
    )


@router.get("/{receipt_id}", response_model=ReceiptDetailResponse)
def get_receipt_detail(receipt_id: str, state: StateDep) -> ReceiptDetailResponse:
    """Return full manifest, report.md, and total sample count for a receipt."""
    receipt_dir = _resolve_receipt_dir(receipt_id, state)
    manifest, samples = _load_manifest_and_samples(receipt_dir)
    report_path = receipt_dir / "report.md"
    report_md = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""

    return ReceiptDetailResponse(
        manifest=manifest.model_dump(mode="json"),
        report_md=report_md,
        total_samples=len(samples),
    )


@router.get("/{receipt_id}/samples", response_model=ReceiptSamplesResponse)
def get_receipt_samples(
    receipt_id: str,
    state: StateDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    q: Annotated[str | None, Query()] = None,
) -> ReceiptSamplesResponse:
    """Return paginated and searchable sample records from samples.jsonl."""
    receipt_dir = _resolve_receipt_dir(receipt_id, state)
    _, samples = _load_manifest_and_samples(receipt_dir)

    if q and q.strip():
        needle = q.strip().casefold()
        filtered: list[ScoredSample] = []
        for s in samples:
            haystack = " ".join(
                [
                    s.sample_id,
                    s.input_text,
                    str(s.target),
                    " ".join(r.text or "" for r in s.raw_responses),
                    " ".join(r.reasoning_trace or "" for r in s.raw_responses),
                    json.dumps(s.filtered_responses, ensure_ascii=False),
                ]
            ).casefold()
            if needle in haystack:
                filtered.append(s)
        samples = filtered

    total = len(samples)
    page = samples[offset : offset + limit]
    return ReceiptSamplesResponse(
        total=total,
        offset=offset,
        limit=limit,
        samples=[s.model_dump(mode="json") for s in page],
    )


@router.post("/{receipt_id}/verify", response_model=VerifyReceiptResponse)
def verify_receipt_endpoint(receipt_id: str, state: StateDep) -> VerifyReceiptResponse:
    """Run cryptographic SHA-256 tamper & secret-hygiene verification on a receipt."""
    receipt_dir = _resolve_receipt_dir(receipt_id, state)
    result = verify_receipt(receipt_dir)
    return VerifyReceiptResponse(valid=result.valid, errors=result.errors)


@router.post("/{receipt_id}/rescore", response_model=RescoreResponse)
def rescore_receipt_endpoint(
    receipt_id: str,
    payload: RescoreRequest,
    state: StateDep,
) -> RescoreResponse:
    """Re-score an existing receipt offline (or with judge/filter/metric overrides)."""
    receipt_dir = _resolve_receipt_dir(receipt_id, state)
    orig_manifest, _ = _load_manifest_and_samples(receipt_dir)

    out_name = payload.output_name or f"{receipt_dir.name}-rescored"
    safe_out = _SAFE_DIR_RE.sub("_", out_name).strip("._") or f"{receipt_dir.name}-rescored"
    out_dir = (state.config.receipts_dir / safe_out).resolve()
    if not out_dir.is_relative_to(state.config.receipts_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid output_name.")
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = (
        build_judge_from_args(
            payload.judge_model,
            judge_base_url=payload.judge_base_url,
            fallback_judge_model=payload.fallback_judge_model,
        )
        if payload.judge_model
        else None
    )

    override_task_path: Path | None = None
    try:
        if payload.metrics or payload.filter_ensembles:
            if orig_manifest.task_path and Path(orig_manifest.task_path).is_file():
                base_spec = load_task_spec(Path(orig_manifest.task_path))
            else:
                base_spec = TaskSpec(
                    name=orig_manifest.task_name,
                    metrics=list(orig_manifest.metrics.keys()) or ["exact_match"],
                )
            update_fields: dict[str, Any] = {}
            if payload.metrics:
                update_fields["metrics"] = payload.metrics
            if payload.filter_ensembles is not None:
                update_fields["filter_ensembles"] = [
                    FilterEnsembleSpec.model_validate(fe) for fe in payload.filter_ensembles
                ]
            spec_copy = base_spec.model_copy(update=update_fields)
            override_task_path = out_dir / "_rescore_spec.yaml"
            override_task_path.write_text(
                yaml.safe_dump(spec_copy.model_dump(mode="json"), sort_keys=False),
                encoding="utf-8",
            )

        new_manifest = rescore_receipt(
            receipt_dir,
            output_dir=out_dir,
            task_path=override_task_path,
            judge=judge,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(f"Offline re-scoring failed: {exc}"),
        ) from exc
    finally:
        if override_task_path is not None:
            override_task_path.unlink(missing_ok=True)

    ws = state.config.workspace_dir.resolve()
    rel_path = (
        out_dir.relative_to(ws).as_posix()
        if out_dir.is_relative_to(ws)
        else out_dir.as_posix()
    )
    return RescoreResponse(
        receipt_id=out_dir.name,
        path=rel_path,
        manifest=new_manifest.model_dump(mode="json"),
    )
