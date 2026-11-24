"""Four-stage N+M evaluation runner and offline receipt re-scorer."""

from __future__ import annotations

import contextlib
import hashlib
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_studio import __version__
from eval_studio.cache.fingerprint import compute_request_fingerprint, resolve_cache_dir
from eval_studio.cache.sqlite_store import SQLiteCacheStore
from eval_studio.cache.wal import JsonlWriteAheadLog
from eval_studio.core.config import EvalRunConfig
from eval_studio.core.registry import get_registry, load_plugins
from eval_studio.core.schema import (
    ChatContentPart,
    ChatMessage,
    EvalRequest,
    EvalResponse,
    FilterEnsembleSpec,
    ImageContentPart,
    OutputType,
    Sample,
    ScoredSample,
    TaskSpec,
    TextContentPart,
)
from eval_studio.gateway.base import InferenceGateway, transfer_trailing_whitespace
from eval_studio.gateway.cost import CostBudgetExceededError, CostGuard, PreFlightCostEstimate
from eval_studio.gateway.http import HttpBYOKGateway
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import SecretRole, redact_object, redact_secrets
from eval_studio.paper.compiler import sha256_file
from eval_studio.receipts.manifest import (
    MetricSummary,
    ReceiptManifest,
    build_reproduce_command,
    compute_manifest_digest,
)
from eval_studio.receipts.render import render_receipt_html, render_receipt_markdown
from eval_studio.scoring.filters import (
    Filter,
    FilterEnsemble,
    MajorityVoteFilter,
    MultiChoiceRegexFilter,
    TakeFirstFilter,
    TakeFirstKFilter,
    ThinkStripFilter,
    build_olmes_dual_prompts,
    select_olmes_best_format,
)
from eval_studio.scoring.judge import JudgeRubric, LLMJudge
from eval_studio.scoring.metrics import (
    acc,
    acc_bytes,
    acc_mutual_info,
    acc_norm,
    bits_per_byte,
    brier_score,
    byte_perplexity,
    classification_f1,
    exact_match,
    numeric_exact_match,
    word_perplexity,
)
from eval_studio.stats.aggregation import SubtaskStats, aggregate_group
from eval_studio.stats.bootstrap import bootstrap_stderr
from eval_studio.stats.fewshot import SampledFewShot, assign_fewshot_sets
from eval_studio.tasks.loader import load_samples, load_task_spec
from eval_studio.tasks.templates import render_sandboxed_template
from eval_studio.tasks.validator import validate_task_spec


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(
    on_progress: ProgressCallback | None,
    payload: dict[str, Any],
) -> None:
    """Safely emit a secret-redacted progress payload without failing the run."""
    if on_progress is None:
        return
    with contextlib.suppress(Exception):
        on_progress(redact_object(payload))


def build_judge_from_args(
    judge_model: str,
    *,
    judge_base_url: str | None = None,
    fallback_judge_model: str | None = None,
    fallback_judge_base_url: str | None = None,
    rubric_name: str = "clinical_judge",
    rubric_criteria: str = (
        "Score 1.0 if the response accurately matches the clinical reference answer, else 0.0."
    ),
) -> LLMJudge:
    """Construct a decoupled LLMJudge with optional fallback_judge gateway."""
    rubric = JudgeRubric(name=rubric_name, criteria=rubric_criteria)
    primary_gw: InferenceGateway
    if judge_model == "mock":
        primary_gw = MockGateway(model_id="mock-judge", judge_mode="json_score")
    elif judge_model == "mock-refusal":
        primary_gw = MockGateway(model_id="mock-judge-refusal", judge_mode="refusal")
    else:
        primary_gw = HttpBYOKGateway(
            model_id=judge_model,
            base_url=judge_base_url,
            role=SecretRole.JUDGE,
        )

    fallback_gw: InferenceGateway | None = None
    if fallback_judge_model:
        if fallback_judge_model == "mock":
            fallback_gw = MockGateway(model_id="mock-fallback-judge", judge_mode="json_score")
        else:
            fallback_gw = HttpBYOKGateway(
                model_id=fallback_judge_model,
                base_url=fallback_judge_base_url or judge_base_url,
                role=SecretRole.FALLBACK_JUDGE,
            )

    return LLMJudge(
        judge_model_id=judge_model,
        rubric=rubric,
        gateway=primary_gw,
        fallback_judge=fallback_gw,
    )


def _instantiate_gateway(
    backend: str,
    model_id: str | None,
    *,
    base_url: str | None = None,
    cost_guard: CostGuard | None = None,
    token_pricing: dict[str, float] | None = None,
) -> InferenceGateway:
    if backend == "mock":
        return MockGateway(
            model_id=model_id or "mock-deterministic",
            token_pricing=token_pricing,
            cost_guard=cost_guard,
        )
    if backend == "http-byok":
        return HttpBYOKGateway(
            model_id=model_id or "gpt-4o-mini",
            base_url=base_url,
            role=SecretRole.TARGET,
            cost_guard=cost_guard,
            token_pricing=token_pricing,
        )
    cls_or_obj: Any = get_registry("eval_studio.models").get(backend)
    if isinstance(cls_or_obj, type):
        return cls_or_obj(model_id=model_id or backend)  # type: ignore[no-any-return]
    return cls_or_obj  # type: ignore[no-any-return]


def _extract_pricing(spec: TaskSpec) -> dict[str, float]:
    raw = spec.metadata.get("pricing")
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _render_sample_prompt(spec: TaskSpec, sample: Sample) -> str:
    ctx: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "input_text": sample.input_text,
        "target": sample.target,
        "choices": sample.choices,
        **sample.metadata,
    }
    rendered = render_sandboxed_template(spec.doc_to_text, ctx)
    if spec.description:
        rendered = f"{spec.description}{spec.fewshot_delimiter}{rendered}"
    return rendered


def _render_sample_messages(
    spec: TaskSpec,
    sample: Sample,
    prompt_text: str,
) -> list[ChatMessage]:
    """Render multimodal ChatMessages from sample.messages or spec.doc_to_messages."""
    if sample.messages:
        return list(sample.messages)
    if not isinstance(spec.doc_to_messages, list):
        return [ChatMessage(role="user", content=prompt_text)]

    ctx: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "input_text": sample.input_text,
        "target": sample.target,
        "choices": sample.choices,
        **sample.metadata,
    }
    rendered_msgs: list[ChatMessage] = []
    for raw_msg in spec.doc_to_messages:
        if not isinstance(raw_msg, dict):
            continue
        role = str(raw_msg.get("role", "user"))
        raw_content = raw_msg.get("content", "")
        if isinstance(raw_content, str):
            rendered_msgs.append(
                ChatMessage(role=role, content=render_sandboxed_template(raw_content, ctx))
            )
            continue
        if isinstance(raw_content, list):
            parts: list[ChatContentPart] = []
            for raw_part in raw_content:
                if not isinstance(raw_part, dict):
                    continue
                p_type = str(raw_part.get("type", "text"))
                if p_type == "text":
                    p_text = render_sandboxed_template(str(raw_part.get("text", "")), ctx)
                    parts.append(TextContentPart(text=p_text))
                elif p_type in ("image_url", "image"):
                    raw_img_url = raw_part.get("image_url")
                    rendered_img_url: str | dict[str, str] | None = None
                    if isinstance(raw_img_url, str):
                        rendered_img_url = render_sandboxed_template(raw_img_url, ctx)
                    elif isinstance(raw_img_url, dict):
                        rendered_img_url = {
                            str(k): render_sandboxed_template(str(v), ctx)
                            for k, v in raw_img_url.items()
                        }
                    raw_path = raw_part.get("path")
                    rendered_path = (
                        render_sandboxed_template(str(raw_path), ctx)
                        if raw_path is not None
                        else None
                    )
                    parts.append(
                        ImageContentPart(
                            type="image" if p_type == "image" else "image_url",
                            image_url=rendered_img_url,
                            path=rendered_path,
                            mime_type=(
                                str(raw_part["mime_type"])
                                if raw_part.get("mime_type") is not None
                                else None
                            ),
                            detail=(
                                str(raw_part["detail"])
                                if raw_part.get("detail") is not None
                                else None
                            ),
                        )
                    )
            rendered_msgs.append(ChatMessage(role=role, content=parts))
    return rendered_msgs or [ChatMessage(role="user", content=prompt_text)]


def _build_filter_ensembles(
    spec: TaskSpec,
    choices: list[str],
    *,
    enable_thinking: bool,
    think_end_token: str | None,
) -> list[FilterEnsemble]:
    specs: list[FilterEnsembleSpec] = list(spec.filter_ensembles)
    if not specs:
        steps: list[Filter] = []
        if enable_thinking and think_end_token:
            steps.append(ThinkStripFilter(think_end_token=think_end_token))
        return [FilterEnsemble("default", steps)]

    ensembles: list[FilterEnsemble] = []
    for es in specs:
        steps = []
        if enable_thinking and think_end_token:
            steps.append(ThinkStripFilter(think_end_token=think_end_token))
        for step_cfg in es.steps:
            name = step_cfg.name
            params = dict(step_cfg.params)
            if name == "multi_choice_regex":
                step_choices = params.pop("choices", None) or choices
                steps.append(MultiChoiceRegexFilter(step_choices, **params))
            elif name == "take_first":
                steps.append(TakeFirstFilter())
            elif name == "take_first_k":
                steps.append(TakeFirstKFilter(k=int(params.get("k", 1))))
            elif name == "majority_vote":
                steps.append(MajorityVoteFilter())
            elif name == "think_strip":
                steps.append(
                    ThinkStripFilter(
                        think_end_token=str(params.get("think_end_token", "</think>"))
                    )
                )
            else:
                flt_cls: Any = get_registry("eval_studio.filters").get(name)
                steps.append(flt_cls(**params) if isinstance(flt_cls, type) else flt_cls)
        ensembles.append(FilterEnsemble(es.name, steps))
    return ensembles


def _apply_thinking_strip(
    resp: EvalResponse,
    *,
    enable_thinking: bool,
    think_end_token: str | None,
) -> EvalResponse:
    if not enable_thinking or not think_end_token:
        return resp
    raw = resp.text or ""
    idx = raw.rfind(think_end_token)
    if idx != -1:
        trace = raw[:idx]
        stripped = raw[idx + len(think_end_token) :].strip()
        return resp.model_copy(
            update={
                "text": stripped,
                "reasoning_trace": trace,
                "thinking_truncated": False,
            }
        )
    if resp.reasoning_trace is not None:
        return resp
    return resp.model_copy(
        update={
            "text": "",
            "reasoning_trace": raw,
            "thinking_truncated": True,
        }
    )


def _dispatch_group(
    gw: InferenceGateway,
    output_type: OutputType,
    reqs: Sequence[EvalRequest],
) -> list[EvalResponse]:
    if output_type == OutputType.LOGLIKELIHOOD:
        return gw.loglikelihood(reqs)
    if output_type == OutputType.LOGLIKELIHOOD_ROLLING:
        return gw.loglikelihood_rolling(reqs)
    if output_type == OutputType.CHAT:
        return gw.chat(reqs)
    return gw.generate_until(reqs)


def _dispatch_cached_requests(
    requests: Sequence[EvalRequest],
    *,
    gw: InferenceGateway,
    task_sha256: str,
    use_cache: bool,
    effective_do_sample: bool,
    wal: JsonlWriteAheadLog,
    sqlite_store: SQLiteCacheStore,
    wal_cache: dict[str, dict[str, Any]],
    enable_thinking: bool,
    think_end_token: str | None,
    dtype: str | None = None,
    quantization: str | None = None,
    strict_budget: bool = False,
) -> tuple[list[EvalResponse], int, int, float]:
    n = len(requests)
    if n == 0:
        return [], 0, 0, 0.0

    results: list[EvalResponse | None] = [None] * n
    misses: list[tuple[int, EvalRequest, str]] = []

    for idx, req in enumerate(requests):
        fp = compute_request_fingerprint(
            task_hash=task_sha256,
            model_id=gw.model_id,
            backend=gw.backend_name,
            request_payload=req.model_dump(mode="json"),
            dtype=dtype,
            quantization=quantization,
        )
        cached: dict[str, Any] | None = None
        if use_cache and not effective_do_sample:
            cached = wal_cache.get(fp) or sqlite_store.get(fp, do_sample=effective_do_sample)
        if cached is not None:
            results[idx] = EvalResponse.model_validate(cached)
        else:
            misses.append((idx, req, fp))

    def _finalize_and_store(idx: int, req: EvalRequest, fp: str, resp: EvalResponse) -> None:
        resp = _apply_thinking_strip(
            resp,
            enable_thinking=enable_thinking,
            think_end_token=think_end_token,
        )
        resp = resp.model_copy(
            update={
                "metadata": {
                    **resp.metadata,
                    "prompt_hash": req.prompt_hash or _sha256_text(req.context),
                }
            }
        )
        if use_cache and not effective_do_sample:
            dumped = resp.model_dump(mode="json")
            wal.append({"fingerprint": fp, "response": dumped})
            sqlite_store.put(fp, dumped, do_sample=effective_do_sample)
            wal_cache[fp] = dumped
        results[idx] = resp

    if strict_budget:
        for idx, req, fp in misses:
            resp = _dispatch_group(gw, req.output_type, [req])[0]
            _finalize_and_store(idx, req, fp, resp)
    elif misses:
        grouped: dict[OutputType, list[tuple[int, EvalRequest, str]]] = {}
        for item in misses:
            grouped.setdefault(item[1].output_type, []).append(item)
        for otype, group_items in grouped.items():
            batch_reqs = [item[1] for item in group_items]
            batch_resps = _dispatch_group(gw, otype, batch_reqs)
            for (idx, req, fp), resp in zip(group_items, batch_resps, strict=True):
                _finalize_and_store(idx, req, fp, resp)

    finalized: list[EvalResponse] = []
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0
    for r in results:
        assert r is not None
        prompt_tokens += r.usage.prompt_tokens
        completion_tokens += r.usage.completion_tokens
        cost_usd += float(r.metadata.get("cost_usd", 0.0))
        finalized.append(r)

    return finalized, prompt_tokens, completion_tokens, cost_usd


def _score_sample(
    spec: TaskSpec,
    sample: Sample,
    raw_responses: list[EvalResponse],
    *,
    enable_thinking: bool = False,
    think_end_token: str | None = None,
    continuations: Sequence[str] | None = None,
    uncond_lls: Sequence[float] | None = None,
    judge: LLMJudge | None = None,
) -> ScoredSample:
    ignore_case = bool(spec.metadata.get("ignore_case", True))
    eff_choices = list(continuations) if continuations is not None else list(sample.choices)
    ensembles = _build_filter_ensembles(
        spec,
        eff_choices,
        enable_thinking=enable_thinking,
        think_end_token=think_end_token,
    )

    # Optional Tier 3 custom scorer registered in eval_studio.metrics
    custom_scorer_name = spec.metadata.get("scorer")
    custom_scorer: Any = None
    if isinstance(custom_scorer_name, str) and custom_scorer_name:
        try:
            sc_obj = get_registry("eval_studio.metrics").get(custom_scorer_name)
            custom_scorer = sc_obj() if isinstance(sc_obj, type) else sc_obj
        except KeyError:
            custom_scorer = None

    filtered_responses: dict[str, list[str]] = {}
    metrics_by_pipeline: dict[str, dict[str, float]] = {}

    if spec.output_type == OutputType.MULTIPLE_CHOICE and eff_choices:
        lls = [r.loglikelihood if r.loglikelihood is not None else -1e9 for r in raw_responses]
        target_idx = int(sample.target) if isinstance(sample.target, int) else 0
        if isinstance(sample.target, str) and sample.target in sample.choices:
            target_idx = sample.choices.index(sample.target)
        elif isinstance(sample.target, str) and sample.target in eff_choices:
            target_idx = eff_choices.index(sample.target)
        elif isinstance(sample.target, str) and sample.target.isdigit():
            target_idx = int(sample.target)

        best_idx = max(range(len(lls)), key=lambda i: lls[i]) if lls else 0
        best_choice_text = eff_choices[best_idx] if eff_choices else ""
        used_uncond = (
            list(uncond_lls)
            if uncond_lls is not None and len(uncond_lls) == len(lls)
            else [0.0 for _ in lls]
        )

        for ens in ensembles:
            filtered = ens.apply([best_choice_text])
            filtered_responses[ens.name] = filtered
            pipe_scores: dict[str, float] = {}
            for m_name in spec.metrics:
                if m_name == "acc":
                    pipe_scores["acc"] = acc(lls, target_idx)
                elif m_name == "acc_bytes":
                    pipe_scores["acc_bytes"] = acc_bytes(lls, eff_choices, target_idx)
                elif m_name == "acc_norm":
                    pipe_scores["acc_norm"] = acc_norm(lls, eff_choices, target_idx)
                elif m_name == "acc_mutual_info":
                    pipe_scores["acc_mutual_info"] = acc_mutual_info(
                        lls, used_uncond, target_idx
                    )
                elif m_name == "brier_score":
                    pipe_scores["brier_score"] = brier_score(lls, target_idx)
                elif m_name in ("classification_f1", "f1"):
                    pipe_scores[m_name] = classification_f1([best_idx], [target_idx])
                elif m_name == "exact_match":
                    gold_text = (
                        eff_choices[target_idx]
                        if 0 <= target_idx < len(eff_choices)
                        else str(sample.target)
                    )
                    pipe_scores["exact_match"] = exact_match(
                        filtered[0], gold_text, ignore_case=ignore_case
                    )
            if judge is not None:
                verdict = judge.grade(
                    prompt=sample.input_text,
                    response=filtered[0],
                    reference=str(sample.target),
                )
                pipe_scores["llm_judge"] = float(verdict.score)
            metrics_by_pipeline[ens.name] = pipe_scores
    else:
        raw_texts = [r.text or "" for r in raw_responses]
        ref_text = str(sample.target)
        for ens in ensembles:
            filtered = ens.apply(raw_texts)
            filtered_responses[ens.name] = filtered
            pred_text = filtered[0] if filtered else ""
            pipe_scores = {}

            if custom_scorer is not None and hasattr(custom_scorer, "score") and raw_responses:
                scored_dict = custom_scorer.score(sample, raw_responses[0])
                if isinstance(scored_dict, dict):
                    for k, v in scored_dict.items():
                        pipe_scores[str(k)] = float(v)

            for m_name in spec.metrics:
                if m_name in pipe_scores:
                    continue
                if m_name == "exact_match":
                    pipe_scores["exact_match"] = exact_match(
                        pred_text, ref_text, ignore_case=ignore_case
                    )
                elif m_name == "numeric_exact_match":
                    pipe_scores["numeric_exact_match"] = numeric_exact_match(pred_text, ref_text)
                elif m_name == "bits_per_byte":
                    ll_val = (
                        raw_responses[0].loglikelihood
                        if raw_responses and raw_responses[0].loglikelihood is not None
                        else -1.0
                    )
                    pipe_scores["bits_per_byte"] = bits_per_byte([ll_val], [sample.input_text])
                elif m_name == "word_perplexity":
                    ll_val = (
                        raw_responses[0].loglikelihood
                        if raw_responses and raw_responses[0].loglikelihood is not None
                        else -1.0
                    )
                    pipe_scores["word_perplexity"] = word_perplexity([ll_val], [sample.input_text])
                elif m_name == "byte_perplexity":
                    ll_val = (
                        raw_responses[0].loglikelihood
                        if raw_responses and raw_responses[0].loglikelihood is not None
                        else -1.0
                    )
                    pipe_scores["byte_perplexity"] = byte_perplexity([ll_val], [sample.input_text])
                else:
                    try:
                        fn: Any = get_registry("eval_studio.metrics").get(m_name)
                        if callable(fn):
                            pipe_scores[m_name] = float(fn(pred_text, ref_text))
                    except Exception:
                        pass

            if judge is not None:
                verdict = judge.grade(
                    prompt=sample.input_text,
                    response=pred_text,
                    reference=ref_text,
                )
                pipe_scores["llm_judge"] = float(verdict.score)

            metrics_by_pipeline[ens.name] = pipe_scores

    return ScoredSample(
        sample_id=sample.sample_id,
        task_name=spec.name,
        input_text=sample.input_text,
        target=sample.target,
        choices=eff_choices,
        uncond_loglikelihoods=list(uncond_lls) if uncond_lls is not None else None,
        raw_responses=raw_responses,
        filtered_responses=filtered_responses,
        metrics=metrics_by_pipeline,
        doc_hash=sample.doc_hash,
        prompt_hash=raw_responses[0].metadata.get("prompt_hash") if raw_responses else None,
        target_hash=_sha256_text(str(sample.target)),
    )


def _aggregate_scored_samples(
    scored: list[ScoredSample],
    *,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> dict[str, MetricSummary]:
    per_metric: dict[str, list[float]] = {}
    for rec in scored:
        if not rec.metrics:
            continue
        first_pipe = next(iter(rec.metrics.values()))
        for metric_name, val in first_pipe.items():
            per_metric.setdefault(metric_name, []).append(float(val))

    summaries: dict[str, MetricSummary] = {}
    for metric_name, vals in per_metric.items():
        mean_val = sum(vals) / len(vals) if vals else 0.0
        boot = bootstrap_stderr(vals, bootstrap_iters=bootstrap_iters, seed=bootstrap_seed)
        summaries[metric_name] = MetricSummary(
            value=mean_val,
            stderr=boot.stderr,
            n_samples=len(vals),
            bootstrap_iters=boot.iters_executed,
            unresolved=False,
        )
    return summaries


def _write_receipt_bundle(
    output_dir: Path,
    manifest: ReceiptManifest,
    scored_samples: list[ScoredSample],
) -> ReceiptManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_lines = [redact_secrets(s.model_dump_json()) for s in scored_samples]
    samples_text = ("\n".join(samples_lines) + "\n") if samples_lines else ""
    samples_bytes = samples_text.encode("utf-8")
    (output_dir / "samples.jsonl").write_bytes(samples_bytes)

    samples_sha = hashlib.sha256(samples_bytes).hexdigest()
    updated = manifest.model_copy(
        update={
            "samples_sha256": samples_sha,
            "redacted_config": redact_object(manifest.redacted_config),
        }
    )
    redacted_updated = ReceiptManifest.model_validate_json(
        redact_secrets(updated.model_dump_json())
    )
    digest = compute_manifest_digest(redacted_updated)
    signed = redacted_updated.model_copy(update={"manifest_sha256": digest})
    canonical_json = signed.model_dump_json(indent=2) + "\n"
    (output_dir / "manifest.json").write_text(canonical_json, encoding="utf-8")
    (output_dir / "receipt.json").write_text(canonical_json, encoding="utf-8")
    (output_dir / "report.md").write_text(render_receipt_markdown(signed), encoding="utf-8")
    (output_dir / "report.html").write_text(render_receipt_html(signed), encoding="utf-8")
    return signed


def estimate_run_cost(
    task_path: Path,
    *,
    limit: int | None = None,
    trust_remote_code: bool = False,
) -> PreFlightCostEstimate:
    """Compute pre-flight cost estimate for `--dry-run --estimate-cost` without inference."""
    spec = load_task_spec(task_path, trust_remote_code=trust_remote_code)
    samples = load_samples(spec, base_dir=task_path.parent)
    if limit is not None:
        samples = samples[:limit]
    pricing = _extract_pricing(spec)
    repeats = int(spec.generation_kwargs.get("repeats", 1))
    max_gen_toks = int(spec.generation_kwargs.get("max_gen_toks", 128))
    avg_prompt_tokens = (
        max(1, sum(len(s.input_text.split()) for s in samples) // max(1, len(samples)))
        if samples
        else 32
    )
    guard = CostGuard()
    return guard.estimate_run(
        n_samples=len(samples),
        avg_prompt_tokens=avg_prompt_tokens,
        max_gen_toks=max_gen_toks,
        repeats=repeats,
        input_usd_per_1m=pricing.get("input_usd_per_1m", 0.0),
        output_usd_per_1m=pricing.get("output_usd_per_1m", 0.0),
        judge_enabled=bool(spec.metadata.get("judge_enabled", False)),
        judge_input_usd_per_1m=pricing.get("judge_input_usd_per_1m", 0.0),
        judge_output_usd_per_1m=pricing.get("judge_output_usd_per_1m", 0.0),
    )


def _build_unconditional_requests(
    sample_id: str,
    task_name: str,
    continuations: Sequence[str],
    *,
    prefix_tag: str,
    generation_kwargs: dict[str, Any],
    do_sample: bool,
) -> list[EvalRequest]:
    uncond_reqs: list[EvalRequest] = []
    for c_idx, cont in enumerate(continuations):
        ctx_clean, cont_clean = transfer_trailing_whitespace("Answer:", cont)
        uncond_reqs.append(
            EvalRequest(
                request_id=f"{sample_id}-{prefix_tag}c{c_idx}-uncond",
                sample_id=sample_id,
                task_name=task_name,
                output_type=OutputType.LOGLIKELIHOOD,
                context=ctx_clean,
                continuation=cont_clean,
                generation_kwargs=dict(generation_kwargs),
                do_sample=do_sample,
                prompt_hash=_sha256_text(ctx_clean),
            )
        )
    return uncond_reqs


def run_evaluation(
    task_path: Path,
    *,
    config: EvalRunConfig,
    output_dir: Path,
    base_url: str | None = None,
    plugins: list[str] | None = None,
    gateway: InferenceGateway | None = None,
    judge: LLMJudge | None = None,
    do_sample: bool = False,
    on_progress: ProgressCallback | None = None,
) -> ReceiptManifest:
    """Execute the full 4-stage evaluation pipeline and write a Portable Eval Receipt bundle."""
    if plugins:
        load_plugins(plugins, trust_remote_code=config.trust_remote_code)

    spec = load_task_spec(task_path, trust_remote_code=config.trust_remote_code)
    val = validate_task_spec(
        spec,
        enable_thinking=config.enable_thinking,
        apply_chat_template=config.apply_chat_template,
        add_bos_token=config.add_bos_token,
        base_dir=task_path.parent,
    )
    if not val.valid:
        raise ValueError("; ".join(val.errors))

    samples = load_samples(spec, base_dir=task_path.parent)
    if config.limit is not None:
        samples = samples[: config.limit]

    run_id = f"{spec.name}-{uuid.uuid4().hex[:8]}"
    task_sha256 = sha256_file(task_path)
    pricing = _extract_pricing(spec)
    cost_guard = CostGuard(max_cost_usd=config.limits.max_cost_usd)
    effective_do_sample = do_sample or bool(spec.generation_kwargs.get("do_sample", False))

    gw: InferenceGateway = (
        gateway
        if gateway is not None
        else _instantiate_gateway(
            config.model_backend,
            config.model_id,
            base_url=base_url,
            cost_guard=cost_guard,
            token_pricing=pricing,
        )
    )

    cache_root = resolve_cache_dir(config.cache_dir)
    wal = JsonlWriteAheadLog(cache_root / "wal.jsonl")
    sqlite_store = SQLiteCacheStore.for_rank(cache_root, rank=0)

    # Hydrate WAL records into lookup map for zero-duplicate mid-run resume
    wal_cache: dict[str, dict[str, Any]] = {}
    if config.use_cache and not effective_do_sample:
        for rec in wal.read_all():
            fp = rec.get("fingerprint")
            resp_data = rec.get("response")
            if isinstance(fp, str) and isinstance(resp_data, dict):
                wal_cache[fp] = resp_data

    # Group task with subtasks: evaluate each subtask and aggregate micro + macro
    if spec.subtasks:
        all_scored: list[ScoredSample] = []
        subtask_manifests: list[ReceiptManifest] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0
        total_group_samples = 0
        sub_info: list[tuple[Path, str, int]] = []
        if on_progress is not None:
            for sub_ref in spec.subtasks:
                sp = Path(sub_ref)
                if not sp.is_absolute() and not sp.is_file():
                    sp = task_path.parent / sp
                s_spec = load_task_spec(sp, trust_remote_code=config.trust_remote_code)
                s_samples = load_samples(s_spec, base_dir=sp.parent)
                if config.limit is not None:
                    s_samples = s_samples[: config.limit]
                n_sub = len(s_samples)
                total_group_samples += n_sub
                sub_info.append((sp, s_spec.name, n_sub))
            _emit_progress(
                on_progress,
                {
                    "event": "run_started",
                    "run_id": run_id,
                    "task_name": spec.name,
                    "total_samples": total_group_samples,
                },
            )

        completed_offset = 0
        pt_offset = 0
        ct_offset = 0
        cost_offset = 0.0
        for sub_idx, sub_ref in enumerate(spec.subtasks):
            sub_path = Path(sub_ref)
            if not sub_path.is_absolute() and not sub_path.is_file():
                sub_path = task_path.parent / sub_path
            sub_dir = output_dir / f"_subtask_{sub_path.stem}"
            sub_cb: ProgressCallback | None = None
            sub_count = 0
            if on_progress is not None:
                _, sub_name, sub_count = sub_info[sub_idx]
                _emit_progress(
                    on_progress,
                    {
                        "event": "subtask_started",
                        "run_id": run_id,
                        "task_name": spec.name,
                        "subtask_name": sub_name,
                        "subtask_index": sub_idx,
                        "total_subtasks": len(spec.subtasks),
                    },
                )

                def _sub_progress(
                    ev: dict[str, Any],
                    *,
                    _cb: ProgressCallback = on_progress,
                    _idx_off: int = completed_offset,
                    _pt_off: int = pt_offset,
                    _ct_off: int = ct_offset,
                    _cost_off: float = cost_offset,
                ) -> None:
                    ev_type = ev.get("event")
                    if ev_type in ("run_started", "completed"):
                        return
                    translated = dict(ev)
                    translated["run_id"] = run_id
                    if ev_type == "sample_scored":
                        translated["index"] = _idx_off + int(ev.get("index", 0))
                        translated["total"] = total_group_samples
                        translated["spent_usd"] = _cost_off + float(ev.get("spent_usd", 0.0))
                        translated["prompt_tokens"] = _pt_off + int(ev.get("prompt_tokens", 0))
                        translated["completion_tokens"] = _ct_off + int(
                            ev.get("completion_tokens", 0)
                        )
                    elif ev_type == "budget_exceeded":
                        translated["index"] = _idx_off + int(ev.get("index", 0))
                        translated["total"] = total_group_samples
                        translated["spent_usd"] = max(
                            _cost_off + float(ev.get("spent_usd", 0.0)),
                            cost_guard.spent_usd,
                        )
                    _emit_progress(_cb, translated)

                sub_cb = _sub_progress

            sub_manifest = run_evaluation(
                sub_path,
                config=config,
                output_dir=sub_dir,
                base_url=base_url,
                plugins=None,
                gateway=gw,
                judge=judge,
                do_sample=effective_do_sample,
                on_progress=sub_cb,
            )
            subtask_manifests.append(sub_manifest)
            total_prompt_tokens += sub_manifest.total_prompt_tokens
            total_completion_tokens += sub_manifest.total_completion_tokens
            total_cost_usd += sub_manifest.total_cost_usd
            if on_progress is not None:
                completed_offset += sub_count
                pt_offset = total_prompt_tokens
                ct_offset = total_completion_tokens
                cost_offset = total_cost_usd
            sub_samples_file = sub_dir / "samples.jsonl"
            if sub_samples_file.is_file():
                for line in sub_samples_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        all_scored.append(ScoredSample.model_validate_json(line))

        # Roll up every metric present across subtasks (never silently drop heterogeneous keys)
        all_metric_names: list[str] = []
        for sm in subtask_manifests:
            for m_name in sm.metrics:
                if m_name not in all_metric_names:
                    all_metric_names.append(m_name)

        group_metrics: dict[str, MetricSummary] = {}
        for m_name in all_metric_names:
            sub_stats = [
                SubtaskStats(
                    name=sm.task_name,
                    score=sm.metrics[m_name].value,
                    stderr=sm.metrics[m_name].stderr or 0.0,
                    n_samples=max(1, sm.metrics[m_name].n_samples),
                )
                for sm in subtask_manifests
                if m_name in sm.metrics
            ]
            agg_micro = aggregate_group(sub_stats, weight_by_size=True)
            agg_macro = aggregate_group(sub_stats, weight_by_size=False)
            primary = agg_micro if spec.weight_by_size else agg_macro
            group_metrics[m_name] = MetricSummary(
                value=primary.score,
                stderr=primary.stderr,
                n_samples=primary.total_samples,
                bootstrap_iters=config.bootstrap_iters,
                unresolved=False,
            )
            group_metrics[f"{m_name}_micro"] = MetricSummary(
                value=agg_micro.score,
                stderr=agg_micro.stderr,
                n_samples=agg_micro.total_samples,
                bootstrap_iters=config.bootstrap_iters,
                unresolved=False,
            )
            group_metrics[f"{m_name}_macro"] = MetricSummary(
                value=agg_macro.score,
                stderr=agg_macro.stderr,
                n_samples=agg_macro.total_samples,
                bootstrap_iters=config.bootstrap_iters,
                unresolved=False,
            )

        repro_cmd = build_reproduce_command(
            task_path=str(task_path),
            model_backend=config.model_backend,
            output_dir=str(output_dir),
            trust_remote_code=config.trust_remote_code,
            plugins=plugins,
            max_cost_usd=config.limits.max_cost_usd,
        )
        group_manifest = ReceiptManifest(
            harness_version=__version__,
            run_id=run_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            backend=gw.backend_name,
            model_id=gw.model_id,
            task_name=spec.name,
            task_sha256=task_sha256,
            samples_sha256="",
            task_path=str(task_path),
            dtype=config.dtype,
            quantization=config.quantization,
            batch_size=config.batch_size,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cost_usd=total_cost_usd,
            reproduce_command=repro_cmd,
            seeds=config.seeds.model_dump(),
            redacted_config={
                **config.model_dump(mode="json"),
                "dtype": config.dtype,
                "quantization": config.quantization,
                "batch_size": config.batch_size,
                "reproduce_command": repro_cmd,
            },
            metrics=group_metrics,
        )
        signed_group = _write_receipt_bundle(output_dir, group_manifest, all_scored)
        _emit_progress(
            on_progress,
            {
                "event": "completed",
                "run_id": signed_group.run_id,
                "task_name": signed_group.task_name,
                "spent_usd": signed_group.total_cost_usd,
                "total_samples": total_group_samples,
                "manifest_sha256": signed_group.manifest_sha256,
            },
        )
        return signed_group

    # Optional Tier 3 custom Solver
    custom_solver: Any = None
    solver_name = spec.metadata.get("solver")
    if isinstance(solver_name, str) and solver_name:
        solver_obj = get_registry("eval_studio.solvers").get(solver_name)
        custom_solver = solver_obj() if isinstance(solver_obj, type) else solver_obj

    # Few-shot rotation sets (strictly excluding active doc)
    fewshot_assignments: list[SampledFewShot] = []
    if spec.num_fewshot > 0 and len(samples) > 1:
        fewshot_assignments = assign_fewshot_sets(
            samples,
            samples,
            num_fewshot=spec.num_fewshot,
            k_sets=max(1, spec.fewshot_rotation_sets),
            seed=config.seeds.fewshot_seed,
        )

    repeats = max(1, int(spec.generation_kwargs.get("repeats", 1)))
    need_pmi = "acc_mutual_info" in spec.metrics
    strict_budget = config.limits.max_cost_usd is not None
    olmes_enabled = (
        spec.olmes is not None
        and spec.olmes.enabled
        and spec.output_type == OutputType.MULTIPLE_CHOICE
    )

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0

    def _build_base_prompt(idx: int, sample: Sample) -> str:
        prompt_text = _render_sample_prompt(spec, sample)
        if fewshot_assignments:
            target_delim = "" if config.apply_chat_template else spec.target_delimiter
            exemplar_blocks = [
                f"{_render_sample_prompt(spec, ex)}{target_delim}{ex.target}"
                for ex in fewshot_assignments[idx].exemplars
            ]
            if exemplar_blocks:
                prefix = spec.fewshot_delimiter.join(exemplar_blocks)
                prompt_text = f"{prefix}{spec.fewshot_delimiter}{prompt_text}"
        if spec.gen_prefix:
            prompt_text = f"{prompt_text}{spec.gen_prefix}"
        return prompt_text

    extra_olmes_meta: dict[str, Any] = {}
    total_samples = len(samples)
    _emit_progress(
        on_progress,
        {
            "event": "run_started",
            "run_id": run_id,
            "task_name": spec.name,
            "total_samples": total_samples,
        },
    )
    current_idx = 0
    budget_emitted = False

    try:
        if olmes_enabled and spec.olmes is not None:
            olmes_formats = [
                fmt.value if hasattr(fmt, "value") else str(fmt) for fmt in spec.olmes.formats
            ]
            if not olmes_formats:
                olmes_formats = ["cloze", "mcf"]

            scored_by_fmt: dict[str, list[ScoredSample]] = {fk: [] for fk in olmes_formats}
            for idx, sample in enumerate(samples):
                current_idx = idx
                base_prompt = _build_base_prompt(idx, sample)
                if not sample.choices:
                    continue
                bundles = build_olmes_dual_prompts(
                    base_prompt,
                    sample.choices,
                    labels=spec.olmes_option_labels,
                )
                for fk in olmes_formats:
                    bundle = bundles[fk]
                    prompt_hash = _sha256_text(bundle.context)
                    cond_reqs: list[EvalRequest] = []
                    for c_idx, cont in enumerate(bundle.continuations):
                        ctx_clean, cont_clean = transfer_trailing_whitespace(bundle.context, cont)
                        cond_reqs.append(
                            EvalRequest(
                                request_id=f"{sample.sample_id}-{fk}-c{c_idx}",
                                sample_id=sample.sample_id,
                                task_name=spec.name,
                                output_type=OutputType.LOGLIKELIHOOD,
                                context=ctx_clean,
                                continuation=cont_clean,
                                generation_kwargs=dict(spec.generation_kwargs),
                                do_sample=effective_do_sample,
                                prompt_hash=prompt_hash,
                            )
                        )
                    cond_resps, pt, ct, cost = _dispatch_cached_requests(
                        cond_reqs,
                        gw=gw,
                        task_sha256=task_sha256,
                        use_cache=config.use_cache,
                        effective_do_sample=effective_do_sample,
                        wal=wal,
                        sqlite_store=sqlite_store,
                        wal_cache=wal_cache,
                        enable_thinking=config.enable_thinking,
                        think_end_token=config.think_end_token,
                        dtype=config.dtype,
                        quantization=config.quantization,
                        strict_budget=strict_budget,
                    )
                    total_prompt_tokens += pt
                    total_completion_tokens += ct
                    total_cost_usd += cost

                    uncond_lls: list[float] | None = None
                    if need_pmi:
                        uncond_reqs = _build_unconditional_requests(
                            sample.sample_id,
                            spec.name,
                            bundle.continuations,
                            prefix_tag=f"{fk}-",
                            generation_kwargs=spec.generation_kwargs,
                            do_sample=effective_do_sample,
                        )
                        uncond_resps, upt, uct, ucost = _dispatch_cached_requests(
                            uncond_reqs,
                            gw=gw,
                            task_sha256=task_sha256,
                            use_cache=config.use_cache,
                            effective_do_sample=effective_do_sample,
                            wal=wal,
                            sqlite_store=sqlite_store,
                            wal_cache=wal_cache,
                            enable_thinking=config.enable_thinking,
                            think_end_token=config.think_end_token,
                            dtype=config.dtype,
                            quantization=config.quantization,
                            strict_budget=strict_budget,
                        )
                        total_prompt_tokens += upt
                        total_completion_tokens += uct
                        total_cost_usd += ucost
                        uncond_lls = [
                            r.loglikelihood if r.loglikelihood is not None else 0.0
                            for r in uncond_resps
                        ]

                    scored_by_fmt[fk].append(
                        _score_sample(
                            spec,
                            sample,
                            cond_resps,
                            enable_thinking=config.enable_thinking,
                            think_end_token=config.think_end_token,
                            continuations=bundle.continuations,
                            uncond_lls=uncond_lls,
                            judge=judge,
                        )
                    )
                primary_fmt = olmes_formats[0]
                if scored_by_fmt[primary_fmt]:
                    _emit_progress(
                        on_progress,
                        {
                            "event": "sample_scored",
                            "run_id": run_id,
                            "index": idx,
                            "total": total_samples,
                            "sample": scored_by_fmt[primary_fmt][-1].model_dump(mode="json"),
                            "spent_usd": total_cost_usd,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                        },
                    )

            format_metrics: dict[str, dict[str, MetricSummary]] = {
                fk: _aggregate_scored_samples(
                    scored_by_fmt[fk],
                    bootstrap_iters=config.bootstrap_iters,
                    bootstrap_seed=config.seeds.bootstrap_seed,
                )
                for fk in olmes_formats
            }

            pref_metric = spec.olmes.metric_preference or "acc_norm"
            best_fmt = olmes_formats[0]
            if (
                "cloze" in format_metrics
                and "mcf" in format_metrics
                and format_metrics["cloze"]
                and format_metrics["mcf"]
            ):
                cmp_metric = (
                    pref_metric
                    if pref_metric in format_metrics["cloze"]
                    else next(iter(format_metrics["cloze"]))
                )
                olmes_sel = select_olmes_best_format(
                    format_metrics["cloze"][cmp_metric].value,
                    format_metrics["mcf"][cmp_metric].value,
                )
                best_fmt = olmes_sel.selected_format.value
                extra_olmes_meta = {
                    "olmes_selection": {
                        "metric": cmp_metric,
                        "selected_format": olmes_sel.selected_format.value,
                        "cloze_score": olmes_sel.cloze_score,
                        "mcf_score": olmes_sel.mcf_score,
                        "best_score": olmes_sel.best_score,
                        "delta_mcf_minus_cloze": olmes_sel.delta_mcf_minus_cloze,
                    }
                }

            scored_samples = scored_by_fmt[best_fmt]
            metrics: dict[str, MetricSummary] = dict(format_metrics[best_fmt])
            for fk in olmes_formats:
                for m_key, m_summary in format_metrics[fk].items():
                    metrics[f"{m_key}_{fk}"] = m_summary
        else:
            sample_plans: list[tuple[Sample, list[EvalRequest], list[EvalRequest]]] = []
            for idx, sample in enumerate(samples):
                uncond_reqs_std: list[EvalRequest] = []
                if custom_solver is not None and hasattr(custom_solver, "build_request"):
                    base_req: EvalRequest = custom_solver.build_request(sample)
                    requests = [base_req]
                else:
                    prompt_text = _build_base_prompt(idx, sample)
                    prompt_hash = _sha256_text(prompt_text)
                    if spec.output_type == OutputType.MULTIPLE_CHOICE and sample.choices:
                        requests = []
                        for c_idx, choice in enumerate(sample.choices):
                            ctx_clean, cont_clean = transfer_trailing_whitespace(
                                prompt_text, choice
                            )
                            requests.append(
                                EvalRequest(
                                    request_id=f"{sample.sample_id}-c{c_idx}",
                                    sample_id=sample.sample_id,
                                    task_name=spec.name,
                                    output_type=OutputType.LOGLIKELIHOOD,
                                    context=ctx_clean,
                                    continuation=cont_clean,
                                    generation_kwargs=dict(spec.generation_kwargs),
                                    do_sample=effective_do_sample,
                                    prompt_hash=prompt_hash,
                                )
                            )
                        if need_pmi:
                            uncond_reqs_std = _build_unconditional_requests(
                                sample.sample_id,
                                spec.name,
                                sample.choices,
                                prefix_tag="",
                                generation_kwargs=spec.generation_kwargs,
                                do_sample=effective_do_sample,
                            )
                    elif spec.output_type == OutputType.CHAT:
                        msgs = _render_sample_messages(spec, sample, prompt_text)
                        requests = [
                            EvalRequest(
                                request_id=(
                                    f"{sample.sample_id}-chat"
                                    if repeats == 1
                                    else f"{sample.sample_id}-chat-r{r}"
                                ),
                                sample_id=sample.sample_id,
                                task_name=spec.name,
                                output_type=OutputType.CHAT,
                                context=prompt_text,
                                messages=msgs,
                                generation_kwargs=dict(spec.generation_kwargs),
                                do_sample=effective_do_sample,
                                repeats=repeats,
                                prompt_hash=prompt_hash,
                            )
                            for r in range(repeats)
                        ]
                    else:
                        requests = [
                            EvalRequest(
                                request_id=(
                                    f"{sample.sample_id}-0"
                                    if repeats == 1
                                    else f"{sample.sample_id}-r{r}"
                                ),
                                sample_id=sample.sample_id,
                                task_name=spec.name,
                                output_type=spec.output_type,
                                context=prompt_text,
                                generation_kwargs=dict(spec.generation_kwargs),
                                do_sample=effective_do_sample,
                                repeats=repeats,
                                prompt_hash=prompt_hash,
                            )
                            for r in range(repeats)
                        ]
                sample_plans.append((sample, requests, uncond_reqs_std))

            scored_samples = []
            if strict_budget or on_progress is not None:
                for idx, (sample, requests, uncond_reqs_std) in enumerate(sample_plans):
                    current_idx = idx
                    sample_responses, pt, ct, cost = _dispatch_cached_requests(
                        requests,
                        gw=gw,
                        task_sha256=task_sha256,
                        use_cache=config.use_cache,
                        effective_do_sample=effective_do_sample,
                        wal=wal,
                        sqlite_store=sqlite_store,
                        wal_cache=wal_cache,
                        enable_thinking=config.enable_thinking,
                        think_end_token=config.think_end_token,
                        dtype=config.dtype,
                        quantization=config.quantization,
                        strict_budget=strict_budget,
                    )
                    total_prompt_tokens += pt
                    total_completion_tokens += ct
                    total_cost_usd += cost

                    uncond_lls_std: list[float] | None = None
                    if uncond_reqs_std:
                        uncond_resps, upt, uct, ucost = _dispatch_cached_requests(
                            uncond_reqs_std,
                            gw=gw,
                            task_sha256=task_sha256,
                            use_cache=config.use_cache,
                            effective_do_sample=effective_do_sample,
                            wal=wal,
                            sqlite_store=sqlite_store,
                            wal_cache=wal_cache,
                            enable_thinking=config.enable_thinking,
                            think_end_token=config.think_end_token,
                            dtype=config.dtype,
                            quantization=config.quantization,
                            strict_budget=strict_budget,
                        )
                        total_prompt_tokens += upt
                        total_completion_tokens += uct
                        total_cost_usd += ucost
                        uncond_lls_std = [
                            r.loglikelihood if r.loglikelihood is not None else 0.0
                            for r in uncond_resps
                        ]

                    scored = _score_sample(
                        spec,
                        sample,
                        sample_responses,
                        enable_thinking=config.enable_thinking,
                        think_end_token=config.think_end_token,
                        uncond_lls=uncond_lls_std,
                        judge=judge,
                    )
                    scored_samples.append(scored)
                    _emit_progress(
                        on_progress,
                        {
                            "event": "sample_scored",
                            "run_id": run_id,
                            "index": idx,
                            "total": total_samples,
                            "sample": scored.model_dump(mode="json"),
                            "spent_usd": total_cost_usd,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                        },
                    )
            else:
                flat_main: list[EvalRequest] = []
                flat_uncond: list[EvalRequest] = []
                for _, reqs, ureqs in sample_plans:
                    flat_main.extend(reqs)
                    flat_uncond.extend(ureqs)

                all_main_resps, pt, ct, cost = _dispatch_cached_requests(
                    flat_main,
                    gw=gw,
                    task_sha256=task_sha256,
                    use_cache=config.use_cache,
                    effective_do_sample=effective_do_sample,
                    wal=wal,
                    sqlite_store=sqlite_store,
                    wal_cache=wal_cache,
                    enable_thinking=config.enable_thinking,
                    think_end_token=config.think_end_token,
                    dtype=config.dtype,
                    quantization=config.quantization,
                    strict_budget=False,
                )
                total_prompt_tokens += pt
                total_completion_tokens += ct
                total_cost_usd += cost

                all_uncond_resps: list[EvalResponse] = []
                if flat_uncond:
                    all_uncond_resps, upt, uct, ucost = _dispatch_cached_requests(
                        flat_uncond,
                        gw=gw,
                        task_sha256=task_sha256,
                        use_cache=config.use_cache,
                        effective_do_sample=effective_do_sample,
                        wal=wal,
                        sqlite_store=sqlite_store,
                        wal_cache=wal_cache,
                        enable_thinking=config.enable_thinking,
                        think_end_token=config.think_end_token,
                        dtype=config.dtype,
                        quantization=config.quantization,
                        strict_budget=False,
                    )
                    total_prompt_tokens += upt
                    total_completion_tokens += uct
                    total_cost_usd += ucost

                main_pos = 0
                uncond_pos = 0
                for sample, reqs, ureqs in sample_plans:
                    n_main = len(reqs)
                    s_resps = all_main_resps[main_pos : main_pos + n_main]
                    main_pos += n_main

                    s_uncond_lls: list[float] | None = None
                    if ureqs:
                        n_uncond = len(ureqs)
                        u_slice = all_uncond_resps[uncond_pos : uncond_pos + n_uncond]
                        uncond_pos += n_uncond
                        s_uncond_lls = [
                            r.loglikelihood if r.loglikelihood is not None else 0.0 for r in u_slice
                        ]

                    scored_samples.append(
                        _score_sample(
                            spec,
                            sample,
                            s_resps,
                            enable_thinking=config.enable_thinking,
                            think_end_token=config.think_end_token,
                            uncond_lls=s_uncond_lls,
                            judge=judge,
                        )
                    )

            metrics = _aggregate_scored_samples(
                scored_samples,
                bootstrap_iters=config.bootstrap_iters,
                bootstrap_seed=config.seeds.bootstrap_seed,
            )
    except CostBudgetExceededError as exc:
        if not budget_emitted:
            budget_emitted = True
            _emit_progress(
                on_progress,
                {
                    "event": "budget_exceeded",
                    "run_id": run_id,
                    "task_name": spec.name,
                    "index": current_idx,
                    "total": total_samples,
                    "spent_usd": max(total_cost_usd, cost_guard.spent_usd),
                    "max_cost_usd": config.limits.max_cost_usd,
                    "error": redact_secrets(str(exc)),
                },
            )
        raise

    source_paper_path_raw = spec.metadata.get("source_path")
    source_paper_sha256 = spec.metadata.get("source_paper_sha256")
    repro_cmd = build_reproduce_command(
        task_path=str(task_path),
        model_backend=config.model_backend,
        output_dir=str(output_dir),
        trust_remote_code=config.trust_remote_code,
        plugins=plugins,
        max_cost_usd=config.limits.max_cost_usd,
    )

    manifest = ReceiptManifest(
        harness_version=__version__,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        backend=gw.backend_name,
        model_id=gw.model_id,
        task_name=spec.name,
        task_sha256=task_sha256,
        samples_sha256="",
        task_path=str(task_path),
        source_paper_path=str(source_paper_path_raw) if source_paper_path_raw else None,
        source_paper_sha256=str(source_paper_sha256) if source_paper_sha256 else None,
        dtype=config.dtype,
        quantization=config.quantization,
        batch_size=config.batch_size,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cost_usd=total_cost_usd,
        reproduce_command=repro_cmd,
        seeds=config.seeds.model_dump(),
        redacted_config={
            **config.model_dump(mode="json"),
            "dtype": config.dtype,
            "quantization": config.quantization,
            "batch_size": config.batch_size,
            "generation_kwargs": dict(spec.generation_kwargs),
            "task_limits": spec.metadata.get("limits"),
            **extra_olmes_meta,
            "reproduce_command": repro_cmd,
        },
        metrics=metrics,
    )
    signed_manifest = _write_receipt_bundle(output_dir, manifest, scored_samples)
    _emit_progress(
        on_progress,
        {
            "event": "completed",
            "run_id": signed_manifest.run_id,
            "task_name": signed_manifest.task_name,
            "spent_usd": signed_manifest.total_cost_usd,
            "total_samples": total_samples,
            "manifest_sha256": signed_manifest.manifest_sha256,
        },
    )
    return signed_manifest


def rescore_receipt(
    receipt_dir: Path,
    *,
    output_dir: Path | None = None,
    task_path: Path | None = None,
    trust_remote_code: bool = False,
    judge: LLMJudge | None = None,
) -> ReceiptManifest:
    """Re-score saved raw responses 100% offline without invoking any inference gateway."""
    manifest_path = receipt_dir / "manifest.json"
    samples_path = receipt_dir / "samples.jsonl"
    if not manifest_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError(
            f"Receipt directory must contain manifest.json and samples.jsonl: {receipt_dir}"
        )

    orig_manifest = ReceiptManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    raw_lines = [
        line for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not raw_lines:
        raise ValueError(f"Receipt samples.jsonl is empty: {samples_path}")
    orig_samples = [ScoredSample.model_validate_json(line) for line in raw_lines]

    if task_path is not None:
        spec = load_task_spec(task_path, trust_remote_code=trust_remote_code)
    elif orig_manifest.task_path and Path(orig_manifest.task_path).is_file():
        spec = load_task_spec(Path(orig_manifest.task_path), trust_remote_code=trust_remote_code)
    else:
        spec = TaskSpec(
            name=orig_manifest.task_name,
            metrics=list(orig_manifest.metrics.keys()) or ["exact_match"],
        )

    rescored: list[ScoredSample] = []
    for rec in orig_samples:
        sample_stub = Sample(
            sample_id=rec.sample_id,
            input_text=rec.input_text,
            target=rec.target,
            choices=rec.choices,
            doc_hash=rec.doc_hash,
        )
        rescored.append(
            _score_sample(
                spec,
                sample_stub,
                rec.raw_responses,
                continuations=rec.choices if rec.choices else None,
                uncond_lls=rec.uncond_loglikelihoods,
                judge=judge,
            )
        )

    first_metric = next(iter(orig_manifest.metrics.values()), None)
    boot_iters = first_metric.bootstrap_iters if first_metric is not None else 1000
    boot_seed = int(orig_manifest.seeds.get("bootstrap_seed", 1234))

    new_metrics = _aggregate_scored_samples(
        rescored,
        bootstrap_iters=boot_iters,
        bootstrap_seed=boot_seed,
    )
    new_manifest = orig_manifest.model_copy(
        update={
            "run_id": f"{orig_manifest.run_id}-rescored",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": new_metrics,
        }
    )
    target_dir = output_dir if output_dir is not None else receipt_dir
    return _write_receipt_bundle(target_dir, new_manifest, rescored)
