"""Pre-flight task specification validator with secret, SSTI, and snapshot checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_studio.core.schema import OutputType, TaskSpec, TrustTier
from eval_studio.gateway.secrets import contains_secret
from eval_studio.tasks.loader import TaskLoadError, load_samples
from eval_studio.tasks.templates import TemplateSecurityError, render_sandboxed_template


@dataclass
class ValidationResult:
    """Structured output of `eval-studio validate`."""

    task_name: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_task_spec(
    spec: TaskSpec,
    *,
    enable_thinking: bool = False,
    apply_chat_template: bool = False,
    add_bos_token: bool | None = None,
    base_dir: Path | None = None,
    snapshot_path: Path | None = None,
) -> ValidationResult:
    """Validate task schema invariants, secret hygiene, template safety, and golden snapshots."""
    errors: list[str] = []
    warnings: list[str] = []

    serialized = spec.model_dump_json()
    if contains_secret(serialized):
        errors.append("Task specification appears to contain an embedded API key or secret.")

    if apply_chat_template and add_bos_token is True:
        errors.append(
            "apply_chat_template=True cannot be combined with add_bos_token=True "
            "(causes duplicate BOS tokens)."
        )

    if enable_thinking and spec.output_type in (
        OutputType.LOGLIKELIHOOD,
        OutputType.LOGLIKELIHOOD_ROLLING,
        OutputType.MULTIPLE_CHOICE,
    ):
        errors.append("enable_thinking=True is incompatible with log-likelihood output_type.")

    if not spec.metrics:
        warnings.append("Task defines no metrics; defaulting to bypass or exact_match recommended.")

    if "date_stamp" not in spec.metadata:
        warnings.append("Task metadata.date_stamp is unset (recommended for auditability).")

    # Dry-render templates inside SandboxedEnvironment for Tiers 1 & 2 (and snapshot check)
    sample_ctx: dict[str, Any] = {
        "sample_id": "dryrun_0",
        "input_text": "Sample clinical question?",
        "target": "A",
        "choices": ["Option A", "Option B"],
    }
    loaded_first = False
    if spec.dataset_path or isinstance(spec.metadata.get("items"), list):
        try:
            samples = load_samples(spec, base_dir=base_dir)
            if samples:
                s0 = samples[0]
                sample_ctx = {
                    "sample_id": s0.sample_id,
                    "input_text": s0.input_text,
                    "target": s0.target,
                    "choices": s0.choices or ["Option A", "Option B"],
                    **s0.metadata,
                }
                loaded_first = True
        except (TaskLoadError, ImportError) as exc:
            errors.append(str(exc))

    rendered_prompt: str | None = None
    if spec.trust_tier in (TrustTier.TIER1_ZERO_CODE, TrustTier.TIER2_DECLARATIVE):
        try:
            rendered_prompt = render_sandboxed_template(spec.doc_to_text, sample_ctx)
        except TemplateSecurityError as exc:
            errors.append(f"Blocked unsafe doc_to_text template: {exc}")
        try:
            render_sandboxed_template(spec.doc_to_target, sample_ctx)
        except TemplateSecurityError as exc:
            errors.append(f"Blocked unsafe doc_to_target template: {exc}")

    if snapshot_path is not None:
        if not snapshot_path.is_file():
            errors.append(f"Golden snapshot file not found: {snapshot_path}")
        else:
            expected = snapshot_path.read_text(encoding="utf-8").rstrip("\n")
            if rendered_prompt is None:
                try:
                    rendered_prompt = render_sandboxed_template(spec.doc_to_text, sample_ctx)
                except TemplateSecurityError as exc:
                    errors.append(f"Template failed during snapshot check: {exc}")
            if rendered_prompt is not None and rendered_prompt.rstrip("\n") != expected:
                errors.append(
                    f"Golden prompt snapshot drift detected against {snapshot_path} "
                    f"(loaded_dataset={loaded_first})."
                )

    return ValidationResult(
        task_name=spec.name,
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
