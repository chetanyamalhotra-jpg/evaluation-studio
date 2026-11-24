"""Top-level evaluation run configuration and safety/resource limits."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval_studio.core.schema import RuntimeLimits, TaskSpec, validate_thinking_output_type

__all__ = [
    "EvalRunConfig",
    "RuntimeLimits",
    "SeedConfig",
    "is_instruct_or_chat_model",
    "load_run_config",
    "resolve_task_formatting",
    "validate_unknown_args",
]

_INSTRUCT_MODEL_RE = re.compile(
    r"(?:^|[-_/])(instruct|chat|it|sft|dpo)(?:$|[-_/.0-9])",
    re.IGNORECASE,
)


def is_instruct_or_chat_model(model_id: str) -> bool:
    """Heuristically detect instruct/chat-tuned model identifiers."""
    return _INSTRUCT_MODEL_RE.search(model_id) is not None


class SeedConfig(BaseModel):
    """Independent RNG seeds for deterministic evaluation runs."""

    model_config = ConfigDict(extra="forbid")

    python_seed: int = 0
    numpy_seed: int = 1234
    fewshot_seed: int = 1234
    bootstrap_seed: int = 1234


class EvalRunConfig(BaseModel):
    """Canonical configuration for a single `eval-studio run` invocation."""

    model_config = ConfigDict(extra="forbid")

    model_backend: str = "mock"
    model_id: str = "mock-deterministic"
    tasks: list[str] = Field(default_factory=list)
    plugins: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=1, gt=0)
    dtype: str | None = None
    quantization: str | None = None
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    target_delimiter: str | None = None
    add_bos_token: bool | None = None
    enable_thinking: bool = False
    think_end_token: str | None = None
    trust_remote_code: bool = False
    use_cache: bool = True
    cache_dir: Path | None = None
    output_dir: Path = Path("receipts")
    bootstrap_iters: int = Field(default=1000, ge=0)
    model_args: dict[str, Any] = Field(default_factory=dict)
    seeds: SeedConfig = Field(default_factory=SeedConfig)
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)

    @model_validator(mode="after")
    def _validate_formatting_invariants(self) -> EvalRunConfig:
        """Enforce anti-bug guardrails on BOS, thinking-token, plugins, and chat formatting."""
        if self.apply_chat_template and self.add_bos_token is True:
            raise ValueError(
                "Invalid configuration: apply_chat_template=True cannot be combined "
                "with add_bos_token=True (causes duplicate BOS tokens)."
            )
        if self.enable_thinking and not self.think_end_token:
            raise ValueError(
                "Invalid configuration: enable_thinking=True requires an explicit "
                "think_end_token delimiter (e.g., '</think>')."
            )
        if self.plugins and not self.trust_remote_code:
            raise ValueError(
                "Invalid configuration: specifying custom plugins requires trust_remote_code=True."
            )
        if self.apply_chat_template:
            self.target_delimiter = ""
        for msg in self.formatting_warnings():
            warnings.warn(msg, UserWarning, stacklevel=2)
        return self

    def formatting_warnings(self) -> list[str]:
        """Return non-fatal formatting warnings for the current configuration."""
        notes: list[str] = []
        if not self.apply_chat_template and is_instruct_or_chat_model(self.model_id):
            notes.append(
                f"Model '{self.model_id}' appears to be instruct/chat-tuned, but "
                "apply_chat_template=False; this can silently degrade evaluation accuracy."
            )
        return notes


def resolve_task_formatting(spec: TaskSpec, config: EvalRunConfig) -> TaskSpec:
    """Resolve task formatting invariants against an EvalRunConfig without mutating input."""
    validate_thinking_output_type(spec.output_type, config.enable_thinking)
    if config.apply_chat_template and spec.target_delimiter != "":
        return spec.model_copy(update={"target_delimiter": ""})
    return spec


def validate_unknown_args(
    args: Mapping[str, Any],
    allowed_keys: AbstractSet[str],
    *,
    context: str = "arguments",
) -> None:
    """Fail loudly when unknown model/generation configuration arguments are supplied."""
    unknown = sorted(set(args) - set(allowed_keys))
    if unknown:
        raise ValueError(
            f"Unknown {context}: {', '.join(unknown)}. Allowed keys: {sorted(allowed_keys)}"
        )


def _contains_secret_in_value(value: Any, checker: Callable[[str], bool]) -> bool:
    if isinstance(value, str):
        return checker(value)
    if isinstance(value, Mapping):
        return any(
            _contains_secret_in_value(k, checker) or _contains_secret_in_value(v, checker)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_secret_in_value(item, checker) for item in value)
    return checker(str(value))


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(dict(existing), value)
        elif isinstance(value, Mapping):
            merged[key] = _deep_merge({}, value)
        else:
            merged[key] = value
    return merged


def load_run_config(
    path: Path,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
) -> EvalRunConfig:
    """Load an EvalRunConfig from a UTF-8 YAML file with secret checking and CLI overrides."""
    from eval_studio.gateway.secrets import contains_secret

    raw_text = path.read_text(encoding="utf-8")
    if contains_secret(raw_text):
        raise ValueError(
            f"Run configuration '{path}' appears to contain an embedded API key or secret."
        )
    loaded: Any = yaml.safe_load(raw_text)
    if loaded is None:
        data: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        data = dict(loaded)
    else:
        raise ValueError(f"Run configuration '{path}' must be a YAML mapping.")

    if _contains_secret_in_value(data, contains_secret) or contains_secret(
        json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    ):
        raise ValueError(
            f"Run configuration '{path}' appears to contain an embedded API key or secret."
        )

    if cli_overrides:
        data = _deep_merge(data, cli_overrides)

    return EvalRunConfig.model_validate(data)
