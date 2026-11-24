"""N+M four-stage intermediate pipeline schemas.

Stages:
1. Sample & TaskSpec — normalized dataset items and declarative task metadata.
2. EvalRequest — tokenization-agnostic inference request with explicit limits.
3. EvalResponse — raw inference response, logprobs, usage, and backend provenance.
4. ScoredSample — post-processed outputs, per-sample metrics, and integrity hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrustTier(str, Enum):
    """BYOT untrusted-code boundary tier."""

    TIER1_ZERO_CODE = "tier1_zero_code"
    TIER2_DECLARATIVE = "tier2_declarative"
    TIER3_CODE_PLUGIN = "tier3_code_plugin"


class OutputType(str, Enum):
    """Supported evaluation request/output modes."""

    LOGLIKELIHOOD = "loglikelihood"
    LOGLIKELIHOOD_ROLLING = "loglikelihood_rolling"
    GENERATE_UNTIL = "generate_until"
    MULTIPLE_CHOICE = "multiple_choice"
    CHAT = "chat"


class MultipleChoiceFormat(str, Enum):
    """OLMES multiple-choice formatting mode."""

    CLOZE = "cloze"
    MCF = "mcf"
    DUAL = "dual"


class OlmesFormat(str, Enum):
    """Individual OLMES evaluation format."""

    CLOZE = "cloze"
    MCF = "mcf"


class OlmesConfig(BaseModel):
    """Configuration for OLMES Cloze + MCF dual-format evaluation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    formats: list[OlmesFormat] = Field(
        default_factory=lambda: [OlmesFormat.CLOZE, OlmesFormat.MCF]
    )
    choice_prefix: str = "\n "
    metric_preference: str = "acc_norm"

    @model_validator(mode="after")
    def _validate_prefix(self) -> OlmesConfig:
        if " " not in self.choice_prefix:
            raise ValueError(
                "OLMES choice_prefix must include a leading space before option labels "
                "(e.g. '\\n ')."
            )
        return self


class OlmesFormatResult(BaseModel):
    """Dual Cloze vs. MCF score report selecting the higher-performing format."""

    cloze_score: float
    mcf_score: float
    selected_format: OlmesFormat
    selected_score: float
    margin: float


class RuntimeLimits(BaseModel):
    """Execution limits recorded in Portable Eval Receipts."""

    model_config = ConfigDict(extra="forbid")

    max_cost_usd: float | None = Field(default=None, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)
    wall_clock_seconds: float | None = Field(default=None, gt=0.0)
    max_tool_calls: int | None = Field(default=None, ge=0)


class TextContentPart(BaseModel):
    """Plain text content block inside a multimodal chat message."""

    type: Literal["text"] = "text"
    text: str


class ImageContentPart(BaseModel):
    """Image content block inside a multimodal chat message."""

    type: Literal["image_url", "image"] = "image_url"
    image_url: str | dict[str, str] | None = None
    path: str | None = None
    mime_type: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _validate_image_source(self) -> ImageContentPart:
        if self.image_url is None and self.path is None:
            raise ValueError("ImageContentPart requires either 'image_url' or 'path'.")
        return self


ChatContentPart = Annotated[TextContentPart | ImageContentPart, Field(discriminator="type")]


class ChatMessage(BaseModel):
    """Unified chat message supporting text or multimodal content parts."""

    role: Literal["system", "developer", "user", "assistant", "tool"] | str
    content: str | list[ChatContentPart]
    name: str | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _validate_role(self) -> ChatMessage:
        if not self.role.strip():
            raise ValueError("ChatMessage.role must be non-empty.")
        return self

    def text_content(self) -> str:
        """Flatten text parts of the message into a single plain string."""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for part in self.content:
            if isinstance(part, TextContentPart):
                parts.append(part.text)
        return " ".join(parts)


ChatMessages = list[ChatMessage]


class FilterConfig(BaseModel):
    """Single post-processing filter step in a FilterEnsemble."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class FilterEnsembleSpec(BaseModel):
    """Named sequence of filters evaluated over shared raw responses."""

    name: str = "default"
    steps: list[FilterConfig] = Field(default_factory=list)


# --- Stage 1: Sample & TaskSpec ---


class Sample(BaseModel):
    """Stage 1 normalized evaluation sample."""

    sample_id: str
    input_text: str = ""
    target: str | int | list[str]
    choices: list[str] = Field(default_factory=list)
    messages: ChatMessages | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_hash: str | None = None


class TaskSpec(BaseModel):
    """Stage 1 declarative task specification."""

    name: str
    version: str = "0.1.0"
    trust_tier: TrustTier = TrustTier.TIER1_ZERO_CODE
    output_type: OutputType = OutputType.GENERATE_UNTIL
    dataset_path: str | None = None
    doc_to_text: str = "{{ input_text }}"
    doc_to_target: str = "{{ target }}"
    doc_to_choice: list[str] | str | None = None
    doc_to_messages: list[dict[str, Any]] | str | None = None
    description: str = ""
    num_fewshot: int = 0
    fewshot_rotation_sets: int = 1
    fewshot_delimiter: str = "\n\n"
    target_delimiter: str = " "
    gen_prefix: str = ""
    mc_format: MultipleChoiceFormat | None = None
    olmes_option_labels: list[str] | None = None
    olmes: OlmesConfig | None = None
    metrics: list[str] = Field(default_factory=lambda: ["exact_match"])
    subtasks: list[str] = Field(default_factory=list)
    weight_by_size: bool = True
    filter_ensembles: list[FilterEnsembleSpec] = Field(default_factory=list)
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)
    unsafe_code: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Stage 2: EvalRequest ---


class EvalRequest(BaseModel):
    """Stage 2 tokenization-agnostic request passed to any inference gateway."""

    request_id: str
    sample_id: str
    task_name: str
    output_type: OutputType
    context: str = ""
    continuation: str | None = None
    messages: ChatMessages | None = None
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    do_sample: bool = False
    repeats: int = Field(default=1, ge=1)
    prompt_hash: str | None = None


# --- Stage 3: EvalResponse ---


class TokenUsage(BaseModel):
    """Token accounting for cost tracking and limit enforcement."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class EvalResponse(BaseModel):
    """Stage 3 raw model response with provenance and usage telemetry."""

    request_id: str
    sample_id: str
    text: str | None = None
    loglikelihood: float | None = None
    is_greedy: bool | None = None
    reasoning_trace: str | None = None
    thinking_truncated: bool = False
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    backend: str = "mock"
    model_id: str = "mock-deterministic"
    dtype: str | None = None
    quantization: str | None = None
    batch_size: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ThinkingExtractionResult:
    """Result of stripping reasoning traces up to the final think_end_token."""

    text: str
    reasoning_trace: str | None = None
    thinking_truncated: bool = False
    warning: str | None = None


def validate_thinking_output_type(output_type: OutputType, enable_thinking: bool) -> None:
    """Reject enable_thinking=True on log-likelihood / multiple-choice tasks."""
    if enable_thinking and output_type in (
        OutputType.LOGLIKELIHOOD,
        OutputType.LOGLIKELIHOOD_ROLLING,
        OutputType.MULTIPLE_CHOICE,
    ):
        raise ValueError(
            f"enable_thinking=True is incompatible with output_type='{output_type.value}'."
        )


def extract_thinking_trace(
    raw_content: str | None,
    *,
    enable_thinking: bool,
    think_end_token: str | None = None,
    reasoning_content: str | None = None,
    none_answer_placeholder: str = "",
) -> ThinkingExtractionResult:
    """Strip reasoning traces up to and including the last think_end_token.

    Handles providers that emit `content=None` alongside `reasoning_content` when
    the reasoning budget hits `max_tokens` before producing a final answer.
    """
    if not enable_thinking:
        if raw_content is None and reasoning_content is not None:
            return ThinkingExtractionResult(
                text=none_answer_placeholder,
                reasoning_trace=reasoning_content,
                thinking_truncated=True,
                warning=(
                    "Received null content with reasoning_content; "
                    "substituted empty placeholder."
                ),
            )
        return ThinkingExtractionResult(
            text=raw_content or "",
            reasoning_trace=reasoning_content,
            thinking_truncated=False,
            warning=None,
        )

    if not think_end_token:
        raise ValueError("enable_thinking=True requires a non-empty think_end_token.")

    if raw_content is None:
        return ThinkingExtractionResult(
            text=none_answer_placeholder,
            reasoning_trace=reasoning_content,
            thinking_truncated=True,
            warning=(
                "Received null content with reasoning_content; substituted empty placeholder."
            ),
        )

    idx = raw_content.rfind(think_end_token)
    if idx == -1:
        think_start = (
            think_end_token.replace("</", "<", 1)
            if think_end_token.startswith("</")
            else None
        )
        if reasoning_content is not None and (
            think_start is None or think_start not in raw_content
        ):
            return ThinkingExtractionResult(
                text=raw_content.strip(),
                reasoning_trace=reasoning_content,
                thinking_truncated=False,
                warning=None,
            )
        return ThinkingExtractionResult(
            text=none_answer_placeholder,
            reasoning_trace=reasoning_content or raw_content,
            thinking_truncated=True,
            warning=f"Thinking trace truncated before '{think_end_token}'.",
        )

    trace = raw_content[: idx + len(think_end_token)]
    answer = raw_content[idx + len(think_end_token) :].lstrip()
    return ThinkingExtractionResult(
        text=answer,
        reasoning_trace=trace if trace else reasoning_content,
        thinking_truncated=False,
        warning=None,
    )


# --- Stage 4: ScoredSample ---


class ScoredSample(BaseModel):
    """Stage 4 scored record persisted in sample logs and Eval Receipts."""

    sample_id: str
    task_name: str
    input_text: str = ""
    target: str | int | list[str]
    choices: list[str] = Field(default_factory=list)
    uncond_loglikelihoods: list[float] | None = None
    raw_responses: list[EvalResponse] = Field(default_factory=list)
    filtered_responses: dict[str, list[str]] = Field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    doc_hash: str | None = None
    prompt_hash: str | None = None
    target_hash: str | None = None
