"""Strict Pydantic v2 request/response schemas for the Evaluation Studio API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eval_studio.core.schema import OutputType, TrustTier
from eval_studio.gateway.secrets import SecretRole

# ---------------------------------------------------------------------------
# Health & Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard redacted error payload."""

    detail: str


class HealthResponse(BaseModel):
    """Readiness and optional-dependency probe response."""

    status: Literal["ok"] = "ok"
    version: str
    pdf_available: bool
    static_ready: bool


# ---------------------------------------------------------------------------
# BYOK (/api/byok/*)
# ---------------------------------------------------------------------------

KeySource = Literal["session", "env", "keyring", "dotenv", "unset"]
KeyStorage = Literal["session", "keyring"]


class ModelPreset(BaseModel):
    """Preset model metadata and reference token pricing."""

    id: str
    label: str
    supports_thinking: bool = False
    input_usd_per_1m: float = 0.0
    output_usd_per_1m: float = 0.0


class ProviderPreset(BaseModel):
    """BYOK provider connection preset."""

    id: str
    name: str
    default_base_url: str
    env_var: str | None = None
    auth_header: str
    models: list[ModelPreset]


class ByokPresetsResponse(BaseModel):
    """Catalog of supported BYOK provider presets."""

    presets: list[ProviderPreset]


class RoleKeyStatus(BaseModel):
    """Masked credential status for a single SecretRole."""

    role: str
    configured: bool
    source: KeySource
    masked_hint: str | None = None
    env_var: str | None = None


class CacheStats(BaseModel):
    """Summary of the local SQLite + WAL response cache."""

    exists: bool
    db_bytes: int
    wal_entries: int


class ByokStatusResponse(BaseModel):
    """Credential status across all roles plus local cache telemetry."""

    roles: list[RoleKeyStatus]
    cache_dir: str
    cache_stats: CacheStats


class SaveKeyRequest(BaseModel):
    """Request to persist a credential in session memory or OS keyring."""

    model_config = ConfigDict(extra="forbid")

    role: SecretRole
    api_key: str = Field(min_length=1)
    storage: KeyStorage = "session"


class KeyActionResponse(BaseModel):
    """Masked key status returned after saving or deleting a key (never echoes secret)."""

    role: str
    configured: bool
    source: KeySource
    masked_hint: str | None = None


class TestConnectionRequest(BaseModel):
    """Request to ping a mock or HTTP BYOK gateway."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["mock", "http-byok"]
    provider: str | None = None
    model_id: str = Field(min_length=1)
    base_url: str | None = None
    role: SecretRole = SecretRole.TARGET


class TestConnectionResponse(BaseModel):
    """Result of a BYOK connection health check."""

    ok: bool
    latency_ms: float
    model_id: str
    reply_preview: str
    error: str | None = None


# ---------------------------------------------------------------------------
# BYOT (/api/byot/*)
# ---------------------------------------------------------------------------

TaskSource = Literal["builtin", "user", "paper"]


class ByotTaskSummary(BaseModel):
    """Summary metadata for a discovered TaskSpec YAML."""

    id: str
    name: str
    path: str
    source: TaskSource
    trust_tier: TrustTier
    output_type: OutputType
    metrics: list[str]
    description: str = ""
    num_samples: int | None = None


class ByotTaskListResponse(BaseModel):
    """List of built-in, user, and compiled paper tasks."""

    tasks: list[ByotTaskSummary]


class ByotTaskDetailResponse(BaseModel):
    """Full YAML, parsed spec, and preview rows for a single task."""

    path: str
    yaml_text: str
    spec: dict[str, Any]
    sample_rows: list[dict[str, Any]]


class DatasetUploadResponse(BaseModel):
    """Summary of an uploaded CSV/JSONL/JSON/Parquet dataset."""

    dataset_path: str
    columns: list[str]
    preview_rows: list[dict[str, Any]]
    total_rows: int


class ByotSaveRequest(BaseModel):
    """Request to validate and save a Tier 1/2 task YAML under tasks_dir."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    yaml_text: str = Field(min_length=1)


class ByotSaveResponse(BaseModel):
    """Confirmation of a saved task YAML."""

    path: str
    task_name: str


class ByotValidateRequest(BaseModel):
    """Request to validate a TaskSpec YAML and render sample #0 preview."""

    model_config = ConfigDict(extra="forbid")

    yaml_text: str
    base_dir: str | None = None
    apply_chat_template: bool = False
    add_bos_token: bool | None = None
    enable_thinking: bool = False


class ByotValidateResponse(BaseModel):
    """Structured task validation outcome with live sandboxed preview."""

    valid: bool
    task_name: str
    errors: list[str]
    warnings: list[str]
    secret_scan_clean: bool
    preview_prompt: str | None = None
    preview_target: str | None = None


class ByotRegistriesResponse(BaseModel):
    """Available filter, metric, and aggregation plugin names."""

    filters: list[str]
    metrics: list[str]
    aggregations: list[str]


class CalibrationRow(BaseModel):
    """Single judge vs. human label pair for Cohen's kappa calibration."""

    model_config = ConfigDict(extra="ignore")

    judge_label: str | int | float | bool
    human_label: str | int | float | bool


class CalibrateRequest(BaseModel):
    """JSON payload for judge-vs-human alignment calibration."""

    model_config = ConfigDict(extra="forbid")

    jsonl_text: str | None = None
    rows: list[CalibrationRow] | None = None
    threshold: float = Field(default=0.40, ge=-1.0, le=1.0)


class CalibrateResponse(BaseModel):
    """Cohen's kappa calibration report."""

    n_samples: int
    cohens_kappa: float
    raw_agreement: float
    passed_threshold: bool
    threshold: float
    status: Literal["ALIGNED", "UNALIGNED"]


# ---------------------------------------------------------------------------
# BYOP (/api/byop/*)
# ---------------------------------------------------------------------------


class PaperSampleSummary(BaseModel):
    """Sample scientific paper shipped in examples/papers/."""

    name: str
    path: str


class CompiledPaperSummary(BaseModel):
    """Compiled & SHA-256 locked paper suite metadata."""

    name: str
    path: str
    suite_sha256: str
    source_sha256: str
    item_count: int


class ByopPapersResponse(BaseModel):
    """Available sample papers and compiled paper suites."""

    samples: list[PaperSampleSummary]
    compiled_suites: list[CompiledPaperSummary]


class ByopUploadResponse(BaseModel):
    """Parsed upload metadata for a Markdown, text, or PDF paper."""

    paper_path: str
    filename: str
    sha256: str
    char_count: int
    page_count: int
    preview_text: str


class ByopCompileRequest(BaseModel):
    """Request to run Stage 1 extraction + Stage 2 verification on a paper."""

    model_config = ConfigDict(extra="forbid")

    paper_path: str = Field(min_length=1)
    suite_name: str = Field(min_length=1)
    mode: Literal["heuristic", "mock", "http-byok"] = "heuristic"
    extractor_model: str | None = None
    extractor_base_url: str | None = None
    verifier_model: str | None = None
    verifier_base_url: str | None = None
    provider: str | None = None


class ByopRejectedItem(BaseModel):
    """Item rejected by Stage 2 answerability verification with reasons."""

    item: dict[str, Any]
    reasons: list[str]


class ByopCompileResponse(BaseModel):
    """Stage 1 + Stage 2 BYOP compilation inspection payload."""

    source_path: str
    source_sha256: str
    suite_name: str
    extractor_model: str
    verifier_model: str
    approved_items: list[dict[str, Any]]
    rejected_items: list[ByopRejectedItem]


class ByopLockRequest(BaseModel):
    """Request to re-verify, SHA-256 lock, and export approved BYOP items."""

    model_config = ConfigDict(extra="forbid")

    paper_path: str = Field(min_length=1)
    suite_name: str = Field(min_length=1)
    extractor_model: str = "mock-extractor"
    verifier_model: str = "mock-verifier"
    items: list[dict[str, Any]] = Field(min_length=1)
    output_filename: str | None = None


class ByopLockResponse(BaseModel):
    """Result of SHA-256 locking and YAML/manifest export for a BYOP suite."""

    yaml_path: str
    manifest_path: str
    suite_sha256: str
    source_sha256: str
    item_count: int
    yaml_text: str


# ---------------------------------------------------------------------------
# Runs & SSE (/api/runs/*)
# ---------------------------------------------------------------------------

RunStatusLiteral = Literal["queued", "running", "completed", "budget_exceeded", "failed"]


class PricingOverride(BaseModel):
    """Optional USD per 1M token pricing override for pre-flight cost estimation."""

    input_usd_per_1m: float | None = Field(default=None, ge=0.0)
    output_usd_per_1m: float | None = Field(default=None, ge=0.0)
    judge_input_usd_per_1m: float | None = Field(default=None, ge=0.0)
    judge_output_usd_per_1m: float | None = Field(default=None, ge=0.0)


class RunLaunchRequest(BaseModel):
    """Request to estimate or launch an evaluation run."""

    model_config = ConfigDict(extra="ignore")

    task_path: str = Field(min_length=1)
    model_backend: str = "mock"
    model_id: str = "mock-deterministic"
    base_url: str | None = None
    judge_model: str | None = None
    judge_base_url: str | None = None
    fallback_judge_model: str | None = None
    limit: int | None = Field(default=None, ge=1)
    repeats: int = Field(default=1, ge=1, le=32)
    olmes_enabled: bool = False
    enable_thinking: bool = False
    think_end_token: str | None = None
    apply_chat_template: bool = False
    add_bos_token: bool | None = None
    bootstrap_iters: int = Field(default=1000, ge=0, le=20000)
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    trust_remote_code: bool = False
    pricing_override: PricingOverride | None = None


class CostEstimateResponse(BaseModel):
    """Pre-flight USD cost and token breakdown without invoking inference."""

    n_samples: int
    repeats: int
    target_prompt_tokens: int
    target_completion_tokens: int
    target_usd: float
    judge_enabled: bool
    judge_prompt_tokens: int
    judge_completion_tokens: int
    judge_usd: float
    total_usd: float
    exceeds_max_cost: bool


class RunLaunchResponse(BaseModel):
    """Initial response when an evaluation run is queued/started."""

    run_id: str
    status: RunStatusLiteral
    output_dir: str


class RunStatusResponse(BaseModel):
    """Current status, progress counters, spend, and manifest of a run."""

    run_id: str
    status: RunStatusLiteral
    total_samples: int
    completed_samples: int
    spent_usd: float
    max_cost_usd: float | None = None
    receipt_dir: str | None = None
    manifest: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Receipts & Analytics (/api/receipts/*)
# ---------------------------------------------------------------------------


class ReceiptListItem(BaseModel):
    """Summary entry for a discovered Portable Eval Receipt."""

    run_id: str
    dir_name: str
    path: str
    created_at_utc: str
    backend: str
    model_id: str
    task_name: str
    total_cost_usd: float
    total_prompt_tokens: int
    total_completion_tokens: int
    metrics: dict[str, dict[str, Any]]
    task_sha256: str
    manifest_sha256: str
    source_paper_path: str | None = None
    source_paper_sha256: str | None = None
    verified: bool | None = None


class ReceiptsListResponse(BaseModel):
    """List of all discovered Portable Eval Receipts."""

    receipts: list[ReceiptListItem]


class ReceiptDetailResponse(BaseModel):
    """Full manifest, rendered Markdown report, and sample count."""

    manifest: dict[str, Any]
    report_md: str
    total_samples: int


class ReceiptSamplesResponse(BaseModel):
    """Paginated & searchable sample-level records from samples.jsonl."""

    total: int
    offset: int
    limit: int
    samples: list[dict[str, Any]]


class VerifyReceiptResponse(BaseModel):
    """Outcome of cryptographic verify-receipt integrity audit."""

    valid: bool
    errors: list[str]


class RescoreRequest(BaseModel):
    """Request to re-score an existing receipt 100% offline (or with an LLM judge)."""

    model_config = ConfigDict(extra="ignore")

    judge_model: str | None = None
    judge_base_url: str | None = None
    fallback_judge_model: str | None = None
    filter_ensembles: list[dict[str, Any]] | None = None
    metrics: list[str] | None = None
    output_name: str | None = None


class RescoreResponse(BaseModel):
    """Result of an offline receipt re-scoring run."""

    receipt_id: str
    path: str
    manifest: dict[str, Any]


class NoiseFloorRequest(BaseModel):
    """Request to compute infrastructure noise floor between two receipts."""

    model_config = ConfigDict(extra="forbid")

    baseline_id: str = Field(min_length=1)
    perturbed_id: str = Field(min_length=1)
    perturbation: str = "restart"


class NoiseFloorResponse(BaseModel):
    """Measured item-level decision flip rate across runs."""

    n_decisions: int
    n_flips: int
    flip_rate: float
    perturbation: str


class CompareRequest(BaseModel):
    """Request to compare two or more receipts with statistical significance tests."""

    model_config = ConfigDict(extra="forbid")

    receipt_ids: list[str] = Field(min_length=1)
    baseline_id: str | None = None
    bootstrap_iters: int = Field(default=1000, ge=1, le=20000)


class CompareMetricEntry(BaseModel):
    """Metric value with 95% confidence interval and unresolved flag."""

    value: float
    stderr: float | None = None
    ci_low: float
    ci_high: float
    n_samples: int
    unresolved: bool


class CompareRunSummary(BaseModel):
    """Per-run summary metrics, cost, tokens, and latency for multi-run comparison."""

    receipt_id: str
    run_id: str
    dir_name: str
    model_id: str
    backend: str
    task_name: str
    total_cost_usd: float
    total_prompt_tokens: int
    total_completion_tokens: int
    mean_latency_ms: float | None = None
    source_paper_path: str | None = None
    metrics: dict[str, CompareMetricEntry]


class PairedBootstrapSummary(BaseModel):
    """Paired bootstrap delta, 95% CI, and p-value."""

    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    stderr: float = 0.0
    iters_executed: int = 0
    seed: int = 1234


class McNemarSummary(BaseModel):
    """Paired binary McNemar significance test summary."""

    n_items: int
    n_01: int
    n_10: int
    statistic: float
    p_value: float
    exact: bool
    delta: float


class ItemDiff(BaseModel):
    """Head-to-head item-level score and output diff between baseline and challenger."""

    sample_id: str
    input_text: str
    target: str | int | list[str]
    baseline_score: float
    challenger_score: float
    baseline_output: str
    challenger_output: str


class PairwiseComparison(BaseModel):
    """Pairwise statistical comparison between a baseline and challenger receipt."""

    baseline_id: str
    challenger_id: str
    metric: str
    paired_bootstrap: PairedBootstrapSummary
    mcnemar: McNemarSummary | None = None
    noise_floor: NoiseFloorResponse | None = None
    unresolved: bool
    item_diffs: list[ItemDiff]


class CategoryBreakdownEntry(BaseModel):
    """Category or paper-level score breakdown across compared runs."""

    group: str
    category: str
    scores: dict[str, float]
    counts: dict[str, int] = Field(default_factory=dict)


class CompareResponse(BaseModel):
    """Full multi-run statistical comparison payload."""

    runs: list[CompareRunSummary]
    pairwise: list[PairwiseComparison]
    category_breakdown: list[CategoryBreakdownEntry]
