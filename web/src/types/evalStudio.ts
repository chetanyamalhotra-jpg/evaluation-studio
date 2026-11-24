// Strict TypeScript schemas aligned with PROJECT.md § IC2 and src/eval_studio/ Python models.

export type SecretRole =
  | "TARGET_API_KEY"
  | "JUDGE_API_KEY"
  | "EXTRACTOR_API_KEY"
  | "FALLBACK_JUDGE_API_KEY";

export type SecretSource = "session" | "env" | "keyring" | "dotenv" | "unset";
export type KeySource = SecretSource;
export type KeyStorage = "session" | "keyring";
export type KeyDeleteStorage = KeyStorage | "all";

export type TrustTier =
  | "tier1_zero_code"
  | "tier2_declarative"
  | "tier3_code_plugin";

export type OutputType =
  | "loglikelihood"
  | "loglikelihood_rolling"
  | "generate_until"
  | "multiple_choice"
  | "chat";

export type MultipleChoiceFormat = "cloze" | "mcf" | "dual";
export type OlmesFormat = "cloze" | "mcf";

export interface OlmesConfig {
  enabled: boolean;
  formats: OlmesFormat[];
  choice_prefix: string;
  metric_preference: string;
}

export interface RuntimeLimits {
  max_cost_usd?: number | null;
  max_tokens?: number | null;
  wall_clock_seconds?: number | null;
  max_tool_calls?: number | null;
}

export interface TextContentPart {
  type: "text";
  text: string;
}

export interface ImageContentPart {
  type: "image_url" | "image";
  image_url?: string | { url: string; [key: string]: unknown } | null;
  path?: string | null;
  mime_type?: string | null;
  detail?: string | null;
}

export type ChatContentPart = TextContentPart | ImageContentPart;

export interface ChatMessage {
  role: "system" | "developer" | "user" | "assistant" | "tool" | (string & {});
  content: string | ChatContentPart[];
  name?: string | null;
  tool_call_id?: string | null;
}

export interface FilterConfig {
  name: string;
  params: Record<string, unknown>;
}

export interface FilterEnsembleSpec {
  name: string;
  steps: FilterConfig[];
}

export interface Sample {
  sample_id: string;
  input_text: string;
  target: string | number | string[];
  choices: string[];
  messages?: ChatMessage[] | null;
  metadata: Record<string, unknown>;
  doc_hash?: string | null;
}

export interface TaskSpec {
  name: string;
  version: string;
  trust_tier: TrustTier;
  output_type: OutputType;
  dataset_path?: string | null;
  doc_to_text: string;
  doc_to_target: string;
  doc_to_choice?: string[] | string | null;
  doc_to_messages?: Record<string, unknown>[] | string | null;
  description: string;
  num_fewshot: number;
  fewshot_rotation_sets: number;
  fewshot_delimiter: string;
  target_delimiter: string;
  gen_prefix: string;
  mc_format?: MultipleChoiceFormat | null;
  olmes_option_labels?: string[] | null;
  olmes?: OlmesConfig | null;
  metrics: string[];
  subtasks: string[];
  weight_by_size: boolean;
  filter_ensembles: FilterEnsembleSpec[];
  generation_kwargs: Record<string, unknown>;
  unsafe_code: boolean;
  metadata: Record<string, unknown>;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number;
}

export interface EvalResponse {
  request_id: string;
  sample_id: string;
  text: string | null;
  loglikelihood: number | null;
  is_greedy: boolean | null;
  reasoning_trace: string | null;
  thinking_truncated: boolean;
  usage: TokenUsage;
  latency_ms: number;
  backend: string;
  model_id: string;
  dtype: string | null;
  quantization: string | null;
  batch_size: number | null;
  metadata: Record<string, unknown>;
}

export interface ScoredSample {
  sample_id: string;
  task_name: string;
  input_text: string;
  target: string | number | string[];
  choices: string[];
  uncond_loglikelihoods: number[] | null;
  raw_responses: EvalResponse[];
  filtered_responses: Record<string, string[]>;
  metrics: Record<string, Record<string, number>>;
  doc_hash: string | null;
  prompt_hash: string | null;
  target_hash: string | null;
}

// ---------------------------------------------------------------------------
// Health & System
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: "ok" | string;
  version: string;
  pdf_available: boolean;
  static_ready: boolean;
}

// ---------------------------------------------------------------------------
// BYOK (/api/byok/*)
// ---------------------------------------------------------------------------

export interface ProviderModelPreset {
  id: string;
  label: string;
  supports_thinking: boolean;
  input_usd_per_1m: number;
  output_usd_per_1m: number;
}

export interface ProviderPreset {
  id: string;
  name: string;
  default_base_url: string;
  env_var: string | null;
  auth_header: string;
  models: ProviderModelPreset[];
}

export interface ByokPresetsResponse {
  presets: ProviderPreset[];
}

export interface KeyStatus {
  role: SecretRole;
  configured: boolean;
  source: SecretSource;
  masked_hint: string | null;
  env_var: string | null;
}

export interface CacheStats {
  exists: boolean;
  db_bytes: number;
  wal_entries: number;
}

export interface ByokStatusResponse {
  roles: KeyStatus[];
  cache_dir: string;
  cache_stats: CacheStats;
}

export interface SaveKeyRequest {
  role: SecretRole;
  api_key: string;
  storage: KeyStorage;
}

export interface SaveKeyResponse {
  role: SecretRole;
  configured: boolean;
  source: SecretSource;
  masked_hint: string | null;
}

export interface DeleteKeyResponse {
  role: SecretRole;
  configured: boolean;
  source: SecretSource;
  masked_hint: string | null;
}

export interface TestConnectionRequest {
  backend: "mock" | "http-byok";
  provider?: string;
  model_id: string;
  base_url?: string;
  role?: SecretRole;
}

export interface TestConnectionResponse {
  ok: boolean;
  latency_ms: number;
  model_id: string;
  reply_preview: string;
  error?: string | null;
}

// ---------------------------------------------------------------------------
// BYOT (/api/byot/*)
// ---------------------------------------------------------------------------

export type TaskSource = "builtin" | "user" | "paper";

export interface TaskSummary {
  id: string;
  name: string;
  path: string;
  source: TaskSource;
  trust_tier: TrustTier;
  output_type: OutputType;
  metrics: string[];
  description: string;
  num_samples?: number | null;
}

export interface TasksListResponse {
  tasks: TaskSummary[];
}

export interface TaskDetailResponse {
  path: string;
  yaml_text: string;
  spec: TaskSpec;
  sample_rows: Record<string, unknown>[];
}

export interface UploadDatasetResponse {
  dataset_path: string;
  columns: string[];
  preview_rows: Record<string, unknown>[];
  total_rows: number;
}

export interface SaveTaskRequest {
  filename: string;
  yaml_text: string;
}

export interface SaveTaskResponse {
  path: string;
  task_name: string;
}

export interface ValidateTaskRequest {
  yaml_text: string;
  base_dir?: string;
  apply_chat_template?: boolean;
  add_bos_token?: boolean | null;
  enable_thinking?: boolean;
}

export interface ValidateTaskResponse {
  valid: boolean;
  task_name: string;
  errors: string[];
  warnings: string[];
  secret_scan_clean: boolean;
  preview_prompt: string | null;
  preview_target: string | null;
}

export interface RegistriesResponse {
  filters: string[];
  metrics: string[];
  aggregations: string[];
}

export interface CalibrationRow {
  judge_label: string | number | boolean;
  human_label: string | number | boolean;
}

export interface CalibrateRequest {
  jsonl_text?: string;
  rows?: CalibrationRow[];
  threshold?: number;
}

export interface CalibrateResponse {
  n_samples: number;
  cohens_kappa: number;
  raw_agreement: number;
  passed_threshold: boolean;
  threshold: number;
  status: "ALIGNED" | "UNALIGNED";
}

// ---------------------------------------------------------------------------
// BYOP (/api/byop/*)
// ---------------------------------------------------------------------------

export type VerificationStatus =
  | "pending"
  | "approved"
  | "needs_review"
  | "rejected";

export type PaperItemCategory =
  | "quantitative_finding"
  | "claim"
  | "method"
  | "limitation"
  | (string & {});

export interface Citation {
  page: number;
  section: string;
  quote: string;
}

export interface ExtractedItem {
  item_id: string;
  question: string;
  reference_answer: string;
  rubric: string;
  category: PaperItemCategory;
  choices: string[];
  citations: Citation[];
  verification_status: VerificationStatus;
  verification_notes: string;
  item_sha256: string | null;
}

export type CompiledItem = ExtractedItem;

export interface RejectedItemEntry {
  item: ExtractedItem;
  reasons: string[];
}

export interface PaperSampleSummary {
  name: string;
  path: string;
}

export interface CompiledSuiteSummary {
  name: string;
  path: string;
  suite_sha256: string;
  source_sha256: string;
  item_count: number;
}

export interface PapersListResponse {
  samples: PaperSampleSummary[];
  compiled_suites: CompiledSuiteSummary[];
}

export interface PaperUploadResponse {
  paper_path: string;
  filename: string;
  sha256: string;
  char_count: number;
  page_count: number;
  preview_text: string;
}

export type PaperCompileMode = "heuristic" | "mock" | "http-byok";

export interface PaperCompileRequest {
  paper_path: string;
  suite_name: string;
  mode: PaperCompileMode;
  extractor_model?: string;
  extractor_base_url?: string;
  verifier_model?: string;
  verifier_base_url?: string;
  provider?: string;
}
export type CompilePaperRequest = PaperCompileRequest;

export interface PaperCompileResponse {
  source_path: string;
  source_sha256: string;
  suite_name: string;
  extractor_model: string;
  verifier_model: string;
  approved_items: ExtractedItem[];
  rejected_items: RejectedItemEntry[];
}

export interface PaperLockRequest {
  paper_path: string;
  suite_name: string;
  extractor_model: string;
  verifier_model: string;
  items: ExtractedItem[];
  output_filename?: string;
}
export type LockPaperRequest = PaperLockRequest;

export interface PaperLockResponse {
  yaml_path: string;
  manifest_path: string;
  suite_sha256: string;
  source_sha256: string;
  item_count: number;
  yaml_text: string;
}

// ---------------------------------------------------------------------------
// Runs & SSE (/api/runs/*)
// ---------------------------------------------------------------------------

export interface PricingOverride {
  input_usd_per_1m?: number;
  output_usd_per_1m?: number;
  judge_input_usd_per_1m?: number;
  judge_output_usd_per_1m?: number;
}

export interface RunLaunchRequest {
  task_path: string;
  model_backend: "mock" | "http-byok" | (string & {});
  model_id: string;
  base_url?: string;
  judge_model?: string;
  judge_base_url?: string;
  fallback_judge_model?: string;
  limit?: number | null;
  repeats?: number;
  olmes_enabled?: boolean;
  enable_thinking?: boolean;
  think_end_token?: string | null;
  apply_chat_template?: boolean;
  add_bos_token?: boolean | null;
  bootstrap_iters?: number;
  max_cost_usd?: number | null;
  trust_remote_code?: boolean;
  pricing_override?: PricingOverride | null;
}

export interface CostEstimateResponse {
  n_samples: number;
  repeats: number;
  target_prompt_tokens: number;
  target_completion_tokens: number;
  target_usd: number;
  judge_enabled: boolean;
  judge_prompt_tokens: number;
  judge_completion_tokens: number;
  judge_usd: number;
  total_usd: number;
  exceeds_max_cost: boolean;
}

export type RunStatus =
  | "queued"
  | "running"
  | "completed"
  | "budget_exceeded"
  | "failed";

export interface RunLaunchResponse {
  run_id: string;
  status: "queued" | "running" | RunStatus;
  output_dir: string;
}

export interface RunStatusResponse {
  run_id: string;
  status: RunStatus;
  total_samples: number;
  completed_samples: number;
  spent_usd: number;
  max_cost_usd: number | null;
  receipt_dir?: string | null;
  manifest?: ReceiptManifest | null;
  error?: string | null;
}

export interface RunStartedEvent {
  event: "run_started";
  run_id: string;
  task_name: string;
  total_samples: number;
}

export interface SubtaskStartedEvent {
  event: "subtask_started";
  run_id: string;
  task_name: string;
  subtask_name: string;
  subtask_index: number;
  total_subtasks: number;
}

export interface SampleScoredEvent {
  event: "sample_scored";
  run_id: string;
  index: number;
  total: number;
  sample: ScoredSample;
  spent_usd: number;
  prompt_tokens?: number;
  completion_tokens?: number;
}

export interface BudgetExceededEvent {
  event: "budget_exceeded";
  run_id: string;
  task_name: string;
  index: number;
  total: number;
  spent_usd: number;
  max_cost_usd: number | null;
  error: string;
}

export interface RunCompletedEvent {
  event: "completed";
  run_id: string;
  task_name: string;
  spent_usd: number;
  total_samples: number;
  manifest_sha256?: string;
  receipt_dir?: string;
}

export interface RunErrorEvent {
  event: "error";
  run_id?: string;
  error: string;
}

export type RunSseEvent =
  | RunStartedEvent
  | SubtaskStartedEvent
  | SampleScoredEvent
  | BudgetExceededEvent
  | RunCompletedEvent
  | RunErrorEvent;

// ---------------------------------------------------------------------------
// Receipts & Analytics (/api/receipts/*)
// ---------------------------------------------------------------------------

export interface MetricSummary {
  value: number;
  stderr: number | null;
  n_samples: number;
  bootstrap_iters: number;
  unresolved: boolean;
  ci_low?: number;
  ci_high?: number;
}

export interface ReceiptManifest {
  schema_version: string;
  harness_version: string;
  run_id: string;
  created_at_utc: string;
  backend: string;
  model_id: string;
  task_name: string;
  task_sha256: string;
  samples_sha256: string;
  task_path: string | null;
  source_paper_path: string | null;
  source_paper_sha256: string | null;
  dtype: string | null;
  quantization: string | null;
  batch_size: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_cost_usd: number;
  reproduce_command: string;
  seeds: Record<string, number>;
  redacted_config: Record<string, unknown>;
  metrics: Record<string, MetricSummary>;
  manifest_sha256: string;
}

export interface ReceiptListItem {
  run_id: string;
  dir_name: string;
  path: string;
  created_at_utc: string;
  backend: string;
  model_id: string;
  task_name: string;
  total_cost_usd: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  metrics: Record<string, MetricSummary>;
  task_sha256: string;
  manifest_sha256: string;
  source_paper_path?: string | null;
  source_paper_sha256?: string | null;
  verified?: boolean | null;
}

export interface ReceiptsListResponse {
  receipts: ReceiptListItem[];
}

export interface ReceiptDetailResponse {
  manifest: ReceiptManifest;
  report_md: string;
  total_samples: number;
}

export interface ReceiptSamplesResponse {
  total: number;
  offset: number;
  limit: number;
  samples: ScoredSample[];
}

export interface VerifyReceiptResponse {
  valid: boolean;
  errors: string[];
}

export interface RescoreRequest {
  judge_model?: string;
  judge_base_url?: string;
  fallback_judge_model?: string;
  filter_ensembles?: FilterEnsembleSpec[];
  metrics?: string[];
  output_name?: string;
}

export interface RescoreResponse {
  receipt_id: string;
  path: string;
  manifest: ReceiptManifest;
}

export interface NoiseFloorRequest {
  baseline_id: string;
  perturbed_id: string;
  perturbation?: string;
}

export interface NoiseFloorResponse {
  n_decisions: number;
  n_flips: number;
  flip_rate: number;
  perturbation: string;
}

export interface CompareRequest {
  receipt_ids: string[];
  baseline_id?: string;
  bootstrap_iters?: number;
}
export type CompareReceiptsRequest = CompareRequest;

export interface CompareMetricEntry {
  value: number;
  stderr: number | null;
  ci_low: number;
  ci_high: number;
  n_samples: number;
  unresolved: boolean;
}

export interface CompareRunSummary {
  receipt_id: string;
  run_id: string;
  dir_name: string;
  model_id: string;
  backend: string;
  task_name: string;
  total_cost_usd: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  mean_latency_ms?: number | null;
  source_paper_path?: string | null;
  metrics: Record<string, CompareMetricEntry>;
}

export interface PairedBootstrapSummary {
  mean_delta: number;
  ci_low: number;
  ci_high: number;
  p_value: number;
  significant: boolean;
  stderr?: number;
  iters_executed?: number;
  seed?: number;
}

export interface McNemarSummary {
  n_items: number;
  n_01: number;
  n_10: number;
  statistic: number;
  p_value: number;
  exact: boolean;
  delta: number;
}

export interface ItemDiff {
  sample_id: string;
  input_text: string;
  target: string | number | string[];
  baseline_score: number;
  challenger_score: number;
  baseline_output: string;
  challenger_output: string;
}

export interface PairwiseComparison {
  baseline_id: string;
  challenger_id: string;
  metric: string;
  paired_bootstrap: PairedBootstrapSummary;
  mcnemar: McNemarSummary | null;
  noise_floor: NoiseFloorResponse | null;
  unresolved: boolean;
  item_diffs: ItemDiff[];
}

export interface CategoryBreakdownEntry {
  group: string;
  category: string;
  scores: Record<string, number>;
  counts?: Record<string, number>;
}

export interface CompareResponse {
  runs: CompareRunSummary[];
  pairwise: PairwiseComparison[];
  category_breakdown: CategoryBreakdownEntry[];
}
