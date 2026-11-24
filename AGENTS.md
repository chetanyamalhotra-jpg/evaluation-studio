# Evaluation Studio — Agent Engineering Invariants

This repository implements **Evaluation Studio** (`eval-studio`), an open-source, local-first evaluation harness built around **BYOK** (Bring Your Own Key), **BYOT** (Bring Your Own Test), **BYOP** (Bring Your Own Paper), and **Portable Eval Receipts**.

All contributors and coding agents MUST uphold the following non-negotiable architectural and methodological invariants distilled from [docs/eval-harness-research.md](docs/eval-harness-research.md).

## 1. Slim Core & Lazy Plugin Architecture
- **Zero heavy ML dependencies in base install:** `pyproject.toml` core dependencies MUST NOT include `torch`, `transformers`, `accelerate`, `vllm`, `sglang`, `docker`, or PDF parsers. All heavy backends and parsers live behind optional extras (`[api]`, `[hf]`, `[vllm]`, `[pdf]`, `[sandbox]`).
- **Lazy entry-point discovery:** Plugins register via `importlib.metadata.entry_points()` under `eval_studio.{models,solvers,filters,metrics,aggregations,sandboxes}`. The `Registry[T]` (guarded by `threading.RLock`) stores lazy `EntryPoint` placeholders and only imports modules on first access. Colliding plugin names MUST NOT shadow built-in registrations.
- **Offline CPU default:** Core tests and `eval-studio --help` / `eval-studio validate` / `eval-studio verify-receipt` MUST run 100% offline on CPU using the deterministic `MockGateway`.

## 2. N+M Four-Stage Pipeline Schema
To prevent N x M pairwise adapter explosion between datasets, solvers, inference engines, and scorers, all evaluation flows MUST pass through four typed intermediate stages (`src/eval_studio/core/schema.py`):
1. **Stage 1 — Sample & Task Spec (`Sample`, `TaskSpec`)**: Normalized document, choices/targets, metadata, and SHA-256 content hashes.
2. **Stage 2 — Request (`EvalRequest`)**: Tokenization-agnostic text/chat request (`loglikelihood`, `loglikelihood_rolling`, `generate_until`, `chat`) with explicit limits (tokens, wall-clock, tool calls).
3. **Stage 3 — Response (`EvalResponse`)**: Raw model outputs, logprobs, token usage, latency, backend/kernel provenance, and stripped reasoning trace metadata.
4. **Stage 4 — Scored Record & Receipt (`ScoredSample`, `EvalReceipt`)**: Multi-pipeline `FilterEnsemble` outputs, per-sample metrics, statistical aggregates, and tamper-evident hashes.

## 3. BYOK Secret Hygiene & Cost Caps
- **No secrets in repo or artifacts:** API keys MUST ONLY be loaded from environment variables or OS `keyring` via `eval_studio.gateway.secrets`. Never hardcode secrets in task YAMLs, run configs, cache entries, logs, or Eval Receipts.
- **Role-based separation:** Support independent credentials for `TARGET_API_KEY`, `JUDGE_API_KEY`, and `EXTRACTOR_API_KEY` (falling back to provider keys like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`).
- **Mandatory redaction:** Every log message, error trace, WAL entry, and Eval Receipt MUST pass through the secret redactor before persistence.
- **Cost circuit breaker:** Every paid API run MUST run a pre-flight cost estimate and enforce `--max-cost-usd` as a hard circuit breaker before dispatching requests that would exceed the budget.

## 4. BYOT 3-Tier Untrusted-Code Boundary
- **Tier 1 (Zero-Code Data):** CSV/JSONL datasets with a minimal YAML pointer. Zero arbitrary code execution; templates rendered strictly inside `jinja2.sandbox.SandboxedEnvironment`.
- **Tier 2 (Declarative YAML):** Declarative `TaskSpec` with `FilterEnsemble`, few-shot rotation, and sandboxed Jinja2 templates. No `!function` Python execution allowed without opting into Tier 3.
- **Tier 3 (Code Plugins & Executable Benchmarks):** Custom Python `Solver`/`Scorer` plugins or code-execution benchmarks MUST require explicit `--trust-remote-code` (or equivalent opt-in flag) and run untrusted generated code inside an isolated container (`eval_studio.sandbox.docker`) with CPU, memory, network, and wall-clock limits—never in-process on the host.

## 5. BYOP Two-Stage `compile-paper` Verification
- **Stage 1 (Grounded Extraction):** Extract structured evaluation items from PDF/Markdown papers with **mandatory page and section citations** for every question, rubric, and gold answer. Items lacking verbatim provenance are rejected.
- **Stage 2 (Answerability Verification Gate):** An independent verification pass MUST confirm that each extracted item is self-contained, unambiguous, and answerable from its stated context without leaking the answer or relying on unextracted figures.
- **SHA-256 Locking:** Compiled paper suites MUST lock the source paper hash, extractor/verifier model metadata, and compiled item hashes into an immutable manifest before evaluation.

## 6. Anti-Bug Guardrails (Silent Validity Defects)
Every PR touching prompts, tokenization, filters, metrics, stats, or sandboxes MUST preserve these guardrails and keep `tests/adversarial/` green:
1. **Whitespace transfer (`_encode_pair`):** Strip trailing whitespace from `context` (`n_spaces = len(context) - len(context.rstrip())`) and prepend it to `continuation` before tokenization so BPE/SentencePiece leading-space tokens are never split.
2. **BOS & chat-template conflict check:** Default `add_bos_token=None` (delegate to tokenizer). Raise a hard validation error if `--apply-chat-template` is combined with `add_bos_token=True`. Reject `enable_thinking=True` on log-likelihood tasks.
3. **Regex longest-first sorting:** Any multi-choice regex alternation (e.g., `MultiChoiceRegexFilter`) MUST sort candidate patterns by descending length before compiling to prevent prefix shadowing (e.g., `"Guilty"` shadowing `"Guilty of Romance"`).
4. **Unicode normalization safety:** Never lowercase or normalize strings with `numpy.char.lower` (which silently truncates expanded Unicode characters). Use Python `str.lower()` / `casefold()` and explicit `unicodedata.normalize`.
5. **Exact `bootstrap_stderr`:** Run the exact requested `bootstrap_iters` count—never floor-divide or silently truncate non-multiples of 1,000 (e.g., `1500 -> 1000`).
6. **Micro vs. macro stderr pairing:** Group aggregation MUST pair `weight_by_size=True` (micro-averaging) with `pooled_sample_stderr` and `weight_by_size=False` (macro-averaging) with `unweighted_mean_stderr`.
7. **Tokenization-agnostic normalization:** Never divide log-likelihoods by token count. Use character-normalized (`acc_norm`), UTF-8 byte-normalized (`acc_bytes`), or PMI (`acc_mutual_info`).
8. **Few-shot hygiene & rotation:** Never leak the active evaluation document into its own few-shot demonstrations; resolve `gen_prefix` against the test document, not exemplars. Support rotating `K` demonstration sets and mark close comparisons as **unresolved** whenever the cross-set CI spans zero.
9. **`__eq__` sandbox hardening:** Never grade untrusted generated code by evaluating `result == expected` on arbitrary Python objects inside the harness process (prevents `__eq__` always returning `True`). Execute assertions inside the sandbox and serialize primitive pass/fail verdicts across the boundary.
10. **Deterministic checkers & exact numeric comparison:** Instruction checkers MUST NOT draw missing parameters from global randomness; integer comparisons MUST NOT pass through lossy `float` casts.

## 7. Crash-Safe User-Dir WAL + SQLite Caching
- Store all caches in a user-owned directory resolved via `platformdirs.user_cache_dir("eval-studio")` (overridable via `EVAL_STUDIO_CACHE_DIR`), NEVER inside `site-packages`.
- Never deserialize untrusted `pickle` or `dill` files. Persist structured JSON/SQLite rows paired with an `fsync`-backed JSONL write-ahead log (WAL).
- Key every cache entry by a SHA-256 fingerprint covering the task YAML/schema, prompt template, model/backend/quantization parameters, and request args so prompt edits automatically invalidate stale hits. Bypass cache when `do_sample=True` (stochastic decoding).

## 8. Portable Eval Receipts
- Every evaluation run emits a self-contained **Eval Receipt** (`manifest.json` + sample log JSONL) recording: harness version/commit, task/paper SHA-256 hashes, backend/model/dtype/quantization/batch-size provenance, seeds, noise-floor metadata, redacted config, raw + filtered responses, and summary metrics with stderrs.
- Receipts MUST support 100% offline re-scoring (`eval-studio score`) and cryptographic tamper verification (`eval-studio verify-receipt`).
