---
trigger: glob
description: Security, BYOK secret hygiene, cost caps, user-dir caching, and 3-tier untrusted-code boundary rules
globs: "**/gateway/**, **/cli/**, **/cache/**, **/receipts/**, **/sandbox/**, **/tasks/**"
---

# Security, BYOK, and Untrusted-Code Rules

## BYOK Secret Hygiene
- Load API keys exclusively via `eval_studio.gateway.secrets` from environment variables (`TARGET_API_KEY`, `JUDGE_API_KEY`, `EXTRACTOR_API_KEY`, provider-specific vars) or OS `keyring`.
- Reject any task YAML or run config that embeds literal API keys, bearer tokens, or credentials.
- Pass all structured logs, exception messages, WAL records, SQLite metadata, and Eval Receipt manifests through `redact_secrets()` before writing to disk or stdout.

## Cost Estimation & Circuit Breaker
- Estimate worst-case token and USD cost before dispatching remote API batches.
- Enforce `--max-cost-usd` in `eval_studio.gateway.cost.CostGuard`: halt dispatch immediately if cumulative + next-batch estimated spend exceeds the configured cap.

## 3-Tier Untrusted-Code Boundary
- **Tiers 1 & 2 (CSV/JSONL and Declarative YAML):** Render all prompt templates exclusively with `jinja2.sandbox.SandboxedEnvironment` in `eval_studio.tasks.templates`. Disallow arbitrary attribute traversal, `__globals__`, `__subclasses__`, and `!function` Python callables.
- **Tier 3 (Code Plugins & Code Execution):** Require explicit `--trust-remote-code` before importing external Python task plugins. Execute all model-generated code inside `eval_studio.sandbox.docker` with disabled networking, read-only rootfs where possible, memory/CPU/PID limits, and wall-clock timeouts.
- Never evaluate untrusted objects with in-process `==` (`__eq__` spoofing); extract primitive JSON-serializable verdicts from the sandbox.

## Cache Safety
- Resolve default cache paths via `platformdirs.user_cache_dir("eval-studio")` or `EVAL_STUDIO_CACHE_DIR`. Never write caches into `site-packages`.
- Never use `pickle` or `dill` for cached requests or responses. Use fsynced JSONL WAL + SQLite keyed by SHA-256 fingerprints.
