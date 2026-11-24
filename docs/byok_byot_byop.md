# BYOK, BYOT, and BYOP Guide

## BYOK (Bring Your Own Key)
- Credentials are loaded exclusively from environment variables, OS `keyring`, or a gitignored local `.env` file via `eval_studio.gateway.secrets`. Hardcoded keys in task YAMLs or run configs are rejected.
- Role separation allows independent keys for `TARGET_API_KEY`, `JUDGE_API_KEY`, `EXTRACTOR_API_KEY`, and `FALLBACK_JUDGE_API_KEY`.
- First-class provider dialects in `HttpBYOKGateway` (`--model http-byok`):
  - **DeepSeek Direct (`deepseek-flash`)**: defaults to `https://api.deepseek.com`, falls back to `DEEPSEEK_API_KEY`, uses `Authorization: Bearer`, forwards `thinking` / `reasoning_effort` / `top_p` / `stop` / `seed` / `response_format`, and parses `reasoning_content` + `reasoning_tokens` + truncation.
  - **Sarvam AI V2 (`glm5.3-flash`)**: defaults to `https://api.sarvam.ai/v2`, falls back to `SARVAM_API_KEY`, sends required `api-subscription-key` header, and scrubs `sk_...` and `api-subscription-key` from logs/receipts.
  - **OpenAI, Anthropic, Gemini, OpenRouter, LiteLLM, and local Ollama / vLLM / LM Studio** servers.
- `--dry-run --estimate-cost` computes pre-flight target + judge USD spend, and `--max-cost-usd` enforces a hard spend circuit breaker with WAL checkpointing and zero-duplicate resume.

## BYOT (Bring Your Own Test)
- **Tier 1 (`examples/tier1_zero_code/`)**: Local CSV/JSONL/JSON/Parquet + minimal YAML. Zero code execution.
- **Tier 2 (`examples/tier2_declarative/`)**: Declarative YAML with sandboxed Jinja2 templates (SSTI and `!function` tags blocked), multimodal `doc_to_messages`, `FilterEnsemble`, `repeats` majority voting, OLMES dual Cloze+MCF, and K-set few-shot rotation.
- **Tier 3 (`examples/tier3_plugin/`)**: Custom Python `Solver`/`Scorer` plugins gated behind `--trust-remote-code` and isolated Docker/subprocess sandbox execution (`eval_studio.sandbox.docker`).

## BYOP (Bring Your Own Paper)
- `eval-studio compile-paper` executes a two-stage pipeline decoupled from `run`:
  1. Multi-page chunked extraction (`[PAGE N]` markers for PDFs via optional `[pdf]` extra, or offline Markdown/text fallback) requiring `page`, `section`, and verbatim `quote` citations.
  2. Independent Stage 2 `AnswerabilityVerifier` gate (structural + optional BYOK `JUDGE_API_KEY` verifier) rejecting uncited, answer-leaking, or dangling-reference items.
- Approved items are SHA-256 locked (`source_sha256`, `item_sha256`, `suite_sha256`) into runnable `.yaml` or `.json` packs verified by `eval-studio verify-receipt`.
