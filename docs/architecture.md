# Architecture Overview

See [docs/eval-harness-research.md](eval-harness-research.md) for the empirical evidence base and methodological invariants motivating this design.

## N+M Four-Stage Pipeline
1. **Stage 1 (`eval_studio.core.schema.Sample`, `TaskSpec`)**: Normalized dataset records (including multimodal `ChatMessages` via `doc_to_messages`) and declarative YAML specs loaded via `eval_studio.tasks`.
2. **Stage 2 (`EvalRequest`)**: Tokenization-agnostic inference requests (`loglikelihood`, `loglikelihood_rolling`, `multiple_choice`, `generate_until`, `chat`) with OLMES dual Cloze+MCF formatting, `_encode_pair` whitespace transfer, K-set few-shot rotation, and explicit `RuntimeLimits`.
3. **Stage 3 (`EvalResponse`)**: Raw model generations, reasoning traces (`reasoning_content` / `think_end_token`), and log-likelihoods returned by any `InferenceGateway`:
   - `mock` (`MockGateway`): deterministic 100% offline CPU engine.
   - `http-byok` (`HttpBYOKGateway`): unified async HTTP router with bounded `asyncio.Semaphore` concurrency, `tenacity` retries, and first-class dialects for OpenAI, DeepSeek Direct, Sarvam AI V2, Anthropic, Gemini, OpenRouter, LiteLLM, and local OpenAI-compatible servers (Ollama, vLLM, LM Studio).
   - `[hf]` / `[vllm]` native Python adapters (`src/eval_studio/gateway/adapters/`): lazy roadmap stubs (install optional extras); for local GPU serving today, point `--model http-byok --base-url http://localhost:8000/v1` at a running vLLM or Ollama server.
   - Cached via crash-safe user-dir `fsync`ed JSONL WAL + per-rank SQLite (`eval_studio.cache`) keyed by SHA-256 fingerprints covering task YAML, prompt, model, backend, `dtype`, and `quantization` (bypassed on `do_sample=True`).
4. **Stage 4 (`ScoredSample`, `ReceiptManifest`)**: Multi-pipeline `FilterEnsemble` outputs (longest-first regex, `MajorityVoteFilter` over `repeats`, `ThinkStripFilter`), tokenization-agnostic metrics (`acc_bytes`, `acc_norm`, real unconditional `acc_mutual_info`, `bits_per_byte`, `brier_score`), optional `LLMJudge` (+ `fallback_judge`), exact bootstrap/group stderr (`eval_studio.stats`), and tamper-evident Portable Eval Receipts (`manifest.json`, `receipt.json`, `samples.jsonl`, `report.md`, `report.html`).

## Lazy Plugin Entry Points
External packages register plugins in `pyproject.toml` under:
- `eval_studio.models`
- `eval_studio.solvers`
- `eval_studio.filters`
- `eval_studio.metrics`
- `eval_studio.aggregations`
- `eval_studio.sandboxes`
