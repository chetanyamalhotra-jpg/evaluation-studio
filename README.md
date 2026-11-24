# Evaluation Studio (`eval-studio`)

An open-source, local-first evaluation harness built around **BYOK** (Bring Your Own Key), **BYOT** (Bring Your Own Test), **BYOP** (Bring Your Own Paper), and **Portable Eval Receipts**.

## Design Pillars
- **Slim Core & Lazy Plugins:** Zero `torch`/`transformers`/`docker`/`pypdf` in base dependencies. Optional extras (`[api]`, `[hf]`, `[vllm]`, `[pdf]`, `[sandbox]`) and zero-config `importlib.metadata` entry points.
- **N+M Four-Stage Pipeline:** Decouples datasets/tasks (`Sample`, `TaskSpec`, multimodal `doc_to_messages`), requests (`EvalRequest`), inference backends (`EvalResponse`), and multi-pipeline scoring (`ScoredSample`, `ReceiptManifest`).
- **BYOK Hygiene & Cost Caps:** Role-separated keys (`TARGET_API_KEY`, `JUDGE_API_KEY`, `EXTRACTOR_API_KEY`, `FALLBACK_JUDGE_API_KEY`), mandatory secret redaction, `--dry-run --estimate-cost`, and a `--max-cost-usd` circuit breaker with WAL resume.
- **BYOT 3-Tier Trust Boundary:** Tier 1 zero-code CSV/JSONL/Parquet, Tier 2 declarative YAML with sandboxed Jinja2, and Tier 3 code plugins isolated behind `--trust-remote-code` + Docker/subprocess sandboxes.
- **BYOP Two-Stage Paper Compiler:** Multi-page chunked extraction with mandatory verbatim page/section citations + independent answerability verification gate + SHA-256 locking.
- **Portable Eval Receipts:** Cryptographically hashed manifests (`manifest.json`, `receipt.json`, `samples.jsonl`, `report.md`, `report.html`) enabling 100% offline re-scoring (`eval-studio score`), tamper auditing (`eval-studio verify-receipt`), and `noise-floor` flip-rate diagnostics.

## Quickstart (Offline CPU)
```bash
pip install -e ".[dev]"
eval-studio --help
eval-studio validate --task examples/tier1_zero_code/task.yaml
eval-studio run --task examples/tier1_zero_code/task.yaml --model mock --output receipts/demo
eval-studio verify-receipt receipts/demo
pytest -m "not gpu and not docker and not live_api"
```

## Web Studio (`eval-studio ui`)
Launch the local-first interactive workbench (FastAPI + SSE backend + dark-first React SPA) to manage BYOK keys/presets, author & validate BYOT tests, compile & audit BYOP papers, stream live runs with pre-flight cost estimates, verify/rescore receipts, and compare models on statistical charts:

```bash
pip install -e ".[web]"
eval-studio ui --host 127.0.0.1 --port 8420
```

## Live BYOK Examples (DeepSeek Direct, Sarvam V2 & Local Servers)
```bash
# DeepSeek Direct (reads DEEPSEEK_API_KEY or TARGET_API_KEY)
eval-studio run --task examples/tier1_zero_code/task.yaml \
  --model http-byok --model-id deepseek-flash --max-cost-usd 1.00 --output receipts/deepseek

# Sarvam AI V2 (reads SARVAM_API_KEY or TARGET_API_KEY via api-subscription-key header)
eval-studio run --task examples/tier1_zero_code/task.yaml \
  --model http-byok --model-id glm5.3-flash --max-cost-usd 1.00 --output receipts/sarvam

# BYOP two-stage paper compilation with BYOK extractor + judge verifier
eval-studio compile-paper examples/papers/sample_paper.md --out paper_pack.yaml \
  --model http-byok --extractor-model deepseek-flash --verifier-model glm5.3-flash

# Local vLLM / Ollama OpenAI-compatible server
eval-studio run --task examples/tier1_zero_code/task.yaml \
  --model http-byok --base-url http://localhost:11434/v1 --model-id llama3.1:8b
```

> **Backend Note:** Native in-process `[hf]` and `[vllm]` adapters are currently roadmap stubs behind optional extras; local GPU/CPU serving via vLLM, Ollama, or LM Studio is supported today through `--model http-byok --base-url <server_url>`.

## Documentation
- [Architecture Overview](docs/architecture.md)
- [BYOK, BYOT, and BYOP Guide](docs/byok_byot_byop.md)
- [Evaluation Harness Research Analysis](docs/eval-harness-research.md)
- [Contributing Guide](CONTRIBUTING.md)
