# Contributing to Evaluation Studio

Thank you for contributing! Please read [AGENTS.md](AGENTS.md) and [docs/architecture.md](docs/architecture.md) before opening a pull request.

## Development Setup
```bash
python -m venv .venv
pip install -e ".[dev]"
```

## Required Checks Before Opening a PR
```bash
ruff check src tests
mypy src
pytest -m "not gpu and not docker and not live_api"
```

## Non-Negotiable Guidelines
1. Keep core dependencies slim—never add `torch`, `transformers`, or `docker` to base `dependencies` in `pyproject.toml`.
2. Add or update an adversarial regression test under `tests/adversarial/` whenever modifying prompt formatting, tokenization, filters, metrics, statistics, caching, or sandboxing.
3. Never commit `.env` files, API keys, SQLite databases, or unredacted receipts.
