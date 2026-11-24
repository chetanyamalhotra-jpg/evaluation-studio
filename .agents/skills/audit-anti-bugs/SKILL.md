---
name: audit-anti-bugs
description: >-
  Checklist and command workflow to run the adversarial regression suite and audit
  Evaluation Studio for silent validity defects documented in docs/eval-harness-research.md.
---

# Audit Anti-Bug Guardrails

Use this workflow before merging changes to prompt construction, tokenization, filters, metrics, statistics, caching, or sandboxing.

## 1. Run Static Checks & Offline Test Suite
```bash
ruff check src tests
mypy src
pytest -m "not gpu and not docker and not live_api"
```

## 2. Run Adversarial Regression Suite Explicitly
```bash
pytest tests/adversarial/ -v
```

## 3. Verify All 10 Silent Validity Guardrails
1. **Whitespace transfer:** `_encode_pair(context, continuation)` shifts trailing spaces from `context` to `continuation`.
2. **BOS / chat-template conflict:** Combining `apply_chat_template=True` with `add_bos_token=True` raises a validation error.
3. **Regex longest-first sorting:** `MultiChoiceRegexFilter` matches `"Guilty of Romance"` rather than prefix-shadowing with `"Guilty"`.
4. **Unicode normalization safety:** Lowercasing/normalizing Unicode strings (e.g., Turkish `İ`, sharp `ß`) never truncates via `np.char.lower`.
5. **Exact `bootstrap_stderr`:** Requesting `bootstrap_iters=1500` runs exactly 1,500 resamples (never truncates to 1,000).
6. **Micro vs. macro stderr pairing:** `weight_by_size=True` uses `pooled_sample_stderr`; `weight_by_size=False` uses `unweighted_mean_stderr`.
7. **Few-shot leakage & rotation:** Active evaluation documents are excluded from few-shot exemplars; cross-set CIs spanning zero are flagged `unresolved`.
8. **`__eq__` sandbox hardening:** Untrusted code returning an object whose `__eq__` always returns `True` cannot spoof `pass@1 = 1.0`.
9. **BYOK & Jinja2 safety:** Secret redaction scrubs API keys from logs/receipts; cost cap halts over-budget runs; Jinja2 sandbox blocks SSTI payloads.
10. **Cache isolation:** Default cache resolves to a user-owned `platformdirs` path, never `site-packages`, with zero `pickle`/`dill` usage.
