---
name: verify-receipt
description: >-
  Workflow for auditing a Portable Eval Receipt's SHA-256 hashes, secret redaction,
  provenance metadata, and offline re-scoring parity.
---

# Verify a Portable Eval Receipt

Use this workflow when auditing an evaluation run artifact (`manifest.json` + `samples.jsonl`) for reproducibility, integrity, and security hygiene.

## 1. Run Cryptographic & Structural Verification
```bash
eval-studio verify-receipt <path/to/receipt_dir>
```
Confirm:
- `manifest.json` conforms to `ReceiptManifest` schema.
- `samples_sha256`, `task_sha256`, and (for BYOP runs) `source_paper_sha256` match recomputed digests.
- Any tampered sample row, modified score, or altered prompt triggers a verification failure.

## 2. Audit Secret Redaction
- Scan `manifest.json` and `samples.jsonl` using `eval_studio.gateway.secrets.contains_secret()` (or the verification command's built-in leak check).
- Confirm zero raw API keys, bearer tokens, or `.env` values appear in config snapshots, error traces, or headers.

## 3. Check Provenance & Statistical Completeness
Verify the receipt records:
- Harness version and git commit (when available).
- Model ID, backend engine, dtype/quantization, batch size, and random seeds.
- Standard errors (`stderr`), bootstrap iteration count, and whether close comparisons are marked `unresolved` against the noise floor.

## 4. Verify Offline Re-Scoring Parity
Re-score the saved raw responses offline without making any model/API calls:
```bash
eval-studio score --receipt <path/to/receipt_dir>
```
Confirm that re-computed aggregate metrics and stderrs match `manifest.json` within floating-point tolerance.
