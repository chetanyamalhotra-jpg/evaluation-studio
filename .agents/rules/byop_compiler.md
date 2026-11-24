---
trigger: model_decision
description: Rules for BYOP two-stage compile-paper extraction, mandatory page/section citations, answerability verification gate, and SHA-256 locking
---

# BYOP (`compile-paper`) Compiler Rules

Apply these rules whenever creating, modifying, or testing the Bring-Your-Own-Paper (`src/eval_studio/paper/`) pipeline or compiling benchmark suites from scientific papers.

## Two-Stage Compilation Pipeline
1. **Stage 1 — Grounded Extraction (`extract.py`, `compiler.py`):**
   - Extract candidate benchmark items (questions, clinical vignettes, rubrics, reference answers) from source PDF or Markdown papers.
   - Every extracted item MUST include a structured `Citation` (`page`, `section`, and verbatim `quote` span). Items without verifiable provenance MUST be discarded, never fabricated.
2. **Stage 2 — Independent Answerability Verification Gate (`verify.py`):**
   - Run a separate verification pass (ideally with an independent verifier prompt/model) checking:
     - Self-containedness (no dangling references like "as shown in Table 2 above" unless Table 2 is embedded in the prompt).
     - Answerability (the gold answer follows strictly from the cited evidence).
     - Non-leakage (the prompt does not inadvertently contain the gold answer label).
   - Mark items as `APPROVED`, `NEEDS_REVIEW`, or `REJECTED` with structured reasons. Only `APPROVED` items enter the compiled executable suite by default.

## Cryptographic Locking & Provenance
- Compute and store the SHA-256 digest of:
  - The raw source paper file (`source_sha256`).
  - Each compiled item's canonical JSON representation (`item_sha256`).
  - The full compiled suite (`suite_sha256`).
- Record extractor/verifier model identifiers, prompt versions, and timestamps (with all API keys redacted) in the compiled manifest so downstream Eval Receipts trace directly back to the exact paper revision.
