# BYOP (`compile-paper`) Examples

Use `eval-studio compile-paper` to convert a scientific paper (PDF or Markdown) into a SHA-256 locked, citation-grounded evaluation suite:

```bash
eval-studio compile-paper examples/papers/sample_paper.md --output compiled_suite.json
```

## Two-Stage Verification Gate
1. **Stage 1 (Extraction):** Extracts candidate questions and reference answers with mandatory `page`, `section`, and verbatim `quote` citations.
2. **Stage 2 (Verification):** Rejects items missing citations, leaking the answer, or depending on unextracted figures/tables.
3. **SHA-256 Locking:** Records `source_sha256`, per-item `item_sha256`, and `suite_sha256` for full provenance inside Portable Eval Receipts.
