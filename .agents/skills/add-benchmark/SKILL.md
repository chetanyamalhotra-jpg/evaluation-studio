---
name: add-benchmark
description: >-
  Workflow for scaffolding a new Tier 1 (CSV/JSONL), Tier 2 (declarative YAML/Jinja2),
  or Tier 3 (code plugin) benchmark test pack, running eval-studio validate, and generating
  a golden prompt snapshot under tests/snapshots/.
---

# Add a Benchmark Test Pack

Use this workflow when adding a new BYOT benchmark to `examples/` or a user task directory.

## 1. Choose the Right Trust Tier
- **Tier 1 (Zero-Code Data):** Tabular `.csv` or `.jsonl` dataset + minimal 10-line YAML pointing to `data_files`, `doc_to_text`, and `doc_to_target`.
- **Tier 2 (Declarative YAML):** Custom Jinja2 prompt templates (sandboxed), `FilterEnsemble` post-processing pipelines, and few-shot rotation config (`num_fewshot`, `rotation_sets`).
- **Tier 3 (Code Plugin):** Custom Python `Solver`, `Scorer`, or code-execution task. Requires `--trust-remote-code` and Docker sandbox isolation for untrusted code execution.

## 2. Scaffold the Task Files
1. Create the task directory and YAML spec (e.g., `examples/tier2_declarative/<task_name>.yaml`).
2. Confirm no API keys or secrets appear anywhere in the YAML or fixture data.
3. Verify `metadata.version` is set and `output_type` matches the requested metrics (`acc_bytes` / `acc_norm` for `multiple_choice`; `exact_match` for `generate_until`).
4. For multi-choice regex extraction, ensure choices are sorted longest-first by the filter pipeline.

## 3. Validate Offline
Run the validator before running any model inference:
```bash
eval-studio validate --task <path/to/task.yaml>
```
Check that:
- YAML schema validates cleanly via Pydantic (`TaskSpec`).
- Jinja2 templates render inside `SandboxedEnvironment` without errors.
- No BOS + chat-template conflict or `enable_thinking` + `loglikelihood` conflict exists.

## 4. Generate & Lock a Golden Prompt Snapshot
1. Render a small dry-run sample (`--limit 3`) using the deterministic offline `mock` gateway.
2. Save the rendered prompt snapshot to `tests/snapshots/<task_name>.prompt.txt`.
3. Add a unit test asserting rendered prompts match the golden snapshot byte-for-byte so future template or whitespace regressions fail CI immediately.
