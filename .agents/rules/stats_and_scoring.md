---
trigger: glob
description: Tokenization-agnostic metrics, exact bootstrap stderr, micro vs macro aggregation, few-shot rotation, and unresolved CI rules
globs: "**/scoring/**, **/stats/**"
---

# Statistical Rigor & Scoring Rules

## Tokenization-Agnostic Metrics
- Never normalize multiple-choice log-likelihoods by token count (violates cross-tokenizer comparability).
- Support and prefer `acc_bytes` (UTF-8 byte-length normalization), `acc_norm` (character-length normalization), and `acc_mutual_info` (unconditional PMI normalization).
- Distinguish corpus-level classification `f1` from token-overlap QA F1 explicitly in metric names and docstrings.

## Exact Bootstrap & Hierarchical Group Stderr
- `bootstrap_stderr` MUST execute the exact `bootstrap_iters` count requested (never truncate non-multiples of 1,000) using an explicit seeded `numpy.random.Generator` (never global RNG state).
- Group aggregation MUST pair:
  - `weight_by_size=True` (micro-averaging) -> `pooled_sample_stderr`
  - `weight_by_size=False` (macro-averaging) -> `unweighted_mean_stderr`
- Never silently drop metrics when subtasks in a group report heterogeneous metric keys; raise a clear validation error or emit explicit per-metric rollups.

## Few-Shot Rotation & Unresolved Comparisons
- Support rotating `K` demonstration sets across `N` items under a fixed single-pass budget.
- When comparing two models/checkpoints across multiple demonstration sets or repeated runs, compute the cross-set confidence interval and explicitly label the comparison **`unresolved`** whenever the interval spans zero or falls within the measured infrastructure noise floor.

## Filter & Scoring Anti-Bug Guardrails
- Sort regex alternatives longest-first before compiling multi-choice extraction patterns to prevent prefix shadowing.
- Normalize/lowercase text using Python `str.casefold()` / `str.lower()` and `unicodedata.normalize`—never `numpy.char.lower`.
- Compare large integers as exact Python `int` (or `Decimal`), never via lossy `float` conversion.
- When using LLM-as-a-judge, expose calibration via Cohen's kappa (`eval_studio.scoring.calibrate`) against human-labeled gold subsets.
