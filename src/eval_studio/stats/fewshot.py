"""Few-shot demonstration hygiene, K-set rotation, and unresolved CI comparison rules."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from eval_studio.core.schema import Sample


class ComparisonStatus(str, Enum):
    """Outcome of comparing two models across demonstration sets."""

    RESOLVED_HIGHER = "resolved_higher"
    RESOLVED_LOWER = "resolved_lower"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class FewShotComparison:
    """Cross-set comparison with explicit unresolved status when CI spans zero."""

    mean_delta: float
    ci_low: float
    ci_high: float
    status: ComparisonStatus
    n_sets: int


@dataclass(frozen=True)
class SampledFewShot:
    """Assigned few-shot demonstration bundle for a single active evaluation sample."""

    sample_id: str
    set_index: int
    exemplars: tuple[Sample, ...]
    excluded_doc_hash: str | None = None


def _effective_doc_hash(sample: Sample) -> str:
    if sample.doc_hash:
        return sample.doc_hash
    canonical = json.dumps(
        {
            "input_text": sample.input_text,
            "target": sample.target,
            "choices": sample.choices,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assign_fewshot_sets(
    test_samples: Sequence[Sample],
    exemplar_pool: Sequence[Sample],
    *,
    num_fewshot: int,
    k_sets: int = 1,
    seed: int = 1234,
) -> list[SampledFewShot]:
    """Assign `num_fewshot` exemplars across `k_sets` under a single-pass budget.

    Guarantees active-document exclusion by both `sample_id` and `doc_hash`.
    """
    if num_fewshot < 0:
        raise ValueError("num_fewshot must be >= 0.")
    if k_sets < 1:
        raise ValueError("k_sets must be >= 1.")

    if num_fewshot == 0:
        return [
            SampledFewShot(
                sample_id=sample.sample_id,
                set_index=i % k_sets,
                exemplars=(),
                excluded_doc_hash=_effective_doc_hash(sample),
            )
            for i, sample in enumerate(test_samples)
        ]

    if not exemplar_pool:
        raise ValueError("exemplar_pool must be non-empty when num_fewshot > 0.")

    rng = np.random.default_rng(seed)
    pool_size = len(exemplar_pool)
    permutations: list[list[int]] = [
        [int(idx) for idx in rng.permutation(pool_size)] for _ in range(k_sets)
    ]

    assignments: list[SampledFewShot] = []
    for i, sample in enumerate(test_samples):
        set_index = i % k_sets
        active_hash = _effective_doc_hash(sample)
        chosen: list[Sample] = []
        for idx in permutations[set_index]:
            candidate = exemplar_pool[idx]
            if candidate.sample_id == sample.sample_id:
                continue
            if _effective_doc_hash(candidate) == active_hash:
                continue
            chosen.append(candidate)
            if len(chosen) == num_fewshot:
                break

        if len(chosen) < num_fewshot:
            raise ValueError(
                f"Insufficient non-leaking exemplars for sample '{sample.sample_id}': "
                f"requested {num_fewshot}, found {len(chosen)}."
            )

        assignments.append(
            SampledFewShot(
                sample_id=sample.sample_id,
                set_index=set_index,
                exemplars=tuple(chosen),
                excluded_doc_hash=active_hash,
            )
        )
    return assignments


def format_fewshot_prompt(
    exemplars: Sequence[Sample],
    test_sample: Sample,
    *,
    render_doc: Callable[[Sample], str],
    render_target: Callable[[Sample], str],
    fewshot_delimiter: str = "\n\n",
    target_delimiter: str = " ",
    description: str = "",
    gen_prefix: str = "",
    render_gen_prefix: Callable[[Sample], str] | None = None,
) -> str:
    """Format few-shot exemplars + active test prompt with `gen_prefix` strictly on test doc."""
    sections: list[str] = []
    if description:
        sections.append(description)
    for ex in exemplars:
        sections.append(f"{render_doc(ex)}{target_delimiter}{render_target(ex)}")

    resolved_prefix = (
        render_gen_prefix(test_sample) if render_gen_prefix is not None else gen_prefix
    )
    sections.append(f"{render_doc(test_sample)}{resolved_prefix}")
    return fewshot_delimiter.join(sections)


def compute_required_k_sets(
    cross_set_std: float,
    min_detectable_gap: float,
    *,
    z_alpha: float = 1.645,
) -> int:
    """Compute minimum demonstration sets K = ceil((z_alpha * sigma / gap)^2)."""
    if cross_set_std < 0.0 or not math.isfinite(cross_set_std):
        raise ValueError("cross_set_std must be finite and >= 0.")
    if min_detectable_gap <= 0.0 or not math.isfinite(min_detectable_gap):
        raise ValueError("min_detectable_gap must be finite and > 0.")
    if z_alpha <= 0.0 or not math.isfinite(z_alpha):
        raise ValueError("z_alpha must be finite and > 0.")
    if cross_set_std == 0.0:
        return 1
    return max(1, math.ceil((z_alpha * cross_set_std / min_detectable_gap) ** 2))


def evaluate_fewshot_comparison(
    deltas_across_sets: Sequence[float] | np.ndarray[Any, Any],
    *,
    z_value: float = 1.96,
    noise_floor: float = 0.0,
) -> FewShotComparison:
    """Compute cross-set confidence interval and mark unresolved if CI spans 0 or <= noise_floor."""
    if len(deltas_across_sets) == 0:
        raise ValueError("deltas_across_sets must be non-empty.")
    if z_value <= 0.0 or not math.isfinite(z_value):
        raise ValueError("z_value must be finite and > 0.")
    if noise_floor < 0.0 or not math.isfinite(noise_floor):
        raise ValueError("noise_floor must be finite and >= 0.")

    arr = np.asarray(deltas_across_sets, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("deltas_across_sets must contain only finite values.")

    n = len(arr)
    mean_delta = float(np.mean(arr))
    if n < 2:
        return FewShotComparison(
            mean_delta=mean_delta,
            ci_low=mean_delta,
            ci_high=mean_delta,
            status=ComparisonStatus.UNRESOLVED,
            n_sets=n,
        )
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    ci_low = mean_delta - z_value * se
    ci_high = mean_delta + z_value * se
    if ci_low <= 0.0 <= ci_high or abs(mean_delta) <= noise_floor:
        status = ComparisonStatus.UNRESOLVED
    elif mean_delta > 0.0:
        status = ComparisonStatus.RESOLVED_HIGHER
    else:
        status = ComparisonStatus.RESOLVED_LOWER
    return FewShotComparison(
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        status=status,
        n_sets=n,
    )


def evaluate_fewshot_scores(
    scores_a_by_set: Sequence[float] | np.ndarray[Any, Any],
    scores_b_by_set: Sequence[float] | np.ndarray[Any, Any],
    *,
    z_value: float = 1.96,
    noise_floor: float = 0.0,
) -> FewShotComparison:
    """Compare per-set scores (`scores_b - scores_a`) across K rotated demonstration sets."""
    if len(scores_a_by_set) == 0 or len(scores_a_by_set) != len(scores_b_by_set):
        raise ValueError("scores_a_by_set and scores_b_by_set must be non-empty and equal length.")
    deltas = [
        float(b) - float(a)
        for a, b in zip(scores_a_by_set, scores_b_by_set, strict=True)
    ]
    return evaluate_fewshot_comparison(deltas, z_value=z_value, noise_floor=noise_floor)
