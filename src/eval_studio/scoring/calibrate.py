"""Cohen's kappa calibration check for LLM-as-a-judge vs. human gold labels."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalibrationReport:
    """Agreement report between automated judge labels and human annotations."""

    n_samples: int
    cohens_kappa: float
    raw_agreement: float
    passed_threshold: bool
    threshold: float


def cohens_kappa(
    judge_labels: Sequence[str | int],
    human_labels: Sequence[str | int],
) -> float:
    """Compute unweighted Cohen's kappa between two equal-length label sequences."""
    if not judge_labels or len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be non-empty and equal length.")
    n = len(judge_labels)
    categories = sorted(set(judge_labels) | set(human_labels), key=str)
    p_o = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if j == h) / n
    p_e = sum(
        (sum(1 for j in judge_labels if j == c) / n)
        * (sum(1 for h in human_labels if h == c) / n)
        for c in categories
    )
    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def calibrate_judge_file(path: Path, *, threshold: float = 0.40) -> CalibrationReport:
    """Load judge vs. human labels from JSONL and compute Cohen's kappa alignment."""
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    judge_labels: list[str | int] = []
    human_labels: list[str | int] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "judge_label" not in row or "human_label" not in row:
            raise ValueError("Calibration JSONL rows must contain 'judge_label' and 'human_label'.")
        judge_labels.append(row["judge_label"])
        human_labels.append(row["human_label"])
    if not judge_labels:
        raise ValueError("Calibration file contains no labeled rows.")
    kappa = cohens_kappa(judge_labels, human_labels)
    raw_agree = sum(
        1 for j, h in zip(judge_labels, human_labels, strict=True) if j == h
    ) / len(judge_labels)
    return CalibrationReport(
        n_samples=len(judge_labels),
        cohens_kappa=kappa,
        raw_agreement=raw_agree,
        passed_threshold=(kappa + 1e-12) >= threshold,
        threshold=threshold,
    )
