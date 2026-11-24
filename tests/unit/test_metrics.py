"""Unit tests for F8 tokenization-agnostic, perplexity, F1, numeric, and LaTeX metrics."""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from eval_studio.scoring.metrics import (
    acc,
    acc_all,
    acc_bytes,
    acc_mutual_info,
    acc_norm,
    bits_per_byte,
    brier_score,
    byte_perplexity,
    classification_f1,
    contains_whole_word,
    exact_match,
    latex_exact_match,
    normalize_latex,
    numeric_exact_match,
    parse_exact_number,
    squad_f1,
    token_f1,
    validate_bypass_usage,
    weighted_perplexity,
    whole_word_match,
    word_perplexity,
)


def test_multiple_choice_metrics_and_brier_score() -> None:
    assert acc([-1.0, -3.0], target_idx=0) == 1.0
    assert acc_norm([-4.0, -5.0], ["no", "yes_longer"], target_idx=1) == 1.0
    assert acc_bytes([-4.0, -5.0], ["no", "yes_longer"], target_idx=1) == 1.0
    assert acc_mutual_info([-2.0, -1.5], [-0.5, -2.0], target_idx=1) == 1.0
    assert acc_all([1.0, True, 1.0]) == 1.0
    assert acc_all([1.0, 0.0, 1.0]) == 0.0

    # Brier score: confident correct -> ~0, confident wrong -> ~2.0, uniform -> (1-0.5)^2*2=0.5
    assert brier_score([100.0, -100.0], target_idx=0) < 1e-12
    assert math.isclose(brier_score([-100.0, 100.0], target_idx=0), 2.0, abs_tol=1e-12)
    assert math.isclose(brier_score([-1.0, -1.0], target_idx=0), 0.5, rel_tol=1e-12)
    assert math.isclose(
        brier_score([-1.0, -1.0], target_idx=0, normalize_by_choices=True),
        0.25,
        rel_tol=1e-12,
    )
    assert math.isclose(brier_score([float("-inf"), float("-inf")], target_idx=0), 0.5)


def test_rolling_perplexity_and_bits_per_byte() -> None:
    texts = ["hello world", "clinical trial"]
    lls = [-12.0, -15.0]
    bpb = bits_per_byte(lls, texts)
    bpp = byte_perplexity(lls, texts)
    wpp = word_perplexity(lls, texts)
    assert math.isclose(bpp, 2.0**bpb, rel_tol=1e-12)
    assert wpp > bpp
    assert weighted_perplexity([-10000.0], [1]) == float("inf")


def test_bypass_requires_log_samples() -> None:
    validate_bypass_usage(log_samples=True)
    with pytest.raises(ValueError, match="log_samples=True"):
        validate_bypass_usage(log_samples=False)


def test_classification_f1_vs_token_f1_distinction() -> None:
    # Corpus classification F1 across binary predictions
    preds = [1, 1, 0, 0]
    refs = [1, 0, 0, 1]
    assert math.isclose(classification_f1(preds, refs, average="binary"), 0.5, rel_tol=1e-12)
    assert math.isclose(classification_f1(preds, refs, average="macro"), 0.5, rel_tol=1e-12)

    # Token-overlap SQuAD F1 removes articles and punctuation
    score = token_f1("the quick brown fox!", "a quick brown fox")
    assert score == 1.0
    assert squad_f1("acute renal failure", "chronic renal failure") == pytest.approx(2.0 / 3.0)


def test_exact_numeric_comparison_avoids_float_loss() -> None:
    # 2**53 == 9007199254740992 vs 9007199254740993 are indistinguishable in IEEE-754 float64
    assert numeric_exact_match("9007199254740992", "9007199254740993") == 0.0
    assert numeric_exact_match("1,000,000", 1000000) == 1.0
    assert numeric_exact_match(Decimal("0.3"), "0.3000") == 1.0
    assert parse_exact_number("123456789012345678901234567890") == 123456789012345678901234567890
    with pytest.raises(ValueError):
        parse_exact_number(True)


def test_whole_word_and_latex_normalization() -> None:
    assert contains_whole_word("category", "cat") is False
    assert whole_word_match("the cat slept", "cat") == 1.0
    assert exact_match("İstanbul", "i\u0307stanbul") == 1.0

    # Indexed roots and ordered tuples must be preserved
    assert normalize_latex(r"\sqrt[3]{8}") != normalize_latex(r"\sqrt{8}")
    assert latex_exact_match(r"\boxed{\sqrt[3]{8}}", r"$\sqrt [ 3 ] {8}$") == 1.0
    assert latex_exact_match("(1, 2)", "(2, 1)") == 0.0
    assert latex_exact_match(r"\boxed{\dfrac12}", r"$\frac{1}{2}$") == 1.0
    assert latex_exact_match("1,000", "1000") == 1.0
