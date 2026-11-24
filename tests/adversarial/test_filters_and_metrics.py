"""Adversarial tests for regex prefix shadowing, Unicode safety, F8 metrics, and F9 filters."""

from __future__ import annotations

from eval_studio.scoring.filters import (
    MajorityVoteFilter,
    MultiChoiceRegexFilter,
    ThinkStripFilter,
    format_olmes_mcf_prompt,
)
from eval_studio.scoring.metrics import (
    acc_bytes,
    contains_whole_word,
    exact_match,
    latex_exact_match,
    normalize_latex,
    numeric_exact_match,
)


def test_multichoice_regex_longest_first_prevents_prefix_shadowing() -> None:
    flt = MultiChoiceRegexFilter(["Guilty", "Guilty of Romance"])
    out = flt.apply(["The best match is Guilty of Romance."])
    assert out == ["Guilty of Romance"]


def test_multichoice_regex_ignores_empty_choice_strings() -> None:
    flt = MultiChoiceRegexFilter(["", "Yes", "No"])
    assert flt.apply(["Irrelevant text"]) == ["[invalid]"]


def test_unicode_exact_match_does_not_truncate_expanded_characters() -> None:
    # Turkish dotted capital I ('İ') and German sharp s ('ß') expand under casefold/lower
    assert exact_match("İstanbul", "i\u0307stanbul") == 1.0
    assert exact_match("Straße", "STRASSE") == 1.0


def test_acc_bytes_normalizes_by_utf8_bytes_not_token_count() -> None:
    # Choice 1 is longer in UTF-8 bytes with higher per-byte log-likelihood
    score = acc_bytes(loglikelihoods=[-4.0, -5.0], continuations=["no", "yes_longer"], target_idx=1)
    assert score == 1.0


def test_numeric_exact_match_avoids_float64_precision_loss() -> None:
    assert numeric_exact_match("9007199254740992", "9007199254740993") == 0.0
    assert numeric_exact_match("100000000000000000000", "100000000000000000001") == 0.0


def test_whole_word_boundary_blocks_substring_false_positives() -> None:
    assert contains_whole_word("category", "cat") is False
    assert contains_whole_word("bobcat", "cat") is False
    assert contains_whole_word("the cat sat", "cat") is True


def test_latex_normalization_preserves_indexed_roots_and_tuples() -> None:
    assert normalize_latex(r"\sqrt[3]{8}") != normalize_latex(r"\sqrt{8}")
    assert latex_exact_match("(1, 2)", "(2, 1)") == 0.0


def test_think_strip_uses_last_delimiter_and_majority_vote_ignores_invalid() -> None:
    flt = ThinkStripFilter("</think>")
    assert flt.apply(["Mentioned </think> in thought </think>Final Answer"]) == ["Final Answer"]

    mv = MajorityVoteFilter()
    assert mv.apply(["[invalid]", "[invalid]", "Option B"]) == ["Option B"]


def test_olmes_mcf_retains_leading_space_before_label() -> None:
    bundle = format_olmes_mcf_prompt("Q?", ["Alpha", "Beta"])
    assert "\n A. Alpha\n B. Beta" in bundle.context
    assert bundle.continuations == [" A", " B"]
