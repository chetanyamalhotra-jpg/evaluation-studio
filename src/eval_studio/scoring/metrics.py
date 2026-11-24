"""Tokenization-agnostic evaluation metrics, exact comparison, and LaTeX normalization."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Literal

from eval_studio.core.registry import get_registry


def _validate_mc_inputs(
    loglikelihoods: Sequence[float],
    target_idx: int,
    n_expected: int | None = None,
) -> None:
    if not loglikelihoods:
        raise ValueError("loglikelihoods must be non-empty.")
    if n_expected is not None and len(loglikelihoods) != n_expected:
        raise ValueError("loglikelihoods and continuations must be equal length.")
    if not (0 <= target_idx < len(loglikelihoods)):
        raise ValueError(
            f"target_idx {target_idx} out of bounds for {len(loglikelihoods)} choices."
        )
    if any(math.isnan(x) or x == float("inf") for x in loglikelihoods):
        raise ValueError("loglikelihoods must not contain NaN or +inf.")


def acc(
    loglikelihoods: Sequence[float],
    target_idx: int,
    continuations: Sequence[str] | None = None,
) -> float:
    """Unnormalized multiple-choice accuracy (argmax over raw log-likelihoods)."""
    _validate_mc_inputs(
        loglikelihoods,
        target_idx,
        n_expected=len(continuations) if continuations is not None else None,
    )
    pred_idx = max(range(len(loglikelihoods)), key=lambda i: loglikelihoods[i])
    return 1.0 if pred_idx == target_idx else 0.0


def acc_bytes(
    loglikelihoods: Sequence[float],
    continuations: Sequence[str],
    target_idx: int,
) -> float:
    """UTF-8 byte-length normalized multiple-choice accuracy (tokenization-agnostic)."""
    _validate_mc_inputs(loglikelihoods, target_idx, n_expected=len(continuations))
    scores = [
        ll / max(1, len(cont.encode("utf-8")))
        for ll, cont in zip(loglikelihoods, continuations, strict=True)
    ]
    pred_idx = max(range(len(scores)), key=lambda i: scores[i])
    return 1.0 if pred_idx == target_idx else 0.0


def acc_norm(
    loglikelihoods: Sequence[float],
    continuations: Sequence[str],
    target_idx: int,
) -> float:
    """Character-length normalized multiple-choice accuracy (tokenization-agnostic)."""
    _validate_mc_inputs(loglikelihoods, target_idx, n_expected=len(continuations))
    scores = [
        ll / max(1, len(cont))
        for ll, cont in zip(loglikelihoods, continuations, strict=True)
    ]
    pred_idx = max(range(len(scores)), key=lambda i: scores[i])
    return 1.0 if pred_idx == target_idx else 0.0


def acc_mutual_info(
    cond_loglikelihoods: Sequence[float],
    uncond_loglikelihoods: Sequence[float],
    target_idx: int,
) -> float:
    """Pointwise mutual information (PMI) multiple-choice accuracy."""
    if not cond_loglikelihoods or len(cond_loglikelihoods) != len(uncond_loglikelihoods):
        raise ValueError("Conditional and unconditional loglikelihoods must match in length.")
    _validate_mc_inputs(cond_loglikelihoods, target_idx)
    if any(math.isnan(x) or x == float("inf") for x in uncond_loglikelihoods):
        raise ValueError("uncond_loglikelihoods must not contain NaN or +inf.")
    scores = [
        c - u for c, u in zip(cond_loglikelihoods, uncond_loglikelihoods, strict=True)
    ]
    pred_idx = max(range(len(scores)), key=lambda i: scores[i])
    return 1.0 if pred_idx == target_idx else 0.0


def acc_all(subquestion_correct: Sequence[bool | float]) -> float:
    """Group accuracy awarding 1.0 only if every subquestion in a bundle is correct."""
    if not subquestion_correct:
        raise ValueError("subquestion_correct must be non-empty.")
    return 1.0 if all(float(v) >= 1.0 for v in subquestion_correct) else 0.0


def brier_score(
    loglikelihoods: Sequence[float],
    target_idx: int,
    *,
    normalize_by_choices: bool = False,
) -> float:
    """Compute Brier score over softmax-normalized choice probabilities (lower is better).

    Uses the standard Brier (1950) / multiclass sum-of-squared-errors formulation in
    `[0.0, 2.0]` by default; set `normalize_by_choices=True` to divide by `len(choices)`.
    """
    _validate_mc_inputs(loglikelihoods, target_idx)
    k = len(loglikelihoods)
    max_ll = max(loglikelihoods)
    if max_ll == float("-inf"):
        probs = [1.0 / k for _ in range(k)]
    else:
        exps = [math.exp(ll - max_ll) for ll in loglikelihoods]
        denom = sum(exps)
        probs = [e / denom for e in exps]
    targets = [1.0 if i == target_idx else 0.0 for i in range(k)]
    score = sum((p - y) ** 2 for p, y in zip(probs, targets, strict=True))
    return score / k if normalize_by_choices else score


def _validate_rolling_inputs(
    loglikelihoods: Sequence[float],
    weights: Sequence[int | float],
) -> None:
    if not loglikelihoods or len(loglikelihoods) != len(weights):
        raise ValueError("loglikelihoods and weights must be non-empty and equal length.")
    if any(math.isnan(x) or math.isinf(x) for x in loglikelihoods):
        raise ValueError("loglikelihoods must be finite.")
    if any(math.isnan(float(w)) or float(w) < 0.0 for w in weights):
        raise ValueError("weights must be non-negative finite values.")


def weighted_perplexity(
    loglikelihoods: Sequence[float],
    weights: Sequence[int | float],
) -> float:
    """Compute corpus-level weighted perplexity `exp(-sum(ll) / sum(weights))`."""
    _validate_rolling_inputs(loglikelihoods, weights)
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Total weight must be > 0.")
    exponent = -sum(loglikelihoods) / total_weight
    if exponent > 709.78:
        return float("inf")
    return math.exp(exponent)


def bits_per_byte(loglikelihoods: Sequence[float], texts: Sequence[str]) -> float:
    """Compute tokenization-agnostic bits per UTF-8 byte over rolling log-likelihoods."""
    byte_counts = [len(t.encode("utf-8")) for t in texts]
    _validate_rolling_inputs(loglikelihoods, byte_counts)
    total_bytes = sum(byte_counts)
    if total_bytes <= 0:
        raise ValueError("Total UTF-8 byte count across texts must be > 0.")
    return -sum(loglikelihoods) / (math.log(2.0) * total_bytes)


def byte_perplexity(loglikelihoods: Sequence[float], texts: Sequence[str]) -> float:
    """Compute UTF-8 byte-level perplexity (`2 ** bits_per_byte`)."""
    byte_counts = [len(t.encode("utf-8")) for t in texts]
    return weighted_perplexity(loglikelihoods, byte_counts)


def word_perplexity(loglikelihoods: Sequence[float], texts: Sequence[str]) -> float:
    """Compute whitespace-delimited word-level perplexity."""
    word_counts = [len(t.split()) for t in texts]
    return weighted_perplexity(loglikelihoods, word_counts)


def validate_bypass_usage(*, log_samples: bool) -> None:
    """Ensure the `bypass` metric is only used when `log_samples=True`."""
    if not log_samples:
        raise ValueError(
            "Metric 'bypass' requires log_samples=True so raw outputs are preserved for "
            "offline scoring."
        )


def bypass(*, log_samples: bool = True) -> float | None:
    """Defer scoring during inference while enforcing `log_samples=True`."""
    validate_bypass_usage(log_samples=log_samples)
    return None


def classification_f1(
    predictions: Sequence[str | int],
    references: Sequence[str | int],
    *,
    average: str = "binary",
    pos_label: str | int = 1,
    zero_division: float = 0.0,
) -> float:
    """Compute corpus-level classification F1 (distinct from QA token-overlap F1)."""
    if not predictions or len(predictions) != len(references):
        raise ValueError("predictions and references must be non-empty and equal length.")
    if average not in {"binary", "macro", "micro", "weighted"}:
        raise ValueError("average must be one of {'binary', 'macro', 'micro', 'weighted'}.")

    def _f1_counts(tp: int, fp: int, fn: int) -> float:
        denom = 2 * tp + fp + fn
        return zero_division if denom == 0 else (2.0 * tp) / denom

    if average == "binary":
        tp = sum(
            1 for p, r in zip(predictions, references, strict=True)
            if p == pos_label and r == pos_label
        )
        fp = sum(
            1 for p, r in zip(predictions, references, strict=True)
            if p == pos_label and r != pos_label
        )
        fn = sum(
            1 for p, r in zip(predictions, references, strict=True)
            if p != pos_label and r == pos_label
        )
        return _f1_counts(tp, fp, fn)

    labels = sorted({*predictions, *references}, key=str)
    if average == "micro":
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0
        for label in labels:
            tp_sum += sum(
                1 for p, r in zip(predictions, references, strict=True)
                if p == label and r == label
            )
            fp_sum += sum(
                1 for p, r in zip(predictions, references, strict=True)
                if p == label and r != label
            )
            fn_sum += sum(
                1 for p, r in zip(predictions, references, strict=True)
                if p != label and r == label
            )
        return _f1_counts(tp_sum, fp_sum, fn_sum)

    per_class_f1: list[float] = []
    supports: list[int] = []
    for label in labels:
        tp = sum(
            1 for p, r in zip(predictions, references, strict=True)
            if p == label and r == label
        )
        fp = sum(
            1 for p, r in zip(predictions, references, strict=True)
            if p == label and r != label
        )
        fn = sum(
            1 for p, r in zip(predictions, references, strict=True)
            if p != label and r == label
        )
        support = sum(1 for r in references if r == label)
        per_class_f1.append(_f1_counts(tp, fp, fn))
        supports.append(support)

    if average == "macro":
        return sum(per_class_f1) / len(per_class_f1)
    total_support = sum(supports)
    if total_support == 0:
        return zero_division
    return sum(
        f1_val * sup for f1_val, sup in zip(per_class_f1, supports, strict=True)
    ) / total_support


def _normalize_qa_tokens(text: str, *, normalize: bool = True) -> list[str]:
    if not normalize:
        return text.split()
    norm = unicodedata.normalize("NFC", text).casefold()
    no_punct = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch for ch in norm
    )
    no_articles = re.sub(r"\b(?:a|an|the)\b", " ", no_punct)
    return no_articles.split()


def token_f1(prediction: str, reference: str, *, normalize: bool = True) -> float:
    """Compute SQuAD-style token-overlap F1 between a predicted and reference span."""
    pred_tokens = _normalize_qa_tokens(prediction, normalize=normalize)
    ref_tokens = _normalize_qa_tokens(reference, normalize=normalize)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return (2.0 * precision * recall) / (precision + recall)


squad_f1 = token_f1


def exact_match(
    prediction: str,
    reference: str,
    *,
    ignore_case: bool = True,
    normalize_unicode: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFC",
) -> float:
    """Unicode-safe exact match using Python str.casefold() (never np.char.lower)."""
    pred = unicodedata.normalize(normalize_unicode, prediction.strip())
    ref = unicodedata.normalize(normalize_unicode, reference.strip())
    if ignore_case:
        pred = pred.casefold().replace("i\u0307", "i")
        ref = ref.casefold().replace("i\u0307", "i")
    return 1.0 if pred == ref else 0.0


_GROUPED_INT_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_PLAIN_INT_RE = re.compile(r"^[+-]?\d+$")


def parse_exact_number(value: str | int | Decimal) -> int | Decimal:
    """Parse an integer or decimal without lossy IEEE-754 float conversion."""
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid exact numbers.")
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Decimal value must be finite.")
        return value
    if not isinstance(value, str):
        raise ValueError(f"Unsupported numeric value type: {type(value).__name__}")

    cleaned = unicodedata.normalize("NFC", value.strip())
    if not cleaned:
        raise ValueError("Empty string cannot be parsed as a number.")
    if _GROUPED_INT_RE.fullmatch(cleaned):
        cleaned = cleaned.replace(",", "")
    if _PLAIN_INT_RE.fullmatch(cleaned):
        return int(cleaned)
    with localcontext() as ctx:
        ctx.prec = 100
        try:
            dec = Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid numeric literal: '{value}'") from exc
        if not dec.is_finite():
            raise ValueError(f"Non-finite numeric literal: '{value}'")
        return dec


def numeric_exact_match(
    prediction: str | int | Decimal,
    reference: str | int | Decimal,
) -> float:
    """Exact integer/Decimal comparison avoiding float64 precision loss (>2^53)."""
    try:
        parsed_pred = parse_exact_number(prediction)
        parsed_ref = parse_exact_number(reference)
    except ValueError:
        return 0.0

    if isinstance(parsed_pred, int) and isinstance(parsed_ref, int):
        return 1.0 if parsed_pred == parsed_ref else 0.0

    with localcontext() as ctx:
        ctx.prec = 100
        dec_pred = Decimal(parsed_pred) if isinstance(parsed_pred, int) else parsed_pred
        dec_ref = Decimal(parsed_ref) if isinstance(parsed_ref, int) else parsed_ref
        return 1.0 if dec_pred == dec_ref else 0.0


def contains_whole_word(
    text: str,
    keyword: str,
    *,
    ignore_case: bool = True,
) -> bool:
    """Check whether `keyword` appears as a whole word (never substring-matching)."""
    if not keyword:
        raise ValueError("keyword must be non-empty.")
    norm_text = unicodedata.normalize("NFC", text)
    norm_kw = unicodedata.normalize("NFC", keyword)
    if ignore_case:
        norm_text = norm_text.casefold()
        norm_kw = norm_kw.casefold()
    pattern = re.compile(rf"(?<!\w){re.escape(norm_kw)}(?!\w)")
    return pattern.search(norm_text) is not None


def whole_word_match(
    prediction: str,
    keyword: str,
    *,
    ignore_case: bool = True,
) -> float:
    """Score 1.0 if `keyword` appears in `prediction` with whole-word boundaries."""
    return 1.0 if contains_whole_word(prediction, keyword, ignore_case=ignore_case) else 0.0


def _unwrap_boxed(expr: str) -> str:
    s = expr.strip()
    while s.startswith(r"\boxed{") and s.endswith("}"):
        depth = 0
        matched_end = -1
        for i, ch in enumerate(s[len(r"\boxed") :], start=len(r"\boxed")):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    matched_end = i
                    break
        if matched_end == len(s) - 1:
            s = s[len(r"\boxed{") : -1].strip()
        else:
            break
    return s


def _strip_math_delimiters(expr: str) -> str:
    s = expr.strip()
    while True:
        if len(s) >= 4 and s.startswith("$$") and s.endswith("$$"):
            s = s[2:-2].strip()
        elif len(s) >= 2 and s.startswith("$") and s.endswith("$"):
            s = s[1:-1].strip()
        elif (
            len(s) >= 4
            and (
                (s.startswith(r"\(") and s.endswith(r"\)"))
                or (s.startswith(r"\[") and s.endswith(r"\]"))
            )
        ):
            s = s[2:-2].strip()
        else:
            break
    return s


def normalize_latex(expr: str) -> str:
    """Canonicalize LaTeX strings while preserving indexed roots (`\\sqrt[n]{x}`) and tuples."""
    s = unicodedata.normalize("NFC", expr.strip())
    s = _strip_math_delimiters(s)
    s = _unwrap_boxed(s)
    s = _strip_math_delimiters(s)
    s = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", s)
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\\frac\s*([0-9A-Za-z])\s*([0-9A-Za-z])", r"\\frac{\1}{\2}", s)
    # Preserve indexed roots \sqrt[n]{x}; never drop [n]
    s = re.sub(r"\\sqrt\s*\[\s*([^\]]+?)\s*\]\s*\{", r"\\sqrt[\1]{", s)
    s = re.sub(r"\\sqrt\s*\[\s*([^\]]+?)\s*\]\s*([0-9A-Za-z])", r"\\sqrt[\1]{\2}", s)
    s = re.sub(r"\\sqrt\s*\{", r"\\sqrt{", s)
    s = re.sub(r"\\sqrt\s+([0-9A-Za-z])", r"\\sqrt{\1}", s)
    # Remove digit-grouping commas (1,000 -> 1000) before normalizing tuple spacing "(1, 2)"
    s = re.sub(r"(?<=\d),(?=\d{3}(?:,\d{3})*(?!\d))", "", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    return s


def latex_exact_match(prediction: str, reference: str) -> float:
    """Compare LaTeX math expressions after indexed-root and tuple-safe normalization."""
    return 1.0 if normalize_latex(prediction) == normalize_latex(reference) else 0.0


def register_builtin_metrics() -> None:
    """Register all built-in metric functions with the global PluginRegistry."""
    registry = get_registry("eval_studio.metrics")
    builtins: dict[str, Any] = {
        "acc": acc,
        "acc_norm": acc_norm,
        "acc_bytes": acc_bytes,
        "acc_mutual_info": acc_mutual_info,
        "acc_all": acc_all,
        "brier_score": brier_score,
        "bits_per_byte": bits_per_byte,
        "word_perplexity": word_perplexity,
        "byte_perplexity": byte_perplexity,
        "weighted_perplexity": weighted_perplexity,
        "bypass": bypass,
        "exact_match": exact_match,
        "numeric_exact_match": numeric_exact_match,
        "whole_word_match": whole_word_match,
        "latex_exact_match": latex_exact_match,
        "classification_f1": classification_f1,
        "f1": classification_f1,
        "token_f1": token_f1,
        "squad_f1": token_f1,
    }
    for metric_name, metric_fn in builtins.items():
        registry.register(metric_name, metric_fn, builtin=True)


register_builtin_metrics()
