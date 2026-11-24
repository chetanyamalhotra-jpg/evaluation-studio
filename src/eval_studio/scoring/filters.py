"""FilterEnsemble post-processing protocols, built-in filters, and OLMES formatting helpers."""

from __future__ import annotations

import inspect
import math
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from eval_studio.core.registry import get_registry
from eval_studio.core.schema import (
    EvalResponse,
    FilterConfig,
    FilterEnsembleSpec,
    MultipleChoiceFormat,
)


@runtime_checkable
class Filter(Protocol):
    """Stateful or stateless response post-processing step."""

    name: str

    def apply(self, responses: Sequence[str]) -> list[str]:
        """Transform raw or intermediate response strings."""
        ...


class MultiChoiceRegexFilter:
    """Regex extractor that sorts choice alternatives longest-first to avoid prefix shadowing."""

    name: str = "multi_choice_regex"

    def __init__(
        self,
        choices: Sequence[str],
        *,
        fallback: str = "[invalid]",
        ignore_case: bool = False,
        word_boundary: bool = False,
    ) -> None:
        self.choices = [c for c in choices if c != ""]
        self.fallback = fallback
        self.ignore_case = ignore_case
        self.word_boundary = word_boundary
        sorted_choices = sorted(dict.fromkeys(self.choices), key=len, reverse=True)
        if word_boundary:
            parts = [rf"(?<!\w){re.escape(c)}(?!\w)" for c in sorted_choices]
        else:
            parts = [re.escape(c) for c in sorted_choices]
        escaped = "|".join(parts)
        flags = re.IGNORECASE if ignore_case else 0
        self._pattern = re.compile(escaped, flags=flags) if escaped else None

    def apply(self, responses: Sequence[str]) -> list[str]:
        if self._pattern is None:
            return [self.fallback for _ in responses]
        extracted: list[str] = []
        for text in responses:
            match = self._pattern.search(text)
            extracted.append(match.group(0) if match else self.fallback)
        return extracted


class RegexFilter:
    """General regex extractor returning the requested capture group or fallback."""

    name: str = "regex"

    def __init__(
        self,
        pattern: str,
        *,
        group: int = 0,
        fallback: str = "[invalid]",
        flags: int = 0,
    ) -> None:
        if not pattern:
            raise ValueError("RegexFilter pattern must be non-empty.")
        self.pattern = pattern
        self.group = group
        self.fallback = fallback
        self.flags = flags
        self._compiled = re.compile(pattern, flags=flags)

    def apply(self, responses: Sequence[str]) -> list[str]:
        extracted: list[str] = []
        for text in responses:
            match = self._compiled.search(text)
            if match is None:
                extracted.append(self.fallback)
                continue
            captured = match.group(self.group)
            extracted.append(captured if captured is not None else self.fallback)
        return extracted


class TakeFirstFilter:
    """Select only the first response from a repeated generation list."""

    name: str = "take_first"

    def __init__(self, *, fallback: str | None = None) -> None:
        self.fallback = fallback

    def apply(self, responses: Sequence[str]) -> list[str]:
        if responses:
            return [responses[0]]
        return [self.fallback] if self.fallback is not None else []


class TakeFirstKFilter:
    """Select up to the first `k` responses from a repeated generation list."""

    name: str = "take_first_k"

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError("k must be >= 1.")
        self.k = k

    def apply(self, responses: Sequence[str]) -> list[str]:
        return list(responses[: self.k])


class MajorityVoteFilter:
    """Aggregate repeated responses by deterministic majority vote (first-seen tie-break)."""

    name: str = "majority_vote"

    def __init__(
        self,
        *,
        ignore_values: Sequence[str] = ("[invalid]", ""),
        fallback: str = "[invalid]",
        normalize: bool = False,
    ) -> None:
        self.ignore_values = tuple(ignore_values)
        self.fallback = fallback
        self.normalize = normalize

    def _canonicalize(self, value: str) -> str:
        return unicodedata.normalize("NFC", value.strip()).casefold()

    def apply(self, responses: Sequence[str]) -> list[str]:
        ignored = (
            frozenset(self._canonicalize(v) for v in self.ignore_values)
            if self.normalize
            else frozenset(self.ignore_values)
        )
        counts: Counter[str] = Counter()
        first_surface: dict[str, str] = {}
        for resp in responses:
            key = self._canonicalize(resp) if self.normalize else resp
            if key in ignored:
                continue
            if key not in first_surface:
                first_surface[key] = resp
            counts[key] += 1
        if not counts:
            return [self.fallback]
        winner_key = counts.most_common(1)[0][0]
        return [first_surface[winner_key]]


def strip_thinking_trace(
    text: str | None,
    think_end_token: str,
    *,
    empty_fallback: str = "",
    strip_whitespace: bool = True,
    allow_unclosed: bool = False,
) -> tuple[str, str | None, bool]:
    """Strip reasoning trace up to and including the last `think_end_token`.

    Returns `(cleaned_answer, reasoning_trace, thinking_truncated)`.
    """
    if not think_end_token:
        raise ValueError("think_end_token must be non-empty.")
    if text is None:
        return (empty_fallback, None, False)
    idx = text.rfind(think_end_token)
    if idx != -1:
        trace = text[:idx]
        answer = text[idx + len(think_end_token) :]
        return (answer.strip() if strip_whitespace else answer, trace, False)
    if allow_unclosed:
        return (empty_fallback, text, True)
    return (text.strip() if strip_whitespace else text, None, False)


class ThinkStripFilter:
    """Filter that strips reasoning traces up to the final `think_end_token`."""

    name: str = "think_strip"

    def __init__(
        self,
        think_end_token: str = "</think>",
        *,
        strip_whitespace: bool = True,
        empty_fallback: str = "",
        allow_unclosed: bool = False,
    ) -> None:
        if not think_end_token:
            raise ValueError("think_end_token must be non-empty.")
        self.think_end_token = think_end_token
        self.strip_whitespace = strip_whitespace
        self.empty_fallback = empty_fallback
        self.allow_unclosed = allow_unclosed

    def apply(self, responses: Sequence[str]) -> list[str]:
        cleaned: list[str] = []
        for resp in responses:
            answer, _, _ = strip_thinking_trace(
                resp,
                self.think_end_token,
                empty_fallback=self.empty_fallback,
                strip_whitespace=self.strip_whitespace,
                allow_unclosed=self.allow_unclosed,
            )
            cleaned.append(answer)
        return cleaned


def build_filter(config: FilterConfig, *, choices: Sequence[str] | None = None) -> Filter:
    """Instantiate a registered filter from a `FilterConfig`, injecting per-sample choices."""
    registry = get_registry("eval_studio.filters")
    factory: Any = registry.get(config.name)
    kwargs: dict[str, Any] = dict(config.params)
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        sig = None
    if (
        sig is not None
        and choices is not None
        and "choices" not in kwargs
        and "choices" in sig.parameters
    ):
        kwargs["choices"] = list(choices)
    flt: Filter = factory(**kwargs)
    return flt


class FilterEnsemble:
    """Executes an ordered sequence of Filter steps over shared raw responses."""

    def __init__(self, name: str, filters: Sequence[Filter]) -> None:
        self.name = name
        self.filters = list(filters)

    @classmethod
    def from_spec(
        cls,
        spec: FilterEnsembleSpec,
        *,
        choices: Sequence[str] | None = None,
    ) -> FilterEnsemble:
        """Build a FilterEnsemble from a declarative FilterEnsembleSpec."""
        built = [build_filter(step, choices=choices) for step in spec.steps]
        return cls(name=spec.name, filters=built)

    def apply(self, responses: Sequence[str]) -> list[str]:
        current = list(responses)
        for step in self.filters:
            current = step.apply(current)
        return current


def apply_filter_ensembles(
    raw_responses: Sequence[str | EvalResponse],
    ensembles: Sequence[FilterEnsemble | FilterEnsembleSpec],
    *,
    choices: Sequence[str] | None = None,
    default_name: str = "none",
) -> dict[str, list[str]]:
    """Run multiple independent filter pipelines over shared raw responses."""
    raw_texts: list[str] = [
        r if isinstance(r, str) else (r.text if r.text is not None else "")
        for r in raw_responses
    ]
    if not ensembles:
        return {default_name: list(raw_texts)}

    outputs: dict[str, list[str]] = {}
    for entry in ensembles:
        ensemble = (
            entry
            if isinstance(entry, FilterEnsemble)
            else FilterEnsemble.from_spec(entry, choices=choices)
        )
        if ensemble.name in outputs:
            raise ValueError(f"Duplicate FilterEnsemble name '{ensemble.name}'.")
        outputs[ensemble.name] = ensemble.apply(list(raw_texts))
    return outputs


# --- OLMES Cloze + Multiple Choice Format (MCF) Dual-Format Helpers ---


@dataclass(frozen=True)
class OLMESPromptBundle:
    """Rendered OLMES prompt context and candidate continuations."""

    format: MultipleChoiceFormat
    context: str
    continuations: list[str]
    labels: list[str]


@dataclass(frozen=True)
class OLMESDualResult:
    """Outcome of dual OLMES Cloze vs. MCF evaluation."""

    cloze_score: float
    mcf_score: float
    selected_format: MultipleChoiceFormat
    best_score: float
    delta_mcf_minus_cloze: float


def olmes_mcf_labels(
    n_choices: int,
    *,
    labels: Sequence[str] | None = None,
) -> list[str]:
    """Return canonical OLMES option labels (`['A', 'B', ...]`) for `n_choices`."""
    if n_choices <= 0:
        raise ValueError("choices must be non-empty.")
    if labels is None:
        if n_choices > len(string.ascii_uppercase):
            raise ValueError(
                f"Default OLMES labels support up to {len(string.ascii_uppercase)} choices."
            )
        return list(string.ascii_uppercase[:n_choices])
    used = list(labels)
    if len(used) < n_choices:
        raise ValueError("Provided option labels fewer than number of choices.")
    return used[:n_choices]


def format_olmes_mcf_prompt(
    question: str,
    choices: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    answer_prefix: str = "Answer:",
    separator: str = "\n",
) -> OLMESPromptBundle:
    """Format an OLMES Multiple-Choice Format (MCF) prompt with `'\\n A'` leading space."""
    used_labels = olmes_mcf_labels(len(choices), labels=labels)
    option_lines = [
        f" {label}. {choice}"
        for label, choice in zip(used_labels, choices, strict=True)
    ]
    options_block = separator.join(option_lines)
    parts = [question.rstrip(), options_block]
    if answer_prefix:
        parts.append(answer_prefix)
    context = separator.join(parts)
    continuations = [f" {label}" for label in used_labels]
    return OLMESPromptBundle(
        format=MultipleChoiceFormat.MCF,
        context=context,
        continuations=continuations,
        labels=used_labels,
    )


def format_olmes_cloze_prompt(
    question: str,
    choices: Sequence[str],
    *,
    answer_prefix: str = "Answer:",
) -> OLMESPromptBundle:
    """Format an OLMES Cloze prompt where candidate continuations carry a leading space."""
    if not choices:
        raise ValueError("choices must be non-empty.")
    context = f"{question.rstrip()}\n{answer_prefix}" if answer_prefix else question.rstrip()
    continuations = [f" {choice}" for choice in choices]
    return OLMESPromptBundle(
        format=MultipleChoiceFormat.CLOZE,
        context=context,
        continuations=continuations,
        labels=[],
    )


def build_olmes_dual_prompts(
    question: str,
    choices: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    answer_prefix: str = "Answer:",
) -> dict[str, OLMESPromptBundle]:
    """Build both Cloze and MCF OLMES prompt bundles for dual evaluation."""
    return {
        "cloze": format_olmes_cloze_prompt(question, choices, answer_prefix=answer_prefix),
        "mcf": format_olmes_mcf_prompt(
            question,
            choices,
            labels=labels,
            answer_prefix=answer_prefix,
        ),
    }


def select_olmes_best_format(cloze_score: float, mcf_score: float) -> OLMESDualResult:
    """Select the higher-scoring OLMES format while preserving both format scores."""
    if math.isnan(cloze_score) or math.isnan(mcf_score):
        raise ValueError("OLMES format scores must not be NaN.")
    selected = (
        MultipleChoiceFormat.CLOZE if cloze_score >= mcf_score else MultipleChoiceFormat.MCF
    )
    return OLMESDualResult(
        cloze_score=float(cloze_score),
        mcf_score=float(mcf_score),
        selected_format=selected,
        best_score=float(max(cloze_score, mcf_score)),
        delta_mcf_minus_cloze=float(mcf_score - cloze_score),
    )


OLMES_TASK_METRICS: dict[str, str] = {
    "arc_challenge": "acc_mutual_info",
    "commonsense_qa": "acc_mutual_info",
    "openbookqa": "acc_mutual_info",
    "arc_easy": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "sciq": "acc_norm",
    "mmlu": "acc_norm",
    "boolq": "acc",
    "winogrande": "acc",
}


def get_olmes_recommended_metric(task_name: str, *, default: str = "acc_bytes") -> str:
    """Return the OLMES recommended normalization metric for a benchmark family."""
    normalized = task_name.strip().casefold().replace("-", "_")
    if normalized in OLMES_TASK_METRICS:
        return OLMES_TASK_METRICS[normalized]
    for prefix, metric_name in OLMES_TASK_METRICS.items():
        if normalized.startswith(f"{prefix}_"):
            return metric_name
    return default


def register_builtin_filters() -> None:
    """Register all built-in filter plugins with the global PluginRegistry."""
    registry = get_registry("eval_studio.filters")
    builtins: dict[str, Any] = {
        "multi_choice_regex": MultiChoiceRegexFilter,
        "regex": RegexFilter,
        "take_first": TakeFirstFilter,
        "take_first_k": TakeFirstKFilter,
        "majority_vote": MajorityVoteFilter,
        "think_strip": ThinkStripFilter,
    }
    for filter_name, filter_cls in builtins.items():
        registry.register(filter_name, filter_cls, builtin=True)


register_builtin_filters()
