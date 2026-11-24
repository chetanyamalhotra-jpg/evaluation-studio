"""Tokenization-agnostic inference gateway protocol and encoding helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from eval_studio.core.schema import EvalRequest, EvalResponse


def transfer_trailing_whitespace(context: str, continuation: str) -> tuple[str, str]:
    """Shift trailing spaces from context onto continuation before tokenization.

    Prevents BPE/SentencePiece boundary splits (e.g., ' Paris' vs 'Paris').
    """
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    return context, continuation


@runtime_checkable
class InferenceGateway(Protocol):
    """Tokenization-agnostic inference backend contract."""

    backend_name: str
    model_id: str

    def loglikelihood(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        """Score conditional log-likelihood of (context, continuation) pairs."""
        ...

    def loglikelihood_rolling(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        """Score rolling full-sequence log-likelihood for perplexity / BPB."""
        ...

    def generate_until(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        """Generate continuations until stop sequences or max_gen_toks."""
        ...

    def chat(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        """Generate chat completions from structured ChatMessages or context."""
        ...
