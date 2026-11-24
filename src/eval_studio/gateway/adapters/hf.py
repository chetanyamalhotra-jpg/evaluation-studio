"""Optional Hugging Face transformers adapter (requires `evaluation-studio[hf]`)."""

from __future__ import annotations

from collections.abc import Sequence

from eval_studio.core.schema import EvalRequest, EvalResponse


class HuggingFaceAdapter:
    """Lazy wrapper for transformers causal/seq2seq models (zero heavy base imports)."""

    backend_name: str = "hf"

    def __init__(self, model_id: str, *, add_bos_token: bool | None = None) -> None:
        self.model_id = model_id
        self.add_bos_token = add_bos_token

    def _require_extra(self) -> None:
        import importlib.util

        if importlib.util.find_spec("torch") is None or (
            importlib.util.find_spec("transformers") is None
        ):
            raise ImportError(
                "HuggingFaceAdapter requires optional dependencies. "
                "Install with: pip install 'evaluation-studio[hf]' (evaluation-studio[hf])."
            )

    def loglikelihood(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        self._require_extra()
        raise NotImplementedError(
            "HF GPU loglikelihood requires an active model runtime (evaluation-studio[hf])."
        )

    def loglikelihood_rolling(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        self._require_extra()
        raise NotImplementedError(
            "HF GPU rolling LL requires an active model runtime (evaluation-studio[hf])."
        )

    def generate_until(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        self._require_extra()
        raise NotImplementedError(
            "HF GPU generate_until requires an active model runtime (evaluation-studio[hf])."
        )

    def chat(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        return self.generate_until(requests)
