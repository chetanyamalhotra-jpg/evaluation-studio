"""Optional vLLM adapter (requires `evaluation-studio[vllm]`)."""

from __future__ import annotations

from collections.abc import Sequence

from eval_studio.core.schema import EvalRequest, EvalResponse


class VLLMAdapter:
    """Lazy wrapper for vLLM high-throughput inference (zero heavy base imports)."""

    backend_name: str = "vllm"

    def __init__(self, model_id: str, *, tensor_parallel_size: int = 1) -> None:
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size

    def _require_extra(self) -> None:
        import importlib.util

        if importlib.util.find_spec("vllm") is None:
            raise ImportError(
                "VLLMAdapter requires optional dependencies. "
                "Install with: pip install 'evaluation-studio[vllm]' (evaluation-studio[vllm])."
            )

    def loglikelihood(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        self._require_extra()
        raise NotImplementedError(
            "vLLM loglikelihood requires GPU runtime (evaluation-studio[vllm])."
        )

    def loglikelihood_rolling(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        self._require_extra()
        raise NotImplementedError("vLLM rolling LL requires GPU runtime (evaluation-studio[vllm]).")

    def generate_until(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        self._require_extra()
        raise NotImplementedError(
            "vLLM generate_until requires GPU runtime (evaluation-studio[vllm])."
        )

    def chat(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        return self.generate_until(requests)
