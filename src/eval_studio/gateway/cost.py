"""Pre-flight USD cost estimator and --max-cost-usd circuit breaker."""

from __future__ import annotations

from dataclasses import dataclass


class CostBudgetExceededError(RuntimeError):
    """Raised when estimated or actual API spend exceeds --max-cost-usd."""


@dataclass(frozen=True)
class CostEstimate:
    """Estimated token and USD cost for a batch or full evaluation run."""

    prompt_tokens: int
    completion_tokens: int
    estimated_usd: float


@dataclass(frozen=True)
class PreFlightCostEstimate:
    """Pre-flight cost breakdown across target passes and optional LLM judge grading."""

    n_samples: int
    repeats: int
    target_prompt_tokens: int
    target_completion_tokens: int
    target_usd: float
    judge_enabled: bool
    judge_prompt_tokens: int
    judge_completion_tokens: int
    judge_usd: float
    total_usd: float


class CostGuard:
    """Enforces a hard USD ceiling across an evaluation run."""

    def __init__(self, max_cost_usd: float | None = None) -> None:
        if max_cost_usd is not None and max_cost_usd < 0.0:
            raise ValueError(f"--max-cost-usd must be >= 0.0, got {max_cost_usd}")
        self.max_cost_usd = max_cost_usd
        self.spent_usd: float = 0.0

    def estimate_batch(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        input_usd_per_1m: float = 0.0,
        output_usd_per_1m: float = 0.0,
    ) -> CostEstimate:
        """Estimate USD spend for a token budget."""
        estimated = (prompt_tokens / 1_000_000.0) * input_usd_per_1m + (
            completion_tokens / 1_000_000.0
        ) * output_usd_per_1m
        return CostEstimate(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_usd=estimated,
        )

    def estimate_run(
        self,
        *,
        n_samples: int,
        avg_prompt_tokens: int,
        max_gen_toks: int,
        repeats: int = 1,
        input_usd_per_1m: float = 0.0,
        output_usd_per_1m: float = 0.0,
        judge_enabled: bool = False,
        judge_input_usd_per_1m: float = 0.0,
        judge_output_usd_per_1m: float = 0.0,
        judge_completion_tokens: int = 128,
    ) -> PreFlightCostEstimate:
        """Compute full pre-flight target*repeats + optional judge USD estimate."""
        total_target_calls = max(0, n_samples) * max(1, repeats)
        target_in = total_target_calls * max(1, avg_prompt_tokens)
        target_out = total_target_calls * max(1, max_gen_toks)
        target_usd = self.estimate_batch(
            target_in,
            target_out,
            input_usd_per_1m=input_usd_per_1m,
            output_usd_per_1m=output_usd_per_1m,
        ).estimated_usd

        judge_in = 0
        judge_out = 0
        judge_usd = 0.0
        if judge_enabled:
            judge_in = total_target_calls * (max(1, avg_prompt_tokens) + max(1, max_gen_toks))
            judge_out = total_target_calls * max(1, judge_completion_tokens)
            judge_usd = self.estimate_batch(
                judge_in,
                judge_out,
                input_usd_per_1m=judge_input_usd_per_1m,
                output_usd_per_1m=judge_output_usd_per_1m,
            ).estimated_usd

        return PreFlightCostEstimate(
            n_samples=n_samples,
            repeats=repeats,
            target_prompt_tokens=target_in,
            target_completion_tokens=target_out,
            target_usd=target_usd,
            judge_enabled=judge_enabled,
            judge_prompt_tokens=judge_in,
            judge_completion_tokens=judge_out,
            judge_usd=judge_usd,
            total_usd=target_usd + judge_usd,
        )

    def check_and_record(self, additional_usd: float) -> None:
        """Raise CostBudgetExceededError if spending additional_usd breaches the cap."""
        if additional_usd < 0.0:
            raise ValueError(f"additional_usd must be >= 0.0, got {additional_usd}")
        projected = self.spent_usd + additional_usd
        if self.max_cost_usd is not None and projected > self.max_cost_usd:
            raise CostBudgetExceededError(
                f"Projected spend ${projected:.6f} exceeds --max-cost-usd=${self.max_cost_usd:.6f}"
            )
        self.spent_usd = projected
