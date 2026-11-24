"""Deterministic offline mock gateway for CPU CI testing, pricing, judge, and compiler tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from eval_studio.core.schema import EvalRequest, EvalResponse, TokenUsage
from eval_studio.gateway.cost import CostGuard


class MockGateway:
    """Zero-network, deterministic inference gateway supporting pricing, judge, and BYOP modes."""

    backend_name: str = "mock"

    def __init__(
        self,
        model_id: str = "mock-deterministic",
        default_reply: str = "A",
        *,
        token_pricing: dict[str, float] | None = None,
        judge_mode: str = "normal",
        compiler_mode: str | None = None,
        cost_guard: CostGuard | None = None,
        thinking_prefix: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.default_reply = default_reply
        self.token_pricing = dict(token_pricing) if token_pricing is not None else {}
        self.judge_mode = judge_mode
        self.compiler_mode = compiler_mode
        self.cost_guard = cost_guard
        self.thinking_prefix = thinking_prefix
        self._call_count = 0

    def _compute_cost(self, usage: TokenUsage) -> float:
        in_rate = float(self.token_pricing.get("input_usd_per_1m", 0.0))
        out_rate = float(self.token_pricing.get("output_usd_per_1m", 0.0))
        cost = (usage.prompt_tokens / 1_000_000.0) * in_rate + (
            usage.completion_tokens / 1_000_000.0
        ) * out_rate
        if self.cost_guard is not None and cost > 0.0:
            self.cost_guard.check_and_record(cost)
        return cost

    def _compiler_payload(self, req: EvalRequest) -> str:
        valid_item = {
            "item_id": f"{req.sample_id}-cited-1",
            "question": (
                "What negative predictive value was achieved by the 0/1-hour high-sensitivity "
                "troponin protocol in the emergency department cohort?"
            ),
            "reference_answer": "99.6%",
            "rubric": "Must state 99.6% negative predictive value for ruling out acute MI.",
            "category": "quantitative_finding",
            "citations": [
                {
                    "page": 1,
                    "section": "Section 2: Cohort Findings",
                    "quote": "achieved a negative predictive value of 99.6%",
                }
            ],
        }
        valid_item_2 = {
            "item_id": f"{req.sample_id}-cited-2",
            "question": (
                "What serum troponin I threshold indicates acute myocardial injury in the study?"
            ),
            "reference_answer": "> 0.04 ng/mL",
            "rubric": "Must identify > 0.04 ng/mL serum troponin I.",
            "category": "claim",
            "citations": [
                {
                    "page": 1,
                    "section": "Section 1: Background",
                    "quote": "Elevated serum troponin I (> 0.04 ng/mL)",
                }
            ],
        }
        uncited_item: dict[str, Any] = {
            "item_id": f"{req.sample_id}-uncited-1",
            "question": "How does this compare to Table 99 from an external unpublished study?",
            "reference_answer": "Unknown",
            "rubric": "Unverifiable external reference.",
            "category": "limitation",
            "citations": [],
        }
        leaky_item = {
            "item_id": f"{req.sample_id}-leaky-1",
            "question": "Why did the cohort achieve a negative predictive value of 99.6%?",
            "reference_answer": "99.6%",
            "rubric": "Leaks reference answer verbatim.",
            "category": "claim",
            "citations": [
                {
                    "page": 1,
                    "section": "Section 2: Cohort Findings",
                    "quote": "achieved a negative predictive value of 99.6%",
                }
            ],
        }

        if self.compiler_mode == "valid_only":
            items = [valid_item, valid_item_2]
        elif self.compiler_mode == "all_invalid":
            items = [uncited_item, leaky_item]
        else:
            items = [valid_item, valid_item_2, uncited_item, leaky_item]
        return json.dumps({"items": items}, ensure_ascii=False)

    def _generate_text(self, req: EvalRequest) -> str:
        self._call_count += 1
        if self.compiler_mode is not None:
            return self._compiler_payload(req)
        if self.judge_mode == "refusal":
            return "I'm sorry, I cannot evaluate this clinical response."
        if self.judge_mode == "malformed_json":
            return "{malformed_json: not valid"
        if self.judge_mode == "fail_then_succeed":
            if self._call_count % 2 == 1:
                return "I'm sorry, I cannot provide a score right now."
            return json.dumps(
                {"score": 1.0, "rationale": "Matches clinical reference answer."},
                ensure_ascii=False,
            )
        if self.judge_mode == "json_score":
            return json.dumps(
                {"score": 1.0, "rationale": "Deterministic aligned judge verdict."},
                ensure_ascii=False,
            )
        reply = self.default_reply
        if self.thinking_prefix:
            return f"{self.thinking_prefix}{reply}"
        return reply

    def loglikelihood(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        responses: list[EvalResponse] = []
        for req in requests:
            payload = f"{req.context}|{req.continuation or ''}".encode()
            digest = int(hashlib.sha256(payload).hexdigest()[:8], 16)
            ll = -float((digest % 1000) / 100.0)
            usage = TokenUsage(prompt_tokens=max(1, len(req.context.split())))
            cost_usd = self._compute_cost(usage)
            responses.append(
                EvalResponse(
                    request_id=req.request_id,
                    sample_id=req.sample_id,
                    loglikelihood=ll,
                    is_greedy=(digest % 2 == 0),
                    backend=self.backend_name,
                    model_id=self.model_id,
                    usage=usage,
                    metadata={"cost_usd": cost_usd},
                )
            )
        return responses

    def loglikelihood_rolling(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        return self.loglikelihood(requests)

    def generate_until(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        responses: list[EvalResponse] = []
        for req in requests:
            text = self._generate_text(req)
            prompt_text = req.context
            if not prompt_text and req.messages:
                prompt_text = " ".join(m.text_content() for m in req.messages)
            usage = TokenUsage(
                prompt_tokens=max(1, len(prompt_text.split())),
                completion_tokens=max(1, len(text.split())),
            )
            cost_usd = self._compute_cost(usage)
            responses.append(
                EvalResponse(
                    request_id=req.request_id,
                    sample_id=req.sample_id,
                    text=text,
                    backend=self.backend_name,
                    model_id=self.model_id,
                    usage=usage,
                    metadata={"cost_usd": cost_usd},
                )
            )
        return responses

    def chat(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        return self.generate_until(requests)
