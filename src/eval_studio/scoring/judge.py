"""LLM-as-a-judge rubric schema and decoupled scorer with fallback_judge failover."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from eval_studio.core.schema import EvalRequest, OutputType
from eval_studio.gateway.base import InferenceGateway
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import redact_secrets

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class JudgeRubric(BaseModel):
    """Structured rubric passed to a role-separated JUDGE_API_KEY model."""

    name: str
    criteria: str
    scale_min: float = 0.0
    scale_max: float = 1.0
    prompt_version: str = "0.1.0"


class JudgeVerdict(BaseModel):
    """Structured verdict returned by an LLM judge."""

    score: float
    rationale: str
    judge_model_id: str
    rubric_name: str
    metadata: dict[str, str] = Field(default_factory=dict)


def _parse_judge_json(raw_text: str, rubric: JudgeRubric) -> tuple[float, str]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty judge response.")
    match = _FENCE_RE.search(text)
    candidate = match.group(1) if match else text
    data = json.loads(candidate)
    if not isinstance(data, dict) or "score" not in data:
        raise ValueError("Judge JSON missing required 'score' field.")
    score = float(data["score"])
    if score < rubric.scale_min or score > rubric.scale_max:
        raise ValueError(
            f"Judge score {score} outside rubric range [{rubric.scale_min}, {rubric.scale_max}]."
        )
    rationale = str(data.get("rationale", "")).strip() or "No rationale provided."
    return score, rationale


class LLMJudge:
    """Decoupled post-hoc LLM-as-a-judge runner with automatic fallback_judge failover."""

    def __init__(
        self,
        judge_model_id: str,
        rubric: JudgeRubric,
        *,
        gateway: InferenceGateway | None = None,
        fallback_judge: InferenceGateway | None = None,
    ) -> None:
        self.judge_model_id = judge_model_id
        self.rubric = rubric
        self.gateway: InferenceGateway = (
            gateway if gateway is not None else MockGateway(model_id=judge_model_id)
        )
        self.fallback_judge = fallback_judge

    def _build_prompt(self, prompt: str, response: str, reference: str | None) -> str:
        ref_block = f"\nReference Answer: {reference}" if reference is not None else ""
        return (
            f"Rubric ({self.rubric.name}): {self.rubric.criteria}\n"
            f"Score range: [{self.rubric.scale_min}, {self.rubric.scale_max}]\n"
            f"Prompt: {prompt}\n"
            f"Model Response: {response}{ref_block}\n"
            'Respond strictly with JSON: {"score": <float>, "rationale": "<brief explanation>"}'
        )

    def _call_gateway(
        self,
        gw: InferenceGateway,
        eval_req: EvalRequest,
        *,
        used_fallback: bool,
    ) -> JudgeVerdict:
        resp = gw.generate_until([eval_req])[0]
        score, rationale = _parse_judge_json(resp.text or "", self.rubric)
        return JudgeVerdict(
            score=score,
            rationale=redact_secrets(rationale),
            judge_model_id=gw.model_id,
            rubric_name=self.rubric.name,
            metadata={"used_fallback": "true" if used_fallback else "false"},
        )

    def grade(self, prompt: str, response: str, reference: str | None = None) -> JudgeVerdict:
        eval_req = EvalRequest(
            request_id=f"judge-{self.rubric.name}",
            sample_id="judge-sample",
            task_name=self.rubric.name,
            output_type=OutputType.GENERATE_UNTIL,
            context=self._build_prompt(prompt, response, reference),
        )
        try:
            return self._call_gateway(self.gateway, eval_req, used_fallback=False)
        except Exception as primary_exc:
            if self.fallback_judge is None:
                raise RuntimeError(
                    redact_secrets(f"Primary judge failed without fallback: {primary_exc}")
                ) from None
            try:
                return self._call_gateway(self.fallback_judge, eval_req, used_fallback=True)
            except Exception as fb_exc:
                raise RuntimeError(
                    redact_secrets(
                        f"Both primary judge ({primary_exc}) and fallback judge ({fb_exc}) failed."
                    )
                ) from None
