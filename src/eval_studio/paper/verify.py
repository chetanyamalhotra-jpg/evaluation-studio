"""Stage 2 BYOP independent answerability verification gate."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from eval_studio.core.schema import EvalRequest, OutputType
from eval_studio.gateway.base import InferenceGateway
from eval_studio.paper.schema import ExtractedItem, VerificationStatus

_DANGLING_REF_RE = re.compile(
    r"\b(?:as shown in (?:fig(?:ure)?|table)|see (?:fig(?:ure)?|table)\s*\d+|"
    r"table\s+\d+\s+above|figure\s+\d+\s+above|above table|below figure|"
    r"external unpublished|unpublished study|supplementary figure)\b",
    re.IGNORECASE,
)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class VerificationOutcome:
    """Result of evaluating a single extracted item through the Stage 2 gate."""

    item_id: str
    status: VerificationStatus
    reasons: list[str]


def _parse_verifier_json(raw_text: str) -> tuple[bool, list[str]]:
    text = raw_text.strip()
    match = _JSON_FENCE_RE.search(text)
    candidate = match.group(1) if match else text
    data = json.loads(candidate)
    if not isinstance(data, dict) or "approved" not in data:
        raise ValueError("Verifier response JSON missing 'approved' field.")
    approved = bool(data["approved"])
    raw_reasons = data.get("reasons") or []
    reasons = [str(r) for r in raw_reasons] if isinstance(raw_reasons, list) else []
    return approved, reasons


class AnswerabilityVerifier:
    """Checks self-containedness, verbatim citation grounding, and answer non-leakage."""

    def __init__(
        self,
        verifier_model: str = "mock-verifier",
        *,
        gateway: InferenceGateway | None = None,
    ) -> None:
        self.verifier_model = verifier_model
        self.gateway = gateway

    def verify_item(
        self,
        item: ExtractedItem,
        *,
        source_text: str | None = None,
    ) -> VerificationOutcome:
        """Run structural, citation-grounding, and optional LLM verifier checks."""
        reasons: list[str] = []

        if not item.citations:
            reasons.append("Missing mandatory page/section citation.")
        else:
            for cite in item.citations:
                if cite.page < 1 or not cite.section.strip() or not cite.quote.strip():
                    reasons.append("Incomplete citation metadata (page/section/quote required).")
                elif source_text is not None and cite.quote not in source_text:
                    reasons.append(
                        f"Citation quote not found verbatim in source paper: '{cite.quote[:40]}'"
                    )

        ref = item.reference_answer.strip()
        ref_core = ref.rstrip(".!?,;:")
        if ref and (
            ref.casefold() in item.question.casefold()
            or (len(ref_core) >= 3 and ref_core.casefold() in item.question.casefold())
        ):
            reasons.append("Prompt leaks reference answer verbatim.")

        if _DANGLING_REF_RE.search(item.question):
            reasons.append(
                "Question contains dangling or externally dependent figure/table reference."
            )

        if not item.question.strip():
            reasons.append("Question text is empty.")

        if not reasons and self.gateway is not None:
            quotes_text = "\n".join(
                f"- p.{c.page} ({c.section}): {c.quote}" for c in item.citations
            )
            prompt = (
                "Verify whether this extracted benchmark question is self-contained, "
                "unambiguous, non-leaky, and answerable strictly from the cited evidence.\n"
                f"Question: {item.question}\n"
                f"Reference Answer: {item.reference_answer}\n"
                f"Citations:\n{quotes_text}\n"
                'Respond strictly with JSON: {"approved": true/false, "reasons": ["..."]}'
            )
            req = EvalRequest(
                request_id=f"verify-{item.item_id}",
                sample_id=item.item_id,
                task_name="byop_verify",
                output_type=OutputType.GENERATE_UNTIL,
                context=prompt,
            )
            try:
                resp = self.gateway.generate_until([req])[0]
                approved, gw_reasons = _parse_verifier_json(resp.text or "")
                if not approved:
                    reasons.extend(
                        gw_reasons or ["Rejected by Stage 2 answerability verifier gateway."]
                    )
            except Exception as exc:
                reasons.append(f"Stage 2 verifier gateway failed closed: {exc}")

        status = VerificationStatus.APPROVED if not reasons else VerificationStatus.REJECTED
        return VerificationOutcome(item_id=item.item_id, status=status, reasons=reasons)
