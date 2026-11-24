"""Minimal Tier 3 custom Solver/Scorer plugin stub.

Requires explicit `--trust-remote-code` and Docker sandbox isolation for untrusted code execution.
"""

from __future__ import annotations

from eval_studio.core.schema import EvalRequest, EvalResponse, OutputType, Sample


class CustomClinicalSolver:
    """Example Tier 3 solver stub translating a Sample into an EvalRequest."""

    name: str = "custom_clinical_solver"

    def build_request(self, sample: Sample) -> EvalRequest:
        return EvalRequest(
            request_id=f"req-{sample.sample_id}",
            sample_id=sample.sample_id,
            task_name="tier3_custom_clinical",
            output_type=OutputType.GENERATE_UNTIL,
            context=f"Clinical Case: {sample.input_text}\nDiagnosis:",
        )


class CustomClinicalScorer:
    """Example Tier 3 scorer stub grading a response against a target."""

    name: str = "custom_clinical_scorer"

    def score(self, sample: Sample, response: EvalResponse) -> dict[str, float]:
        pred = (response.text or "").strip().casefold()
        ref = str(sample.target).strip().casefold()
        return {"exact_match": 1.0 if pred == ref else 0.0}
