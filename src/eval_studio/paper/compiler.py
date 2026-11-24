"""Orchestrator for the two-stage `eval-studio compile-paper` pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval_studio.gateway.base import InferenceGateway
from eval_studio.paper.extract import read_paper_text
from eval_studio.paper.schema import CompiledPaperSuite, ExtractedItem, VerificationStatus
from eval_studio.paper.verify import AnswerabilityVerifier


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compile_paper(
    source_path: Path,
    *,
    items: list[ExtractedItem],
    suite_name: str,
    extractor_model: str = "mock-extractor",
    verifier_model: str = "mock-verifier",
    extraction_mode: str = "mode_a_comprehension",
    verifier_gateway: InferenceGateway | None = None,
) -> CompiledPaperSuite:
    """Run Stage 2 verification gate and SHA-256 lock approved items into a suite."""
    source_text: str | None = None
    try:
        source_text = read_paper_text(source_path)
    except Exception:
        source_text = None

    verifier = AnswerabilityVerifier(verifier_model=verifier_model, gateway=verifier_gateway)
    approved: list[ExtractedItem] = []

    for item in items:
        outcome = verifier.verify_item(item, source_text=source_text)
        if outcome.status == VerificationStatus.APPROVED:
            canonical = item.model_dump_json(exclude={"item_sha256", "verification_status"})
            locked_item = item.model_copy(
                update={
                    "verification_status": VerificationStatus.APPROVED,
                    "item_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                }
            )
            approved.append(locked_item)

    source_hash = sha256_file(source_path)
    suite_payload = "|".join(
        [source_hash, suite_name, *(i.item_sha256 or "" for i in approved)]
    ).encode("utf-8")
    return CompiledPaperSuite(
        suite_name=suite_name,
        source_path=str(source_path),
        source_sha256=source_hash,
        extractor_model=extractor_model,
        verifier_model=verifier_model,
        extraction_mode=extraction_mode,
        items=approved,
        suite_sha256=hashlib.sha256(suite_payload).hexdigest(),
    )


def suite_to_task_yaml_dict(suite: CompiledPaperSuite) -> dict[str, Any]:
    """Convert a SHA-256 locked CompiledPaperSuite into a runnable Tier 2 TaskSpec dict."""
    items_payload = [
        {
            "sample_id": item.item_id,
            "input_text": item.question,
            "target": item.reference_answer,
            "rubric": item.rubric,
            "category": item.category,
            "citations": [c.model_dump() for c in item.citations],
            "item_sha256": item.item_sha256,
        }
        for item in suite.items
    ]
    return {
        "name": suite.suite_name,
        "version": "1.0.0",
        "trust_tier": "tier2_declarative",
        "output_type": "generate_until",
        "doc_to_text": "Question: {{ input_text }}\nAnswer:",
        "doc_to_target": "{{ target }}",
        "metrics": ["exact_match"],
        "metadata": {
            "date_stamp": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "source_path": suite.source_path,
            "source_paper_sha256": suite.source_sha256,
            "suite_sha256": suite.suite_sha256,
            "extractor_model": suite.extractor_model,
            "verifier_model": suite.verifier_model,
            "extraction_mode": suite.extraction_mode,
            "items": items_payload,
        },
    }


def write_compiled_pack(suite: CompiledPaperSuite, output_path: Path) -> None:
    """Persist a CompiledPaperSuite as runnable YAML (.yaml/.yml) or locked JSON (.json)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in (".yaml", ".yml"):
        payload = suite_to_task_yaml_dict(suite)
        output_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    else:
        output_path.write_text(suite.model_dump_json(indent=2) + "\n", encoding="utf-8")
