"""Adversarial tests for BYOP compile-paper verification gate and Eval Receipt tampering."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_studio.paper.compiler import compile_paper
from eval_studio.paper.schema import Citation, ExtractedItem, VerificationStatus
from eval_studio.receipts.manifest import MetricSummary, ReceiptManifest, compute_manifest_digest
from eval_studio.receipts.verify import verify_receipt


def test_byop_item_requires_citation_and_rejects_answer_leak(examples_dir: Path) -> None:
    with pytest.raises(ValidationError):
        ExtractedItem(
            item_id="bad-1",
            question="What is the NPV?",
            reference_answer="99.6%",
            citations=[],
        )

    citation = Citation(page=1, section="Section 2", quote="negative predictive value of 99.6%")
    leaky = ExtractedItem(
        item_id="leaky-1",
        question="Why is the negative predictive value 99.6%?",
        reference_answer="99.6%",
        citations=[citation],
    )
    clean = ExtractedItem(
        item_id="clean-1",
        question="What negative predictive value did the 0/1-hour troponin protocol achieve?",
        reference_answer="99.6%",
        citations=[citation],
    )

    suite = compile_paper(
        examples_dir / "papers" / "sample_paper.md",
        items=[leaky, clean],
        suite_name="troponin_suite",
    )
    assert len(suite.items) == 1
    assert suite.items[0].item_id == "clean-1"
    assert suite.items[0].verification_status == VerificationStatus.APPROVED
    assert suite.items[0].item_sha256 is not None
    assert len(suite.source_sha256) == 64


def test_receipt_verification_detects_sample_tampering(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_bytes = b'{"sample_id":"s1","score":1.0}\n'
    samples_path.write_bytes(samples_bytes)

    manifest = ReceiptManifest(
        run_id="run-001",
        created_at_utc="2026-11-22T00:00:00Z",
        backend="mock",
        model_id="mock-deterministic",
        task_name="tier1_med_qa_sample",
        task_sha256="a" * 64,
        samples_sha256=hashlib.sha256(samples_bytes).hexdigest(),
        metrics={"exact_match": MetricSummary(value=1.0, stderr=0.0, n_samples=1)},
    )
    manifest = manifest.model_copy(update={"manifest_sha256": compute_manifest_digest(manifest)})
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    assert verify_receipt(tmp_path).valid is True

    # Tamper with samples.jsonl and verify detection
    samples_path.write_bytes(b'{"sample_id":"s1","score":0.0}\n')
    tampered = verify_receipt(tmp_path)
    assert tampered.valid is False
    assert any("Samples SHA-256 mismatch" in err for err in tampered.errors)
