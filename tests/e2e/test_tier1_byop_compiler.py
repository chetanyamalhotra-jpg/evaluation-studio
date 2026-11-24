"""Tier 1 E2E tests: BYOP two-stage `compile-paper` pipeline & SHA-256 locking (F22)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eval_studio.cli.app import app
from eval_studio.gateway.mock import MockGateway
from eval_studio.paper.compiler import compile_paper, sha256_file
from eval_studio.paper.extract import PaperExtractor
from eval_studio.paper.schema import Citation, CompiledPaperSuite, ExtractedItem, VerificationStatus
from eval_studio.receipts.manifest import ReceiptManifest

pytestmark = pytest.mark.e2e


def test_t1_compile_paper_markdown_offline_cli(
    examples_dir: Path,
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-BYOP-01: `eval-studio compile-paper <paper.md> --out <json>` works 100% offline."""
    paper = examples_dir / "papers" / "sample_paper.md"
    out_json = tmp_path / "compiled_suite.json"

    res_out = cli_runner.invoke(app, ["compile-paper", str(paper), "--out", str(out_json)])
    assert res_out.exit_code == 0, res_out.stdout
    assert out_json.is_file()

    suite = CompiledPaperSuite.model_validate_json(out_json.read_text(encoding="utf-8"))
    assert len(suite.items) > 0
    assert all(item.verification_status == VerificationStatus.APPROVED for item in suite.items)

    out_flag = tmp_path / "compiled_suite_o.json"
    res_o = cli_runner.invoke(app, ["compile-paper", str(paper), "-o", str(out_flag)])
    assert res_o.exit_code == 0, res_o.stdout
    suite_o = CompiledPaperSuite.model_validate_json(out_flag.read_text(encoding="utf-8"))
    assert suite_o.suite_sha256 == suite.suite_sha256


def test_t1_compile_paper_mandatory_citations_and_sha256_lock(
    examples_dir: Path,
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-BYOP-02: Mandatory verbatim citations and deterministic SHA-256 suite locking."""
    paper = examples_dir / "papers" / "sample_paper.md"
    paper_text = paper.read_text(encoding="utf-8")
    expected_source_hash = sha256_file(paper)
    assert len(expected_source_hash) == 64

    out1 = tmp_path / "suite1.json"
    out2 = tmp_path / "suite2.json"
    assert cli_runner.invoke(app, ["compile-paper", str(paper), "--out", str(out1)]).exit_code == 0
    assert cli_runner.invoke(app, ["compile-paper", str(paper), "--out", str(out2)]).exit_code == 0

    s1 = CompiledPaperSuite.model_validate_json(out1.read_text(encoding="utf-8"))
    s2 = CompiledPaperSuite.model_validate_json(out2.read_text(encoding="utf-8"))

    assert s1.source_sha256 == expected_source_hash
    assert s1.suite_sha256 == s2.suite_sha256
    assert len(s1.suite_sha256) == 64

    for item in s1.items:
        assert len(item.citations) >= 1
        assert item.item_sha256 is not None and len(item.item_sha256) == 64
        for cite in item.citations:
            assert cite.page >= 1
            assert cite.section.strip() != ""
            assert cite.quote in paper_text


def test_t1_compile_paper_two_stage_mixed_filtering(
    examples_dir: Path,
    tmp_path: Path,
) -> None:
    """T1-BYOP-03: Stage 2 AnswerabilityVerifier filters uncited & leaky candidates."""
    paper = examples_dir / "papers" / "sample_paper.md"

    # Exercise MockGateway compiler_mode producing mixed valid + invalid payloads
    comp_gw = MockGateway(model_id="mock-compiler", compiler_mode="mixed")
    extractor = PaperExtractor(extractor_model="mock-extractor", gateway=comp_gw)
    extracted = extractor.extract_from_path(paper)
    assert len(extracted) >= 2

    # Also explicitly test compile_paper with 2 valid + 1 ungrounded + 1 answer-leaking item
    valid_1 = ExtractedItem(
        item_id="valid-1",
        question="What troponin I threshold indicates acute myocardial injury?",
        reference_answer="> 0.04 ng/mL",
        citations=[
            Citation(
                page=1,
                section="Section 1: Background",
                quote="Elevated serum troponin I (> 0.04 ng/mL) is a sensitive",
            )
        ],
    )
    valid_2 = ExtractedItem(
        item_id="valid-2",
        question="What NPV did the 0/1-hour high-sensitivity troponin protocol achieve?",
        reference_answer="99.6%",
        citations=[
            Citation(
                page=1,
                section="Section 2: Cohort Findings",
                quote="achieved a negative predictive value of 99.6%",
            )
        ],
    )
    leaky_item = ExtractedItem(
        item_id="leaky-1",
        question="Why did the cohort achieve a negative predictive value of 99.6%?",
        reference_answer="99.6%",
        citations=[
            Citation(
                page=1,
                section="Section 2: Cohort Findings",
                quote="achieved a negative predictive value of 99.6%",
            )
        ],
    )
    # Construct uncited candidate via model_construct to bypass min_length=1 and test Stage 2 gate
    uncited_item = ExtractedItem.model_construct(
        item_id="uncited-1",
        question="How many patients were enrolled?",
        reference_answer="240",
        choices=[],
        citations=[],
        verification_status=VerificationStatus.PENDING,
        verification_notes="",
        item_sha256=None,
    )

    suite = compile_paper(
        paper,
        items=[valid_1, valid_2, leaky_item, uncited_item],
        suite_name="mixed_gate_check",
        extractor_model="stage1-extractor-v1",
        verifier_model="stage2-verifier-v1",
    )
    assert [item.item_id for item in suite.items] == ["valid-1", "valid-2"]
    assert suite.extractor_model == "stage1-extractor-v1"
    assert suite.verifier_model == "stage2-verifier-v1"


def test_t1_compile_paper_yaml_and_json_output_runnable(
    examples_dir: Path,
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-BYOP-04: Compiled `.yaml` pack validates, runs with mock, and passes verify-receipt."""
    paper = examples_dir / "papers" / "sample_paper.md"
    compiled_yaml = tmp_path / "paper_eval_pack.yaml"

    comp_res = cli_runner.invoke(
        app,
        ["compile-paper", str(paper), "--output", str(compiled_yaml)],
    )
    assert comp_res.exit_code == 0, comp_res.stdout
    assert compiled_yaml.is_file()

    val_res = cli_runner.invoke(app, ["validate", "--task", str(compiled_yaml)])
    assert val_res.exit_code == 0, val_res.stdout

    receipt_dir = tmp_path / "paper_receipt"
    run_res = cli_runner.invoke(
        app,
        ["run", "--task", str(compiled_yaml), "--model", "mock", "--output", str(receipt_dir)],
    )
    assert run_res.exit_code == 0, run_res.stdout

    ver_res = cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)])
    assert ver_res.exit_code == 0, ver_res.stdout
    assert "VERIFIED" in ver_res.stdout

    manifest = ReceiptManifest.model_validate_json(
        (receipt_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.source_paper_sha256 == sha256_file(paper)


def test_t1_compile_paper_pdf_extra_and_mode_bc_extensibility(
    examples_dir: Path,
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-BYOP-05: Missing `[pdf]` extra gives actionable error; Mode B/C metadata supported."""
    pdf_path = tmp_path / "dummy_study.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n")
    out_json = tmp_path / "pdf_suite.json"

    has_pypdf = importlib.util.find_spec("pypdf") is not None

    if not has_pypdf:
        res = cli_runner.invoke(app, ["compile-paper", str(pdf_path), "--out", str(out_json)])
        assert res.exit_code != 0
        combined = (res.output + (f" {res.exception}" if res.exception else "")).casefold()
        assert "evaluation-studio[pdf]" in combined or "pypdf" in combined

        with pytest.raises(ImportError) as exc_info:
            PaperExtractor().extract_from_path(pdf_path)
        assert "evaluation-studio[pdf]" in str(exc_info.value)

    # Verify Mode B/C extensibility hook (`extraction_mode`) in CompiledPaperSuite
    paper = examples_dir / "papers" / "sample_paper.md"
    suite = CompiledPaperSuite(
        suite_name="mode_b_clinical",
        source_path=str(paper),
        source_sha256=sha256_file(paper),
        extractor_model="mock-extractor",
        verifier_model="mock-verifier",
        extraction_mode="mode_b_evidence_synthesis",
        items=[],
        suite_sha256="0" * 64,
    )
    roundtrip = json.loads(suite.model_dump_json())
    assert roundtrip["extraction_mode"] == "mode_b_evidence_synthesis"
