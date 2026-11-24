"""Tier 2 boundary & negative E2E tests for BYOT, BYOP, Receipts, and Windows (15 tests)."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import socket
import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from eval_studio.cache.sqlite_store import SQLiteCacheStore
from eval_studio.cache.wal import JsonlWriteAheadLog
from eval_studio.cli.app import app
from eval_studio.paper.compiler import compile_paper
from eval_studio.paper.schema import Citation, ExtractedItem, VerificationStatus
from eval_studio.paper.verify import AnswerabilityVerifier
from eval_studio.receipts.manifest import MetricSummary, ReceiptManifest, compute_manifest_digest
from eval_studio.sandbox.docker import DockerSandbox
from eval_studio.sandbox.limits import SandboxLimits
from eval_studio.scoring.calibrate import cohens_kappa
from eval_studio.stats.noise_floor import measure_noise_floor
from eval_studio.tasks.templates import TemplateSecurityError, render_sandboxed_template

pytestmark = pytest.mark.e2e


def _write_simple_tier1_task(base_dir: Path, *, name: str = "sample_task") -> tuple[Path, Path]:
    csv_path = base_dir / f"{name}.csv"
    csv_path.write_text(
        "sample_id,input_text,target\n"
        's1,"Which vitamin deficiency causes scurvy?",Vitamin C\n'
        's2,"Which organ produces insulin?",Pancreas\n',
        encoding="utf-8",
    )
    yaml_path = base_dir / f"{name}.yaml"
    yaml_path.write_text(
        f"name: {name}\n"
        'version: "1.0.0"\n'
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_path}\n"
        'doc_to_text: "Question: {{ input_text }}\\nAnswer:"\n'
        'doc_to_target: "{{ target }}"\n'
        "metrics:\n"
        "  - exact_match\n",
        encoding="utf-8",
    )
    return yaml_path, csv_path


# ---------------------------------------------------------------------------
# Area 2C: BYOT, Sandbox, validate & calibrate Boundaries (T2-BYOT-01..05)
# ---------------------------------------------------------------------------


def test_t2_ssti_and_function_tag_blocked_in_tier1_tier2(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-BYOT-01: Jinja2 SSTI payloads and !function YAML tags are blocked in Tiers 1-2."""
    pwn_marker = tmp_path / "pwned_marker.txt"
    pwn_marker_posix = pwn_marker.as_posix()
    payloads = [
        "{{ ''.__class__.__mro__[1].__subclasses__() }}",
        f"{{{{ lipsum.__globals__['os'].popen('echo pwn > \"{pwn_marker_posix}\"').read() }}}}",
    ]
    for expr in payloads:
        with pytest.raises(TemplateSecurityError):
            render_sandboxed_template(expr, {})

    csv_path = tmp_path / "ssti.csv"
    csv_path.write_text('sample_id,input_text,target\ns1,"hi","ok"\n', encoding="utf-8")

    for idx, expr in enumerate(payloads):
        yaml_path = tmp_path / f"ssti_{idx}.yaml"
        task_data = {
            "name": f"ssti_{idx}",
            "trust_tier": "tier2_declarative",
            "output_type": "generate_until",
            "dataset_path": csv_path.as_posix(),
            "doc_to_text": expr,
            "doc_to_target": "{{ target }}",
        }
        yaml_path.write_text(yaml.safe_dump(task_data, sort_keys=False), encoding="utf-8")

        res_val = cli_runner.invoke(app, ["validate", "--task", str(yaml_path)])
        val_out = res_val.stdout + getattr(res_val, "stderr", "") + (
            f" {res_val.exception}" if res_val.exception else ""
        )
        assert res_val.exit_code != 0
        assert "no such option" not in val_out.lower()
        assert "stub:" not in val_out.lower()
        assert any(
            kw in val_out.lower()
            for kw in ("security", "sandbox", "unsafe", "template", "blocked", "forbidden")
        )

        out_dir = tmp_path / f"ssti_out_{idx}"
        res_run = cli_runner.invoke(
            app,
            ["run", "--task", str(yaml_path), "--model", "mock", "--output", str(out_dir)],
        )
        run_out = res_run.stdout + getattr(res_run, "stderr", "") + (
            f" {res_run.exception}" if res_run.exception else ""
        )
        assert res_run.exit_code != 0
        assert "no such option" not in run_out.lower()
        assert "stub:" not in run_out.lower()
        assert any(
            kw in run_out.lower()
            for kw in ("security", "sandbox", "unsafe", "template", "blocked", "forbidden")
        )
        assert not (out_dir / "manifest.json").exists()

    func_yaml = tmp_path / "tier2_func_tag.yaml"
    func_yaml.write_text(
        "name: bad_func_tag\n"
        "trust_tier: tier2_declarative\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_path.as_posix()}\n"
        "doc_to_text: !function os.system\n",
        encoding="utf-8",
    )
    res_func = cli_runner.invoke(app, ["validate", "--task", str(func_yaml)])
    func_out = res_func.stdout + getattr(res_func, "stderr", "") + (
        f" {res_func.exception}" if res_func.exception else ""
    )
    assert res_func.exit_code != 0
    assert "no such option" not in func_out.lower()
    assert "stub:" not in func_out.lower()
    assert any(
        kw in func_out.lower()
        for kw in ("!function", "tag", "constructor", "unsafe", "yaml", "tier", "blocked")
    )
    assert not pwn_marker.exists()


def test_t2_tier3_blocked_without_trust_remote_code(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-BYOT-02: Tier 3 tasks and --plugins are rejected unless --trust-remote-code is set."""
    import_marker = tmp_path / "plugin_imported.txt"
    plugin_path = tmp_path / "unsafe_plugin.py"
    plugin_path.write_text(
        "from pathlib import Path\n"
        f"Path(r'{import_marker}').write_text('imported', encoding='utf-8')\n"
        "EVAL_STUDIO_PLUGINS = {}\n",
        encoding="utf-8",
    )

    tier3_yaml = tmp_path / "tier3.yaml"
    tier3_yaml.write_text(
        "name: unsafe_tier3_task\n"
        "trust_tier: tier3_code_plugin\n"
        "output_type: generate_until\n"
        "unsafe_code: true\n",
        encoding="utf-8",
    )

    res_val = cli_runner.invoke(app, ["validate", "--task", str(tier3_yaml)])
    val_out = res_val.stdout + getattr(res_val, "stderr", "") + (
        f" {res_val.exception}" if res_val.exception else ""
    )
    assert res_val.exit_code != 0
    assert "no such option" not in val_out.lower()
    assert "stub:" not in val_out.lower()
    assert "--trust-remote-code" in val_out or "trust-remote-code" in val_out

    out_dir = tmp_path / "tier3_out"
    res_run = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(tier3_yaml),
            "--model",
            "mock",
            "--plugins",
            str(plugin_path),
            "--output",
            str(out_dir),
        ],
    )
    run_out = res_run.stdout + getattr(res_run, "stderr", "") + (
        f" {res_run.exception}" if res_run.exception else ""
    )
    assert res_run.exit_code != 0
    assert "no such option" not in run_out.lower()
    assert "stub:" not in run_out.lower()
    assert "trust" in run_out.lower()
    assert not import_marker.exists()
    assert not (out_dir / "manifest.json").exists()


def test_t2_sandbox_blocks_eq_spoofing_timeout_and_network() -> None:
    """T2-BYOT-03: Sandbox isolates __eq__ spoofing, enforces timeouts, and disables network."""
    default_limits = SandboxLimits()
    assert default_limits.network_disabled is True
    assert default_limits.read_only_rootfs is True

    sandbox = DockerSandbox()
    spoof_code = (
        "class AlwaysEqual:\n"
        "    def __eq__(self, other):\n"
        "        return True\n"
        "result = AlwaysEqual()\n"
    )
    spoof_harness = "assert type(result) is int and result == 42"
    spoof_verdict = sandbox.run_code(spoof_code, test_harness=spoof_harness, limits=default_limits)
    assert type(spoof_verdict.passed) is bool
    assert spoof_verdict.passed is False
    assert spoof_verdict.exit_code != 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        net_code = (
            "import socket\n"
            f"s = socket.create_connection(('127.0.0.1', {port}), timeout=1.0)\n"
            "s.close()\n"
            "result = True\n"
        )
        net_verdict = sandbox.run_code(
            net_code,
            test_harness="assert result is True",
            limits=default_limits,
        )
    assert type(net_verdict.passed) is bool
    assert net_verdict.passed is False
    assert net_verdict.exit_code != 0

    t0 = time.monotonic()
    timeout_limits = SandboxLimits(timeout_seconds=0.5)
    loop_verdict = sandbox.run_code(
        "while True:\n    pass\n",
        test_harness="assert True",
        limits=timeout_limits,
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0
    assert loop_verdict.timed_out is True
    assert loop_verdict.passed is False


def test_t2_calibrate_kappa_below_040_warning_and_boundary(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-BYOT-04: calibrate computes exact Cohen's kappa and flags < 0.40 as unaligned."""
    # Exact kappa = 0.20 (n=10, balanced 5/5: p_o=0.60, p_e=0.50 -> 0.20)
    human_low = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    judge_low = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
    assert cohens_kappa(judge_low, human_low) == pytest.approx(0.20, abs=1e-9)

    low_jsonl = tmp_path / "calib_low.jsonl"
    low_jsonl.write_text(
        "\n".join(
            json.dumps({"sample_id": f"c{i}", "judge_label": j, "human_label": h})
            for i, (j, h) in enumerate(zip(judge_low, human_low, strict=True))
        )
        + "\n",
        encoding="utf-8",
    )
    res_low = cli_runner.invoke(app, ["calibrate", str(low_jsonl)])
    low_out = res_low.stdout + getattr(res_low, "stderr", "")
    assert "stub:" not in low_out.lower()
    assert "0.20" in low_out
    assert "unaligned" in low_out.lower() or "false" in low_out.lower()

    # Exact kappa = 0.40 boundary (n=20, balanced 10/10: p_o=14/20=0.70, p_e=0.50 -> 0.40)
    human_b = [1] * 10 + [0] * 10
    judge_b = [1] * 7 + [0] * 3 + [1] * 3 + [0] * 7
    assert cohens_kappa(judge_b, human_b) == pytest.approx(0.40, abs=1e-9)

    boundary_jsonl = tmp_path / "calib_boundary.jsonl"
    boundary_jsonl.write_text(
        "\n".join(
            json.dumps({"sample_id": f"b{i}", "judge_label": j, "human_label": h})
            for i, (j, h) in enumerate(zip(judge_b, human_b, strict=True))
        )
        + "\n",
        encoding="utf-8",
    )
    res_b = cli_runner.invoke(app, ["calibrate", str(boundary_jsonl)])
    b_out = res_b.stdout + getattr(res_b, "stderr", "")
    assert res_b.exit_code == 0
    assert "stub:" not in b_out.lower()
    assert "0.40" in b_out
    assert "unaligned" not in b_out.lower()

    # Empty and mismatched inputs must fail loudly
    empty_jsonl = tmp_path / "calib_empty.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")
    res_empty = cli_runner.invoke(app, ["calibrate", str(empty_jsonl)])
    empty_out = res_empty.stdout + getattr(res_empty, "stderr", "")
    assert res_empty.exit_code != 0
    assert "no such option" not in empty_out.lower()
    assert "stub:" not in empty_out.lower()

    with pytest.raises(ValueError):
        cohens_kappa([], [])
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 0])


def test_t2_validate_catches_snapshot_drift_and_invalid_schema(
    tmp_path: Path,
    cli_runner: CliRunner,
    repo_root: Path,
) -> None:
    """T2-BYOT-05: eval-studio validate detects 1-char golden snapshot drift and invalid schemas."""
    snapshot_path = repo_root / "tests" / "snapshots" / "tier1_sample.prompt.txt"
    csv_src = repo_root / "examples" / "tier1_zero_code" / "sample.csv"
    csv_dst = tmp_path / "sample.csv"
    shutil.copyfile(csv_src, csv_dst)

    drifted_yaml = tmp_path / "drifted_task.yaml"
    drifted_yaml.write_text(
        "name: tier1_med_qa_sample\n"
        'version: "0.1.0"\n'
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_dst}\n"
        'doc_to_text: "Question: {{ input_text }}\\nAnswer: "\n'
        'doc_to_target: "{{ target }}"\n'
        "metrics:\n"
        "  - exact_match\n",
        encoding="utf-8",
    )
    res_drift = cli_runner.invoke(
        app,
        ["validate", "--task", str(drifted_yaml), "--snapshot", str(snapshot_path)],
    )
    drift_out = res_drift.stdout + getattr(res_drift, "stderr", "")
    assert res_drift.exit_code == 1
    assert "no such option" not in drift_out.lower()
    assert "snapshot" in drift_out.lower()

    invalid_schema_yaml = tmp_path / "invalid_schema.yaml"
    invalid_schema_yaml.write_text(
        "name: bad_schema\n"
        "trust_tier: tier1_zero_code\n"
        'output_type: "invalid_mode"\n',
        encoding="utf-8",
    )
    res_bad = cli_runner.invoke(app, ["validate", "--task", str(invalid_schema_yaml)])
    bad_out = res_bad.stdout + getattr(res_bad, "stderr", "")
    assert res_bad.exit_code != 0
    assert "no such option" not in bad_out.lower()
    assert "stub:" not in bad_out.lower()


# ---------------------------------------------------------------------------
# Area 2D: BYOP compile-paper Boundaries (T2-BYOP-01..05)
# ---------------------------------------------------------------------------


def test_t2_compile_paper_rejects_uncited_and_blank_citation_items(
    examples_dir: Path,
) -> None:
    """T2-BYOP-01: Uncited, page=0, blank-section, and blank-quote items are rejected."""
    with pytest.raises(ValidationError):
        ExtractedItem(
            item_id="no-cite",
            question="What biomarker was evaluated?",
            reference_answer="Troponin I",
            citations=[],
        )

    with pytest.raises(ValidationError):
        Citation(page=0, section="Section 1", quote="Elevated serum troponin I")

    with pytest.raises(ValidationError):
        Citation(page=1, section="   ", quote="Elevated serum troponin I")

    with pytest.raises(ValidationError):
        Citation(page=1, section="Section 1", quote="\t\n")

    valid_cite = Citation(
        page=1,
        section="Section 2: Cohort Findings",
        quote="negative predictive value of 99.6%",
    )
    clean_item = ExtractedItem(
        item_id="valid-1",
        question="What NPV was observed in the ED chest pain cohort?",
        reference_answer="99.6%",
        citations=[valid_cite],
    )
    suite = compile_paper(
        examples_dir / "papers" / "sample_paper.md",
        items=[clean_item],
        suite_name="citation_gate_check",
    )
    assert len(suite.items) == 1
    assert suite.items[0].item_sha256 is not None


def test_t2_compile_paper_rejects_answer_leakage_and_dangling_refs(
    examples_dir: Path,
) -> None:
    """T2-BYOP-02: Stage 2 gate rejects answer-leaking questions and dangling references."""
    cite = Citation(
        page=1,
        section="Section 2: Cohort Findings",
        quote="negative predictive value of 99.6%",
    )
    leaky = ExtractedItem(
        item_id="leaky-1",
        question="Why did the cohort protocol achieve a 99.6% NPV?",
        reference_answer="99.6%",
        citations=[cite],
    )
    dangling = ExtractedItem(
        item_id="dangling-1",
        question="What does Table 2 above show for subgroup sensitivity?",
        reference_answer="94.2%",
        citations=[cite],
    )

    verifier = AnswerabilityVerifier()
    leaky_outcome = verifier.verify_item(leaky)
    assert leaky_outcome.status in (VerificationStatus.REJECTED, VerificationStatus.NEEDS_REVIEW)
    assert leaky_outcome.reasons

    dangling_outcome = verifier.verify_item(dangling)
    assert dangling_outcome.status in (VerificationStatus.REJECTED, VerificationStatus.NEEDS_REVIEW)
    assert dangling_outcome.reasons

    suite = compile_paper(
        examples_dir / "papers" / "sample_paper.md",
        items=[leaky, dangling],
        suite_name="rejected_suite",
    )
    assert len(suite.items) == 0


def test_t2_compile_paper_all_items_rejected_behavior(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-BYOP-03: 100% Stage-2 rejection emits 0 items without fabricating citations."""
    paper_path = tmp_path / "empty_yield_paper.md"
    paper_path.write_text("# Brief Note\n\nNo quantitative claims reported.\n", encoding="utf-8")

    cite = Citation(page=1, section="Brief Note", quote="No quantitative claims reported.")
    leaky_only = ExtractedItem(
        item_id="only-leaky",
        question="Does this brief note contain No quantitative claims reported?",
        reference_answer="No quantitative claims reported.",
        citations=[cite],
    )
    suite = compile_paper(paper_path, items=[leaky_only], suite_name="zero_approved")
    assert len(suite.items) == 0

    out_suite = tmp_path / "compiled_zero.json"
    res = cli_runner.invoke(
        app,
        ["compile-paper", str(paper_path), "--out", str(out_suite)],
    )
    cli_out = res.stdout + getattr(res, "stderr", "")
    assert "stub:" not in cli_out.lower()
    assert "no such option" not in cli_out.lower()
    if res.exit_code == 0 and out_suite.exists():
        data = json.loads(out_suite.read_text(encoding="utf-8"))
        assert data.get("items", []) == []
    else:
        assert res.exit_code != 0


def test_t2_compile_paper_nonexistent_and_corrupt_input_rejected(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2-BYOP-04: Missing and corrupt paper inputs fail cleanly without leaking secrets."""
    monkeypatch.setenv("EXTRACTOR_API_KEY", "sk-extract-secret1234567890")
    missing_md = tmp_path / "does_not_exist.md"
    res_missing = cli_runner.invoke(
        app,
        ["compile-paper", str(missing_md), "--out", str(tmp_path / "out1.json")],
    )
    out1 = res_missing.stdout + getattr(res_missing, "stderr", "") + (
        f" {res_missing.exception}" if res_missing.exception else ""
    )
    assert res_missing.exit_code != 0
    assert "sk-extract-secret1234567890" not in out1
    assert "stub:" not in out1.lower()
    assert "no such option" not in out1.lower()
    assert any(
        kw in out1.lower() for kw in ("not found", "exist", "missing", "no such file")
    )

    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4\n\x00\xff\xfeCORRUPT_BINARY_STREAM")
    res_corrupt = cli_runner.invoke(
        app,
        ["compile-paper", str(corrupt_pdf), "--out", str(tmp_path / "out2.json")],
    )
    out2 = res_corrupt.stdout + getattr(res_corrupt, "stderr", "") + (
        f" {res_corrupt.exception}" if res_corrupt.exception else ""
    )
    assert res_corrupt.exit_code != 0
    assert "sk-extract-secret1234567890" not in out2
    assert "stub:" not in out2.lower()
    assert "no such option" not in out2.lower()
    assert any(
        kw in out2.lower()
        for kw in ("pdf", "corrupt", "invalid", "parse", "pypdf", "evaluation-studio[pdf]")
    )


def test_t2_compile_paper_source_1char_edit_breaks_hash_chain(
    tmp_path: Path,
    cli_runner: CliRunner,
    examples_dir: Path,
) -> None:
    """T2-BYOP-05: 1-char edit to source paper invalidates downstream Eval Receipt."""
    paper_copy = tmp_path / "paper.md"
    shutil.copyfile(examples_dir / "papers" / "sample_paper.md", paper_copy)

    suite_yaml = tmp_path / "compiled_suite.yaml"
    res_comp = cli_runner.invoke(
        app,
        ["compile-paper", str(paper_copy), "--out", str(suite_yaml)],
    )
    assert res_comp.exit_code == 0, res_comp.stdout

    receipt_dir = tmp_path / "byop_receipt"
    res_run = cli_runner.invoke(
        app,
        ["run", "--task", str(suite_yaml), "--model", "mock", "--output", str(receipt_dir)],
    )
    assert res_run.exit_code == 0, res_run.stdout

    res_ok = cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)])
    assert res_ok.exit_code == 0, res_ok.stdout

    # Mutate 1 character in the source paper
    original = paper_copy.read_text(encoding="utf-8")
    assert "99.6%" in original
    paper_copy.write_text(original.replace("99.6%", "99.5%"), encoding="utf-8")

    res_tampered = cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)])
    assert res_tampered.exit_code == 1
    assert "sha256" in res_tampered.stdout.lower() or "mismatch" in res_tampered.stdout.lower()


# ---------------------------------------------------------------------------
# Area 2E: Portable Eval Receipts & Windows Boundaries (T2-RCP-01..05)
# ---------------------------------------------------------------------------


def test_t2_verify_receipt_detects_1char_tamper_across_all_artifacts(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-RCP-01: verify-receipt catches 1-char tamper in samples, manifest, YAML, or missing."""
    task_yaml, _ = _write_simple_tier1_task(tmp_path, name="tamper_target")
    receipt_dir = tmp_path / "valid_receipt"

    res_run = cli_runner.invoke(
        app,
        ["run", "--task", str(task_yaml), "--model", "mock", "--output", str(receipt_dir)],
    )
    assert res_run.exit_code == 0, res_run.stdout
    assert cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)]).exit_code == 0

    # (a) 1-char edit in samples.jsonl
    copy_a = tmp_path / "copy_a"
    shutil.copytree(receipt_dir, copy_a)
    s_text = (copy_a / "samples.jsonl").read_text(encoding="utf-8")
    (copy_a / "samples.jsonl").write_text(s_text.replace("s1", "s9", 1), encoding="utf-8")
    assert cli_runner.invoke(app, ["verify-receipt", str(copy_a)]).exit_code == 1

    # (b) 1-char edit in manifest.json metric float
    copy_b = tmp_path / "copy_b"
    shutil.copytree(receipt_dir, copy_b)
    m_data = json.loads((copy_b / "manifest.json").read_text(encoding="utf-8"))
    m_data["metrics"]["exact_match"]["value"] = round(
        m_data["metrics"]["exact_match"]["value"] + 0.001, 4
    )
    (copy_b / "manifest.json").write_text(json.dumps(m_data, indent=2), encoding="utf-8")
    assert cli_runner.invoke(app, ["verify-receipt", str(copy_b)]).exit_code == 1

    # (c) 1-char edit in linked task YAML prompt
    copy_c = tmp_path / "copy_c"
    shutil.copytree(receipt_dir, copy_c)
    yaml_orig = task_yaml.read_text(encoding="utf-8")
    try:
        task_yaml.write_text(yaml_orig.replace("Answer:", "Answer?"), encoding="utf-8")
        assert cli_runner.invoke(app, ["verify-receipt", str(copy_c)]).exit_code == 1
    finally:
        task_yaml.write_text(yaml_orig, encoding="utf-8")

    # (d) Missing samples.jsonl
    copy_d = tmp_path / "copy_d"
    shutil.copytree(receipt_dir, copy_d)
    (copy_d / "samples.jsonl").unlink()
    assert cli_runner.invoke(app, ["verify-receipt", str(copy_d)]).exit_code == 1


def test_t2_verify_receipt_detects_secret_leak_even_with_valid_hashes(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-RCP-02: verify-receipt rejects leaked secrets even when all SHA-256 hashes match."""
    leaked_line = (
        b'{"sample_id":"s1","note":"Authorization sk-proj-secret1234567890abcdef"}\n'
    )
    (tmp_path / "samples.jsonl").write_bytes(leaked_line)

    manifest = ReceiptManifest(
        run_id="run-secret-check",
        created_at_utc="2026-11-22T00:00:00Z",
        backend="mock",
        model_id="mock-deterministic",
        task_name="secret_audit_task",
        task_sha256="b" * 64,
        samples_sha256=hashlib.sha256(leaked_line).hexdigest(),
        metrics={"exact_match": MetricSummary(value=1.0, stderr=0.0, n_samples=1)},
    )
    manifest = manifest.model_copy(update={"manifest_sha256": compute_manifest_digest(manifest)})
    payload = manifest.model_dump_json(indent=2)
    (tmp_path / "manifest.json").write_text(payload, encoding="utf-8")
    (tmp_path / "receipt.json").write_text(payload, encoding="utf-8")

    res = cli_runner.invoke(app, ["verify-receipt", str(tmp_path)])
    assert res.exit_code == 1
    assert "secret" in res.stdout.lower()


def test_t2_score_missing_or_corrupt_receipt_fails_loudly(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-RCP-03: eval-studio score fails cleanly on missing or corrupt receipt files."""
    empty_dir = tmp_path / "empty_receipt"
    empty_dir.mkdir()
    res_empty = cli_runner.invoke(app, ["score", "--receipt", str(empty_dir)])
    empty_out = res_empty.stdout + getattr(res_empty, "stderr", "")
    assert res_empty.exit_code == 1
    assert "stub:" not in empty_out.lower()

    corrupt_dir = tmp_path / "corrupt_receipt"
    corrupt_dir.mkdir()
    corrupt_bytes = b"{not_valid_json\n"
    (corrupt_dir / "samples.jsonl").write_bytes(corrupt_bytes)
    manifest = ReceiptManifest(
        run_id="run-corrupt",
        created_at_utc="2026-11-22T00:00:00Z",
        backend="mock",
        model_id="mock-deterministic",
        task_name="corrupt_task",
        task_sha256="c" * 64,
        samples_sha256=hashlib.sha256(corrupt_bytes).hexdigest(),
    )
    manifest = manifest.model_copy(update={"manifest_sha256": compute_manifest_digest(manifest)})
    (corrupt_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    res_corrupt = cli_runner.invoke(app, ["score", "--receipt", str(corrupt_dir)])
    corrupt_out = res_corrupt.stdout + getattr(res_corrupt, "stderr", "")
    assert res_corrupt.exit_code == 1
    assert "stub:" not in corrupt_out.lower()


def _write_synthetic_receipt(receipt_dir: Path, sample_ids: list[str]) -> None:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "sample_id": sid,
                "task_name": "nf_task",
                "target": "A",
                "raw_responses": [
                    {
                        "request_id": f"req-{sid}",
                        "sample_id": sid,
                        "text": "A",
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    }
                ],
                "filtered_responses": {"default": ["A"]},
                "metrics": {"default": {"exact_match": 1.0}},
            }
        )
        for sid in sample_ids
    ]
    raw_bytes = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    (receipt_dir / "samples.jsonl").write_bytes(raw_bytes)
    manifest = ReceiptManifest(
        run_id=f"run-{receipt_dir.name}",
        created_at_utc="2026-11-22T00:00:00Z",
        backend="mock",
        model_id="mock-deterministic",
        task_name="nf_task",
        task_sha256="d" * 64,
        samples_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    manifest = manifest.model_copy(update={"manifest_sha256": compute_manifest_digest(manifest)})
    payload = manifest.model_dump_json(indent=2)
    (receipt_dir / "manifest.json").write_text(payload, encoding="utf-8")
    (receipt_dir / "receipt.json").write_text(payload, encoding="utf-8")


def test_t2_noise_floor_mismatched_lengths_and_empty_fails(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T2-RCP-04: noise-floor rejects mismatched sample counts and empty receipts."""
    with pytest.raises(ValueError):
        measure_noise_floor(["A", "B"], ["A"])
    with pytest.raises(ValueError):
        measure_noise_floor([], [])

    base_dir = tmp_path / "base_5"
    pert_dir = tmp_path / "pert_4"
    empty_dir = tmp_path / "empty_0"
    _write_synthetic_receipt(base_dir, [f"s{i}" for i in range(5)])
    _write_synthetic_receipt(pert_dir, [f"s{i}" for i in range(4)])
    _write_synthetic_receipt(empty_dir, [])

    res_mismatch = cli_runner.invoke(
        app,
        ["noise-floor", "--baseline", str(base_dir), "--perturbed", str(pert_dir)],
    )
    mismatch_out = res_mismatch.stdout + getattr(res_mismatch, "stderr", "")
    assert res_mismatch.exit_code == 1
    assert "stub:" not in mismatch_out.lower()

    res_empty = cli_runner.invoke(
        app,
        ["noise-floor", "--baseline", str(empty_dir), "--perturbed", str(empty_dir)],
    )
    empty_out = res_empty.stdout + getattr(res_empty, "stderr", "")
    assert res_empty.exit_code == 1
    assert "stub:" not in empty_out.lower()


def test_t2_windows_non_ascii_and_reserved_chars_and_lock_cleanup(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2-RCP-05: Windows UTF-8, spaces/parens paths, quoted reproduce_command, and handle close."""
    suite_dir = tmp_path / "Med Eval   Suite (v1.0)"
    suite_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = suite_dir / "cache store (Café)"
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(cache_dir))

    # Exercise SQLite + WAL directly first and confirm immediate Windows deletion succeeds
    db_path = cache_dir / "eval_rank0.db"
    store = SQLiteCacheStore(db_path)
    store.put("fp-utf8", {"text": "α-thalassemia — ≥ 95% Café Straße İstanbul 🧬"})
    assert store.get("fp-utf8") == {"text": "α-thalassemia — ≥ 95% Café Straße İstanbul 🧬"}
    wal = JsonlWriteAheadLog(cache_dir / "wal.jsonl")
    wal.append({"note": "α-thalassemia — ≥ 95% Café Straße İstanbul 🧬"})

    csv_path = suite_dir / "utf8 data (v1).csv"
    csv_path.write_text(
        "sample_id,input_text,target\n"
        'w1,"Interpret: α-thalassemia — ≥ 95% Café Straße İstanbul 🧬","α-thalassemia"\n',
        encoding="utf-8",
    )
    yaml_path = suite_dir / "utf8 task (v1).yaml"
    yaml_path.write_text(
        "name: windows_utf8_suite\n"
        'version: "1.0.0"\n'
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {csv_path}\n"
        'doc_to_text: "Case: {{ input_text }}\\nAnswer:"\n'
        'doc_to_target: "{{ target }}"\n'
        "metrics:\n"
        "  - exact_match\n",
        encoding="utf-8",
    )
    receipt_dir = suite_dir / "receipt out (v1)"

    assert cli_runner.invoke(app, ["validate", "--task", str(yaml_path)]).exit_code == 0
    res_run = cli_runner.invoke(
        app,
        ["run", "--task", str(yaml_path), "--model", "mock", "--output", str(receipt_dir)],
    )
    assert res_run.exit_code == 0, res_run.stdout
    assert cli_runner.invoke(app, ["score", "--receipt", str(receipt_dir)]).exit_code == 0
    assert cli_runner.invoke(app, ["verify-receipt", str(receipt_dir)]).exit_code == 0

    manifest = json.loads((receipt_dir / "manifest.json").read_text(encoding="utf-8"))
    repro_cmd = manifest.get("reproduce_command", "")
    assert repro_cmd
    tokens = [tok.strip("\"'") for tok in shlex.split(repro_cmd, posix=False)]
    assert any(str(yaml_path) in tok or yaml_path.name in tok for tok in tokens)

    # Immediate rmtree must succeed on Windows without WinError 32 file-lock failures
    shutil.rmtree(suite_dir)
    assert not suite_dir.exists()
