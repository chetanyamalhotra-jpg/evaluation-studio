"""Adversarial & E2E verification tests for the 6 production-hardening pass fixes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from eval_studio.cli.app import app
from eval_studio.core.schema import (
    ChatMessage,
    EvalRequest,
    EvalResponse,
    ImageContentPart,
    OutputType,
    TextContentPart,
    TokenUsage,
)
from eval_studio.gateway.http import HttpBYOKGateway
from eval_studio.gateway.mock import MockGateway
from eval_studio.receipts.manifest import ReceiptManifest


def test_olmes_dual_cloze_and_mcf_produce_distinct_prompts_and_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gap 1: OLMES dual mode builds separate Cloze and MCF requests and scores independently."""
    runner = CliRunner()
    data_path = tmp_path / "olmes_dual.jsonl"
    data_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": f"ol_{i}",
                    "input_text": f"Which drug treats condition {i}?",
                    "choices": ["Aspirin", "Placebo"],
                    "target": 0,
                }
            )
            for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    task_yaml = tmp_path / "olmes_dual.yaml"
    task_yaml.write_text(
        f"""name: olmes_dual_real
trust_tier: tier2_declarative
output_type: multiple_choice
dataset_path: "{data_path.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
olmes:
  enabled: true
  formats:
    - cloze
    - mcf
  metric_preference: acc_norm
metrics:
  - acc_norm
  - acc_bytes
""",
        encoding="utf-8",
    )

    seen_contexts: dict[str, list[str]] = {"cloze": [], "mcf": []}

    def _mock_ll(self: MockGateway, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        out: list[EvalResponse] = []
        for req in requests:
            is_mcf = "-mcf-" in req.request_id
            fmt = "mcf" if is_mcf else "cloze"
            seen_contexts[fmt].append(f"{req.context}::{req.continuation}")
            # Score MCF 100% correct (choice 0 wins) and Cloze 0% correct (choice 1 wins)
            is_choice_0 = req.request_id.endswith("-c0")
            ll = (-0.1 if is_choice_0 else -5.0) if is_mcf else (-5.0 if is_choice_0 else -0.1)
            out.append(
                EvalResponse(
                    request_id=req.request_id,
                    sample_id=req.sample_id,
                    loglikelihood=ll,
                    is_greedy=True,
                    backend=self.backend_name,
                    model_id=self.model_id,
                    usage=TokenUsage(prompt_tokens=8),
                )
            )
        return out

    monkeypatch.setattr(MockGateway, "loglikelihood", _mock_ll)

    out_dir = tmp_path / "olmes_out"
    res = runner.invoke(
        app,
        ["run", "--task", str(task_yaml), "--model", "mock", "--output", str(out_dir)],
    )
    assert res.exit_code == 0, res.stdout
    assert len(seen_contexts["cloze"]) == 8
    assert len(seen_contexts["mcf"]) == 8
    assert seen_contexts["cloze"] != seen_contexts["mcf"]

    manifest = ReceiptManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.metrics["acc_norm_mcf"].value == pytest.approx(1.0)
    assert manifest.metrics["acc_norm_cloze"].value == pytest.approx(0.0)
    assert manifest.metrics["acc_norm"].value == pytest.approx(1.0)
    sel = manifest.redacted_config["olmes_selection"]
    assert sel["selected_format"] == "mcf"
    assert sel["mcf_score"] == pytest.approx(1.0)
    assert sel["cloze_score"] == pytest.approx(0.0)


def test_acc_mutual_info_dispatches_real_unconditional_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gap 2: acc_mutual_info runs real unconditional loglikelihood requests (not [0.0])."""
    runner = CliRunner()
    data_path = tmp_path / "pmi_mc.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "sample_id": "pmi_0",
                "input_text": "Most specific marker for myocardial necrosis?",
                "choices": ["Troponin I", "Albumin"],
                "target": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task_yaml = tmp_path / "pmi_task.yaml"
    task_yaml.write_text(
        f"""name: pmi_check
trust_tier: tier2_declarative
output_type: multiple_choice
dataset_path: "{data_path.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics:
  - acc
  - acc_mutual_info
""",
        encoding="utf-8",
    )

    uncond_ids: list[str] = []

    def _pmi_ll(self: MockGateway, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        out: list[EvalResponse] = []
        for req in requests:
            if req.request_id.endswith("-uncond"):
                uncond_ids.append(req.request_id)
                # Choice 0 is rare unconditionally (-4.0), Choice 1 is common (-0.1)
                ll = -4.0 if "-c0-" in req.request_id else -0.1
            else:
                # Conditionally Choice 1 slightly edges Choice 0 (-1.2 vs -1.5),
                # so unnormalized acc=0.0 while PMI flips Choice 0 (+2.5 vs -1.1) -> 1.0!
                ll = -1.5 if req.request_id.endswith("-c0") else -1.2
            out.append(
                EvalResponse(
                    request_id=req.request_id,
                    sample_id=req.sample_id,
                    loglikelihood=ll,
                    is_greedy=True,
                    backend=self.backend_name,
                    model_id=self.model_id,
                    usage=TokenUsage(prompt_tokens=6),
                )
            )
        return out

    monkeypatch.setattr(MockGateway, "loglikelihood", _pmi_ll)

    out_dir = tmp_path / "pmi_out"
    res = runner.invoke(
        app,
        ["run", "--task", str(task_yaml), "--model", "mock", "--output", str(out_dir)],
    )
    assert res.exit_code == 0, res.stdout
    assert len(uncond_ids) == 2

    manifest = ReceiptManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.metrics["acc"].value == pytest.approx(0.0)
    assert manifest.metrics["acc_mutual_info"].value == pytest.approx(1.0)


def test_repeats_majority_vote_judge_config_and_dtype_quantization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gap 3: repeats, MajorityVoteFilter, --config, dtype/quantization, and LLMJudge + fallback."""
    runner = CliRunner()
    csv_path = tmp_path / "rep.csv"
    csv_path.write_text(
        'sample_id,input_text,target\n'
        'r1,"Organ making insulin?",Pancreas\n'
        'r2,"Organ making insulin?",Pancreas\n',
        encoding="utf-8",
    )

    task_yaml = tmp_path / "rep_task.yaml"
    task_yaml.write_text(
        f"""name: repeats_judge_task
trust_tier: tier2_declarative
output_type: generate_until
dataset_path: "{csv_path.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics:
  - exact_match
generation_kwargs:
  repeats: 3
filter_ensembles:
  - name: mv_pipe
    steps:
      - name: majority_vote
""",
        encoding="utf-8",
    )

    run_cfg = tmp_path / "run_config.yaml"
    out_dir = tmp_path / "cfg_out"
    run_cfg.write_text(
        f"""model_backend: mock
model_id: mock-configured
tasks:
  - "{task_yaml.as_posix()}"
bootstrap_iters: 250
output_dir: "{out_dir.as_posix()}"
""",
        encoding="utf-8",
    )

    target_replies = iter(
        ["Liver", "Pancreas", "Pancreas", "Liver", "Pancreas", "Pancreas"]
    )

    def _gen(self: MockGateway, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        results: list[EvalResponse] = []
        for req in requests:
            if req.task_name == "clinical_judge":
                if "refusal" in self.model_id:
                    text = "I'm sorry, I cannot grade this."
                else:
                    text = '{"score": 1.0, "rationale": "Verified Pancreas"}'
            else:
                text = next(target_replies)
            results.append(
                EvalResponse(
                    request_id=req.request_id,
                    sample_id=req.sample_id,
                    text=text,
                    backend=self.backend_name,
                    model_id=self.model_id,
                    usage=TokenUsage(prompt_tokens=5, completion_tokens=2),
                )
            )
        return results

    monkeypatch.setattr(MockGateway, "generate_until", _gen)

    res = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(run_cfg),
            "--dtype",
            "bf16",
            "--quantization",
            "fp8",
            "--judge-model",
            "mock-refusal",
            "--fallback-judge-model",
            "mock",
            "--no-cache",
        ],
    )
    assert res.exit_code == 0, res.stdout
    manifest = ReceiptManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.dtype == "bf16"
    assert manifest.quantization == "fp8"
    assert manifest.metrics["exact_match"].value == pytest.approx(1.0)
    assert manifest.metrics["exact_match"].bootstrap_iters == 250
    assert manifest.metrics["llm_judge"].value == pytest.approx(1.0)

    sample_row = json.loads((out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(sample_row["raw_responses"]) == 3
    assert sample_row["filtered_responses"]["mv_pipe"] == ["Pancreas"]


def test_offline_score_mcqa_and_rolling_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gap 4: Offline `eval-studio score` re-scores MCQA (acc_bytes/acc_norm/PMI) & rolling BPB."""
    runner = CliRunner()
    mc_jsonl = tmp_path / "mc.jsonl"
    mc_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "m1",
                        "input_text": "Cause of scurvy?",
                        "choices": ["Vitamin C deficiency", "Iron excess"],
                        "target": 0,
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "m2",
                        "input_text": "Organ producing insulin?",
                        "choices": ["Pancreas", "Spleen"],
                        "target": 0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    mc_yaml = tmp_path / "mc.yaml"
    mc_yaml.write_text(
        f"""name: mc_rescore_parity
trust_tier: tier1_zero_code
output_type: multiple_choice
dataset_path: "{mc_jsonl.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics:
  - acc_bytes
  - acc_norm
  - acc_mutual_info
""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "mc_receipt"
    res_run = runner.invoke(
        app,
        ["run", "--task", str(mc_yaml), "--model", "mock", "--output", str(out_dir)],
    )
    assert res_run.exit_code == 0, res_run.stdout
    orig_manifest = ReceiptManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )

    def _forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Offline `eval-studio score` must not call MockGateway")

    monkeypatch.setattr(MockGateway, "loglikelihood", _forbid)
    monkeypatch.setattr(MockGateway, "generate_until", _forbid)

    rescored_dir = tmp_path / "mc_rescored"
    res_score = runner.invoke(
        app,
        ["score", "--receipt", str(out_dir), "--output", str(rescored_dir)],
    )
    assert res_score.exit_code == 0, res_score.stdout
    new_manifest = ReceiptManifest.model_validate_json(
        (rescored_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for m in ("acc_bytes", "acc_norm", "acc_mutual_info"):
        assert new_manifest.metrics[m].value == pytest.approx(orig_manifest.metrics[m].value)
        assert new_manifest.metrics[m].stderr == pytest.approx(
            orig_manifest.metrics[m].stderr or 0.0
        )

    # Also verify rolling perplexity (`bits_per_byte`) offline re-scoring parity
    monkeypatch.undo()
    roll_jsonl = tmp_path / "roll.jsonl"
    roll_jsonl.write_text(
        json.dumps(
            {"sample_id": "rl1", "input_text": "Troponin elevation indicates injury.", "target": ""}
        )
        + "\n"
        + json.dumps(
            {"sample_id": "rl2", "input_text": "Pancreas secretes insulin.", "target": ""}
        )
        + "\n",
        encoding="utf-8",
    )
    roll_yaml = tmp_path / "roll.yaml"
    roll_yaml.write_text(
        f"""name: roll_rescore_parity
trust_tier: tier1_zero_code
output_type: loglikelihood_rolling
dataset_path: "{roll_jsonl.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics:
  - bits_per_byte
""",
        encoding="utf-8",
    )
    roll_out = tmp_path / "roll_receipt"
    assert runner.invoke(
        app, ["run", "--task", str(roll_yaml), "--model", "mock", "--output", str(roll_out)]
    ).exit_code == 0
    orig_roll = ReceiptManifest.model_validate_json(
        (roll_out / "manifest.json").read_text(encoding="utf-8")
    )

    monkeypatch.setattr(MockGateway, "loglikelihood", _forbid)
    monkeypatch.setattr(MockGateway, "loglikelihood_rolling", _forbid)

    roll_rescored = tmp_path / "roll_rescored"
    res_roll_score = runner.invoke(
        app,
        ["score", "--receipt", str(roll_out), "--output", str(roll_rescored)],
    )
    assert res_roll_score.exit_code == 0, res_roll_score.stdout
    new_roll = ReceiptManifest.model_validate_json(
        (roll_rescored / "manifest.json").read_text(encoding="utf-8")
    )
    assert new_roll.metrics["bits_per_byte"].value == pytest.approx(
        orig_roll.metrics["bits_per_byte"].value
    )


def test_compile_paper_with_http_byok_gateway_extractor_and_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gap 5: `compile-paper` wires HttpBYOKGateway for extractor & Stage 2 verifier."""
    runner = CliRunner()
    monkeypatch.setenv("EXTRACTOR_API_KEY", "sk-extractor-mocktest1234567890")
    monkeypatch.setenv("JUDGE_API_KEY", "sk-judge-mocktest1234567890")

    paper = tmp_path / "study.md"
    paper.write_text(
        "# Section 1: Results\n\n"
        "The diagnostic assay achieved 98.4% sensitivity in the cohort.\n",
        encoding="utf-8",
    )

    recorded_auth: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        recorded_auth.append(request.headers.get("authorization", ""))
        body = json.loads(request.content.decode("utf-8"))
        prompt = body["messages"][0]["content"]
        if "Verify whether this extracted benchmark question" in prompt:
            approved = "ambiguous endpoint" not in prompt.casefold()
            verdict = {
                "approved": approved,
                "reasons": [] if approved else ["Ambiguous question rejected by verifier."],
            }
            content = json.dumps(verdict)
        else:
            content = json.dumps(
                {
                    "items": [
                        {
                            "item_id": "keep-1",
                            "question": "What sensitivity did the diagnostic assay achieve?",
                            "reference_answer": "98.4%",
                            "citations": [
                                {
                                    "page": 1,
                                    "section": "Section 1: Results",
                                    "quote": "achieved 98.4% sensitivity in the cohort",
                                }
                            ],
                        },
                        {
                            "item_id": "reject-1",
                            "question": "Is this ambiguous endpoint superior in all populations?",
                            "reference_answer": "Yes",
                            "citations": [
                                {
                                    "page": 1,
                                    "section": "Section 1: Results",
                                    "quote": "achieved 98.4% sensitivity in the cohort",
                                }
                            ],
                        },
                    ]
                }
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 20},
            },
        )

    transport = httpx.MockTransport(_handler)
    orig_async_init = httpx.AsyncClient.__init__

    def _patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("transport") is None:
            kwargs["transport"] = transport
        orig_async_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _patched_init)

    out_yaml = tmp_path / "compiled_byok.yaml"
    res = runner.invoke(
        app,
        [
            "compile-paper",
            str(paper),
            "--out",
            str(out_yaml),
            "--model",
            "http-byok",
            "--base-url",
            "https://api.openai.com/v1",
            "--extractor-model",
            "gpt-4o-mini",
            "--verifier-model",
            "gpt-4o-mini",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "sk-extractor-mocktest1234567890" in recorded_auth[0]
    assert any("sk-judge-mocktest1234567890" in h for h in recorded_auth[1:])
    text = out_yaml.read_text(encoding="utf-8")
    assert "keep-1" in text
    assert "reject-1" not in text


def test_http_gateway_bounded_async_concurrency_and_multimodal_dialects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gap 6: Bounded async Semaphore concurrency preserves order & multimodal parts."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-conc1234567890")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-conc1234567890")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyConcMock1234567890abcdef")

    active = 0
    peak = 0

    class _SlowAsyncTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            body = json.loads(request.content.decode("utf-8"))
            text = body["messages"][0]["content"]
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": f"reply:{text}"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                },
            )

    gw = HttpBYOKGateway(
        model_id="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        max_concurrent=2,
        transport=_SlowAsyncTransport(),
    )
    reqs = [
        EvalRequest(
            request_id=f"c-{i}",
            sample_id=f"s-{i}",
            task_name="conc",
            output_type=OutputType.GENERATE_UNTIL,
            context=f"prompt-{i}",
        )
        for i in range(6)
    ]
    resps = asyncio.run(gw.agenerate_until(reqs))
    assert [r.text for r in resps] == [f"reply:prompt-{i}" for i in range(6)]
    assert peak == 2

    # Multimodal serialization check across OpenAI, Anthropic, and Gemini
    mm_req = EvalRequest(
        request_id="mm-1",
        sample_id="s-mm",
        task_name="mm_test",
        output_type=OutputType.CHAT,
        context="",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    TextContentPart(text="Describe ECG."),
                    ImageContentPart(image_url="data:image/png;base64,AAAA"),
                ],
            )
        ],
    )

    captured: list[tuple[str, dict[str, Any]]] = []

    class _InspectTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            captured.append((str(request.url), payload))
            url_s = str(request.url)
            if "anthropic" in url_s:
                return httpx.Response(
                    200,
                    json={"content": [{"type": "text", "text": "Normal"}], "usage": {}},
                )
            if "googleapis" in url_s:
                return httpx.Response(
                    200,
                    json={
                        "candidates": [{"content": {"parts": [{"text": "Normal"}]}}],
                        "usageMetadata": {},
                    },
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Normal"}}], "usage": {}},
            )

    t = _InspectTransport()
    for prov, mid, burl in (
        ("openai", "gpt-4o", "https://api.openai.com/v1"),
        ("anthropic", "claude-3-5-sonnet", "https://api.anthropic.com/v1"),
        ("gemini", "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta"),
    ):
        g = HttpBYOKGateway(model_id=mid, base_url=burl, provider=prov, transport=t)
        assert g.chat([mm_req])[0].text == "Normal"

    openai_blocks = captured[0][1]["messages"][0]["content"]
    assert openai_blocks[0] == {"type": "text", "text": "Describe ECG."}
    assert openai_blocks[1]["type"] == "image_url"
    assert openai_blocks[1]["image_url"]["url"] == "data:image/png;base64,AAAA"

    anth_blocks = captured[1][1]["messages"][0]["content"]
    assert anth_blocks[1]["type"] == "image"
    assert anth_blocks[1]["source"]["type"] == "base64"
    assert anth_blocks[1]["source"]["media_type"] == "image/png"
    assert anth_blocks[1]["source"]["data"] == "AAAA"

    gem_parts = captured[2][1]["contents"][0]["parts"]
    assert gem_parts[0] == {"text": "Describe ECG."}
    assert gem_parts[1]["inlineData"] == {"mimeType": "image/png", "data": "AAAA"}
