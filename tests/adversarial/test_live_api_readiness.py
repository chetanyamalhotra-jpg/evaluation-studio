"""Offline mock-transport tests for live-API readiness (DeepSeek, Sarvam V2, batching, BYOP)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from eval_studio.core.config import EvalRunConfig
from eval_studio.core.runner import run_evaluation
from eval_studio.core.schema import EvalRequest, OutputType
from eval_studio.gateway.http import HttpBYOKGateway
from eval_studio.gateway.secrets import (
    REDACTED_PLACEHOLDER,
    SecretRole,
    contains_secret,
    load_api_key,
    redact_secrets,
)
from eval_studio.paper.extract import PaperExtractor


def test_deepseek_direct_dialect_kwargs_and_reasoning_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepSeek Direct: URL/Bearer/kwargs forwarding, max_tokens=2048 default, truncation."""
    monkeypatch.delenv("TARGET_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-livecheck1234567890")

    captured: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    class _DeepSeekTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            captured.append((str(request.url), dict(request.headers), body))
            if body["messages"][0]["content"] == "truncate-me":
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {
                                    "content": None,
                                    "reasoning_content": "Long clinical differential...",
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 2048,
                            "completion_tokens_details": {"reasoning_tokens": 2048},
                        },
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "Pancreas",
                                "reasoning_content": "Insulin is produced by beta cells.",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 15,
                        "completion_tokens": 42,
                        "output_tokens_details": {"reasoning_tokens": 35},
                    },
                },
            )

    gw = HttpBYOKGateway(model_id="deepseek-flash", transport=_DeepSeekTransport())
    assert gw.provider == "deepseek"
    assert gw.base_url == "https://api.deepseek.com"

    req_ok = EvalRequest(
        request_id="ds-1",
        sample_id="s-1",
        task_name="ds_test",
        output_type=OutputType.GENERATE_UNTIL,
        context="Which organ produces insulin?",
        generation_kwargs={
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "top_p": 0.95,
            "stop": ["\n\n"],
            "seed": 42,
            "response_format": {"type": "json_object"},
        },
    )
    req_trunc = EvalRequest(
        request_id="ds-2",
        sample_id="s-2",
        task_name="ds_test",
        output_type=OutputType.GENERATE_UNTIL,
        context="truncate-me",
        generation_kwargs={"max_tokens": 4096},
    )

    r_ok, r_trunc = gw.generate_until([req_ok, req_trunc])

    url0, hdrs0, body0 = captured[0]
    assert url0 == "https://api.deepseek.com/chat/completions"
    assert "sk-deepseek-livecheck1234567890" in hdrs0.get("authorization", "")
    assert body0["max_tokens"] == 2048
    assert body0["thinking"] == {"type": "enabled"}
    assert body0["reasoning_effort"] == "high"
    assert body0["top_p"] == pytest.approx(0.95)
    assert body0["stop"] == ["\n\n"]
    assert body0["seed"] == 42
    assert body0["response_format"] == {"type": "json_object"}

    assert r_ok.text == "Pancreas"
    assert r_ok.reasoning_trace == "Insulin is produced by beta cells."
    assert r_ok.usage.reasoning_tokens == 35
    assert r_ok.thinking_truncated is False
    assert r_ok.metadata["finish_reason"] == "stop"

    assert captured[1][2]["max_tokens"] == 4096
    assert r_trunc.text == ""
    assert r_trunc.reasoning_trace == "Long clinical differential..."
    assert r_trunc.usage.reasoning_tokens == 2048
    assert r_trunc.thinking_truncated is True
    assert r_trunc.metadata["finish_reason"] == "length"


def test_sarvam_v2_dialect_header_and_secret_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sarvam V2: api-subscription-key header, sk_ redaction, and .env-loaded secret redaction."""
    monkeypatch.delenv("TARGET_API_KEY", raising=False)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    sarvam_key = "sk_sarvam_live_secret_9876543210"
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(f"SARVAM_API_KEY={sarvam_key}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    captured_headers: list[dict[str, str]] = []
    captured_urls: list[str] = []

    class _SarvamTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            captured_headers.append({k.lower(): v for k, v in request.headers.items()})
            return httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": "A"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            )

    gw = HttpBYOKGateway(model_id="glm5.3-flash", transport=_SarvamTransport())
    assert gw.provider == "sarvam"
    assert gw.base_url == "https://api.sarvam.ai/v2"

    req = EvalRequest(
        request_id="sv-1",
        sample_id="s-1",
        task_name="sv_test",
        output_type=OutputType.GENERATE_UNTIL,
        context="Hello",
    )
    resp = gw.generate_until([req])[0]
    assert resp.text == "A"
    assert captured_urls[0] == "https://api.sarvam.ai/v2/chat/completions"
    assert captured_headers[0].get("api-subscription-key") == sarvam_key
    assert "authorization" not in captured_headers[0]

    # Redaction covers sk_..., api-subscription-key header, and runtime-loaded .env secrets
    custom_key = "custom-sarvam-token-xyz123"
    dotenv_file.write_text(f"SARVAM_API_KEY={custom_key}\n", encoding="utf-8")
    loaded = load_api_key(SecretRole.TARGET, fallback_env="SARVAM_API_KEY", dotenv_path=dotenv_file)
    assert loaded == custom_key

    assert contains_secret(f"key={sarvam_key}") is True
    assert contains_secret("api-subscription-key: some_token_value") is True
    assert sarvam_key not in redact_secrets(f"leaked {sarvam_key}")
    assert custom_key not in redact_secrets(f"leaked {custom_key}")
    assert REDACTED_PLACEHOLDER in redact_secrets("Header api-subscription-key: secret_abc_123")


def test_run_evaluation_batched_concurrency_and_dtype_cache_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_evaluation batches misses across samples (exercising Semaphore) & invalidates dtype."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-batch-test1234567890")

    csv_path = tmp_path / "batch6.csv"
    rows = ["sample_id,input_text,target"] + [
        f's{i},"Question number {i}?",ans_{i}' for i in range(6)
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    task_yaml = tmp_path / "batch_task.yaml"
    task_yaml.write_text(
        f"""name: batch_conc_task
trust_tier: tier1_zero_code
output_type: generate_until
dataset_path: "{csv_path.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
""",
        encoding="utf-8",
    )

    state = {"active": 0, "peak": 0, "calls": 0}

    class _ConcurrentTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["calls"] += 1
            await asyncio.sleep(0.02)
            state["active"] -= 1
            body = json.loads(request.content.decode("utf-8"))
            q = body["messages"][0]["content"]
            idx_str = q.split()[-1].rstrip("?")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": f"ans_{idx_str}"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                },
            )

    gw = HttpBYOKGateway(
        model_id="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        max_concurrent=3,
        transport=_ConcurrentTransport(),
    )
    cache_dir = tmp_path / "cache"

    cfg_bf16 = EvalRunConfig(
        model_backend="http-byok",
        model_id="gpt-4o-mini",
        dtype="bf16",
        quantization="none",
        cache_dir=cache_dir,
        bootstrap_iters=50,
    )
    m1 = run_evaluation(task_yaml, config=cfg_bf16, output_dir=tmp_path / "out1", gateway=gw)
    assert m1.metrics["exact_match"].value == pytest.approx(1.0)
    assert state["calls"] == 6
    assert 2 <= state["peak"] <= 3

    # Repeat with same dtype/quantization -> 100% cache hit (0 new HTTP calls)
    run_evaluation(task_yaml, config=cfg_bf16, output_dir=tmp_path / "out2", gateway=gw)
    assert state["calls"] == 6

    # Change dtype to fp8 -> cache invalidates (6 new HTTP calls)
    cfg_fp8 = EvalRunConfig(
        model_backend="http-byok",
        model_id="gpt-4o-mini",
        dtype="fp8",
        quantization="none",
        cache_dir=cache_dir,
        bootstrap_iters=50,
    )
    run_evaluation(task_yaml, config=cfg_fp8, output_dir=tmp_path / "out3", gateway=gw)
    assert state["calls"] == 12


def test_doc_to_messages_multimodal_passthrough_in_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 2 CHAT task with doc_to_messages passes rendered ImageContentPart to HttpBYOKGateway."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-mm-test1234567890")

    jsonl_path = tmp_path / "mm.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "sample_id": "mm1",
                "input_text": "Identify the rhythm.",
                "target": "Sinus",
                "img_data": "data:image/png;base64,BBBB",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task_yaml = tmp_path / "mm_task.yaml"
    task_yaml.write_text(
        f"""name: mm_chat_task
trust_tier: tier2_declarative
output_type: chat
dataset_path: "{jsonl_path.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
doc_to_messages:
  - role: user
    content:
      - type: text
        text: "ECG prompt: {{{{ input_text }}}}"
      - type: image_url
        image_url:
          url: "{{{{ img_data }}}}"
metrics: [exact_match]
""",
        encoding="utf-8",
    )

    seen_payloads: list[dict[str, Any]] = []

    class _MMTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Sinus"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                },
            )

    gw = HttpBYOKGateway(
        model_id="gpt-4o",
        base_url="https://api.openai.com/v1",
        transport=_MMTransport(),
    )
    cfg = EvalRunConfig(
        model_backend="http-byok",
        model_id="gpt-4o",
        use_cache=False,
        bootstrap_iters=0,
    )
    manifest = run_evaluation(task_yaml, config=cfg, output_dir=tmp_path / "mm_out", gateway=gw)
    assert manifest.metrics["exact_match"].value == pytest.approx(1.0)
    content = seen_payloads[0]["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "ECG prompt: Identify the rhythm."}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,BBBB"},
    }


def test_paper_extractor_fenced_json_and_multi_chunk_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PaperExtractor chunks >12k-char papers and strips markdown ```json``` fences."""
    monkeypatch.setenv("EXTRACTOR_API_KEY", "sk-ext-chunk1234567890")

    p1 = "[PAGE 1]\n# Section 1\n" + ("Clinical background sentence one. " * 220)
    p2 = "[PAGE 2]\n# Section 2\n" + ("Cohort achieved 97.5% specificity in validation. " * 220)
    paper = tmp_path / "long_paper.md"
    paper.write_text(f"{p1}\n\n{p2}\n", encoding="utf-8")
    assert len(paper.read_text(encoding="utf-8")) > 15000

    chunk_prompts: list[str] = []

    class _ChunkTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            prompt = body["messages"][0]["content"]
            idx = len(chunk_prompts) + 1
            chunk_prompts.append(prompt)
            fenced = (
                "Here are the extracted items:\n"
                "```json\n"
                + json.dumps(
                    {
                        "items": [
                            {
                                "item_id": f"chunk-{idx}-item-1",
                                "question": f"What was reported in chunk {idx}?",
                                "reference_answer": "97.5%" if idx > 1 else "background",
                                "rubric": "Match citation.",
                                "category": "quantitative_finding",
                                "citations": [
                                    {
                                        "page": idx,
                                        "section": f"Section {idx}",
                                        "quote": (
                                            "Cohort achieved 97.5% specificity in validation."
                                            if idx > 1
                                            else "Clinical background sentence one."
                                        ),
                                    }
                                ],
                            }
                        ]
                    }
                )
                + "\n```"
            )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": fenced}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                },
            )

    gw = HttpBYOKGateway(
        model_id="deepseek-flash",
        role=SecretRole.EXTRACTOR,
        transport=_ChunkTransport(),
    )
    extractor = PaperExtractor(extractor_model="deepseek-flash", gateway=gw, chunk_max_chars=12000)
    items = extractor.extract_from_path(paper)

    assert len(chunk_prompts) >= 2
    assert all('"citations"' in p for p in chunk_prompts)
    assert len(items) == len(chunk_prompts)
    assert items[0].item_id == "chunk-1-item-1"
    assert items[-1].citations[0].page == len(chunk_prompts)
