"""Tier 1 E2E tests: BYOK gateways, multi-key routing, cost caps & secret hygiene (F13–F17)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import keyring
import pytest
from typer.testing import CliRunner

from eval_studio.cache.wal import JsonlWriteAheadLog
from eval_studio.cli.app import app
from eval_studio.core.schema import ChatMessage, EvalRequest, OutputType
from eval_studio.gateway.adapters.hf import HuggingFaceAdapter
from eval_studio.gateway.adapters.vllm import VLLMAdapter
from eval_studio.gateway.cost import CostGuard
from eval_studio.gateway.http import HttpBYOKGateway
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import SecretRole, contains_secret, load_api_key, redact_secrets

pytestmark = pytest.mark.e2e


def _bind_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
) -> None:
    """Bind an offline MockTransport onto httpx Client/AsyncClient when none is supplied."""
    orig_async_init = httpx.AsyncClient.__init__
    orig_sync_init = httpx.Client.__init__

    def _patched_async_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("transport") is None:
            kwargs["transport"] = transport
        orig_async_init(self, *args, **kwargs)

    def _patched_sync_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("transport") is None:
            kwargs["transport"] = transport
        orig_sync_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _patched_async_init)
    monkeypatch.setattr(httpx.Client, "__init__", _patched_sync_init)


def test_t1_mock_gateway_modes_and_pricing() -> None:
    """T1-BYOK-01: MockGateway deterministic modes, token pricing, judge & compiler modes."""
    pricing = {"input_usd_per_1m": 2.0, "output_usd_per_1m": 6.0}
    gw = MockGateway(model_id="mock-priced", token_pricing=pricing)

    ll_req = EvalRequest(
        request_id="ll-1",
        sample_id="s-1",
        task_name="byok_mock",
        output_type=OutputType.LOGLIKELIHOOD,
        context="Normal adult heart rate is",
        continuation=" 60-100 bpm",
    )
    gen_req = EvalRequest(
        request_id="gen-1",
        sample_id="s-1",
        task_name="byok_mock",
        output_type=OutputType.GENERATE_UNTIL,
        context="Question: Which organ produces insulin?\nAnswer:",
    )
    chat_req = EvalRequest(
        request_id="chat-1",
        sample_id="s-1",
        task_name="byok_mock",
        output_type=OutputType.CHAT,
        context="Triage chest pain.",
        messages=[ChatMessage(role="user", content="Triage chest pain.")],
    )

    ll_first = gw.loglikelihood([ll_req])[0]
    ll_second = gw.loglikelihood([ll_req])[0]
    assert ll_first.loglikelihood == ll_second.loglikelihood
    gen_r1 = gw.generate_until([gen_req])[0]
    gen_r2 = gw.generate_until([gen_req])[0]
    assert gen_r1.text == gen_r2.text
    assert gen_r1.usage.prompt_tokens > 0
    assert gen_r1.usage.completion_tokens > 0

    chat_r = gw.chat([chat_req])[0]
    assert chat_r.text is not None

    guard = CostGuard()
    est = guard.estimate_batch(
        gen_r1.usage.prompt_tokens,
        gen_r1.usage.completion_tokens,
        input_usd_per_1m=pricing["input_usd_per_1m"],
        output_usd_per_1m=pricing["output_usd_per_1m"],
    )
    expected_usd = (
        gen_r1.usage.prompt_tokens / 1_000_000.0
    ) * 2.0 + (gen_r1.usage.completion_tokens / 1_000_000.0) * 6.0
    assert est.estimated_usd == pytest.approx(expected_usd)

    judge_gw = MockGateway(judge_mode="fail_then_succeed")
    judge_req = EvalRequest(
        request_id="j-1",
        sample_id="s-1",
        task_name="judge_eval",
        output_type=OutputType.GENERATE_UNTIL,
        context="Grade clinical answer.",
    )
    first_judge = judge_gw.generate_until([judge_req])[0]
    second_judge = judge_gw.generate_until([judge_req])[0]
    assert "sorry" in (first_judge.text or "").casefold() or first_judge.text != second_judge.text
    parsed_second = json.loads(second_judge.text or "{}")
    assert "score" in parsed_second

    comp_gw = MockGateway(compiler_mode="mixed")
    comp_req = EvalRequest(
        request_id="c-1",
        sample_id="paper-1",
        task_name="compile_paper",
        output_type=OutputType.GENERATE_UNTIL,
        context="Extract benchmark items from clinical study.",
    )
    comp_resp = comp_gw.generate_until([comp_req])[0]
    comp_payload = json.loads(comp_resp.text or "{}")
    items = comp_payload if isinstance(comp_payload, list) else comp_payload.get("items", [])
    assert len(items) >= 2
    assert any(item.get("citations") for item in items)
    assert any(not item.get("citations") for item in items)


def test_t1_http_byok_gateway_all_providers_asgi_mock(
    make_mock_transport: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-BYOK-02: HttpBYOKGateway formats 6 provider dialects and retries 429 offline."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-mocktest1234567890")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-mocktest1234567890")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-mocktest1234567890")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyMockTest1234567890abcdef")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-mocktest1234567890abcdef")

    dialects: list[tuple[str, str, str]] = [
        ("openai", "gpt-4o-mini", "https://api.openai.com/v1"),
        ("anthropic", "claude-3-5-sonnet", "https://api.anthropic.com/v1"),
        ("gemini", "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta"),
        ("openrouter", "openrouter/auto", "https://openrouter.ai/api/v1"),
        ("litellm", "litellm/gpt-4o", "http://localhost:4000/v1"),
        ("ollama", "llama3.1:8b", "http://localhost:11434/v1"),
    ]

    for provider, model_id, base_url in dialects:
        attempts = {"count": 0}

        def _handler(
            request: httpx.Request,
            recorded: list[dict[str, Any]],
            *,
            prov: str = provider,
            ctr: dict[str, int] = attempts,
        ) -> httpx.Response:
            ctr["count"] += 1
            if prov == "openai" and ctr["count"] == 1:
                return httpx.Response(429, json={"error": "rate_limited"})
            if prov == "anthropic":
                return httpx.Response(
                    200,
                    json={
                        "content": [{"type": "text", "text": "Pancreas"}],
                        "usage": {"input_tokens": 11, "output_tokens": 3},
                    },
                )
            if prov == "gemini":
                return httpx.Response(
                    200,
                    json={
                        "candidates": [{"content": {"parts": [{"text": "Pancreas"}]}}],
                        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Pancreas"}, "text": "Pancreas"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 3},
                },
            )

        transport, recorded = make_mock_transport(_handler)
        _bind_mock_transport(monkeypatch, transport)
        gw = HttpBYOKGateway(
            model_id=model_id,
            base_url=base_url,
            role=SecretRole.TARGET,
            max_retries=2,
        )
        req = EvalRequest(
            request_id=f"req-{provider}",
            sample_id="s-1",
            task_name="byok_http",
            output_type=OutputType.GENERATE_UNTIL,
            context="Which organ produces insulin?",
        )
        resps = gw.generate_until([req])
        assert len(resps) == 1
        assert resps[0].text == "Pancreas"
        assert resps[0].usage.prompt_tokens == 11
        assert resps[0].usage.completion_tokens == 3
        if provider == "openai":
            assert len(recorded) == 2
        else:
            assert len(recorded) == 1

        final_req = recorded[-1]
        req_url = final_req["url"]
        req_headers = {k.lower(): v for k, v in final_req["headers"].items()}
        req_json = final_req["json"]
        assert isinstance(req_json, dict)

        if provider == "openai":
            assert "api.openai.com" in req_url
            assert req_url.endswith(("/chat/completions", "/completions"))
            assert "sk-target-mocktest1234567890" in req_headers.get("authorization", "")
            assert req_json.get("model") == "gpt-4o-mini"
            assert "messages" in req_json or "prompt" in req_json
        elif provider == "anthropic":
            assert "api.anthropic.com" in req_url
            assert "/messages" in req_url
            ant_key = req_headers.get("x-api-key") or req_headers.get("authorization", "")
            assert any(
                k in ant_key
                for k in ("sk-target-mocktest1234567890", "sk-ant-mocktest1234567890")
            )
            assert req_json.get("model") == "claude-3-5-sonnet"
            assert "messages" in req_json
        elif provider == "gemini":
            assert "generativelanguage.googleapis.com" in req_url
            assert "gemini-2.5-flash" in req_url
            assert "generateContent" in req_url
            gem_auth = (
                req_url
                + " "
                + req_headers.get("x-goog-api-key", "")
                + " "
                + req_headers.get("authorization", "")
            )
            assert any(
                k in gem_auth
                for k in ("sk-target-mocktest1234567890", "AIzaSyMockTest1234567890abcdef")
            )
            assert "contents" in req_json
        elif provider == "openrouter":
            assert "openrouter.ai" in req_url
            assert "/chat/completions" in req_url
            or_auth = req_headers.get("authorization", "")
            assert any(
                k in or_auth
                for k in ("sk-target-mocktest1234567890", "sk-or-v1-mocktest1234567890abcdef")
            )
            assert req_json.get("model") == "openrouter/auto"
            assert "messages" in req_json
        elif provider == "litellm":
            assert "localhost:4000" in req_url
            assert "/chat/completions" in req_url
            assert req_json.get("model") == "litellm/gpt-4o"
            assert "messages" in req_json
        elif provider == "ollama":
            assert "localhost:11434" in req_url
            assert any(ep in req_url for ep in ("/chat/completions", "/api/chat", "/api/generate"))
            assert req_json.get("model") == "llama3.1:8b"
            assert "messages" in req_json or "prompt" in req_json


def test_t1_role_separated_keys_and_env_keyring_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_mock_transport: Any,
    cli_runner: CliRunner,
) -> None:
    """T1-BYOK-03: Role-separated keys (env, .env, keyring) and zero raw secrets in artifacts."""
    target_secret = "sk-target-aaaaaaaaaaaaaaaa"
    judge_secret = "sk-judge-bbbbbbbbbbbbbbbb"
    extractor_secret = "sk-extract-cccccccccccccccc"
    fallback_secret = "sk-fallback-dddddddddddddddd"
    keyring_secret = "sk-keyring-eeeeeeeeeeeeeeee"
    dotenv_secret = "sk-dotenv-ffffffffffffffff"

    monkeypatch.setenv("TARGET_API_KEY", target_secret)
    monkeypatch.setenv("JUDGE_API_KEY", judge_secret)
    monkeypatch.setenv("EXTRACTOR_API_KEY", extractor_secret)
    monkeypatch.setenv("FALLBACK_JUDGE_API_KEY", fallback_secret)

    assert load_api_key(SecretRole.TARGET) == target_secret
    assert load_api_key(SecretRole.JUDGE) == judge_secret
    assert load_api_key(SecretRole.EXTRACTOR) == extractor_secret
    assert load_api_key(SecretRole.FALLBACK_JUDGE) == fallback_secret

    # Keyring fallback when env var is cleared
    monkeypatch.delenv("TARGET_API_KEY", raising=False)
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda service, username: keyring_secret if username == "TARGET_API_KEY" else None,
    )
    assert load_api_key(SecretRole.TARGET) == keyring_secret

    # F16: .env loading when env var and keyring entry are absent
    monkeypatch.setattr(keyring, "get_password", lambda service, username: None)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(f"TARGET_API_KEY={dotenv_secret}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_api_key(SecretRole.TARGET) == dotenv_secret
    dotenv_path.unlink()

    # Restore env for transport routing checks
    monkeypatch.setenv("TARGET_API_KEY", target_secret)

    for role, expected_secret in (
        (SecretRole.TARGET, target_secret),
        (SecretRole.JUDGE, judge_secret),
        (SecretRole.EXTRACTOR, extractor_secret),
    ):
        def _ok_handler(request: httpx.Request, recorded: list[dict[str, Any]]) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "A"}, "text": "A"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            )

        transport, recorded = make_mock_transport(_ok_handler)
        _bind_mock_transport(monkeypatch, transport)
        gw = HttpBYOKGateway(
            model_id="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            role=role,
        )
        gw.generate_until(
            [
                EvalRequest(
                    request_id=f"req-{role.name}",
                    sample_id="s1",
                    task_name="role_check",
                    output_type=OutputType.GENERATE_UNTIL,
                    context="Hello",
                )
            ]
        )
        assert len(recorded) == 1
        assert "api.openai.com" in recorded[0]["url"]
        req_headers = {k.lower(): v for k, v in recorded[0]["headers"].items()}
        assert expected_secret in req_headers.get("authorization", "")
        assert recorded[0]["json"]["model"] == "gpt-4o-mini"

    # Verify WAL & receipt bundle redact secrets completely
    wal_path = tmp_path / "wal.jsonl"
    wal = JsonlWriteAheadLog(wal_path)
    wal.append({"error": f"Auth failed with Bearer {target_secret}", "role": "target"})
    wal_text = wal_path.read_text(encoding="utf-8")
    assert target_secret not in wal_text
    assert not contains_secret(wal_text)
    assert "[REDACTED_SECRET]" in redact_secrets(f"key={judge_secret}")

    csv_path = tmp_path / "s.csv"
    csv_path.write_text('sample_id,input_text,target\ns1,"2+2?",A\n', encoding="utf-8")
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        f"""name: secret_redaction_check
trust_tier: tier1_zero_code
output_type: generate_until
dataset_path: "{csv_path.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "receipt_secrets"
    res = cli_runner.invoke(
        app,
        ["run", "--task", str(task_path), "--model", "mock", "--output", str(out_dir)],
    )
    assert res.exit_code == 0, res.stdout
    for artifact in ("samples.jsonl", "manifest.json", "receipt.json", "report.md", "report.html"):
        content = (out_dir / artifact).read_text(encoding="utf-8")
        for secret in (target_secret, judge_secret, extractor_secret, fallback_secret):
            assert secret not in content
        assert not contains_secret(content)


def test_t1_dry_run_estimate_cost_target_plus_judge(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-BYOK-04: `--dry-run --estimate-cost` computes target*repeats + judge estimate."""
    monkeypatch.setenv("TARGET_API_KEY", "sk-target-dryrun1234567890")
    monkeypatch.setenv("JUDGE_API_KEY", "sk-judge-dryrun1234567890")

    csv_path = tmp_path / "dryrun.csv"
    csv_path.write_text(
        "sample_id,input_text,target\n"
        's1,"Summarize troponin biomarker kinetics in acute MI.",99.6%\n'
        's2,"State normal resting adult heart rate range.",60-100\n',
        encoding="utf-8",
    )
    task_path = tmp_path / "dryrun_task.yaml"
    task_path.write_text(
        f"""name: t1_dryrun_cost
trust_tier: tier2_declarative
output_type: generate_until
dataset_path: "{csv_path.as_posix()}"
doc_to_text: "Question: {{{{ input_text }}}}\\nAnswer:"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
generation_kwargs:
  repeats: 3
  max_gen_toks: 128
metadata:
  judge_enabled: true
  pricing:
    input_usd_per_1m: 2.50
    output_usd_per_1m: 10.00
    judge_input_usd_per_1m: 5.00
    judge_output_usd_per_1m: 15.00
""",
        encoding="utf-8",
    )

    called = {"http": False}

    def _forbid_send(*args: Any, **kwargs: Any) -> Any:
        called["http"] = True
        raise AssertionError("Dry-run must not dispatch HTTP inference requests")

    monkeypatch.setattr(httpx.Client, "send", _forbid_send, raising=False)
    monkeypatch.setattr(httpx.AsyncClient, "send", _forbid_send, raising=False)

    result = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_path),
            "--model",
            "http-byok",
            "--dry-run",
            "--estimate-cost",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert called["http"] is False
    out_lower = result.stdout.casefold()
    assert "usd" in out_lower or "$" in result.stdout
    assert "judge" in out_lower
    assert "target" in out_lower


def test_t1_optional_hf_vllm_lazy_degradation() -> None:
    """T1-BYOK-05: Optional [hf]/[vllm] adapters import lazily and degrade cleanly offline."""
    hf = HuggingFaceAdapter("meta-llama/Llama-3.1-8B-Instruct")
    vllm = VLLMAdapter("meta-llama/Llama-3.1-8B-Instruct")

    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "vllm" not in sys.modules

    req = EvalRequest(
        request_id="opt-1",
        sample_id="s-1",
        task_name="lazy_degrade",
        output_type=OutputType.GENERATE_UNTIL,
        context="Hello",
    )

    for adapter, extra_tag in ((hf, "evaluation-studio[hf]"), (vllm, "evaluation-studio[vllm]")):
        for method_name in ("loglikelihood", "loglikelihood_rolling", "generate_until"):
            with pytest.raises((ImportError, NotImplementedError)) as exc_info:
                getattr(adapter, method_name)([req])
            assert extra_tag in str(exc_info.value)
