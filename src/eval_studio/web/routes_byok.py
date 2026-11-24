"""BYOK credential management, provider presets, cache stats, and connection probes."""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from eval_studio.core.schema import EvalRequest, OutputType
from eval_studio.gateway.http import HttpBYOKGateway
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import (
    SecretRole,
    clear_session_key,
    delete_keyring_key,
    get_key_status,
    redact_secrets,
    save_keyring_key,
    set_session_key,
)
from eval_studio.web.schemas import (
    ByokPresetsResponse,
    ByokStatusResponse,
    CacheStats,
    KeyActionResponse,
    ModelPreset,
    ProviderPreset,
    RoleKeyStatus,
    SaveKeyRequest,
    TestConnectionRequest,
    TestConnectionResponse,
)
from eval_studio.web.server import AppState, get_studio_state

router = APIRouter(prefix="/api/byok", tags=["byok"])

StateDep = Annotated[AppState, Depends(get_studio_state)]

_PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        id="deepseek",
        name="DeepSeek Direct",
        default_base_url="https://api.deepseek.com",
        env_var="DEEPSEEK_API_KEY",
        auth_header="Authorization: Bearer",
        models=[
            ModelPreset(
                id="deepseek-flash",
                label="DeepSeek Flash",
                supports_thinking=True,
                input_usd_per_1m=0.15,
                output_usd_per_1m=0.60,
            ),
            ModelPreset(
                id="deepseek-chat",
                label="DeepSeek Chat (V3)",
                supports_thinking=False,
                input_usd_per_1m=0.27,
                output_usd_per_1m=1.10,
            ),
            ModelPreset(
                id="deepseek-reasoner",
                label="DeepSeek Reasoner (R1)",
                supports_thinking=True,
                input_usd_per_1m=0.55,
                output_usd_per_1m=2.19,
            ),
        ],
    ),
    ProviderPreset(
        id="sarvam",
        name="Sarvam AI V2",
        default_base_url="https://api.sarvam.ai/v2",
        env_var="SARVAM_API_KEY",
        auth_header="api-subscription-key",
        models=[
            ModelPreset(
                id="glm5.3-flash",
                label="GLM 5.3 Flash",
                supports_thinking=True,
                input_usd_per_1m=0.16,
                output_usd_per_1m=0.54,
            ),
            ModelPreset(
                id="sarvam-m",
                label="Sarvam M",
                supports_thinking=False,
                input_usd_per_1m=0.16,
                output_usd_per_1m=0.54,
            ),
        ],
    ),
    ProviderPreset(
        id="openai",
        name="OpenAI",
        default_base_url="https://api.openai.com/v1",
        env_var="OPENAI_API_KEY",
        auth_header="Authorization: Bearer",
        models=[
            ModelPreset(
                id="gpt-4o-mini",
                label="GPT-4o Mini",
                supports_thinking=False,
                input_usd_per_1m=0.15,
                output_usd_per_1m=0.60,
            ),
            ModelPreset(
                id="gpt-4o",
                label="GPT-4o",
                supports_thinking=False,
                input_usd_per_1m=2.50,
                output_usd_per_1m=10.00,
            ),
            ModelPreset(
                id="o3-mini",
                label="o3 Mini",
                supports_thinking=True,
                input_usd_per_1m=1.10,
                output_usd_per_1m=4.40,
            ),
        ],
    ),
    ProviderPreset(
        id="anthropic",
        name="Anthropic",
        default_base_url="https://api.anthropic.com/v1",
        env_var="ANTHROPIC_API_KEY",
        auth_header="x-api-key",
        models=[
            ModelPreset(
                id="claude-3-5-sonnet-latest",
                label="Claude 3.5 Sonnet",
                supports_thinking=False,
                input_usd_per_1m=3.00,
                output_usd_per_1m=15.00,
            ),
            ModelPreset(
                id="claude-3-5-haiku-latest",
                label="Claude 3.5 Haiku",
                supports_thinking=False,
                input_usd_per_1m=0.80,
                output_usd_per_1m=4.00,
            ),
            ModelPreset(
                id="claude-3-7-sonnet-latest",
                label="Claude 3.7 Sonnet",
                supports_thinking=True,
                input_usd_per_1m=3.00,
                output_usd_per_1m=15.00,
            ),
        ],
    ),
    ProviderPreset(
        id="gemini",
        name="Google Gemini",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
        env_var="GEMINI_API_KEY",
        auth_header="x-goog-api-key",
        models=[
            ModelPreset(
                id="gemini-2.5-flash",
                label="Gemini 2.5 Flash",
                supports_thinking=True,
                input_usd_per_1m=0.15,
                output_usd_per_1m=0.60,
            ),
            ModelPreset(
                id="gemini-2.5-pro",
                label="Gemini 2.5 Pro",
                supports_thinking=True,
                input_usd_per_1m=1.25,
                output_usd_per_1m=10.00,
            ),
        ],
    ),
    ProviderPreset(
        id="openrouter",
        name="OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        env_var="OPENROUTER_API_KEY",
        auth_header="Authorization: Bearer",
        models=[
            ModelPreset(
                id="openrouter/auto",
                label="OpenRouter Auto",
                supports_thinking=False,
                input_usd_per_1m=0.0,
                output_usd_per_1m=0.0,
            ),
            ModelPreset(
                id="deepseek/deepseek-r1",
                label="DeepSeek R1 (OpenRouter)",
                supports_thinking=True,
                input_usd_per_1m=0.55,
                output_usd_per_1m=2.19,
            ),
        ],
    ),
    ProviderPreset(
        id="litellm",
        name="LiteLLM Proxy",
        default_base_url="http://localhost:4000/v1",
        env_var="OPENAI_API_KEY",
        auth_header="Authorization: Bearer",
        models=[
            ModelPreset(
                id="litellm/gpt-4o-mini",
                label="LiteLLM GPT-4o Mini",
                supports_thinking=False,
                input_usd_per_1m=0.15,
                output_usd_per_1m=0.60,
            ),
            ModelPreset(
                id="litellm/gpt-4o",
                label="LiteLLM GPT-4o",
                supports_thinking=False,
                input_usd_per_1m=2.50,
                output_usd_per_1m=10.00,
            ),
        ],
    ),
    ProviderPreset(
        id="ollama",
        name="Ollama (Local)",
        default_base_url="http://localhost:11434/v1",
        env_var=None,
        auth_header="Authorization: Bearer (optional)",
        models=[
            ModelPreset(
                id="llama3.1:8b",
                label="Llama 3.1 8B",
                supports_thinking=False,
                input_usd_per_1m=0.0,
                output_usd_per_1m=0.0,
            ),
            ModelPreset(
                id="qwen2.5:7b",
                label="Qwen 2.5 7B",
                supports_thinking=False,
                input_usd_per_1m=0.0,
                output_usd_per_1m=0.0,
            ),
            ModelPreset(
                id="deepseek-r1:8b",
                label="DeepSeek R1 8B",
                supports_thinking=True,
                input_usd_per_1m=0.0,
                output_usd_per_1m=0.0,
            ),
        ],
    ),
)


@router.get("/presets", response_model=ByokPresetsResponse)
def get_presets() -> ByokPresetsResponse:
    """Return built-in BYOK provider and model preset catalog."""
    return ByokPresetsResponse(presets=list(_PROVIDER_PRESETS))


@router.get("/status", response_model=ByokStatusResponse)
def get_status(state: StateDep) -> ByokStatusResponse:
    """Return masked role key status and local cache statistics."""
    roles = [RoleKeyStatus.model_validate(get_key_status(role)) for role in SecretRole]
    cache_dir = state.effective_cache_dir()
    exists = cache_dir.is_dir()
    db_bytes = 0
    wal_entries = 0
    if exists:
        db_bytes = sum(p.stat().st_size for p in cache_dir.glob("*.sqlite3") if p.is_file())
        wal_path = cache_dir / "wal.jsonl"
        if wal_path.is_file():
            try:
                wal_entries = sum(
                    1
                    for line in wal_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            except OSError:
                wal_entries = 0
    return ByokStatusResponse(
        roles=roles,
        cache_dir=str(cache_dir),
        cache_stats=CacheStats(exists=exists, db_bytes=db_bytes, wal_entries=wal_entries),
    )


@router.post("/keys", response_model=KeyActionResponse)
def save_key(body: SaveKeyRequest) -> KeyActionResponse:
    """Save an API key in session memory or OS keyring without ever echoing the raw key."""
    cleaned = body.api_key.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="API key value must be non-empty.")

    if body.storage == "session":
        set_session_key(body.role, cleaned)
    else:
        saved = save_keyring_key(body.role, cleaned)
        if not saved:
            raise HTTPException(status_code=503, detail="OS keyring backend is unavailable.")

    return KeyActionResponse.model_validate(get_key_status(body.role))


@router.delete("/keys/{role}", response_model=KeyActionResponse)
def delete_key(
    role: SecretRole,
    storage: Annotated[Literal["session", "keyring", "all"], Query()] = "all",
) -> KeyActionResponse:
    """Clear a role's API key from session memory, OS keyring, or both."""
    if storage in ("session", "all"):
        clear_session_key(role)
    if storage in ("keyring", "all"):
        delete_keyring_key(role)
    return KeyActionResponse.model_validate(get_key_status(role))


@router.post("/test", response_model=TestConnectionResponse)
async def test_connection(body: TestConnectionRequest) -> TestConnectionResponse:
    """Ping a mock or HTTP BYOK gateway and return redacted latency/reply diagnostics."""
    req = EvalRequest(
        request_id=f"ping-{uuid.uuid4().hex[:8]}",
        sample_id="ping",
        task_name="byok_test",
        output_type=OutputType.GENERATE_UNTIL,
        context="Reply with OK.",
        generation_kwargs={"max_tokens": 16, "max_gen_toks": 16, "temperature": 0.0},
    )
    start = time.perf_counter()
    try:
        if body.backend == "mock":
            mock_gw = MockGateway(model_id=body.model_id, default_reply="OK")
            responses = mock_gw.generate_until([req])
        else:
            http_gw = HttpBYOKGateway(
                model_id=body.model_id,
                base_url=body.base_url,
                provider=body.provider,
                role=body.role,
                timeout_seconds=20.0,
                max_retries=1,
            )
            responses = await http_gw.agenerate_until([req])
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        raw_reply = (responses[0].text or "").strip() if responses else ""
        preview = redact_secrets(raw_reply)[:200]
        return TestConnectionResponse(
            ok=True,
            latency_ms=elapsed_ms,
            model_id=body.model_id,
            reply_preview=preview,
            error=None,
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return TestConnectionResponse(
            ok=False,
            latency_ms=elapsed_ms,
            model_id=body.model_id,
            reply_preview="",
            error=redact_secrets(str(exc)),
        )
