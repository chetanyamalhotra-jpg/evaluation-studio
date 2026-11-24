"""Unified async/sync HTTP BYOK gateway supporting OpenAI, Anthropic, Gemini, OpenRouter,
LiteLLM, and local OpenAI-compatible servers (Ollama, LM Studio, vLLM)."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import mimetypes
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from eval_studio.core.schema import (
    ChatMessage,
    EvalRequest,
    EvalResponse,
    ImageContentPart,
    TextContentPart,
    TokenUsage,
)
from eval_studio.gateway.cost import CostGuard
from eval_studio.gateway.secrets import SecretRole, load_api_key, redact_secrets

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


class _RetryableHttpError(RuntimeError):
    """Internal retryable HTTP status error."""


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (_RetryableHttpError, httpx.TransportError, httpx.TimeoutException))


def _infer_provider(base_url: str | None, model_id: str, explicit_provider: str | None) -> str:
    if explicit_provider:
        return explicit_provider.lower()
    url = (base_url or "").lower()
    mid = model_id.lower()
    if "api.deepseek.com" in url or "deepseek" in mid:
        return "deepseek"
    if "api.sarvam.ai" in url or "sarvam" in mid or "glm5.3" in mid:
        return "sarvam"
    if "anthropic" in url or mid.startswith("claude"):
        return "anthropic"
    if "generativelanguage.googleapis.com" in url or mid.startswith("gemini"):
        return "gemini"
    if "openrouter.ai" in url or mid.startswith("openrouter/"):
        return "openrouter"
    if "11434" in url or "ollama" in url:
        return "ollama"
    if "litellm" in mid or "4000" in url:
        return "litellm"
    return "openai"


def _provider_fallback_env(provider: str) -> str | None:
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "sarvam": "SARVAM_API_KEY",
        "litellm": "OPENAI_API_KEY",
        "ollama": None,
    }.get(provider)


def _default_base_url(provider: str) -> str:
    return {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "openrouter": "https://openrouter.ai/api/v1",
        "deepseek": "https://api.deepseek.com",
        "sarvam": "https://api.sarvam.ai/v2",
        "litellm": "http://localhost:4000/v1",
        "ollama": "http://localhost:11434/v1",
    }.get(provider, "https://api.openai.com/v1")


def _resolve_image_url_and_mime(part: ImageContentPart) -> tuple[str, str | None]:
    mime = part.mime_type
    if isinstance(part.image_url, str):
        url = part.image_url
    elif isinstance(part.image_url, dict):
        url = str(part.image_url.get("url", ""))
    elif part.path is not None:
        img_path = Path(part.path)
        guessed_mime, _ = mimetypes.guess_type(str(img_path))
        mime = mime or guessed_mime or "image/png"
        if img_path.is_file():
            b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
            url = f"data:{mime};base64,{b64}"
        else:
            url = f"file://{img_path.as_posix()}"
    else:
        url = ""

    if url.startswith("data:") and ";base64," in url and mime is None:
        header, _, _ = url[5:].partition(";base64,")
        if header:
            mime = header
    return url, mime


def _parse_data_url(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, _, b64_data = url[5:].partition(";base64,")
    media_type = header.strip() or "image/png"
    return media_type, b64_data


def _serialize_openai_messages(req: EvalRequest) -> list[dict[str, Any]]:
    msgs = req.messages if req.messages else [ChatMessage(role="user", content=req.context)]
    serialized: list[dict[str, Any]] = []
    for m in msgs:
        if isinstance(m.content, str):
            serialized.append({"role": m.role, "content": m.content})
            continue
        blocks: list[dict[str, Any]] = []
        for part in m.content:
            if isinstance(part, TextContentPart):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContentPart):
                url, _ = _resolve_image_url_and_mime(part)
                img_obj: dict[str, Any] = {"url": url}
                if part.detail:
                    img_obj["detail"] = part.detail
                blocks.append({"type": "image_url", "image_url": img_obj})
        serialized.append({"role": m.role, "content": blocks})
    return serialized


def _serialize_anthropic_messages(req: EvalRequest) -> list[dict[str, Any]]:
    msgs = req.messages if req.messages else [ChatMessage(role="user", content=req.context)]
    serialized: list[dict[str, Any]] = []
    for m in msgs:
        if isinstance(m.content, str):
            serialized.append({"role": m.role, "content": m.content})
            continue
        blocks: list[dict[str, Any]] = []
        for part in m.content:
            if isinstance(part, TextContentPart):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContentPart):
                url, _ = _resolve_image_url_and_mime(part)
                parsed_data = _parse_data_url(url)
                if parsed_data is not None:
                    media_type, b64_data = parsed_data
                    source: dict[str, Any] = {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    }
                else:
                    source = {"type": "url", "url": url}
                blocks.append({"type": "image", "source": source})
        serialized.append({"role": m.role, "content": blocks})
    return serialized


def _serialize_gemini_contents(req: EvalRequest) -> list[dict[str, Any]]:
    msgs = req.messages if req.messages else [ChatMessage(role="user", content=req.context)]
    contents: list[dict[str, Any]] = []
    for m in msgs:
        role = "model" if m.role == "assistant" else "user"
        if isinstance(m.content, str):
            contents.append({"role": role, "parts": [{"text": m.content}]})
            continue
        parts: list[dict[str, Any]] = []
        for part in m.content:
            if isinstance(part, TextContentPart):
                parts.append({"text": part.text})
            elif isinstance(part, ImageContentPart):
                url, mime = _resolve_image_url_and_mime(part)
                parsed_data = _parse_data_url(url)
                if parsed_data is not None:
                    media_type, b64_data = parsed_data
                    parts.append({"inlineData": {"mimeType": media_type, "data": b64_data}})
                else:
                    parts.append(
                        {"fileData": {"mimeType": mime or "image/png", "fileUri": url}}
                    )
        contents.append({"role": role, "parts": parts})
    return contents


def _run_coro_sync(coro: Any) -> Any:
    """Execute an async coroutine safely whether or not an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class HttpBYOKGateway:
    """Unified async/sync HTTP router with role-separated BYOK keys, retries, and cost caps."""

    backend_name: str = "http-byok"

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        provider: str | None = None,
        role: SecretRole = SecretRole.TARGET,
        cost_guard: CostGuard | None = None,
        token_pricing: dict[str, float] | None = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 3,
        max_concurrent: int = 8,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model_id = model_id
        self.provider = _infer_provider(base_url, model_id, provider)
        self.base_url = (base_url or _default_base_url(self.provider)).rstrip("/")
        self.role = role
        self.cost_guard = cost_guard
        self.token_pricing = dict(token_pricing) if token_pricing is not None else {}
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.max_concurrent = max(1, max_concurrent)
        self._transport = transport

    def _resolve_api_key(self) -> str | None:
        return load_api_key(self.role, fallback_env=_provider_fallback_env(self.provider))

    def _build_request(self, req: EvalRequest) -> tuple[str, dict[str, str], dict[str, Any]]:
        api_key = self._resolve_api_key()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        raw_max_toks = req.generation_kwargs.get(
            "max_tokens", req.generation_kwargs.get("max_gen_toks", 2048)
        )
        max_tokens = int(raw_max_toks)
        temperature = float(req.generation_kwargs.get("temperature", 0.0))

        if self.provider == "anthropic":
            url = f"{self.base_url}/messages"
            headers["anthropic-version"] = "2023-06-01"
            if api_key:
                headers["x-api-key"] = api_key
            payload: dict[str, Any] = {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": _serialize_anthropic_messages(req),
            }
            return url, headers, payload

        if self.provider == "gemini":
            url = f"{self.base_url}/models/{self.model_id}:generateContent"
            if api_key:
                headers["x-goog-api-key"] = api_key
            payload = {
                "contents": _serialize_gemini_contents(req),
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
            return url, headers, payload

        # OpenAI, DeepSeek, Sarvam V2, OpenRouter, LiteLLM, Ollama, vLLM, LM Studio
        url = f"{self.base_url}/chat/completions"
        if api_key:
            if self.provider == "sarvam":
                headers["api-subscription-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.model_id,
            "messages": _serialize_openai_messages(req),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for opt_key in (
            "thinking",
            "reasoning_effort",
            "top_p",
            "stop",
            "seed",
            "response_format",
        ):
            opt_val = req.generation_kwargs.get(opt_key)
            if opt_val is not None:
                payload[opt_key] = opt_val
        return url, headers, payload

    def _parse_response(
        self, req: EvalRequest, data: dict[str, Any], latency_ms: float
    ) -> EvalResponse:
        text = ""
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        reasoning_content: str | None = None
        finish_reason: str | None = None
        thinking_truncated = False

        if self.provider == "anthropic":
            content_blocks = data.get("content") or []
            if isinstance(content_blocks, list) and content_blocks:
                first = content_blocks[0]
                if isinstance(first, dict):
                    text = str(first.get("text", ""))
            usage_raw = data.get("usage") or {}
            prompt_tokens = int(usage_raw.get("input_tokens", 0))
            completion_tokens = int(usage_raw.get("output_tokens", 0))
        elif self.provider == "gemini":
            candidates = data.get("candidates") or []
            if isinstance(candidates, list) and candidates:
                parts = (candidates[0].get("content") or {}).get("parts") or []
                if parts and isinstance(parts[0], dict):
                    text = str(parts[0].get("text", ""))
            usage_raw = data.get("usageMetadata") or {}
            prompt_tokens = int(usage_raw.get("promptTokenCount", 0))
            completion_tokens = int(usage_raw.get("candidatesTokenCount", 0))
        else:
            choices = data.get("choices") or []
            raw_content: Any = None
            if isinstance(choices, list) and choices:
                choice0 = choices[0]
                if isinstance(choice0, dict):
                    raw_fr = choice0.get("finish_reason")
                    finish_reason = str(raw_fr) if raw_fr is not None else None
                    msg = choice0.get("message")
                    if isinstance(msg, dict):
                        raw_content = msg.get("content")
                        if raw_content is not None:
                            text = str(raw_content)
                        raw_rc = msg.get("reasoning_content")
                        if isinstance(raw_rc, str) and raw_rc:
                            reasoning_content = raw_rc
                    elif choice0.get("text") is not None:
                        raw_content = choice0.get("text")
                        text = str(raw_content)
            usage_raw = data.get("usage") or {}
            prompt_tokens = int(usage_raw.get("prompt_tokens", 0))
            completion_tokens = int(usage_raw.get("completion_tokens", 0))
            for details_key in ("completion_tokens_details", "output_tokens_details"):
                details = usage_raw.get(details_key)
                if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
                    reasoning_tokens = int(details["reasoning_tokens"])
                    break
            if (
                finish_reason == "length"
                and (raw_content is None or not str(raw_content).strip())
                and reasoning_content
            ):
                thinking_truncated = True

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        in_rate = float(self.token_pricing.get("input_usd_per_1m", 0.0))
        out_rate = float(self.token_pricing.get("output_usd_per_1m", 0.0))
        cost_usd = (usage.prompt_tokens / 1_000_000.0) * in_rate + (
            usage.completion_tokens / 1_000_000.0
        ) * out_rate
        if self.cost_guard is not None and cost_usd > 0.0:
            self.cost_guard.check_and_record(cost_usd)

        resp_meta: dict[str, Any] = {"provider": self.provider, "cost_usd": cost_usd}
        if finish_reason is not None:
            resp_meta["finish_reason"] = finish_reason

        return EvalResponse(
            request_id=req.request_id,
            sample_id=req.sample_id,
            text=text,
            reasoning_trace=reasoning_content,
            thinking_truncated=thinking_truncated,
            usage=usage,
            latency_ms=latency_ms,
            backend=self.backend_name,
            model_id=self.model_id,
            metadata=resp_meta,
        )

    def _precheck_cost(self, req: EvalRequest) -> None:
        if self.cost_guard is None or self.cost_guard.max_cost_usd is None:
            return
        in_rate = float(self.token_pricing.get("input_usd_per_1m", 0.0))
        out_rate = float(self.token_pricing.get("output_usd_per_1m", 0.0))
        if in_rate <= 0.0 and out_rate <= 0.0:
            return
        ctx_text = req.context or (
            " ".join(m.text_content() for m in req.messages) if req.messages else ""
        )
        est_in = max(1, len(ctx_text.split()))
        est_out = int(req.generation_kwargs.get("max_gen_toks", 64))
        est_usd = (est_in / 1_000_000.0) * in_rate + (est_out / 1_000_000.0) * out_rate
        if self.cost_guard.spent_usd + est_usd > self.cost_guard.max_cost_usd:
            self.cost_guard.check_and_record(est_usd)

    async def _dispatch_one(self, client: httpx.AsyncClient, req: EvalRequest) -> EvalResponse:
        self._precheck_cost(req)
        url, headers, payload = self._build_request(req)
        start = time.perf_counter()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=0.01, min=0.01, max=0.5),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                except Exception as exc:
                    raise httpx.TransportError(redact_secrets(str(exc))) from None
                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    raise _RetryableHttpError(
                        redact_secrets(f"HTTP {resp.status_code} from {self.provider}: {resp.text}")
                    )
                if resp.status_code >= 400:
                    raise RuntimeError(
                        redact_secrets(f"HTTP {resp.status_code} from {self.provider}: {resp.text}")
                    )
                data: dict[str, Any] = resp.json()
                latency_ms = (time.perf_counter() - start) * 1000.0
                return self._parse_response(req, data, latency_ms)

        raise RuntimeError(f"Exhausted retries for request {req.request_id}")

    async def agenerate_until(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        if not requests:
            return []
        client_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        # Sequential dispatch when a hard budget cap is active prevents over-spending in flight
        strict_budget = self.cost_guard is not None and self.cost_guard.max_cost_usd is not None
        eff_concurrency = 1 if strict_budget else self.max_concurrent
        async with httpx.AsyncClient(**client_kwargs) as client:
            if eff_concurrency == 1:
                return [await self._dispatch_one(client, req) for req in requests]
            sem = asyncio.Semaphore(eff_concurrency)

            async def _bounded(req: EvalRequest) -> EvalResponse:
                async with sem:
                    return await self._dispatch_one(client, req)

            return list(await asyncio.gather(*(_bounded(req) for req in requests)))

    async def achat(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        return await self.agenerate_until(requests)

    def generate_until(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        result: list[EvalResponse] = _run_coro_sync(self.agenerate_until(requests))
        return result

    def chat(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        return self.generate_until(requests)

    async def _dispatch_loglikelihood(
        self, client: httpx.AsyncClient, req: EvalRequest
    ) -> EvalResponse:
        self._precheck_cost(req)
        api_key = self._resolve_api_key()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{self.base_url}/completions"
        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": f"{req.context}{req.continuation or ''}",
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
        }
        start = time.perf_counter()
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=0.01, min=0.01, max=0.5),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                except Exception as exc:
                    raise httpx.TransportError(redact_secrets(str(exc))) from None
                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    raise _RetryableHttpError(
                        redact_secrets(f"HTTP {resp.status_code} from {self.provider}: {resp.text}")
                    )
                if resp.status_code >= 400:
                    raise RuntimeError(
                        redact_secrets(f"HTTP {resp.status_code} from {self.provider}: {resp.text}")
                    )
                data: dict[str, Any] = resp.json()
                latency_ms = (time.perf_counter() - start) * 1000.0
                ll_val = -1.0
                choices = data.get("choices") or []
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    c0 = choices[0]
                    if "loglikelihood" in c0:
                        ll_val = float(c0["loglikelihood"])
                    elif isinstance(c0.get("logprobs"), dict):
                        tok_lps = c0["logprobs"].get("token_logprobs") or []
                        nums = [float(x) for x in tok_lps if isinstance(x, (int, float))]
                        if nums:
                            ll_val = sum(nums)
                usage_raw = data.get("usage") or {}
                default_prompt_toks = max(1, len(req.context.split()))
                usage = TokenUsage(
                    prompt_tokens=int(usage_raw.get("prompt_tokens", default_prompt_toks)),
                    completion_tokens=int(usage_raw.get("completion_tokens", 0)),
                )
                return EvalResponse(
                    request_id=req.request_id,
                    sample_id=req.sample_id,
                    loglikelihood=ll_val,
                    is_greedy=True,
                    usage=usage,
                    latency_ms=latency_ms,
                    backend=self.backend_name,
                    model_id=self.model_id,
                    metadata={"provider": self.provider, "cost_usd": 0.0},
                )
        raise RuntimeError(f"Exhausted retries for loglikelihood {req.request_id}")

    async def aloglikelihood(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        if not requests:
            return []
        client_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        strict_budget = self.cost_guard is not None and self.cost_guard.max_cost_usd is not None
        eff_concurrency = 1 if strict_budget else self.max_concurrent
        async with httpx.AsyncClient(**client_kwargs) as client:
            if eff_concurrency == 1:
                return [await self._dispatch_loglikelihood(client, req) for req in requests]
            sem = asyncio.Semaphore(eff_concurrency)

            async def _bounded(req: EvalRequest) -> EvalResponse:
                async with sem:
                    return await self._dispatch_loglikelihood(client, req)

            return list(await asyncio.gather(*(_bounded(req) for req in requests)))

    def loglikelihood(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        result: list[EvalResponse] = _run_coro_sync(self.aloglikelihood(requests))
        return result

    def loglikelihood_rolling(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        return self.loglikelihood(requests)
