"""BYOK secret loader (env / OS keyring / gitignored .env) and mandatory redactor."""

from __future__ import annotations

import os
import re
import threading
from enum import Enum
from pathlib import Path
from typing import Any

import keyring


class SecretRole(str, Enum):
    """Role-separated BYOK credential slots."""

    TARGET = "TARGET_API_KEY"
    JUDGE = "JUDGE_API_KEY"
    EXTRACTOR = "EXTRACTOR_API_KEY"
    FALLBACK_JUDGE = "FALLBACK_JUDGE_API_KEY"


PROVIDER_ENV_VARS: tuple[str, ...] = (
    "TARGET_API_KEY",
    "JUDGE_API_KEY",
    "EXTRACTOR_API_KEY",
    "FALLBACK_JUDGE_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "SARVAM_API_KEY",
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk_[A-Za-z0-9_-]{12,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{12,}"),
    re.compile(r"(?i)api-subscription-key\s*[:=]\s*\S+"),
)

REDACTED_PLACEHOLDER = "[REDACTED_SECRET]"

# Thread-safe runtime registries for loaded secrets and in-memory session keys
_SECRETS_LOCK = threading.RLock()
_LOADED_SECRETS: set[str] = set()
_SESSION_KEYS: dict[str, str] = {}


def _remember_secret(value: str | None) -> str | None:
    if value and len(value) >= 6:
        with _SECRETS_LOCK:
            _LOADED_SECRETS.add(value)
    return value


def mask_secret(value: str) -> str:
    """Return a non-reversible masked hint for an API key."""
    cleaned = value.strip()
    if len(cleaned) >= 8:
        return f"{cleaned[:3]}...{cleaned[-4:]}"
    return "***"


def set_session_key(role: SecretRole, value: str) -> None:
    """Store an in-memory session-scoped API key for the given role."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("API key value must be non-empty.")
    with _SECRETS_LOCK:
        _SESSION_KEYS[role.value] = cleaned
        _remember_secret(cleaned)


def clear_session_key(role: SecretRole) -> None:
    """Clear any in-memory session-scoped API key for the given role."""
    with _SECRETS_LOCK:
        _SESSION_KEYS.pop(role.value, None)


def save_keyring_key(
    role: SecretRole,
    value: str,
    *,
    keyring_service: str = "eval-studio",
) -> bool:
    """Persist an API key to the OS keyring; returns False if unavailable."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("API key value must be non-empty.")
    _remember_secret(cleaned)
    try:
        keyring.set_password(keyring_service, role.value, cleaned)
        return True
    except Exception:
        return False


def delete_keyring_key(
    role: SecretRole,
    *,
    keyring_service: str = "eval-studio",
) -> bool:
    """Delete an API key from the OS keyring; returns False if missing or unavailable."""
    try:
        keyring.delete_password(keyring_service, role.value)
        return True
    except Exception:
        return False


def _read_dotenv_value(var_name: str, dotenv_path: Path | None = None) -> str | None:
    """Read a single key from a local .env file without mutating os.environ."""
    path = dotenv_path if dotenv_path is not None else Path.cwd() / ".env"
    if not path.is_file():
        return None
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() == var_name:
                cleaned = val.strip().strip('"').strip("'").strip()
                return cleaned or None
    except OSError:
        return None
    return None


def _resolve_api_key(
    role: SecretRole,
    *,
    fallback_env: str | None = None,
    keyring_service: str = "eval-studio",
    dotenv_path: Path | None = None,
) -> tuple[str | None, str, str | None]:
    """Resolve an API key and return (secret_value, source, env_var)."""
    with _SECRETS_LOCK:
        session_val = _SESSION_KEYS.get(role.value)
    if session_val:
        cleaned_session = session_val.strip()
        if cleaned_session:
            return _remember_secret(cleaned_session), "session", None

    if value := os.environ.get(role.value):
        cleaned_env = value.strip()
        if cleaned_env:
            return _remember_secret(cleaned_env), "env", role.value

    if fallback_env and (fallback_val := os.environ.get(fallback_env)):
        cleaned_fallback = fallback_val.strip()
        if cleaned_fallback:
            return _remember_secret(cleaned_fallback), "env", fallback_env

    try:
        if kr_val := keyring.get_password(keyring_service, role.value):
            cleaned_kr = kr_val.strip()
            if cleaned_kr:
                return _remember_secret(cleaned_kr), "keyring", None
    except Exception:
        pass

    if env_file_val := _read_dotenv_value(role.value, dotenv_path=dotenv_path):
        return _remember_secret(env_file_val), "dotenv", role.value

    if fallback_env and (
        fallback_file_val := _read_dotenv_value(fallback_env, dotenv_path=dotenv_path)
    ):
        return _remember_secret(fallback_file_val), "dotenv", fallback_env

    return None, "unset", None


def load_api_key(
    role: SecretRole,
    *,
    fallback_env: str | None = None,
    keyring_service: str = "eval-studio",
    dotenv_path: Path | None = None,
) -> str | None:
    """Resolve an API key from session store, env vars, OS keyring, then local .env file."""
    value, _, _ = _resolve_api_key(
        role,
        fallback_env=fallback_env,
        keyring_service=keyring_service,
        dotenv_path=dotenv_path,
    )
    return value


def get_key_status(
    role: SecretRole,
    *,
    fallback_env: str | None = None,
    keyring_service: str = "eval-studio",
    dotenv_path: Path | None = None,
) -> dict[str, Any]:
    """Return masked configuration status for a credential role without exposing raw keys."""
    value, source, env_var = _resolve_api_key(
        role,
        fallback_env=fallback_env,
        keyring_service=keyring_service,
        dotenv_path=dotenv_path,
    )
    return {
        "role": role.value,
        "configured": value is not None,
        "source": source,
        "masked_hint": mask_secret(value) if value is not None else None,
        "env_var": env_var,
    }


def active_env_secrets() -> tuple[str, ...]:
    """Return non-empty secret values currently present in known provider env vars."""
    secrets: list[str] = []
    for var in PROVIDER_ENV_VARS:
        val = os.environ.get(var)
        cleaned = val.strip() if val else ""
        if cleaned and len(cleaned) >= 6:
            secrets.append(cleaned)
    return tuple(secrets)


def redact_secrets(text: str, extra_secrets: tuple[str, ...] = ()) -> str:
    """Scrub known API key patterns, env secrets, and runtime-loaded secrets from text."""
    redacted = text
    with _SECRETS_LOCK:
        loaded_snapshot = tuple(_LOADED_SECRETS)
        session_snapshot = tuple(_SESSION_KEYS.values())
    all_literals = (*extra_secrets, *active_env_secrets(), *loaded_snapshot, *session_snapshot)
    for secret in sorted(set(all_literals), key=len, reverse=True):
        if secret and len(secret) >= 6:
            redacted = redacted.replace(secret, REDACTED_PLACEHOLDER)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(REDACTED_PLACEHOLDER, redacted)
    return redacted


def redact_object(obj: Any, extra_secrets: tuple[str, ...] = ()) -> Any:
    """Recursively redact secrets across nested dicts, lists, and strings."""
    if isinstance(obj, str):
        return redact_secrets(obj, extra_secrets=extra_secrets)
    if isinstance(obj, dict):
        return {k: redact_object(v, extra_secrets=extra_secrets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_object(v, extra_secrets=extra_secrets) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_object(v, extra_secrets=extra_secrets) for v in obj)
    return obj


def contains_secret(text: str) -> bool:
    """Return True if text appears to contain an unredacted API key or token."""
    return any(pattern.search(text) is not None for pattern in _SECRET_PATTERNS)
