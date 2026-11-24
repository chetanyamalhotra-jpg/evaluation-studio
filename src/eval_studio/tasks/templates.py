"""Sandboxed Jinja2 template renderer for Tier 1 and Tier 2 BYOT tasks."""

from __future__ import annotations

import re
from typing import Any

from jinja2 import StrictUndefined
from jinja2.exceptions import SecurityError, TemplateSyntaxError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment


class TemplateSecurityError(RuntimeError):
    """Raised when a Jinja2 template attempts unsafe attribute or code access."""


_FORBIDDEN_TEMPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"__[a-zA-Z0-9_]+__"),
    re.compile(r"\b(?:cycler|joiner|namespace|lipsum|request|config|self)\s*\.\s*__"),
    re.compile(r"\bos\.system\b|\bsubprocess\b|\b__import__\b|\beval\s*\(|\bexec\s*\("),
)

_SANDBOX_ENV = SandboxedEnvironment(
    undefined=StrictUndefined,
    autoescape=False,
)
_SANDBOX_ENV.globals.clear()


def render_sandboxed_template(template_str: str, context: dict[str, Any]) -> str:
    """Render a Jinja2 prompt template strictly inside SandboxedEnvironment."""
    for pattern in _FORBIDDEN_TEMPLATE_PATTERNS:
        if pattern.search(template_str):
            raise TemplateSecurityError(
                f"Blocked unsafe Jinja2 expression matching forbidden pattern: {pattern.pattern}"
            )
    try:
        template = _SANDBOX_ENV.from_string(template_str)
        return template.render(**context)
    except SecurityError as exc:
        raise TemplateSecurityError(f"Blocked unsafe Jinja2 expression: {exc}") from exc
    except (TemplateSyntaxError, UndefinedError) as exc:
        raise TemplateSecurityError(f"Template rendering error: {exc}") from exc
