"""Adversarial tests for BYOK redaction, cost cap, Jinja2 SSTI, Tier 3 gate, and cache safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_studio.cache.fingerprint import resolve_cache_dir
from eval_studio.gateway.cost import CostBudgetExceededError, CostGuard
from eval_studio.gateway.secrets import REDACTED_PLACEHOLDER, contains_secret, redact_secrets
from eval_studio.sandbox.base import ExecutionResult
from eval_studio.tasks.loader import TaskLoadError, load_task_spec
from eval_studio.tasks.templates import TemplateSecurityError, render_sandboxed_template


def test_byok_secret_redaction_scrubs_api_keys() -> None:
    raw = "Authorization: Bearer sk-proj-1234567890abcdef1234567890"
    assert contains_secret(raw) is True
    scrubbed = redact_secrets(raw)
    assert "sk-proj-" not in scrubbed
    assert REDACTED_PLACEHOLDER in scrubbed


def test_cost_guard_circuit_breaker_halts_over_budget() -> None:
    guard = CostGuard(max_cost_usd=0.05)
    guard.check_and_record(0.03)
    with pytest.raises(CostBudgetExceededError):
        guard.check_and_record(0.03)


def test_jinja2_sandbox_blocks_ssti_payload() -> None:
    payload = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    with pytest.raises(TemplateSecurityError):
        render_sandboxed_template(payload, {})


def test_tier3_plugin_blocked_without_trust_remote_code(tmp_path: Path) -> None:
    task_file = tmp_path / "tier3.yaml"
    task_file.write_text(
        "name: unsafe_tier3\ntrust_tier: tier3_code_plugin\n",
        encoding="utf-8",
    )
    with pytest.raises(TaskLoadError, match="--trust-remote-code"):
        load_task_spec(task_file, trust_remote_code=False)


def test_sandbox_verdict_is_primitive_bool_immune_to_eq_spoofing() -> None:
    class AlwaysEqual:
        def __eq__(self, other: object) -> bool:
            return True

    verdict = ExecutionResult(passed=False, exit_code=1)
    assert type(verdict.passed) is bool
    assert verdict.passed is False
    # Demonstrates why untrusted objects must never be compared in-process
    assert AlwaysEqual() == 42


def test_default_cache_dir_is_never_inside_site_packages() -> None:
    cache_dir = resolve_cache_dir()
    assert "site-packages" not in str(cache_dir).lower()
