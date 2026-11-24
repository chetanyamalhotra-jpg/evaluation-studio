"""Unit tests for M1 core engine hooks, secrets helpers, packaging, and `eval-studio ui` CLI."""

from __future__ import annotations

import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import keyring
import pytest
import tomllib
from typer.testing import CliRunner

from eval_studio.cli.app import app
from eval_studio.core.config import EvalRunConfig, RuntimeLimits
from eval_studio.core.runner import build_judge_from_args, run_evaluation
from eval_studio.core.schema import EvalResponse
from eval_studio.gateway.cost import CostBudgetExceededError
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import (
    PROVIDER_ENV_VARS,
    REDACTED_PLACEHOLDER,
    SecretRole,
    clear_session_key,
    delete_keyring_key,
    get_key_status,
    load_api_key,
    mask_secret,
    redact_secrets,
    save_keyring_key,
    set_session_key,
)
from eval_studio.receipts.verify import verify_receipt

runner = CliRunner()


def _clear_all_roles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for role in SecretRole:
        clear_session_key(role)
    monkeypatch.setattr(keyring, "get_password", lambda service, username: None)


def _write_tier1_csv_task(
    root: Path,
    *,
    n_samples: int = 4,
    extra_yaml: str = "",
    name: str = "m1_progress_task",
) -> Path:
    data_path = root / f"{name}.csv"
    rows = ["id,question,answer"]
    for i in range(n_samples):
        rows.append(f"s{i},Question {i}?,Pancreas")
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    task_path = root / f"{name}.yaml"
    task_path.write_text(
        f"name: {name}\n"
        "trust_tier: tier1_zero_code\n"
        "output_type: generate_until\n"
        f"dataset_path: {data_path.name}\n"
        'doc_to_text: "{{ question }}"\n'
        'doc_to_target: "{{ answer }}"\n'
        "metrics:\n"
        "  - exact_match\n"
        f"{extra_yaml}",
        encoding="utf-8",
    )
    return task_path


# ---------------------------------------------------------------------------
# 1. secrets.py helpers & thread safety
# ---------------------------------------------------------------------------


def test_mask_secret_formatting_and_short_inputs() -> None:
    assert mask_secret("sk-proj-1234a1b2") == "sk-...a1b2"
    assert mask_secret("  12345678  ") == "123...5678"
    assert mask_secret("1234567") == "***"
    assert mask_secret("") == "***"
    assert mask_secret("   ") == "***"
    assert mask_secret("        ") == "***"


def test_session_key_precedence_and_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text('TARGET_API_KEY="sk-dotenv-9999zzzz"\n', encoding="utf-8")
    monkeypatch.setenv("TARGET_API_KEY", "sk-env-8888yyyy")
    monkeypatch.setattr(keyring, "get_password", lambda s, u: "sk-keyring-7777xxxx")

    with pytest.raises(ValueError, match="non-empty"):
        set_session_key(SecretRole.TARGET, "   ")

    session_secret = "sk-session-1111aaaa"
    set_session_key(SecretRole.TARGET, f"  {session_secret}  ")
    assert load_api_key(SecretRole.TARGET, dotenv_path=dotenv_file) == session_secret

    clear_session_key(SecretRole.TARGET)
    assert load_api_key(SecretRole.TARGET, dotenv_path=dotenv_file) == "sk-env-8888yyyy"
    # Cleared session key remains registered for redaction for process lifetime
    assert redact_secrets(f"leaked={session_secret}") == f"leaked={REDACTED_PLACEHOLDER}"


def test_save_and_delete_keyring_key_handles_success_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)
    store: dict[tuple[str, str], str] = {}

    def _set_pw(service: str, username: str, password: str) -> None:
        store[(service, username)] = password

    def _del_pw(service: str, username: str) -> None:
        if (service, username) not in store:
            raise RuntimeError("not found")
        del store[(service, username)]

    monkeypatch.setattr(keyring, "set_password", _set_pw)
    monkeypatch.setattr(keyring, "delete_password", _del_pw)

    with pytest.raises(ValueError, match="non-empty"):
        save_keyring_key(SecretRole.JUDGE, "   ")

    kr_secret = "sk-keyring-save-1234"
    assert save_keyring_key(SecretRole.JUDGE, kr_secret) is True
    assert store[("eval-studio", SecretRole.JUDGE.value)] == kr_secret
    assert redact_secrets(f"token={kr_secret}") == f"token={REDACTED_PLACEHOLDER}"

    assert delete_keyring_key(SecretRole.JUDGE) is True
    assert delete_keyring_key(SecretRole.JUDGE) is False

    def _broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("keyring daemon unavailable")

    monkeypatch.setattr(keyring, "set_password", _broken)
    assert save_keyring_key(SecretRole.JUDGE, "sk-keyring-fail-5678") is False


def test_get_key_status_all_five_sources_and_never_exposes_raw_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)
    dotenv_file = tmp_path / ".env"

    # 1. Unset (and whitespace-only env/.env ignored)
    monkeypatch.setenv("TARGET_API_KEY", "   ")
    dotenv_file.write_text('TARGET_API_KEY="   "\n', encoding="utf-8")
    unset_status = get_key_status(SecretRole.TARGET, dotenv_path=dotenv_file)
    assert unset_status == {
        "role": "TARGET_API_KEY",
        "configured": False,
        "source": "unset",
        "masked_hint": None,
        "env_var": None,
    }
    monkeypatch.delenv("TARGET_API_KEY", raising=False)

    # 2. Dotenv fallback + primary
    dotenv_fallback = "sk-dotenv-fb-1234"
    dotenv_file.write_text(f"OPENAI_API_KEY={dotenv_fallback}\n", encoding="utf-8")
    st_df = get_key_status(
        SecretRole.TARGET, fallback_env="OPENAI_API_KEY", dotenv_path=dotenv_file
    )
    assert st_df["configured"] is True
    assert st_df["source"] == "dotenv"
    assert st_df["env_var"] == "OPENAI_API_KEY"
    assert st_df["masked_hint"] == "sk-...1234"
    assert dotenv_fallback not in str(st_df)

    dotenv_primary = "sk-dotenv-pr-5678"
    dotenv_file.write_text(
        f"TARGET_API_KEY={dotenv_primary}\nOPENAI_API_KEY={dotenv_fallback}\n",
        encoding="utf-8",
    )
    st_dp = get_key_status(
        SecretRole.TARGET, fallback_env="OPENAI_API_KEY", dotenv_path=dotenv_file
    )
    assert st_dp["source"] == "dotenv"
    assert st_dp["env_var"] == "TARGET_API_KEY"
    assert st_dp["masked_hint"] == "sk-...5678"
    assert dotenv_primary not in str(st_dp)

    # 3. Keyring overrides dotenv
    kr_secret = "sk-keyring-9999wxyz"
    monkeypatch.setattr(keyring, "get_password", lambda s, u: kr_secret)
    st_kr = get_key_status(
        SecretRole.TARGET, fallback_env="OPENAI_API_KEY", dotenv_path=dotenv_file
    )
    assert st_kr["source"] == "keyring"
    assert st_kr["env_var"] is None
    assert st_kr["masked_hint"] == "sk-...wxyz"
    assert kr_secret not in str(st_kr)

    # 4. Env (fallback + primary) overrides keyring
    env_fb = "sk-env-fb-4321"
    monkeypatch.setenv("OPENAI_API_KEY", env_fb)
    st_ef = get_key_status(
        SecretRole.TARGET, fallback_env="OPENAI_API_KEY", dotenv_path=dotenv_file
    )
    assert st_ef["source"] == "env"
    assert st_ef["env_var"] == "OPENAI_API_KEY"
    assert st_ef["masked_hint"] == "sk-...4321"
    assert env_fb not in str(st_ef)

    env_pr = "sk-env-pr-8765"
    monkeypatch.setenv("TARGET_API_KEY", env_pr)
    st_ep = get_key_status(
        SecretRole.TARGET, fallback_env="OPENAI_API_KEY", dotenv_path=dotenv_file
    )
    assert st_ep["source"] == "env"
    assert st_ep["env_var"] == "TARGET_API_KEY"
    assert st_ep["masked_hint"] == "sk-...8765"
    assert env_pr not in str(st_ep)

    # 5. Session overrides env
    sess_secret = "sk-session-0000abcd"
    set_session_key(SecretRole.TARGET, sess_secret)
    st_sess = get_key_status(
        SecretRole.TARGET, fallback_env="OPENAI_API_KEY", dotenv_path=dotenv_file
    )
    assert st_sess["source"] == "session"
    assert st_sess["env_var"] is None
    assert st_sess["masked_hint"] == "sk-...abcd"
    assert sess_secret not in str(st_sess)
    clear_session_key(SecretRole.TARGET)


def test_secrets_thread_safety_concurrent_ops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)

    def _worker(i: int) -> None:
        secret = f"sk-thread-secret-{i:04d}-xyz"
        set_session_key(SecretRole.EXTRACTOR, secret)
        _ = load_api_key(SecretRole.EXTRACTOR)
        status = get_key_status(SecretRole.EXTRACTOR)
        assert status["configured"] is True
        assert secret not in str(status)
        red = redact_secrets(f"err: {secret}")
        assert secret not in red
        if i % 3 == 0:
            clear_session_key(SecretRole.EXTRACTOR)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_worker, range(40)))
    clear_session_key(SecretRole.EXTRACTOR)


# ---------------------------------------------------------------------------
# 2. runner.py on_progress callback hooks
# ---------------------------------------------------------------------------


def test_runner_on_progress_incremental_without_budget_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)
    task_path = _write_tier1_csv_task(
        tmp_path,
        n_samples=4,
        extra_yaml=(
            "metadata:\n"
            "  pricing:\n"
            "    input_usd_per_1m: 1000.0\n"
            "    output_usd_per_1m: 2000.0\n"
        ),
    )
    out_dir = tmp_path / "out"
    cfg = EvalRunConfig(
        model_backend="mock",
        model_id="mock-deterministic",
        tasks=[str(task_path)],
        use_cache=False,
        cache_dir=tmp_path / "cache",
        output_dir=out_dir,
        limits=RuntimeLimits(max_cost_usd=None),
    )
    events: list[dict[str, Any]] = []
    disk_ready_on_completed: dict[str, bool] = {}

    def _cb(ev: dict[str, Any]) -> None:
        events.append(ev)
        if ev["event"] == "completed":
            disk_ready_on_completed["manifest"] = (out_dir / "manifest.json").is_file()
            disk_ready_on_completed["samples"] = (out_dir / "samples.jsonl").is_file()

    manifest = run_evaluation(
        task_path,
        config=cfg,
        output_dir=out_dir,
        gateway=MockGateway(
            default_reply="Pancreas",
            token_pricing={"input_usd_per_1m": 1000.0, "output_usd_per_1m": 2000.0},
        ),
        on_progress=_cb,
    )

    assert [e["event"] for e in events] == [
        "run_started",
        "sample_scored",
        "sample_scored",
        "sample_scored",
        "sample_scored",
        "completed",
    ]
    assert events[0]["run_id"] == manifest.run_id
    assert events[0]["total_samples"] == 4
    spent_series = [e["spent_usd"] for e in events[1:5]]
    assert [e["index"] for e in events[1:5]] == [0, 1, 2, 3]
    assert all(e["total"] == 4 for e in events[1:5])
    assert all(isinstance(e["sample"], dict) for e in events[1:5])
    assert spent_series == sorted(spent_series)
    assert spent_series[-1] > 0.0
    assert events[-1]["run_id"] == manifest.run_id
    assert events[-1]["manifest_sha256"] == manifest.manifest_sha256
    assert disk_ready_on_completed == {"manifest": True, "samples": True}


def test_runner_on_progress_parity_and_batching_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)
    task_path = _write_tier1_csv_task(tmp_path, n_samples=4)

    class _SpyGateway(MockGateway):
        def __init__(self) -> None:
            super().__init__(default_reply="Pancreas")
            self.batch_sizes: list[int] = []

        def generate_until(self, requests: Any) -> list[EvalResponse]:
            self.batch_sizes.append(len(requests))
            return super().generate_until(requests)

    gw_batched = _SpyGateway()
    out_batched = tmp_path / "out_batched"
    cfg_batched = EvalRunConfig(
        model_backend="mock",
        tasks=[str(task_path)],
        use_cache=False,
        cache_dir=tmp_path / "cache1",
        output_dir=out_batched,
        limits=RuntimeLimits(max_cost_usd=None),
    )
    m_batched = run_evaluation(
        task_path, config=cfg_batched, output_dir=out_batched, gateway=gw_batched
    )

    gw_stream = _SpyGateway()
    out_stream = tmp_path / "out_stream"
    cfg_stream = EvalRunConfig(
        model_backend="mock",
        tasks=[str(task_path)],
        use_cache=False,
        cache_dir=tmp_path / "cache2",
        output_dir=out_stream,
        limits=RuntimeLimits(max_cost_usd=None),
    )
    events: list[dict[str, Any]] = []
    m_stream = run_evaluation(
        task_path,
        config=cfg_stream,
        output_dir=out_stream,
        gateway=gw_stream,
        on_progress=events.append,
    )

    assert gw_batched.batch_sizes == [4]
    assert gw_stream.batch_sizes == [1, 1, 1, 1]
    assert (out_batched / "samples.jsonl").read_bytes() == (
        out_stream / "samples.jsonl"
    ).read_bytes()
    assert m_batched.samples_sha256 == m_stream.samples_sha256
    assert m_batched.total_prompt_tokens == m_stream.total_prompt_tokens
    assert m_batched.total_completion_tokens == m_stream.total_completion_tokens
    assert m_batched.total_cost_usd == m_stream.total_cost_usd
    assert m_batched.metrics == m_stream.metrics


def test_runner_on_progress_repeats_fewshot_thinking_and_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)
    task_path = _write_tier1_csv_task(
        tmp_path,
        n_samples=3,
        name="m1_repeats_task",
        extra_yaml=(
            "num_fewshot: 1\n"
            "generation_kwargs:\n"
            "  repeats: 3\n"
            "filter_ensembles:\n"
            "  - name: mv_pipe\n"
            "    steps:\n"
            "      - name: majority_vote\n"
        ),
    )
    out_dir = tmp_path / "out_repeats"
    cfg = EvalRunConfig(
        model_backend="mock",
        tasks=[str(task_path)],
        enable_thinking=True,
        think_end_token="</think>",
        use_cache=False,
        cache_dir=tmp_path / "cache_rep",
        output_dir=out_dir,
    )
    events: list[dict[str, Any]] = []
    run_evaluation(
        task_path,
        config=cfg,
        output_dir=out_dir,
        gateway=MockGateway(default_reply="Pancreas", thinking_prefix="<think>plan</think>"),
        judge=build_judge_from_args("mock"),
        on_progress=events.append,
    )

    scored_events = [e for e in events if e["event"] == "sample_scored"]
    assert len(scored_events) == 3
    for ev in scored_events:
        raw_resps = ev["sample"]["raw_responses"]
        assert len(raw_resps) == 3
        assert raw_resps[0]["reasoning_trace"] == "<think>plan"
        assert "llm_judge" in ev["sample"]["metrics"]["mv_pipe"]


def test_runner_on_progress_budget_cap_emits_budget_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)
    task_path = _write_tier1_csv_task(
        tmp_path,
        n_samples=5,
        name="m1_budget_task",
        extra_yaml=(
            "metadata:\n"
            "  pricing:\n"
            "    input_usd_per_1m: 1000000.0\n"
            "    output_usd_per_1m: 1000000.0\n"
        ),
    )
    # MockGateway uses ~3-4 tokens per request; at $1/token that's ~$3-$4/sample.
    out_dir = tmp_path / "out_budget"
    cache_dir = tmp_path / "cache_budget"
    cfg = EvalRunConfig(
        model_backend="mock",
        tasks=[str(task_path)],
        use_cache=True,
        cache_dir=cache_dir,
        output_dir=out_dir,
        limits=RuntimeLimits(max_cost_usd=6.0),
    )
    events: list[dict[str, Any]] = []
    with pytest.raises(CostBudgetExceededError):
        run_evaluation(
            task_path,
            config=cfg,
            output_dir=out_dir,
            on_progress=events.append,
        )

    ev_names = [e["event"] for e in events]
    assert ev_names[0] == "run_started"
    assert ev_names[-1] == "budget_exceeded"
    assert "completed" not in ev_names
    assert any(name == "sample_scored" for name in ev_names)
    budget_ev = events[-1]
    assert budget_ev["max_cost_usd"] == 6.0
    assert budget_ev["spent_usd"] <= 6.0
    assert (cache_dir / "wal.jsonl").is_file()


def test_runner_on_progress_callback_exception_and_secret_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)
    secret = "sk-secret-callback-test-123456"
    monkeypatch.setenv("TARGET_API_KEY", secret)
    task_path = _write_tier1_csv_task(tmp_path, n_samples=2, name="m1_redact_task")
    out_dir = tmp_path / "out_redact"
    cfg = EvalRunConfig(
        model_backend="mock",
        tasks=[str(task_path)],
        use_cache=False,
        cache_dir=tmp_path / "cache_red",
        output_dir=out_dir,
    )
    seen: list[dict[str, Any]] = []

    def _hostile_cb(ev: dict[str, Any]) -> None:
        seen.append(dict(ev))
        ev["mutated"] = True
        if ev["event"] == "sample_scored":
            raise RuntimeError("SSE client disconnected")

    run_evaluation(
        task_path,
        config=cfg,
        output_dir=out_dir,
        gateway=MockGateway(default_reply=f"Echo {secret}"),
        on_progress=_hostile_cb,
    )

    assert [e["event"] for e in seen] == [
        "run_started",
        "sample_scored",
        "sample_scored",
        "completed",
    ]
    serialized = str(seen)
    assert secret not in serialized
    assert REDACTED_PLACEHOLDER in serialized
    assert verify_receipt(out_dir).valid is True


def test_runner_on_progress_cache_hit_and_subtasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_all_roles(monkeypatch, tmp_path)
    # Test A: cache hit still emits full event lifecycle
    task_path = _write_tier1_csv_task(
        tmp_path,
        n_samples=2,
        name="m1_cache_task",
        extra_yaml=(
            "metadata:\n"
            "  pricing:\n"
            "    input_usd_per_1m: 1000.0\n"
            "    output_usd_per_1m: 2000.0\n"
        ),
    )
    cache_dir = tmp_path / "cache_hit"
    cfg1 = EvalRunConfig(
        model_backend="mock",
        tasks=[str(task_path)],
        use_cache=True,
        cache_dir=cache_dir,
        output_dir=tmp_path / "out_c1",
    )
    events1: list[dict[str, Any]] = []
    m1 = run_evaluation(
        task_path, config=cfg1, output_dir=tmp_path / "out_c1", on_progress=events1.append
    )

    cfg2 = EvalRunConfig(
        model_backend="mock",
        tasks=[str(task_path)],
        use_cache=True,
        cache_dir=cache_dir,
        output_dir=tmp_path / "out_c2",
    )
    events2: list[dict[str, Any]] = []
    m2 = run_evaluation(
        task_path, config=cfg2, output_dir=tmp_path / "out_c2", on_progress=events2.append
    )
    assert [e["event"] for e in events2] == [
        "run_started",
        "sample_scored",
        "sample_scored",
        "completed",
    ]
    assert events2[-1]["spent_usd"] == pytest.approx(m1.total_cost_usd)
    assert events2[-1]["run_id"] == m2.run_id

    # Test B: group task with 2 subtasks (2 samples each)
    sub_a = _write_tier1_csv_task(tmp_path, n_samples=2, name="sub_a")
    sub_b = _write_tier1_csv_task(tmp_path, n_samples=2, name="sub_b")
    group_path = tmp_path / "group_task.yaml"
    group_path.write_text(
        "name: m1_group_task\n"
        "trust_tier: tier1_zero_code\n"
        "subtasks:\n"
        f"  - {sub_a.name}\n"
        f"  - {sub_b.name}\n",
        encoding="utf-8",
    )
    group_out = tmp_path / "out_group"
    group_cfg = EvalRunConfig(
        model_backend="mock",
        tasks=[str(group_path)],
        use_cache=False,
        cache_dir=tmp_path / "cache_grp",
        output_dir=group_out,
    )
    grp_events: list[dict[str, Any]] = []
    grp_manifest = run_evaluation(
        group_path, config=group_cfg, output_dir=group_out, on_progress=grp_events.append
    )

    assert [e["event"] for e in grp_events] == [
        "run_started",
        "subtask_started",
        "sample_scored",
        "sample_scored",
        "subtask_started",
        "sample_scored",
        "sample_scored",
        "completed",
    ]
    assert grp_events[0]["total_samples"] == 4
    scored_grp = [e for e in grp_events if e["event"] == "sample_scored"]
    assert [e["index"] for e in scored_grp] == [0, 1, 2, 3]
    assert all(e["total"] == 4 for e in scored_grp)
    assert all(e["run_id"] == grp_manifest.run_id for e in grp_events)


# ---------------------------------------------------------------------------
# 3. `eval-studio ui` CLI & `[web]` packaging invariants
# ---------------------------------------------------------------------------


def test_cli_ui_help_and_lazy_import_overhead() -> None:
    res_ui = runner.invoke(app, ["ui", "--help"])
    assert res_ui.exit_code == 0
    for flag in ("--host", "--port", "--open", "--no-open", "--reload"):
        assert flag in res_ui.stdout

    probe = (
        "import sys; "
        "from eval_studio.cli.app import app; "
        "from typer.testing import CliRunner; "
        "res = CliRunner().invoke(app, ['--help']); "
        "assert res.exit_code == 0 and 'ui' in res.stdout; "
        "forbidden = ('fastapi', 'uvicorn', 'python_multipart', 'eval_studio.web', "
        "'torch', 'transformers', 'vllm', 'docker', 'pypdf'); "
        "loaded = [m for m in sys.modules if m.startswith(forbidden)]; "
        "assert not loaded, f'Eagerly loaded: {loaded}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_cli_ui_missing_web_extra_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-cli-ui-secret-999999"
    monkeypatch.setenv("TARGET_API_KEY", secret)
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    # Trigger ImportError carrying a secret if an import hook or module raises
    res = runner.invoke(app, ["ui", "--no-open"])
    assert res.exit_code == 1
    assert "evaluation-studio[web]" in res.stdout
    assert secret not in res.stdout

    # Also test unexpected exception carrying a secret in its message
    def _boom_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "fastapi":
            raise ImportError(f"broken wheel {secret}")
        raise AssertionError(name)

    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)
    monkeypatch.setattr("importlib.import_module", _boom_import)
    res2 = runner.invoke(app, ["ui", "--no-open"])
    assert res2.exit_code == 1
    assert secret not in res2.stdout
    assert REDACTED_PLACEHOLDER in res2.stdout


def test_cli_ui_missing_server_module_and_mocked_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_fastapi = types.ModuleType("fastapi")
    fake_multipart = types.ModuleType("python_multipart")
    run_calls: list[dict[str, Any]] = []
    fake_uvicorn: Any = types.ModuleType("uvicorn")

    def _fake_run(app_obj: Any, **kwargs: Any) -> None:
        run_calls.append({"app": app_obj, **kwargs})

    fake_uvicorn.run = _fake_run

    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "python_multipart", fake_multipart)

    # Missing eval_studio.web.server
    monkeypatch.setitem(sys.modules, "eval_studio.web.server", None)
    res_missing = runner.invoke(app, ["ui", "--no-open"])
    assert res_missing.exit_code == 1
    assert "eval_studio.web.server" in res_missing.stdout
    assert "evaluation-studio[web]" not in res_missing.stdout

    # Present eval_studio.web.server with create_app()
    fake_server: Any = types.ModuleType("eval_studio.web.server")
    sentinel_app = object()
    fake_server.create_app = lambda: sentinel_app
    monkeypatch.setitem(sys.modules, "eval_studio.web.server", fake_server)

    opened_urls: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url) or True)

    res_no_open = runner.invoke(app, ["ui", "--host", "0.0.0.0", "--port", "9000", "--no-open"])
    assert res_no_open.exit_code == 0, res_no_open.stdout
    assert opened_urls == []
    assert run_calls == [{"app": sentinel_app, "host": "0.0.0.0", "port": 9000}]

    run_calls.clear()
    res_reload = runner.invoke(
        app, ["ui", "--host", "0.0.0.0", "--port", "8420", "--open", "--reload"]
    )
    assert res_reload.exit_code == 0, res_reload.stdout
    assert opened_urls == ["http://127.0.0.1:8420"]
    assert run_calls == [
        {
            "app": "eval_studio.web.server:create_app",
            "host": "0.0.0.0",
            "port": 8420,
            "reload": True,
            "factory": True,
        }
    ]


def test_pyproject_web_extra_dev_package_data_and_gitignore(repo_root: Path) -> None:
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    def _pkg_names(reqs: list[str]) -> set[str]:
        names: set[str] = set()
        for r in reqs:
            s = r
            for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
                s = s.split(sep, 1)[0]
            names.add(s.strip().lower())
        return names

    core_deps = _pkg_names(data["project"]["dependencies"])
    assert core_deps.isdisjoint(
        {
            "fastapi",
            "uvicorn",
            "python-multipart",
            "torch",
            "transformers",
            "vllm",
            "docker",
            "pypdf",
        }
    )

    extras = data["project"]["optional-dependencies"]
    assert {"fastapi", "uvicorn", "python-multipart"} <= _pkg_names(extras["web"])
    assert {"fastapi", "uvicorn", "python-multipart"} <= _pkg_names(extras["dev"])

    pkg_data = data["tool"]["setuptools"]["package-data"]["eval_studio"]
    assert "py.typed" in pkg_data
    assert "web/static/**/*" in pkg_data

    assert (repo_root / "src" / "eval_studio" / "web" / "static" / ".gitkeep").is_file()
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "web/node_modules/" in gitignore
    assert "web/dist/" in gitignore
    assert "src/eval_studio/web/static/" not in {
        line.strip() for line in gitignore.splitlines() if not line.strip().startswith("#")
    }
