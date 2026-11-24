"""Adversarial tests for whitespace transfer, BOS/chat-template conflicts, and config guards."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_studio.core.config import (
    EvalRunConfig,
    load_run_config,
    resolve_task_formatting,
    validate_unknown_args,
)
from eval_studio.core.schema import OutputType, TaskSpec
from eval_studio.gateway.base import transfer_trailing_whitespace
from eval_studio.tasks.validator import validate_task_spec


def test_whitespace_transfer_moves_trailing_spaces_to_continuation() -> None:
    ctx, cont = transfer_trailing_whitespace("Capital of France:  ", "Paris")
    assert ctx == "Capital of France:"
    assert cont == "  Paris"

    ctx_tab, cont_tab = transfer_trailing_whitespace("Q:\t ", "A")
    assert ctx_tab == "Q:"
    assert cont_tab == "\t A"


def test_bos_and_chat_template_conflict_raises() -> None:
    cfg_default = EvalRunConfig()
    assert cfg_default.add_bos_token is None

    with pytest.raises(ValidationError, match="apply_chat_template"):
        EvalRunConfig(apply_chat_template=True, add_bos_token=True)


def test_enable_thinking_requires_delimiter_and_blocks_loglikelihood() -> None:
    with pytest.raises(ValidationError, match="think_end_token"):
        EvalRunConfig(enable_thinking=True, think_end_token=None)

    spec = TaskSpec(name="mc_task", output_type=OutputType.MULTIPLE_CHOICE)
    result = validate_task_spec(spec, enable_thinking=True)
    assert result.valid is False
    assert any("enable_thinking" in e for e in result.errors)


def test_apply_chat_template_clears_target_delimiter_and_warns_on_instruct() -> None:
    with pytest.warns(UserWarning, match="instruct/chat-tuned"):
        cfg_warn = EvalRunConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            apply_chat_template=False,
        )
    assert cfg_warn.formatting_warnings()

    cfg_chat = EvalRunConfig(model_id="meta-llama/Llama-3.1-8B-Instruct", apply_chat_template=True)
    assert cfg_chat.target_delimiter == ""

    spec = TaskSpec(name="qa", output_type=OutputType.GENERATE_UNTIL, target_delimiter=" ")
    resolved = resolve_task_formatting(spec, cfg_chat)
    assert resolved.target_delimiter == ""
    assert spec.target_delimiter == " "


def test_config_rejects_unknown_keys_and_embedded_secrets(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        EvalRunConfig(nonexistent_flag=True)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="Unknown generation_kwargs"):
        validate_unknown_args(
            {"temp": 0.0},
            {"temperature", "max_tokens"},
            context="generation_kwargs",
        )

    cfg_file = tmp_path / "run_secret.yaml"
    cfg_file.write_text(
        "model_backend: http-byok\nmodel_id: sk-abcdefghijklmnopqrstuvwx\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="secret"):
        load_run_config(cfg_file)

    clean_file = tmp_path / "run_clean.yaml"
    clean_file.write_text("model_backend: mock\nbootstrap_iters: 250\n", encoding="utf-8")
    loaded = load_run_config(clean_file, cli_overrides={"limit": 5})
    assert loaded.bootstrap_iters == 250
    assert loaded.limit == 5


def test_load_run_config_rejects_yaml_continuation_secret_and_deep_merges_overrides(
    tmp_path: Path,
) -> None:
    cont_secret = tmp_path / "run_cont_secret.yaml"
    cont_secret.write_text(
        'model_backend: mock\nmodel_id: "sk-abcdefgh\\\nijklmnopqrstuvwx"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="secret"):
        load_run_config(cont_secret)

    nested_cfg = tmp_path / "run_nested.yaml"
    nested_cfg.write_text(
        "model_backend: mock\n"
        "limits:\n"
        "  max_tokens: 256\n"
        "  max_cost_usd: 5.0\n"
        "seeds:\n"
        "  numpy_seed: 77\n"
        "model_args:\n"
        "  temperature: 0.0\n",
        encoding="utf-8",
    )
    merged = load_run_config(
        nested_cfg,
        cli_overrides={
            "limits": {"max_cost_usd": 1.0, "max_tokens": None},
            "seeds": {"bootstrap_seed": 99},
            "model_args": {"top_p": 0.9},
        },
    )
    assert merged.limits.max_tokens == 256
    assert merged.limits.max_cost_usd == 1.0
    assert merged.seeds.numpy_seed == 77
    assert merged.seeds.bootstrap_seed == 99
    assert merged.model_args == {"temperature": 0.0, "top_p": 0.9}
