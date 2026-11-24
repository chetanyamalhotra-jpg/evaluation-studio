"""CLI smoke tests verifying `eval-studio --help` and registered subcommands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from eval_studio.cli.app import app

runner = CliRunner()


def test_cli_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for subcmd in (
        "run",
        "ls",
        "validate",
        "calibrate",
        "compile-paper",
        "score",
        "verify-receipt",
        "noise-floor",
    ):
        assert subcmd in result.stdout


def test_cli_validate_tier1_example(examples_dir: Path) -> None:
    task_path = examples_dir / "tier1_zero_code" / "task.yaml"
    result = runner.invoke(app, ["validate", "--task", str(task_path)])
    assert result.exit_code == 0
    assert "VALID" in result.stdout
