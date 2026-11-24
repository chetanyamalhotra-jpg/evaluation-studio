"""Typer CLI registering all Evaluation Studio subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape

from eval_studio import __version__
from eval_studio.core.config import EvalRunConfig, RuntimeLimits, load_run_config
from eval_studio.core.registry import ENTRY_POINT_GROUPS, PluginLoadError, get_registry
from eval_studio.core.runner import (
    build_judge_from_args,
    estimate_run_cost,
    rescore_receipt,
    run_evaluation,
)
from eval_studio.core.schema import ScoredSample
from eval_studio.gateway.base import InferenceGateway
from eval_studio.gateway.cost import CostBudgetExceededError
from eval_studio.gateway.http import HttpBYOKGateway
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import SecretRole, redact_secrets
from eval_studio.paper.compiler import compile_paper, write_compiled_pack
from eval_studio.paper.extract import PaperExtractor
from eval_studio.receipts.verify import verify_receipt
from eval_studio.scoring.calibrate import calibrate_judge_file
from eval_studio.stats.noise_floor import measure_noise_floor
from eval_studio.tasks.loader import TaskLoadError, load_task_spec
from eval_studio.tasks.validator import validate_task_spec

app = typer.Typer(
    name="eval-studio",
    help=(
        "Evaluation Studio — local-first BYOK/BYOT/BYOP evaluation "
        "with Portable Eval Receipts."
    ),
    no_args_is_help=True,
)
console = Console()


def _is_cli_set(ctx: typer.Context, param_name: str) -> bool:
    source = ctx.get_parameter_source(param_name)
    return bool(getattr(source, "name", "") == "COMMANDLINE")


@app.callback(invoke_without_command=True)
def _root_callback(
    version: Annotated[
        bool,
        typer.Option("--version", help="Print harness version and exit."),
    ] = False,
) -> None:
    if version:
        console.print(f"eval-studio {__version__}")
        raise typer.Exit(code=0)


@app.command("run")
def cmd_run(
    ctx: typer.Context,
    task: Annotated[
        Path | None, typer.Option("--task", help="Path to a Tier 1/2/3 task YAML.")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML run configuration file.")
    ] = None,
    model: Annotated[str, typer.Option("--model", help="Model backend name.")] = "mock",
    model_id: Annotated[
        str | None, typer.Option("--model-id", help="Model identifier passed to backend.")
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Optional HTTP base URL for BYOK gateway.")
    ] = None,
    dtype: Annotated[
        str | None, typer.Option("--dtype", help="Pinned model dtype recorded in receipt.")
    ] = None,
    quantization: Annotated[
        str | None,
        typer.Option("--quantization", help="Pinned quantization recipe recorded in receipt."),
    ] = None,
    judge_model: Annotated[
        str | None, typer.Option("--judge-model", help="Optional LLM-as-a-judge model ID.")
    ] = None,
    judge_base_url: Annotated[
        str | None, typer.Option("--judge-base-url", help="Optional HTTP base URL for judge.")
    ] = None,
    fallback_judge_model: Annotated[
        str | None,
        typer.Option("--fallback-judge-model", help="Optional fallback judge model ID."),
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for Portable Eval Receipt.")
    ] = Path("receipts"),
    limit: Annotated[
        int | None, typer.Option("--limit", help="Maximum number of samples to evaluate.")
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Inference batch size.")] = 1,
    max_cost_usd: Annotated[
        float | None,
        typer.Option("--max-cost-usd", help="Hard USD spend cap circuit breaker."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and estimate cost without dispatching requests."),
    ] = False,
    estimate_cost: Annotated[
        bool,
        typer.Option("--estimate-cost", help="Print pre-flight target + judge USD cost estimate."),
    ] = False,
    trust_remote_code: Annotated[
        bool,
        typer.Option("--trust-remote-code", help="Opt in to Tier 3 Python plugins."),
    ] = False,
    plugins: Annotated[
        list[str] | None,
        typer.Option("--plugins", help="Local plugin module or .py file to load."),
    ] = None,
    apply_chat_template: Annotated[
        bool, typer.Option("--apply-chat-template", help="Apply tokenizer chat template.")
    ] = False,
    add_bos_token: Annotated[
        bool | None, typer.Option("--add-bos-token", help="Explicitly toggle BOS token.")
    ] = None,
    enable_thinking: Annotated[
        bool, typer.Option("--enable-thinking", help="Strip reasoning traces before scoring.")
    ] = False,
    think_end_token: Annotated[
        str | None, typer.Option("--think-end-token", help="Reasoning end delimiter.")
    ] = None,
    use_cache: Annotated[
        bool, typer.Option("--use-cache/--no-cache", help="Enable/disable WAL+SQLite cache.")
    ] = True,
    cache_dir: Annotated[
        Path | None, typer.Option("--cache-dir", help="Override user cache directory.")
    ] = None,
    bootstrap_iters: Annotated[
        int, typer.Option("--bootstrap-iters", help="Exact bootstrap resample iterations.")
    ] = 1000,
    do_sample: Annotated[
        bool, typer.Option("--do-sample", help="Stochastic decoding (bypasses cache).")
    ] = False,
) -> None:
    """Execute an evaluation run and emit a Portable Eval Receipt."""
    try:
        if config is not None:
            overrides: dict[str, Any] = {}
            if task is not None:
                overrides["tasks"] = [str(task)]
            if _is_cli_set(ctx, "model"):
                overrides["model_backend"] = model
            if _is_cli_set(ctx, "model_id") and model_id is not None:
                overrides["model_id"] = model_id
            if _is_cli_set(ctx, "limit") and limit is not None:
                overrides["limit"] = limit
            if _is_cli_set(ctx, "batch_size"):
                overrides["batch_size"] = batch_size
            if _is_cli_set(ctx, "dtype") and dtype is not None:
                overrides["dtype"] = dtype
            if _is_cli_set(ctx, "quantization") and quantization is not None:
                overrides["quantization"] = quantization
            if _is_cli_set(ctx, "apply_chat_template"):
                overrides["apply_chat_template"] = apply_chat_template
            if _is_cli_set(ctx, "add_bos_token"):
                overrides["add_bos_token"] = add_bos_token
            if _is_cli_set(ctx, "enable_thinking"):
                overrides["enable_thinking"] = enable_thinking
            if _is_cli_set(ctx, "think_end_token") and think_end_token is not None:
                overrides["think_end_token"] = think_end_token
            if _is_cli_set(ctx, "trust_remote_code"):
                overrides["trust_remote_code"] = trust_remote_code
            if _is_cli_set(ctx, "plugins") and plugins is not None:
                overrides["plugins"] = plugins
            if _is_cli_set(ctx, "use_cache"):
                overrides["use_cache"] = use_cache
            if _is_cli_set(ctx, "cache_dir") and cache_dir is not None:
                overrides["cache_dir"] = str(cache_dir)
            if _is_cli_set(ctx, "output"):
                overrides["output_dir"] = str(output)
            if _is_cli_set(ctx, "bootstrap_iters"):
                overrides["bootstrap_iters"] = bootstrap_iters
            if _is_cli_set(ctx, "max_cost_usd") and max_cost_usd is not None:
                overrides["limits"] = {"max_cost_usd": max_cost_usd}
            cfg = load_run_config(config, cli_overrides=overrides)
        else:
            if task is None:
                raise ValueError("Provide either --task <path.yaml> or --config <run_config.yaml>.")
            cfg = EvalRunConfig(
                model_backend=model,
                model_id=model_id or ("mock-deterministic" if model == "mock" else model),
                tasks=[str(task)],
                plugins=plugins or [],
                limit=limit,
                batch_size=batch_size,
                dtype=dtype,
                quantization=quantization,
                apply_chat_template=apply_chat_template,
                add_bos_token=add_bos_token,
                enable_thinking=enable_thinking,
                think_end_token=think_end_token,
                trust_remote_code=trust_remote_code,
                use_cache=use_cache,
                cache_dir=cache_dir,
                output_dir=output,
                bootstrap_iters=bootstrap_iters,
                limits=RuntimeLimits(max_cost_usd=max_cost_usd),
            )

        resolved_task = task if task is not None else (Path(cfg.tasks[0]) if cfg.tasks else None)
        if resolved_task is None:
            raise ValueError("No task path specified in --task or --config.")

        if dry_run or estimate_cost:
            est = estimate_run_cost(
                resolved_task,
                limit=cfg.limit,
                trust_remote_code=cfg.trust_remote_code,
            )
            console.print(
                f"Pre-flight Cost Estimate (samples={est.n_samples}, repeats={est.repeats}):\n"
                f"  Target: ${est.target_usd:.6f} USD "
                f"({est.target_prompt_tokens} in / {est.target_completion_tokens} out)\n"
                f"  Judge:  ${est.judge_usd:.6f} USD (enabled={est.judge_enabled})\n"
                f"  Total:  ${est.total_usd:.6f} USD"
            )
            if dry_run:
                return

        judge = (
            build_judge_from_args(
                judge_model,
                judge_base_url=judge_base_url,
                fallback_judge_model=fallback_judge_model,
            )
            if judge_model
            else None
        )
        manifest = run_evaluation(
            resolved_task,
            config=cfg,
            output_dir=cfg.output_dir,
            base_url=base_url,
            plugins=cfg.plugins or None,
            judge=judge,
            do_sample=do_sample,
        )
        console.print(
            f"[bold green]COMPLETED:[/bold green] {manifest.task_name} "
            f"({manifest.run_id}) -> {cfg.output_dir}"
        )
    except (TaskLoadError, PluginLoadError, CostBudgetExceededError, ValueError) as exc:
        console.print(f"[bold red]ERROR:[/bold red] {redact_secrets(str(exc))}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/bold red] {escape(redact_secrets(str(exc)))}")
        raise typer.Exit(code=1) from None


@app.command("ls")
def cmd_ls() -> None:
    """List registered plugin groups and lazily discovered entry points."""
    for group in ENTRY_POINT_GROUPS:
        names = get_registry(group).list_names()
        joined = ", ".join(names) if names else "(none)"
        console.print(f"[bold]{group}[/bold]: {joined}")


@app.command("validate")
def cmd_validate(
    task: Annotated[Path, typer.Option("--task", help="Path to a task YAML to validate.")],
    snapshot: Annotated[
        Path | None, typer.Option("--snapshot", help="Optional golden prompt snapshot file.")
    ] = None,
    plugins: Annotated[
        list[str] | None,
        typer.Option("--plugins", help="Local plugin module or .py file to load."),
    ] = None,
    trust_remote_code: Annotated[
        bool,
        typer.Option("--trust-remote-code", help="Allow loading Tier 3 plugin specs."),
    ] = False,
    apply_chat_template: Annotated[
        bool, typer.Option("--apply-chat-template", help="Check chat-template compatibility.")
    ] = False,
    add_bos_token: Annotated[
        bool | None, typer.Option("--add-bos-token", help="Check BOS token compatibility.")
    ] = None,
    enable_thinking: Annotated[
        bool, typer.Option("--enable-thinking", help="Check thinking-mode compatibility.")
    ] = False,
) -> None:
    """Validate a task YAML schema, secret hygiene, template safety, and golden snapshots."""
    try:
        if plugins:
            from eval_studio.core.registry import load_plugins

            load_plugins(plugins, trust_remote_code=trust_remote_code)
        spec = load_task_spec(task, trust_remote_code=trust_remote_code)
    except (TaskLoadError, PluginLoadError, ValueError) as exc:
        console.print(f"[bold red]ERROR:[/bold red] {redact_secrets(str(exc))}")
        raise typer.Exit(code=1) from None

    result = validate_task_spec(
        spec,
        enable_thinking=enable_thinking,
        apply_chat_template=apply_chat_template,
        add_bos_token=add_bos_token,
        base_dir=task.parent,
        snapshot_path=snapshot,
    )
    for warn in result.warnings:
        console.print(f"[bold yellow]WARNING:[/bold yellow] {warn}")
    if not result.valid:
        for err in result.errors:
            console.print(f"[bold red]ERROR:[/bold red] {redact_secrets(err)}")
        raise typer.Exit(code=1)
    console.print(f"[bold green]VALID:[/bold green] {result.task_name}")


@app.command("calibrate")
def cmd_calibrate(
    judge_file: Annotated[Path, typer.Argument(help="Path to judge vs. human labels JSONL.")],
    threshold: Annotated[
        float, typer.Option("--threshold", help="Minimum Cohen's kappa alignment threshold.")
    ] = 0.40,
) -> None:
    """Compute Cohen's kappa calibration between an LLM judge and human gold labels."""
    try:
        report = calibrate_judge_file(judge_file, threshold=threshold)
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/bold red] {escape(redact_secrets(str(exc)))}")
        raise typer.Exit(code=1) from None

    console.print(
        f"Calibration ({report.n_samples} samples): "
        f"Cohen's kappa = {report.cohens_kappa:.4f}, "
        f"raw_agreement = {report.raw_agreement:.4f}"
    )
    if not report.passed_threshold:
        console.print(
            f"[bold yellow]WARNING: UNALIGNED[/bold yellow] "
            f"(kappa={report.cohens_kappa:.4f} < {report.threshold:.2f})"
        )


@app.command("compile-paper")
def cmd_compile_paper(
    paper: Annotated[Path, typer.Argument(help="Path to a PDF or Markdown scientific paper.")],
    output: Annotated[
        Path,
        typer.Option("--out", "--output", "-o", help="Output compiled suite (.yaml or .json)."),
    ] = Path("compiled_suite.json"),
    model: Annotated[
        str,
        typer.Option("--model", help="Extraction backend: heuristic, mock, or http-byok."),
    ] = "heuristic",
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Optional HTTP base URL for extractor.")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Optional HTTP provider dialect.")
    ] = None,
    verifier_base_url: Annotated[
        str | None,
        typer.Option("--verifier-base-url", help="Optional HTTP base URL for verifier."),
    ] = None,
    extractor_model: Annotated[
        str, typer.Option("--extractor-model", help="Stage 1 extractor model ID.")
    ] = "mock-extractor",
    verifier_model: Annotated[
        str, typer.Option("--verifier-model", help="Stage 2 answerability verifier model ID.")
    ] = "mock-verifier",
) -> None:
    """Run two-stage BYOP extraction, answerability verification, and SHA-256 locking."""
    try:
        extractor_gw: InferenceGateway | None = None
        verifier_gw: InferenceGateway | None = None

        if model == "mock":
            extractor_gw = MockGateway(model_id=extractor_model, compiler_mode="mixed")
        elif model == "http-byok" or base_url is not None or provider is not None:
            extractor_gw = HttpBYOKGateway(
                model_id=extractor_model,
                base_url=base_url,
                provider=provider,
                role=SecretRole.EXTRACTOR,
            )
            if verifier_model != "mock-verifier" or verifier_base_url is not None:
                verifier_gw = HttpBYOKGateway(
                    model_id=verifier_model,
                    base_url=verifier_base_url or base_url,
                    provider=provider,
                    role=SecretRole.JUDGE,
                )

        extractor = PaperExtractor(extractor_model=extractor_model, gateway=extractor_gw)
        candidates = extractor.extract_from_path(paper)
        suite = compile_paper(
            paper,
            items=candidates,
            suite_name=f"byop_{paper.stem}",
            extractor_model=extractor_model,
            verifier_model=verifier_model,
            verifier_gateway=verifier_gw,
        )
        if not suite.items:
            console.print(
                "[bold red]ERROR:[/bold red] All candidate items were rejected by Stage 2 "
                "answerability verification gate."
            )
            raise typer.Exit(code=1)
        write_compiled_pack(suite, output)
        console.print(
            f"[bold green]COMPILED:[/bold green] {len(suite.items)} approved items -> {output} "
            f"(suite_sha256={suite.suite_sha256[:12]}...)"
        )
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/bold red] {escape(redact_secrets(str(exc)))}")
        raise typer.Exit(code=1) from None


@app.command("score")
def cmd_score(
    receipt: Annotated[
        Path, typer.Option("--receipt", help="Path to a Portable Eval Receipt dir.")
    ],
    task: Annotated[
        Path | None, typer.Option("--task", help="Optional updated task YAML for re-scoring.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output dir for rescored receipt.")
    ] = None,
    trust_remote_code: Annotated[
        bool, typer.Option("--trust-remote-code", help="Allow Tier 3 task specs.")
    ] = False,
    judge_model: Annotated[
        str | None, typer.Option("--judge-model", help="Optional LLM-as-a-judge model ID.")
    ] = None,
    judge_base_url: Annotated[
        str | None, typer.Option("--judge-base-url", help="Optional HTTP base URL for judge.")
    ] = None,
    fallback_judge_model: Annotated[
        str | None,
        typer.Option("--fallback-judge-model", help="Optional fallback judge model ID."),
    ] = None,
) -> None:
    """Re-score saved raw responses offline without calling any model backend."""
    try:
        judge = (
            build_judge_from_args(
                judge_model,
                judge_base_url=judge_base_url,
                fallback_judge_model=fallback_judge_model,
            )
            if judge_model
            else None
        )
        manifest = rescore_receipt(
            receipt,
            output_dir=output,
            task_path=task,
            trust_remote_code=trust_remote_code,
            judge=judge,
        )
        metrics_str = ", ".join(
            f"{k}={v.value:.4f}" for k, v in sorted(manifest.metrics.items())
        )
        console.print(
            f"[bold green]SCORED OFFLINE:[/bold green] {manifest.task_name} "
            f"({metrics_str}) -> {output or receipt}"
        )
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/bold red] {escape(redact_secrets(str(exc)))}")
        raise typer.Exit(code=1) from None


@app.command("verify-receipt")
def cmd_verify_receipt(
    receipt: Annotated[Path, typer.Argument(help="Path to a Portable Eval Receipt directory.")],
    task: Annotated[
        Path | None, typer.Option("--task", help="Optional task YAML to cross-check SHA-256.")
    ] = None,
    source_paper: Annotated[
        Path | None,
        typer.Option("--source-paper", help="Optional source paper to cross-check SHA-256."),
    ] = None,
) -> None:
    """Verify SHA-256 integrity hashes and secret redaction for an Eval Receipt."""
    result = verify_receipt(receipt, task_path=task, source_paper_path=source_paper)
    if not result.valid:
        for err in result.errors:
            console.print(f"[bold red]FAILED:[/bold red] {redact_secrets(err)}")
        raise typer.Exit(code=1)
    console.print(f"[bold green]VERIFIED:[/bold green] {receipt}")


def _extract_receipt_decisions(receipt_dir: Path) -> list[str]:
    samples_path = receipt_dir / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(f"Missing samples.jsonl in {receipt_dir}")
    decisions: list[str] = []
    for raw_line in samples_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rec = ScoredSample.model_validate_json(line)
        if rec.filtered_responses:
            first_pipe = next(iter(rec.filtered_responses.values()))
            decisions.append(first_pipe[0] if first_pipe else "")
        elif rec.raw_responses:
            decisions.append(rec.raw_responses[0].text or "")
        else:
            decisions.append("")
    return decisions


@app.command("noise-floor")
def cmd_noise_floor(
    baseline: Annotated[Path, typer.Option("--baseline", help="Baseline Eval Receipt dir.")],
    perturbed: Annotated[Path, typer.Option("--perturbed", help="Perturbed Eval Receipt dir.")],
    perturbation: Annotated[
        str, typer.Option("--perturbation", help="Label for perturbation type.")
    ] = "restart",
) -> None:
    """Measure item-level flip rate across restarts, batch sizes, or engines."""
    try:
        base_dec = _extract_receipt_decisions(baseline)
        pert_dec = _extract_receipt_decisions(perturbed)
        report = measure_noise_floor(base_dec, pert_dec, perturbation=perturbation)
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/bold red] {escape(redact_secrets(str(exc)))}")
        raise typer.Exit(code=1) from None

    console.print(
        f"Noise Floor ({report.perturbation}): "
        f"n_decisions={report.n_decisions}, n_flips={report.n_flips}, "
        f"flip_rate={report.flip_rate:.4f} ({report.flip_rate * 100:.1f}%)"
    )


@app.command("ui")
def cmd_ui(
    host: Annotated[
        str, typer.Option("--host", help="Host interface to bind the Web Studio server.")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", help="Port for the Web Studio server.")
    ] = 8420,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open Web Studio in your default browser."),
    ] = True,
    reload: Annotated[
        bool, typer.Option("--reload", help="Enable auto-reload for backend development.")
    ] = False,
) -> None:
    """Launch the local-first Evaluation Studio server and SPA."""
    import contextlib
    import importlib
    import webbrowser

    try:
        importlib.import_module("fastapi")
        uvicorn = importlib.import_module("uvicorn")
        try:
            importlib.import_module("python_multipart")
        except ImportError:
            importlib.import_module("multipart")
    except ImportError as exc:
        msg = redact_secrets(
            "Web Studio requires optional dependencies `evaluation-studio[web]` "
            f"(missing: {exc.name or exc}). "
            "Install with: pip install 'evaluation-studio[web]'."
        )
        console.print(f"[bold red]ERROR:[/bold red] {escape(msg)}")
        raise typer.Exit(code=1) from None

    try:
        server_mod = importlib.import_module("eval_studio.web.server")
    except ModuleNotFoundError as exc:
        if exc.name in {"eval_studio.web", "eval_studio.web.server"}:
            msg = redact_secrets(
                "Web Studio backend module 'eval_studio.web.server' is not available yet."
            )
        else:
            msg = redact_secrets(
                f"Web Studio failed to import '{exc.name}': {exc}. "
                "Install with: pip install 'evaluation-studio[web]'."
            )
        console.print(f"[bold red]ERROR:[/bold red] {escape(msg)}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/bold red] {escape(redact_secrets(str(exc)))}")
        raise typer.Exit(code=1) from None

    create_app = getattr(server_mod, "create_app", None)
    if create_app is None or not callable(create_app):
        console.print(
            "[bold red]ERROR:[/bold red] "
            + escape(
                redact_secrets("eval_studio.web.server does not expose a callable create_app().")
            )
        )
        raise typer.Exit(code=1) from None

    browse_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    url = f"http://{browse_host}:{port}"
    console.print(f"[bold cyan]Evaluation Studio[/bold cyan] -> {url}")

    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)

    try:
        if reload:
            uvicorn.run(
                "eval_studio.web.server:create_app",
                host=host,
                port=port,
                reload=True,
                factory=True,
            )
        else:
            web_app = create_app()
            uvicorn.run(web_app, host=host, port=port)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/bold red] {escape(redact_secrets(str(exc)))}")
        raise typer.Exit(code=1) from None


def main() -> None:
    """Console script entry point."""
    app()
