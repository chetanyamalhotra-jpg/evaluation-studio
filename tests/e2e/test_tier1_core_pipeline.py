"""Tier 1 E2E tests: Core architecture, N+M schema, plugins, stats, and cache (F1–F12)."""

from __future__ import annotations

import importlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from eval_studio.cli.app import app
from eval_studio.core.config import EvalRunConfig, RuntimeLimits, resolve_task_formatting
from eval_studio.core.registry import ENTRY_POINT_GROUPS, get_registry
from eval_studio.core.schema import (
    ChatMessage,
    ChatMessages,
    EvalRequest,
    ImageContentPart,
    MultipleChoiceFormat,
    OutputType,
    TaskSpec,
    TextContentPart,
    extract_thinking_trace,
)
from eval_studio.gateway.base import transfer_trailing_whitespace
from eval_studio.gateway.mock import MockGateway
from eval_studio.receipts.manifest import ReceiptManifest
from eval_studio.scoring.metrics import (
    acc_all,
    bypass,
    exact_match,
    latex_exact_match,
    token_f1,
    whole_word_match,
)
from eval_studio.stats.aggregation import (
    SubtaskStats,
    aggregate_group,
    clustered_stderr,
    mcnemar_test,
)
from eval_studio.stats.bootstrap import bootstrap_stderr, paired_bootstrap_test
from eval_studio.stats.fewshot import ComparisonStatus, evaluate_fewshot_comparison
from eval_studio.tasks.loader import load_task_spec

pytestmark = pytest.mark.e2e


def _extract_pkg_name(req: str) -> str:
    for sep in ("[", "<", ">", "=", "!", "~", ";", " "):
        req = req.split(sep, 1)[0]
    return req.strip().lower()


def _parse_pyproject_project_section(text: str) -> dict[str, Any]:
    """Pure-Python fallback parser for pyproject.toml [project] deps/extras on Python 3.10."""
    section = ""
    deps: list[str] = []
    extras: dict[str, list[str]] = {}
    in_deps = False
    active_extra: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            in_deps = False
            active_extra = None
            continue
        if section == "project":
            if re.match(r"^dependencies\s*=\s*\[", line):
                in_deps = True
                deps = []
                rest = line.split("[", 1)[1]
                for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', rest):
                    deps.append(m.group(1) or m.group(2))
                if "]" in rest:
                    in_deps = False
            elif in_deps:
                for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', line):
                    deps.append(m.group(1) or m.group(2))
                if "]" in line:
                    in_deps = False
        elif section == "project.optional-dependencies":
            key_match = re.match(r"^([A-Za-z0-9_-]+)\s*=\s*\[", line)
            if key_match:
                active_extra = key_match.group(1)
                extras.setdefault(active_extra, [])
                rest = line.split("[", 1)[1]
                for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', rest):
                    extras[active_extra].append(m.group(1) or m.group(2))
                if "]" in rest:
                    active_extra = None
            elif active_extra is not None:
                for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', line):
                    extras[active_extra].append(m.group(1) or m.group(2))
                if "]" in line:
                    active_extra = None

    return {"project": {"dependencies": deps, "optional-dependencies": extras}}


def _load_pyproject(pyproject_path: Path) -> dict[str, Any]:
    """Load pyproject.toml compatibly across Python 3.10+ without static `import tomllib`."""
    text = pyproject_path.read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        tomllib_mod = importlib.import_module("tomllib")
        loaded: dict[str, Any] = tomllib_mod.loads(text)
        return loaded
    try:
        tomli_mod = importlib.import_module("tomli")
        loaded_tomli: dict[str, Any] = tomli_mod.loads(text)
        return loaded_tomli
    except ModuleNotFoundError:
        return _parse_pyproject_project_section(text)


def test_t1_slim_core_and_lazy_plugin_discovery(
    repo_root: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-CORE-01: Slim core deps, optional extras, lazy registry, and `eval-studio ls`."""
    pyproject = _load_pyproject(repo_root / "pyproject.toml")
    base_deps = {_extract_pkg_name(dep) for dep in pyproject["project"]["dependencies"]}
    heavy_forbidden = {
        "torch",
        "transformers",
        "accelerate",
        "vllm",
        "sglang",
        "docker",
        "pypdf",
    }
    assert base_deps.isdisjoint(heavy_forbidden)

    extras = set(pyproject["project"]["optional-dependencies"].keys())
    assert {"api", "hf", "vllm", "pdf", "sandbox"}.issubset(extras)

    result = cli_runner.invoke(app, ["ls"])
    assert result.exit_code == 0, result.stdout
    for group in ENTRY_POINT_GROUPS:
        assert group in result.stdout
        names = get_registry(group).list_names()
        assert isinstance(names, list)

    assert "mock" in get_registry("eval_studio.models").list_names()
    assert "docker" in get_registry("eval_studio.sandboxes").list_names()

    for heavy_mod in ("torch", "transformers", "docker", "vllm", "pypdf"):
        assert heavy_mod not in sys.modules


def test_t1_local_plugins_flag_and_four_stage_run(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-CORE-02: `--plugins` local loader + 4-stage pipeline emitting full receipt."""
    plugin_path = tmp_path / "local_clin_plugin.py"
    plugin_path.write_text(
        '''"""Local Tier 3 clinical filter/metric plugin for E2E test."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


class ClinicalUpperFilter:
    name: str = "clinical_upper"

    def apply(self, responses: Sequence[str]) -> list[str]:
        return [f"CLINICAL:{r.strip().upper()}" for r in responses]


def clinical_prefix_match(prediction: str, reference: str) -> float:
    cleaned = prediction.strip().removeprefix("CLINICAL:")
    return 1.0 if cleaned.strip().casefold() == reference.strip().casefold() else 0.0


def register_plugins(registry_getter: Callable[[str], Any]) -> None:
    registry_getter("eval_studio.filters").register(
        "clinical_upper", ClinicalUpperFilter, builtin=False
    )
    registry_getter("eval_studio.metrics").register(
        "clinical_prefix_match", clinical_prefix_match, builtin=False
    )
''',
        encoding="utf-8",
    )

    dataset_path = tmp_path / "clin.csv"
    dataset_path.write_text(
        'sample_id,input_text,target\ns1,"Which organ produces insulin?",A\n',
        encoding="utf-8",
    )
    task_path = tmp_path / "tier3_plugin_task.yaml"
    task_path.write_text(
        f"""name: t1_local_plugin_task
version: "0.1.0"
trust_tier: tier3_code_plugin
output_type: generate_until
dataset_path: "{dataset_path.as_posix()}"
doc_to_text: "Question: {{{{ input_text }}}}\\nAnswer:"
doc_to_target: "{{{{ target }}}}"
unsafe_code: true
metrics:
  - clinical_prefix_match
filter_ensembles:
  - name: upper_pipe
    steps:
      - name: clinical_upper
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "receipt_plugin"

    result = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_path),
            "--model",
            "mock",
            "--plugins",
            str(plugin_path),
            "--trust-remote-code",
            "--output",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    for fname in ("manifest.json", "receipt.json", "samples.jsonl", "report.md", "report.html"):
        assert (out_dir / fname).is_file(), f"Missing {fname}"

    samples = [
        json.loads(line)
        for line in (out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(samples) == 1
    rec = samples[0]
    assert rec["sample_id"] == "s1"
    assert rec["raw_responses"]
    assert "upper_pipe" in rec["filtered_responses"]
    assert rec["filtered_responses"]["upper_pipe"] == ["CLINICAL:A"]
    assert "clinical_prefix_match" in rec["metrics"] or any(
        "clinical_prefix_match" in m for m in rec["metrics"].values()
    )


def test_t1_five_output_types_and_chat_messages(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-CORE-03: 5 OutputTypes, multimodal ChatMessages, F6 null-content, F7 whitespace/BOS."""
    limits = RuntimeLimits(max_tokens=128, wall_clock_seconds=30.0, max_tool_calls=2)
    chat_msgs: ChatMessages = [
        ChatMessage(
            role="user",
            content=[
                TextContentPart(text="Interpret ECG finding."),
                ImageContentPart(image_url={"url": "data:image/png;base64,AAAA"}),
            ],
        )
    ]
    chat_req = EvalRequest(
        request_id="req-chat-1",
        sample_id="chat-1",
        task_name="chat_smoke",
        output_type=OutputType.CHAT,
        context="Interpret ECG finding.",
        messages=chat_msgs,
        limits=limits,
    )
    assert chat_req.output_type == OutputType.CHAT
    assert chat_req.limits.max_tokens == 128
    chat_resps = MockGateway().chat([chat_req])
    assert len(chat_resps) == 1
    assert chat_resps[0].text is not None

    # F6: content=null with reasoning_content fallback when reasoning budget hits max_tokens
    null_content_res = extract_thinking_trace(
        None,
        enable_thinking=True,
        think_end_token="</think>",
        reasoning_content="Differential diagnosis truncated at token limit",
    )
    assert null_content_res.text == ""
    assert null_content_res.reasoning_trace == "Differential diagnosis truncated at token limit"
    assert null_content_res.thinking_truncated is True
    assert null_content_res.warning is not None

    # F7: _encode_pair whitespace transfer + apply_chat_template=True clearing target_delimiter
    ctx_clean, cont_prefixed = transfer_trailing_whitespace("Question: Capital? ", "Paris")
    assert (ctx_clean, cont_prefixed) == ("Question: Capital?", " Paris")
    ctx_multi, cont_multi = transfer_trailing_whitespace("Prompt  ", "A")
    assert (ctx_multi, cont_multi) == ("Prompt", "  A")

    chat_cfg = EvalRunConfig(apply_chat_template=True)
    assert chat_cfg.target_delimiter == ""
    resolved_spec = resolve_task_formatting(
        TaskSpec(name="f7_delimiter_check", target_delimiter=" "),
        chat_cfg,
    )
    assert resolved_spec.target_delimiter == ""

    ll_jsonl = tmp_path / "ll.jsonl"
    ll_jsonl.write_text(
        json.dumps({"sample_id": "s1", "input_text": "2+2=", "target": "4"}) + "\n",
        encoding="utf-8",
    )
    mc_jsonl = tmp_path / "mc.jsonl"
    mc_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "input_text": "Normal resting heart rate?",
                "choices": ["60-100 bpm", "10-20 bpm"],
                "target": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    chat_jsonl = tmp_path / "chat.jsonl"
    chat_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "input_text": "Triage patient with chest pain.",
                "target": "ECG immediately",
                "image_ref": "ecg_01.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    configs: list[tuple[OutputType, str]] = [
        (
            OutputType.LOGLIKELIHOOD,
            f"""name: task_ll
trust_tier: tier1_zero_code
output_type: loglikelihood
dataset_path: "{ll_jsonl.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
""",
        ),
        (
            OutputType.LOGLIKELIHOOD_ROLLING,
            f"""name: task_ll_rolling
trust_tier: tier1_zero_code
output_type: loglikelihood_rolling
dataset_path: "{ll_jsonl.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics: [bits_per_byte]
""",
        ),
        (
            OutputType.MULTIPLE_CHOICE,
            f"""name: task_mc
trust_tier: tier1_zero_code
output_type: multiple_choice
dataset_path: "{mc_jsonl.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics: [acc_bytes]
""",
        ),
        (
            OutputType.GENERATE_UNTIL,
            f"""name: task_gen
trust_tier: tier1_zero_code
output_type: generate_until
dataset_path: "{ll_jsonl.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
""",
        ),
        (
            OutputType.CHAT,
            f"""name: task_chat
trust_tier: tier2_declarative
output_type: chat
dataset_path: "{chat_jsonl.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
doc_to_messages:
  - role: user
    content:
      - type: text
        text: "{{{{ input_text }}}}"
      - type: image_url
        image_url:
          url: "file://{{{{ image_ref }}}}"
metrics: [exact_match]
metadata:
  limits:
    max_tokens: 128
    wall_clock_seconds: 30.0
    max_tool_calls: 2
""",
        ),
    ]

    for output_type, yaml_body in configs:
        t_path = tmp_path / f"{output_type.value}.yaml"
        t_path.write_text(yaml_body, encoding="utf-8")
        out_dir = tmp_path / f"out_{output_type.value}"
        res = cli_runner.invoke(
            app,
            ["run", "--task", str(t_path), "--model", "mock", "--output", str(out_dir)],
        )
        assert res.exit_code == 0, f"{output_type.value} failed: {res.stdout}"
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["task_name"]
        if output_type == OutputType.CHAT:
            cfg_str = json.dumps(manifest["redacted_config"])
            assert "128" in cfg_str


def test_t1_thinking_trace_stripping_and_token_metrics(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-CORE-04: Last `</think>` stripping + tokenization-agnostic & calibration metrics."""
    thinking_plugin = tmp_path / "thinking_mock_plugin.py"
    thinking_plugin.write_text(
        '''"""Mock gateway returning multi-tag CoT output for thinking-strip test."""

from __future__ import annotations

from collections.abc import Sequence

from eval_studio.core.registry import get_registry
from eval_studio.core.schema import EvalRequest, EvalResponse, TokenUsage
from eval_studio.gateway.mock import MockGateway


class ThinkingMockGateway(MockGateway):
    backend_name: str = "thinking_mock"

    def generate_until(self, requests: Sequence[EvalRequest]) -> list[EvalResponse]:
        return [
            EvalResponse(
                request_id=req.request_id,
                sample_id=req.sample_id,
                text="<think>step1</think>Draft<think>step2</think>Pancreas",
                backend=self.backend_name,
                model_id=self.model_id,
                usage=TokenUsage(prompt_tokens=8, completion_tokens=12, reasoning_tokens=6),
            )
            for req in requests
        ]


get_registry("eval_studio.models").register("thinking_mock", ThinkingMockGateway, builtin=False)
''',
        encoding="utf-8",
    )

    gen_csv = tmp_path / "gen.csv"
    gen_csv.write_text(
        'sample_id,input_text,target\nm1,"Which organ produces insulin?",Pancreas\n',
        encoding="utf-8",
    )
    gen_yaml = tmp_path / "gen_thinking.yaml"
    gen_yaml.write_text(
        f"""name: t1_thinking_gen
trust_tier: tier3_code_plugin
output_type: generate_until
dataset_path: "{gen_csv.as_posix()}"
doc_to_text: "Question: {{{{ input_text }}}}\\nAnswer:"
doc_to_target: "{{{{ target }}}}"
unsafe_code: true
metrics: [exact_match]
""",
        encoding="utf-8",
    )
    gen_out = tmp_path / "gen_out"
    res_gen = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(gen_yaml),
            "--model",
            "thinking_mock",
            "--plugins",
            str(thinking_plugin),
            "--trust-remote-code",
            "--enable-thinking",
            "--think-end-token",
            "</think>",
            "--output",
            str(gen_out),
        ],
    )
    assert res_gen.exit_code == 0, res_gen.stdout
    sample_row = json.loads(
        (gen_out / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    raw0 = sample_row["raw_responses"][0]
    assert raw0["text"] == "Pancreas"
    assert raw0["reasoning_trace"] is not None
    assert "step2" in raw0["reasoning_trace"]
    gen_manifest = ReceiptManifest.model_validate_json(
        (gen_out / "manifest.json").read_text(encoding="utf-8")
    )
    assert gen_manifest.metrics["exact_match"].value == pytest.approx(1.0)

    mc_jsonl = tmp_path / "mc_metrics.jsonl"
    mc_rows = [
        {
            "sample_id": f"mc-{i}",
            "input_text": f"Clinical vignette {i}",
            "choices": ["Option A", "Option B", "Option C"],
            "target": 0,
        }
        for i in range(4)
    ]
    mc_jsonl.write_text("\n".join(json.dumps(r) for r in mc_rows) + "\n", encoding="utf-8")
    mc_yaml = tmp_path / "mc_metrics.yaml"
    mc_yaml.write_text(
        f"""name: t1_token_metrics_mc
trust_tier: tier1_zero_code
output_type: multiple_choice
dataset_path: "{mc_jsonl.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics:
  - acc_bytes
  - acc_norm
  - acc_mutual_info
  - brier_score
  - classification_f1
""",
        encoding="utf-8",
    )
    mc_out = tmp_path / "mc_out"
    res_mc = cli_runner.invoke(
        app,
        ["run", "--task", str(mc_yaml), "--model", "mock", "--output", str(mc_out)],
    )
    assert res_mc.exit_code == 0, res_mc.stdout
    mc_manifest = ReceiptManifest.model_validate_json(
        (mc_out / "manifest.json").read_text(encoding="utf-8")
    )
    for metric_name in (
        "acc_bytes",
        "acc_norm",
        "acc_mutual_info",
        "brier_score",
        "classification_f1",
    ):
        assert metric_name in mc_manifest.metrics
        assert math.isfinite(mc_manifest.metrics[metric_name].value)
        assert mc_manifest.metrics[metric_name].stderr is not None

    ll_yaml = tmp_path / "ll_metrics.yaml"
    ll_yaml.write_text(
        f"""name: t1_rolling_metrics
trust_tier: tier1_zero_code
output_type: loglikelihood_rolling
dataset_path: "{gen_csv.as_posix()}"
doc_to_text: "{{{{ input_text }}}}"
doc_to_target: "{{{{ target }}}}"
metrics:
  - bits_per_byte
  - word_perplexity
  - byte_perplexity
""",
        encoding="utf-8",
    )
    ll_out = tmp_path / "ll_out"
    res_ll = cli_runner.invoke(
        app,
        ["run", "--task", str(ll_yaml), "--model", "mock", "--output", str(ll_out)],
    )
    assert res_ll.exit_code == 0, res_ll.stdout
    ll_manifest = ReceiptManifest.model_validate_json(
        (ll_out / "manifest.json").read_text(encoding="utf-8")
    )
    for metric_name in ("bits_per_byte", "word_perplexity", "byte_perplexity"):
        assert metric_name in ll_manifest.metrics
        assert math.isfinite(ll_manifest.metrics[metric_name].value)

    # F8: Additional tokenization-agnostic / exact / calibration metrics
    assert latex_exact_match(r"\boxed{\frac{1}{2}}", r"\dfrac12") == 1.0
    assert latex_exact_match(r"\sqrt[3]{8}", r"\sqrt{8}") == 0.0
    assert whole_word_match("The category is benign.", "cat") == 0.0
    assert whole_word_match("A cat was observed.", "cat") == 1.0
    assert token_f1("the pancreas", "Pancreas") == pytest.approx(1.0)
    assert 0.0 < token_f1("pancreas gland", "pancreas") < 1.0
    assert acc_all([1.0, 1.0, 1.0]) == 1.0
    assert acc_all([1.0, 0.0, 1.0]) == 0.0
    assert bypass(log_samples=True) is None
    with pytest.raises(ValueError, match="log_samples=True"):
        bypass(log_samples=False)
    assert exact_match("  İSTANBUL  ", "istanbul") == 1.0


def test_t1_olmes_dual_cloze_mcf_and_filter_ensemble(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    """T1-CORE-05: OLMES dual Cloze + MCF (`\\n A`) and multi-pipeline FilterEnsemble."""
    mcqa_jsonl = tmp_path / "olmes_mcqa.jsonl"
    rows = [
        {
            "sample_id": f"olmes-{i}",
            "input_text": f"Patient {i} presents with fever and cough.",
            "choices": ["Pneumonia", "Appendicitis", "Fracture", "Migraine"],
            "target": 0,
        }
        for i in range(6)
    ]
    mcqa_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    task_yaml = tmp_path / "olmes_task.yaml"
    task_yaml.write_text(
        f"""name: t1_olmes_dual
version: "0.1.0"
trust_tier: tier2_declarative
output_type: multiple_choice
dataset_path: "{mcqa_jsonl.as_posix()}"
doc_to_text: |-
  Question: {{{{ input_text }}}}
  Choices:
   A. {{{{ choices[0] }}}}
   B. {{{{ choices[1] }}}}
   C. {{{{ choices[2] }}}}
   D. {{{{ choices[3] }}}}
  Answer:
doc_to_target: "{{{{ target }}}}"
metrics:
  - acc_bytes
  - acc_norm
filter_ensembles:
  - name: strict
    steps:
      - name: multi_choice_regex
        params:
          fallback: "[invalid]"
      - name: take_first
  - name: voted
    steps:
      - name: multi_choice_regex
        params:
          fallback: "[invalid]"
      - name: majority_vote
generation_kwargs:
  repeats: 3
mc_format: dual
olmes_option_labels: ["A", "B", "C", "D"]
olmes:
  enabled: true
  formats: [cloze, mcf]
  choice_prefix: "\\n "
  metric_preference: acc_norm
""",
        encoding="utf-8",
    )
    loaded_olmes_spec = load_task_spec(task_yaml)
    assert loaded_olmes_spec.mc_format == MultipleChoiceFormat.DUAL
    assert loaded_olmes_spec.olmes is not None and loaded_olmes_spec.olmes.enabled is True
    assert loaded_olmes_spec.olmes_option_labels == ["A", "B", "C", "D"]

    out_dir = tmp_path / "olmes_out"
    result = cli_runner.invoke(
        app,
        ["run", "--task", str(task_yaml), "--model", "mock", "--output", str(out_dir)],
    )
    assert result.exit_code == 0, result.stdout

    samples = [
        json.loads(line)
        for line in (out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(samples) == 6
    for rec in samples:
        assert "strict" in rec["filtered_responses"]
        assert "voted" in rec["filtered_responses"]
        assert len(rec["raw_responses"]) >= 1

    manifest_text = (out_dir / "manifest.json").read_text(encoding="utf-8").casefold()
    assert "cloze" in manifest_text
    assert "mcf" in manifest_text


def test_t1_stats_suite_and_fewshot_rotation(
    tmp_path: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-CORE-06: Few-shot active-doc exclusion, K-set rotation, micro/macro SE, unresolved CI."""
    sub_a = tmp_path / "subtask_a.jsonl"
    sub_b = tmp_path / "subtask_b.jsonl"
    items_a = [
        {
            "sample_id": f"s0_{i}",
            "input_text": f"UNIQUE_DOC_0_{i}: Identify biomarker {i}.",
            "target": "A",
        }
        for i in range(6)
    ]
    sub_a.write_text("\n".join(json.dumps(x) for x in items_a) + "\n", encoding="utf-8")

    items_b = [
        {
            "sample_id": f"s1_{i}",
            "input_text": f"UNIQUE_DOC_1_{i}: Identify secondary marker {i}.",
            "target": "A" if i < 2 else "B",
        }
        for i in range(4)
    ]
    sub_b.write_text("\n".join(json.dumps(x) for x in items_b) + "\n", encoding="utf-8")

    task_a = tmp_path / "subtask_a.yaml"
    task_a.write_text(
        f"""name: subtask_a
trust_tier: tier2_declarative
output_type: generate_until
dataset_path: "{sub_a.as_posix()}"
doc_to_text: "Case: {{{{ input_text }}}}\\nAnswer:"
doc_to_target: "{{{{ target }}}}"
num_fewshot: 2
fewshot_rotation_sets: 4
gen_prefix: "Answer:"
metrics: [exact_match]
""",
        encoding="utf-8",
    )
    task_b = tmp_path / "subtask_b.yaml"
    task_b.write_text(
        f"""name: subtask_b
trust_tier: tier2_declarative
output_type: generate_until
dataset_path: "{sub_b.as_posix()}"
doc_to_text: "Case: {{{{ input_text }}}}\\nAnswer:"
doc_to_target: "{{{{ target }}}}"
num_fewshot: 2
fewshot_rotation_sets: 2
gen_prefix: "Answer:"
metrics: [exact_match]
""",
        encoding="utf-8",
    )

    captured_contexts: dict[str, list[tuple[str, str]]] = {"subtask_a": [], "subtask_b": []}
    orig_generate_until = MockGateway.generate_until

    def _spy_generate_until(self: MockGateway, requests: Any) -> Any:
        for req in requests:
            captured_contexts.setdefault(req.task_name, []).append((req.sample_id, req.context))
        return orig_generate_until(self, requests)

    monkeypatch.setattr(MockGateway, "generate_until", _spy_generate_until)

    out_a = tmp_path / "out_a"
    res_a = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_a),
            "--model",
            "mock",
            "--bootstrap-iters",
            "1000",
            "--output",
            str(out_a),
        ],
    )
    assert res_a.exit_code == 0, res_a.stdout
    samples_raw_a = (out_a / "samples.jsonl").read_text(encoding="utf-8")
    assert len(samples_raw_a.splitlines()) == 6

    out_b = tmp_path / "out_b"
    res_b = cli_runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_b),
            "--model",
            "mock",
            "--bootstrap-iters",
            "1000",
            "--output",
            str(out_b),
        ],
    )
    assert res_b.exit_code == 0, res_b.stdout
    samples_raw_b = (out_b / "samples.jsonl").read_text(encoding="utf-8")
    assert len(samples_raw_b.splitlines()) == 4

    # F11: Verify active doc never leaks into its own few-shot exemplars & K sets rotate
    assert len(captured_contexts["subtask_a"]) == 6
    prefixes_a: set[str] = set()
    for sample_id, prompt_ctx in captured_contexts["subtask_a"]:
        idx_str = sample_id.split("_", 1)[1]
        active_marker = f"UNIQUE_DOC_0_{idx_str}:"
        assert prompt_ctx.count(active_marker) == 1
        other_markers = [
            f"UNIQUE_DOC_0_{j}:"
            for j in range(6)
            if j != int(idx_str) and f"UNIQUE_DOC_0_{j}:" in prompt_ctx
        ]
        assert len(other_markers) == 2
        prefixes_a.add(prompt_ctx.split(active_marker, 1)[0])
    assert len(prefixes_a) >= 2

    assert len(captured_contexts["subtask_b"]) == 4
    for sample_id, prompt_ctx in captured_contexts["subtask_b"]:
        idx_str = sample_id.split("_", 1)[1]
        active_marker = f"UNIQUE_DOC_1_{idx_str}:"
        assert prompt_ctx.count(active_marker) == 1

    # Verify statistical engine invariants across subtasks and close model deltas
    manifest_a = ReceiptManifest.model_validate_json(
        (out_a / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_b = ReceiptManifest.model_validate_json(
        (out_b / "manifest.json").read_text(encoding="utf-8")
    )

    scores_a = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    scores_b = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    boot_a = bootstrap_stderr(scores_a, bootstrap_iters=1000, seed=1234)
    assert boot_a.iters_executed == 1000

    subtasks = [
        SubtaskStats(
            name="subtask_a",
            score=manifest_a.metrics["exact_match"].value,
            stderr=manifest_a.metrics["exact_match"].stderr or 0.05,
            n_samples=manifest_a.metrics["exact_match"].n_samples,
        ),
        SubtaskStats(
            name="subtask_b",
            score=manifest_b.metrics["exact_match"].value,
            stderr=manifest_b.metrics["exact_match"].stderr or 0.10,
            n_samples=manifest_b.metrics["exact_match"].n_samples,
        ),
    ]
    micro = aggregate_group(subtasks, weight_by_size=True)
    macro = aggregate_group(subtasks, weight_by_size=False)
    assert micro.stderr_method == "pooled_sample_stderr"
    assert macro.stderr_method == "unweighted_mean_stderr"
    assert micro.score != pytest.approx(macro.score)

    c_se = clustered_stderr(scores_a, cluster_ids=["c1", "c1", "c2", "c2", "c3", "c3"])
    assert math.isfinite(c_se.stderr) and c_se.stderr >= 0.0

    mcnemar_res = mcnemar_test(scores_a, scores_b)
    pboot_res = paired_bootstrap_test(scores_a, scores_b, bootstrap_iters=500, seed=42)
    assert 0.0 <= mcnemar_res.p_value <= 1.0
    assert pboot_res.ci_low <= pboot_res.ci_high

    comparison = evaluate_fewshot_comparison([0.02, -0.01, 0.01, -0.015])
    assert comparison.status == ComparisonStatus.UNRESOLVED


def test_t1_wal_sqlite_cache_hit_and_invalidation(
    tmp_path: Path,
    examples_dir: Path,
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1-CORE-07: User-dir WAL + SQLite cache hits and SHA-256 prompt invalidation."""
    import eval_studio

    cache_dir = tmp_path / "wal_sqlite_cache"
    monkeypatch.setenv("EVAL_STUDIO_CACHE_DIR", str(cache_dir))

    csv_copy = tmp_path / "sample.csv"
    shutil.copy2(examples_dir / "tier1_zero_code" / "sample.csv", csv_copy)
    task_yaml = tmp_path / "cached_task.yaml"
    task_yaml.write_text(
        f"""name: t1_cache_task
version: "0.1.0"
trust_tier: tier1_zero_code
output_type: generate_until
dataset_path: "{csv_copy.as_posix()}"
doc_to_text: "Question: {{{{ input_text }}}}\\nAnswer:"
doc_to_target: "{{{{ target }}}}"
metrics: [exact_match]
""",
        encoding="utf-8",
    )

    out1 = tmp_path / "run1"
    r1 = cli_runner.invoke(
        app,
        ["run", "--task", str(task_yaml), "--model", "mock", "--output", str(out1)],
    )
    assert r1.exit_code == 0, r1.stdout

    pkg_dir = Path(eval_studio.__file__).resolve().parent
    assert list(pkg_dir.rglob("*.db")) == []
    assert list(pkg_dir.rglob("*.sqlite*")) == []

    all_cached_files = list(cache_dir.rglob("*"))
    assert any(p.suffix in (".db", ".sqlite", ".sqlite3") for p in all_cached_files)
    assert any(p.suffix in (".jsonl", ".wal") for p in all_cached_files)
    assert not any(p.suffix in (".pkl", ".pickle", ".dill") for p in all_cached_files)

    wal_files = [p for p in all_cached_files if p.suffix in (".jsonl", ".wal")]
    wal_lines_after_r1 = sum(
        len(p.read_text(encoding="utf-8").splitlines()) for p in wal_files if p.is_file()
    )

    out2 = tmp_path / "run2"
    r2 = cli_runner.invoke(
        app,
        ["run", "--task", str(task_yaml), "--model", "mock", "--output", str(out2)],
    )
    assert r2.exit_code == 0, r2.stdout
    wal_lines_after_r2 = sum(
        len(p.read_text(encoding="utf-8").splitlines()) for p in wal_files if p.is_file()
    )
    assert wal_lines_after_r2 == wal_lines_after_r1

    # Mutate prompt template -> SHA-256 fingerprint must invalidate cache
    edited_yaml = task_yaml.read_text(encoding="utf-8").replace(
        "Question: {{ input_text }}", "Prompt: {{ input_text }}"
    )
    task_yaml.write_text(edited_yaml, encoding="utf-8")

    out3 = tmp_path / "run3"
    r3 = cli_runner.invoke(
        app,
        ["run", "--task", str(task_yaml), "--model", "mock", "--output", str(out3)],
    )
    assert r3.exit_code == 0, r3.stdout
    wal_lines_after_r3 = sum(
        len(p.read_text(encoding="utf-8").splitlines()) for p in wal_files if p.is_file()
    )
    assert wal_lines_after_r3 > wal_lines_after_r2

    # Verify Windows deterministic handle cleanup
    shutil.rmtree(cache_dir)
    assert not cache_dir.exists()
