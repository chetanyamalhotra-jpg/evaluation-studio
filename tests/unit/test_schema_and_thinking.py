"""Unit tests for F4–F6: multimodal ChatMessages, RuntimeLimits, and thinking extraction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eval_studio.core.schema import (
    ChatMessage,
    EvalRequest,
    EvalResponse,
    ImageContentPart,
    OutputType,
    RuntimeLimits,
    Sample,
    TaskSpec,
    TextContentPart,
    extract_thinking_trace,
    validate_thinking_output_type,
)


def test_multimodal_chat_messages_and_stage_schemas() -> None:
    msg = ChatMessage(
        role="user",
        content=[
            TextContentPart(text="Describe this chest X-ray."),
            ImageContentPart(type="image_url", image_url="https://example.org/cxr.png"),
        ],
    )
    sample = Sample(sample_id="s1", target="Normal", messages=[msg])
    spec = TaskSpec(
        name="cxr_chat",
        output_type=OutputType.CHAT,
        doc_to_messages=[{"role": "user", "content": "{{ input_text }}"}],
    )
    req = EvalRequest(
        request_id="r1",
        sample_id=sample.sample_id,
        task_name=spec.name,
        output_type=OutputType.CHAT,
        messages=[msg],
        limits=RuntimeLimits(max_tokens=512, max_cost_usd=1.0),
        do_sample=True,
    )
    resp = EvalResponse(
        request_id="r1",
        sample_id="s1",
        text="Normal",
        thinking_truncated=False,
        dtype="bfloat16",
        quantization="fp8",
        batch_size=8,
    )
    assert req.output_type == OutputType.CHAT
    assert req.do_sample is True
    assert req.limits.max_tokens == 512
    assert resp.dtype == "bfloat16"
    assert resp.quantization == "fp8"
    assert resp.batch_size == 8


def test_chat_message_and_image_part_validation() -> None:
    with pytest.raises(ValidationError):
        ImageContentPart(type="image_url")
    with pytest.raises(ValidationError):
        ChatMessage(role="   ", content="hello")
    with pytest.raises(ValidationError):
        RuntimeLimits(unknown_limit=10)  # type: ignore[call-arg]


def test_extract_thinking_trace_uses_last_delimiter_and_handles_null_content() -> None:
    raw = "Let me consider </think> internally first.\nFinal thinking.</think>Answer: B"
    extracted = extract_thinking_trace(
        raw,
        enable_thinking=True,
        think_end_token="</think>",
    )
    assert extracted.text == "Answer: B"
    assert extracted.thinking_truncated is False
    assert extracted.warning is None
    assert extracted.reasoning_trace is not None
    assert extracted.reasoning_trace.endswith("</think>")

    # Provider returns content=None + reasoning_content when max_tokens exhausts during thinking
    null_res = extract_thinking_trace(
        None,
        enable_thinking=True,
        think_end_token="</think>",
        reasoning_content="step 1... step 2...",
    )
    assert null_res.text == ""
    assert null_res.thinking_truncated is True
    assert null_res.reasoning_trace == "step 1... step 2..."
    assert null_res.warning is not None

    # Unclosed thinking trace without end token
    unclosed = extract_thinking_trace(
        "<think>still reasoning...",
        enable_thinking=True,
        think_end_token="</think>",
    )
    assert unclosed.text == ""
    assert unclosed.thinking_truncated is True
    assert unclosed.warning is not None

    # API-split reasoning_content + clean content must preserve final answer
    api_split = extract_thinking_trace(
        "  Diagnosis: Pneumonia  ",
        enable_thinking=True,
        think_end_token="</think>",
        reasoning_content="1. Fever + infiltrate -> Pneumonia.",
    )
    assert api_split.text == "Diagnosis: Pneumonia"
    assert api_split.reasoning_trace == "1. Fever + infiltrate -> Pneumonia."
    assert api_split.thinking_truncated is False
    assert api_split.warning is None

    # Unclosed inline <think> inside content still marks truncation even with reasoning_content
    unclosed_with_reasoning = extract_thinking_trace(
        "<think>still thinking without closing",
        enable_thinking=True,
        think_end_token="</think>",
        reasoning_content="partial reasoning",
    )
    assert unclosed_with_reasoning.text == ""
    assert unclosed_with_reasoning.thinking_truncated is True
    assert unclosed_with_reasoning.warning is not None


def test_validate_thinking_output_type_blocks_ll_and_mc() -> None:
    for bad_type in (
        OutputType.LOGLIKELIHOOD,
        OutputType.LOGLIKELIHOOD_ROLLING,
        OutputType.MULTIPLE_CHOICE,
    ):
        with pytest.raises(ValueError, match="enable_thinking"):
            validate_thinking_output_type(bad_type, True)
    validate_thinking_output_type(OutputType.GENERATE_UNTIL, True)
    validate_thinking_output_type(OutputType.CHAT, True)
