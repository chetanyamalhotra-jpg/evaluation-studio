"""Stage 1 BYOP paper parser and candidate item extractor (Markdown/text + optional [pdf])."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from eval_studio.core.schema import EvalRequest, OutputType
from eval_studio.gateway.base import InferenceGateway
from eval_studio.paper.schema import Citation, ExtractedItem, VerificationStatus

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBER_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?%?(?:\s*(?:ng/mL|mg/dL|bpm|hours?|patients?))?\b")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", re.DOTALL)

_EXTRACTION_SCHEMA_PROMPT = """\
You are a scientific benchmark compiler. Extract self-contained, unambiguous evaluation items \
(claims, methods, quantitative table findings, and limitations) strictly grounded in the source \
paper chunk below.

Rules:
- Every item MUST include at least one verbatim citation with 1-based `page` (use `[PAGE N]` \
markers when present, else 1), exact `section`, and an exact verbatim `quote` substring copied \
character-for-character from the text.
- Questions MUST NOT leak `reference_answer` verbatim and MUST NOT reference unseen figures/tables.
- Respond strictly with valid JSON matching this schema:
{{
  "items": [
    {{
      "item_id": "paper-chunk{chunk_idx}-001",
      "question": "Self-contained clinical/scientific question",
      "reference_answer": "Concise gold answer",
      "rubric": "Grading criteria grounded in the citation",
      "category": "claim | method | quantitative_finding | limitation",
      "citations": [
        {{
          "page": 1,
          "section": "Section heading",
          "quote": "Exact verbatim substring from source text"
        }}
      ]
    }}
  ]
}}

Source paper chunk {chunk_idx}/{total_chunks}:
{chunk_text}
"""


def read_paper_text(source_path: Path) -> str:
    """Read source paper text from Markdown/text or optional [pdf] extra (with [PAGE N] markers)."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Source paper not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdf  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "PDF extraction requires optional extra evaluation-studio[pdf] "
                "(install with: pip install 'evaluation-studio[pdf]' / pypdf)."
            ) from exc
        reader = pypdf.PdfReader(str(source_path))
        pages: list[str] = []
        for idx, page in enumerate(reader.pages, start=1):
            page_txt = (page.extract_text() or "").strip()
            if page_txt:
                pages.append(f"[PAGE {idx}]\n{page_txt}")
        text = "\n\n".join(pages).strip()
        if not text:
            raise ValueError(f"PDF file contains no extractable text: {source_path}")
        return text

    raw_bytes = source_path.read_bytes()
    if b"\x00" in raw_bytes:
        raise ValueError(f"Corrupt or binary non-text input rejected: {source_path}")
    text = raw_bytes.decode("utf-8").strip()
    if not text:
        raise ValueError(f"Source paper is empty: {source_path}")
    return text


def _chunk_paper_text(text: str, *, max_chars: int = 12000) -> list[str]:
    """Split multi-page paper text into paragraph-aligned chunks up to max_chars."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for start in range(0, para_len, max_chars):
                chunks.append(para[start : start + max_chars])
            continue

        added_len = para_len + (2 if current else 0)
        if current and current_len + added_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += added_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _extract_deterministic_markdown_items(text: str, stem: str) -> list[ExtractedItem]:
    """Deterministically extract citation-grounded items from Markdown/plain text offline."""
    sections: list[tuple[str, list[str]]] = []
    current_section = "Introduction"
    current_lines: list[str] = []

    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            if current_lines:
                sections.append((current_section, current_lines))
                current_lines = []
            current_section = m.group(2).strip()
        elif line.strip():
            current_lines.append(line.strip())
    if current_lines:
        sections.append((current_section, current_lines))

    items: list[ExtractedItem] = []
    idx = 0
    for section_name, lines in sections:
        paragraph = " ".join(lines)
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", paragraph)
            if len(s.strip().split()) >= 6
            and not s.strip().casefold().startswith("no quantitative")
        ]
        for sentence in sentences:
            idx += 1
            num_match = _NUMBER_TOKEN_RE.search(sentence)
            if num_match:
                ref_answer = num_match.group(0).strip()
                question = (
                    f"According to {section_name}, what quantitative value is reported "
                    f"regarding the primary finding in item {idx}?"
                )
                category = "quantitative_finding"
            else:
                words = sentence.split()
                ref_answer = " ".join(words[-4:]).rstrip(".!?")
                question = (
                    f"Based on {section_name}, summarize the key clinical claim "
                    f"reported in statement {idx}."
                )
                category = "claim"

            # Ensure question never leaks reference_answer verbatim
            if ref_answer and ref_answer.casefold() in question.casefold():
                question = f"What key finding is reported in {section_name} (item {idx})?"

            items.append(
                ExtractedItem(
                    item_id=f"{stem}-item-{idx:03d}",
                    question=question,
                    reference_answer=ref_answer,
                    rubric=f"Answer must match '{ref_answer}' supported by {section_name}.",
                    category=category,
                    citations=[
                        Citation(
                            page=1,
                            section=section_name,
                            quote=sentence,
                        )
                    ],
                )
            )
    return items


def _parse_gateway_items(raw_text: str) -> list[ExtractedItem]:
    stripped = raw_text.strip()
    match = _JSON_FENCE_RE.search(stripped)
    candidate = match.group(1) if match else stripped
    data: Any = json.loads(candidate)
    raw_items = data if isinstance(data, list) else data.get("items", [])
    results: list[ExtractedItem] = []
    for i, entry in enumerate(raw_items):
        if not isinstance(entry, dict):
            continue
        raw_cites = entry.get("citations") or []
        parsed_cites: list[Citation] = []
        for c in raw_cites:
            if isinstance(c, dict):
                try:
                    parsed_cites.append(Citation.model_validate(c))
                except Exception:
                    continue
        if parsed_cites:
            results.append(
                ExtractedItem(
                    item_id=str(entry.get("item_id") or f"item-{i + 1:03d}"),
                    question=str(entry.get("question", "")),
                    reference_answer=str(entry.get("reference_answer", "")),
                    rubric=str(entry.get("rubric", "")),
                    category=str(entry.get("category", "claim")),
                    choices=[str(x) for x in (entry.get("choices") or [])],
                    citations=parsed_cites,
                )
            )
        else:
            # Preserve uncited items via model_construct so Stage 2 gate audits & rejects them
            results.append(
                ExtractedItem.model_construct(
                    item_id=str(entry.get("item_id") or f"item-{i + 1:03d}"),
                    question=str(entry.get("question", "")),
                    reference_answer=str(entry.get("reference_answer", "")),
                    rubric=str(entry.get("rubric", "")),
                    category=str(entry.get("category", "claim")),
                    choices=[str(x) for x in (entry.get("choices") or [])],
                    citations=[],
                    verification_status=VerificationStatus.PENDING,
                    verification_notes="",
                    item_sha256=None,
                )
            )
    return results


class PaperExtractor:
    """Extracts grounded candidate items with mandatory page/section citations."""

    def __init__(
        self,
        extractor_model: str = "mock-extractor",
        *,
        gateway: InferenceGateway | None = None,
        chunk_max_chars: int = 12000,
    ) -> None:
        self.extractor_model = extractor_model
        self.gateway = gateway
        self.chunk_max_chars = max(1000, chunk_max_chars)

    def extract_from_path(self, source_path: Path) -> list[ExtractedItem]:
        """Parse PDF/Markdown and extract candidate items across all paper chunks."""
        text = read_paper_text(source_path)
        if self.gateway is not None:
            chunks = _chunk_paper_text(text, max_chars=self.chunk_max_chars)
            total_chunks = len(chunks)
            chunk_reqs = [
                EvalRequest(
                    request_id=f"extract-{source_path.stem}-chunk{idx}",
                    sample_id=f"{source_path.stem}-chunk{idx}",
                    task_name="compile_paper",
                    output_type=OutputType.GENERATE_UNTIL,
                    context=_EXTRACTION_SCHEMA_PROMPT.format(
                        chunk_idx=idx,
                        total_chunks=total_chunks,
                        chunk_text=chunk,
                    ),
                )
                for idx, chunk in enumerate(chunks, start=1)
            ]
            responses = self.gateway.generate_until(chunk_reqs)
            all_items: list[ExtractedItem] = []
            for resp in responses:
                all_items.extend(_parse_gateway_items(resp.text or "{}"))
            return all_items
        return _extract_deterministic_markdown_items(text, source_path.stem)
