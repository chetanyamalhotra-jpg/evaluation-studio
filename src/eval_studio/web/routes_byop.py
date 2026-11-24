"""BYOP Paper Studio endpoints: list, upload, two-stage compile, and SHA-256 lock."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import ValidationError

from eval_studio.gateway.base import InferenceGateway
from eval_studio.gateway.http import HttpBYOKGateway
from eval_studio.gateway.mock import MockGateway
from eval_studio.gateway.secrets import SecretRole, redact_secrets
from eval_studio.paper.compiler import (
    compile_paper,
    sha256_file,
    suite_to_task_yaml_dict,
    write_compiled_pack,
)
from eval_studio.paper.extract import PaperExtractor, read_paper_text
from eval_studio.paper.schema import CompiledPaperSuite, ExtractedItem, VerificationStatus
from eval_studio.paper.verify import AnswerabilityVerifier
from eval_studio.web.schemas import (
    ByopCompileRequest,
    ByopCompileResponse,
    ByopLockRequest,
    ByopLockResponse,
    ByopPapersResponse,
    ByopRejectedItem,
    ByopUploadResponse,
    CompiledPaperSummary,
    PaperSampleSummary,
)
from eval_studio.web.server import AppState, get_studio_state

router = APIRouter(prefix="/api/byop", tags=["byop"])

StateDep = Annotated[AppState, Depends(get_studio_state)]

_ALLOWED_PAPER_SUFFIXES: frozenset[str] = frozenset({".md", ".txt", ".pdf"})
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_PAGE_MARKER_RE = re.compile(r"\[PAGE\s+\d+\]")


def _sanitize_filename(name: str, default_ext: str = ".md") -> str:
    base = Path(name).name.strip()
    if not base:
        base = f"paper{default_ext}"
    cleaned = _SAFE_STEM_RE.sub("_", base).strip("._")
    if not cleaned:
        cleaned = f"paper{default_ext}"
    if "." not in cleaned:
        cleaned = f"{cleaned}{default_ext}"
    return cleaned


def _format_rel_path(resolved: Path, state: AppState) -> str:
    ws = state.config.workspace_dir.resolve()
    if resolved.is_relative_to(ws):
        return resolved.relative_to(ws).as_posix()
    ex = state.config.examples_dir.resolve()
    if resolved.is_relative_to(ex):
        return f"examples/{resolved.relative_to(ex).as_posix()}"
    return resolved.as_posix()


def _resolve_paper_path(raw_path: str, state: AppState) -> Path:
    cleaned = raw_path.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="paper_path must be non-empty.")

    candidate = Path(cleaned)
    ws = state.config.workspace_dir.resolve()
    ex = state.config.examples_dir.resolve()
    upl = state.config.uploads_dir.resolve()
    tsk = state.config.tasks_dir.resolve()

    allowed_roots = [ws, ex, upl, tsk]
    resolved_candidates: list[Path] = []
    if candidate.is_absolute():
        resolved_candidates.append(candidate.resolve())
    else:
        resolved_candidates.append((ws / candidate).resolve())
        resolved_candidates.append((upl / candidate).resolve())
        if cleaned.startswith("examples/"):
            rel = cleaned[len("examples/") :]
            resolved_candidates.append((ex / rel).resolve())
        else:
            resolved_candidates.append((ex / "papers" / candidate).resolve())

    for resolved in resolved_candidates:
        if any(resolved.is_relative_to(root) for root in allowed_roots) and resolved.is_file():
            return resolved

    # Explicitly reject traversal outside workspace/allowed roots
    for resolved in resolved_candidates:
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(
                status_code=400,
                detail="Access denied: paper_path must reside inside the workspace.",
            )
    raise HTTPException(status_code=404, detail=f"Paper file not found: {raw_path}")


@router.get("/papers", response_model=ByopPapersResponse)
def list_papers(state: StateDep) -> ByopPapersResponse:
    """List sample papers under examples/papers, uploaded papers, and compiled suites."""
    samples: list[PaperSampleSummary] = []
    seen_paths: set[Path] = set()

    papers_dirs = [
        state.config.examples_dir / "papers",
        state.config.uploads_dir,
    ]
    for directory in papers_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _ALLOWED_PAPER_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            samples.append(
                PaperSampleSummary(
                    name=path.name,
                    path=_format_rel_path(resolved, state),
                )
            )

    compiled_suites: list[CompiledPaperSummary] = []
    compiled_dirs = [
        state.config.tasks_dir / "compiled",
        state.config.tasks_dir,
    ]
    seen_suites: set[Path] = set()
    for directory in compiled_dirs:
        if not directory.is_dir():
            continue
        for json_path in sorted(directory.glob("*.json")):
            resolved = json_path.resolve()
            if resolved in seen_suites:
                continue
            try:
                suite = CompiledPaperSuite.model_validate_json(
                    json_path.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            seen_suites.add(resolved)
            yaml_sibling = json_path.with_suffix(".yaml")
            display_path = yaml_sibling if yaml_sibling.is_file() else json_path
            compiled_suites.append(
                CompiledPaperSummary(
                    name=suite.suite_name,
                    path=_format_rel_path(display_path.resolve(), state),
                    suite_sha256=suite.suite_sha256,
                    source_sha256=suite.source_sha256,
                    item_count=len(suite.items),
                )
            )

    return ByopPapersResponse(samples=samples, compiled_suites=compiled_suites)


@router.post("/upload", response_model=ByopUploadResponse)
async def upload_paper(
    state: StateDep,
    file: Annotated[UploadFile, File(...)],
) -> ByopUploadResponse:
    """Upload a .md, .txt, or .pdf paper and extract preview text + SHA-256."""
    raw_name = file.filename or "paper.md"
    suffix = Path(raw_name).suffix.lower()
    if suffix not in _ALLOWED_PAPER_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported paper extension '{suffix or '(none)'}'. "
                "Allowed extensions: .md, .txt, .pdf."
            ),
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded paper file is empty.")

    safe_name = _sanitize_filename(raw_name, default_ext=suffix)
    uploads_dir = state.config.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = (uploads_dir / safe_name).resolve()
    if not dest.is_relative_to(uploads_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid upload filename.")

    dest.write_bytes(payload)
    try:
        text = read_paper_text(dest)
    except ImportError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(
                f"{exc} Install optional PDF support via: "
                "pip install -e '.[pdf]' (evaluation-studio[pdf])."
            ),
        ) from exc
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(f"Failed to parse uploaded paper: {exc}"),
        ) from exc

    page_markers = _PAGE_MARKER_RE.findall(text)
    page_count = max(1, len(page_markers))
    digest = sha256_file(dest)

    return ByopUploadResponse(
        paper_path=_format_rel_path(dest, state),
        filename=safe_name,
        sha256=digest,
        char_count=len(text),
        page_count=page_count,
        preview_text=text[:2000],
    )


def _build_byop_gateways(
    req: ByopCompileRequest,
) -> tuple[InferenceGateway | None, InferenceGateway | None, str, str]:
    if req.mode == "heuristic":
        return (
            None,
            None,
            req.extractor_model or "heuristic-extractor",
            req.verifier_model or "heuristic-verifier",
        )
    if req.mode == "mock":
        extractor_label = req.extractor_model or "mock-extractor"
        verifier_label = req.verifier_model or "mock-verifier"
        mock_extractor = MockGateway(model_id=extractor_label, compiler_mode="mixed")
        return mock_extractor, None, extractor_label, verifier_label

    # http-byok mode
    extractor_model = req.extractor_model or "gpt-4o-mini"
    verifier_model = req.verifier_model or extractor_model
    extractor_gw: InferenceGateway = HttpBYOKGateway(
        model_id=extractor_model,
        base_url=req.extractor_base_url,
        provider=req.provider,
        role=SecretRole.EXTRACTOR,
    )
    verifier_gw: InferenceGateway = HttpBYOKGateway(
        model_id=verifier_model,
        base_url=req.verifier_base_url or req.extractor_base_url,
        provider=req.provider,
        role=SecretRole.JUDGE,
    )
    return extractor_gw, verifier_gw, extractor_model, verifier_model


@router.post("/compile", response_model=ByopCompileResponse)
def compile_paper_endpoint(
    payload: ByopCompileRequest,
    state: StateDep,
) -> ByopCompileResponse:
    """Run Stage 1 grounded extraction and Stage 2 answerability verification gate."""
    paper_path = _resolve_paper_path(payload.paper_path, state)
    try:
        source_text = read_paper_text(paper_path)
    except ImportError as exc:
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(f"{exc} (requires evaluation-studio[pdf])"),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=redact_secrets(f"Failed to read paper: {exc}"),
        ) from exc

    extractor_gw, verifier_gw, extractor_model, verifier_model = _build_byop_gateways(payload)

    try:
        extractor = PaperExtractor(extractor_model=extractor_model, gateway=extractor_gw)
        candidates = extractor.extract_from_path(paper_path)
    except Exception as exc:
        raise HTTPException(
            status_code=502 if payload.mode == "http-byok" else 422,
            detail=redact_secrets(f"Stage 1 paper extraction failed: {exc}"),
        ) from exc

    verifier = AnswerabilityVerifier(verifier_model=verifier_model, gateway=verifier_gw)
    approved_items: list[dict[str, Any]] = []
    rejected_items: list[ByopRejectedItem] = []

    for item in candidates:
        outcome = verifier.verify_item(item, source_text=source_text)
        item_dict = item.model_dump(mode="json")
        if outcome.status == VerificationStatus.APPROVED:
            item_dict["verification_status"] = VerificationStatus.APPROVED.value
            item_dict["verification_notes"] = ""
            approved_items.append(item_dict)
        else:
            item_dict["verification_status"] = VerificationStatus.REJECTED.value
            item_dict["verification_notes"] = "; ".join(outcome.reasons)
            rejected_items.append(ByopRejectedItem(item=item_dict, reasons=outcome.reasons))

    return ByopCompileResponse(
        source_path=_format_rel_path(paper_path, state),
        source_sha256=sha256_file(paper_path),
        suite_name=payload.suite_name,
        extractor_model=extractor_model,
        verifier_model=verifier_model,
        approved_items=approved_items,
        rejected_items=rejected_items,
    )


def _parse_extracted_items(raw_items: list[dict[str, Any]]) -> list[ExtractedItem]:
    parsed: list[ExtractedItem] = []
    for idx, raw in enumerate(raw_items):
        try:
            parsed.append(ExtractedItem.model_validate(raw))
        except ValidationError as exc:
            # Allow uncited/partial items to be audited by Stage 2 gate cleanly
            if raw.get("citations") in (None, []):
                parsed.append(
                    ExtractedItem.model_construct(
                        item_id=str(raw.get("item_id") or f"item-{idx + 1:03d}"),
                        question=str(raw.get("question", "")),
                        reference_answer=str(raw.get("reference_answer", "")),
                        rubric=str(raw.get("rubric", "")),
                        category=str(raw.get("category", "claim")),
                        choices=[str(c) for c in (raw.get("choices") or [])],
                        citations=[],
                        verification_status=VerificationStatus.PENDING,
                        verification_notes="",
                        item_sha256=None,
                    )
                )
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid item at index {idx}: {exc}",
                ) from exc
    return parsed


@router.post("/lock", response_model=ByopLockResponse)
def lock_paper_suite(
    payload: ByopLockRequest,
    state: StateDep,
) -> ByopLockResponse:
    """Re-verify items through Stage 2 gate, SHA-256 lock, and export YAML + JSON manifest."""
    paper_path = _resolve_paper_path(payload.paper_path, state)
    items = _parse_extracted_items(payload.items)

    suite = compile_paper(
        paper_path,
        items=items,
        suite_name=payload.suite_name,
        extractor_model=payload.extractor_model,
        verifier_model=payload.verifier_model,
    )
    if not suite.items:
        raise HTTPException(
            status_code=422,
            detail=(
                "No items passed the Stage 2 answerability & citation verification gate. "
                "Every item must have a verbatim quote from the paper and non-leaky prompt."
            ),
        )

    out_dir = state.config.tasks_dir / "compiled"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_stem = payload.output_filename or payload.suite_name
    safe_stem = _SAFE_STEM_RE.sub("_", Path(raw_stem).stem).strip("._") or "compiled_suite"
    yaml_path = (out_dir / f"{safe_stem}.yaml").resolve()
    manifest_path = (out_dir / f"{safe_stem}.json").resolve()

    if not yaml_path.is_relative_to(out_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid output_filename.")

    write_compiled_pack(suite, yaml_path)
    write_compiled_pack(suite, manifest_path)
    yaml_text = yaml.safe_dump(
        suite_to_task_yaml_dict(suite),
        sort_keys=False,
        allow_unicode=True,
    )

    # Ensure JSON manifest is valid UTF-8 JSON
    _ = json.loads(manifest_path.read_text(encoding="utf-8"))

    return ByopLockResponse(
        yaml_path=_format_rel_path(yaml_path, state),
        manifest_path=_format_rel_path(manifest_path, state),
        suite_sha256=suite.suite_sha256,
        source_sha256=suite.source_sha256,
        item_count=len(suite.items),
        yaml_text=yaml_text,
    )


# Keep MockGateway referenced cleanly for future mock-extractor extensions
_ = MockGateway
