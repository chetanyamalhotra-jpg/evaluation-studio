"""Schemas for BYOP extracted items, mandatory citations, and SHA-256 locked suites."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class VerificationStatus(str, Enum):
    """Stage 2 answerability verification gate outcome."""

    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class Citation(BaseModel):
    """Mandatory page and section provenance for every BYOP benchmark item."""

    page: int = Field(ge=1)
    section: str = Field(min_length=1)
    quote: str = Field(min_length=1)

    @field_validator("section", "quote")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Citation section and quote must be non-blank.")
        return value.strip()


class ExtractedItem(BaseModel):
    """Single candidate item emitted by Stage 1 extraction."""

    item_id: str
    question: str
    reference_answer: str
    rubric: str = ""
    category: str = "claim"
    choices: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(min_length=1)
    verification_status: VerificationStatus = VerificationStatus.PENDING
    verification_notes: str = ""
    item_sha256: str | None = None


class CompiledPaperSuite(BaseModel):
    """Immutable SHA-256 locked BYOP benchmark suite."""

    suite_name: str
    source_path: str
    source_sha256: str
    extractor_model: str
    verifier_model: str
    extraction_mode: str = "mode_a_comprehension"
    items: list[ExtractedItem] = Field(default_factory=list)
    suite_sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)
