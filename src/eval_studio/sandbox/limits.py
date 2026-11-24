"""Resource and network isolation limits for Tier 3 code sandboxes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SandboxLimits(BaseModel):
    """Strict container limits applied to untrusted code execution."""

    timeout_seconds: float = Field(default=10.0, gt=0.0)
    memory_mb: int = Field(default=512, ge=64)
    cpu_quota: float = Field(default=1.0, gt=0.0)
    pids_limit: int = Field(default=64, ge=8)
    network_disabled: bool = True
    read_only_rootfs: bool = True
