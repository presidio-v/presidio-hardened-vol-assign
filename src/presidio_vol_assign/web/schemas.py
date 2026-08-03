"""Request/response shapes for the demo API.

The pydantic model here only describes the wire format so that the generated
OpenAPI docs are accurate. The authority on what is actually permitted is
:func:`presidio_vol_assign.web.runner.build_request`, which clamps every knob
against its scenario and enforces the server-side caps.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunPayload(BaseModel):
    """Body of ``POST /api/run``."""

    model_config = {"extra": "forbid"}

    scenario: str = Field(description="Scenario id, e.g. 'relief-centres'.")
    knobs: dict[str, float] = Field(
        default_factory=dict,
        description="Slider values; unknown keys are ignored and out-of-range values clamped.",
    )
    solver: str = Field(default="nsga2", description="One of 'nsga2', 'nrga', 'both'.")
    seed: int = Field(default=42, ge=0, description="Reproducibility seed.")
    pop_size: int = Field(default=100, ge=10, le=200, description="GA population size.")
    generations: int = Field(default=120, ge=5, le=200, description="GA generation count.")
    evidence: bool = Field(
        default=False,
        description="Request a signed evidence record (ignored if the server has no key).",
    )

    def to_request_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ErrorResponse(BaseModel):
    """Uniform error body."""

    detail: str
