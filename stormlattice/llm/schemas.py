"""Step 8: structured output the LLM must produce when parsing aviation text products.

The LLM fills these; code validates geometry and times before anything reaches the router.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Hazard(BaseModel):
    source_id: str = Field(description="NOTAM number, SIGMET id or PIREP id, copied verbatim")
    kind: Literal["tfr", "closure", "obstacle", "convective_sigmet", "sigmet", "pirep", "other"]
    polygon_lonlat: list[tuple[float, float]] | None = None
    center_lonlat: tuple[float, float] | None = None
    radius_nm: float | None = None
    floor_ft: float | None = None
    ceiling_ft: float | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    summary: str = Field(max_length=200)


class ParsedProducts(BaseModel):
    hazards: list[Hazard]
    unparsed_ids: list[str] = Field(default_factory=list, description="Products the model could not parse")
