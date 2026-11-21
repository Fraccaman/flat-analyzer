"""Canonical, portal-agnostic data models.

Every portal adapter converts its raw payload into `Listing`. Everything
downstream (store, analysis, site) only ever sees `Listing` / its dict form.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SearchSpec(BaseModel):
    """One configured search. Structured filters are translated to a URL by the
    portal adapter; `url` overrides them when set (escape hatch)."""

    label: str
    portal: str = "homegate"
    locations: list[str] = Field(default_factory=list)  # e.g. ["lugano", "paradiso"]
    rooms_min: Optional[float] = None
    rooms_max: Optional[float] = None
    rent_min: Optional[int] = None
    rent_max: Optional[int] = None
    size_min: Optional[int] = None
    size_max: Optional[int] = None
    url: Optional[str] = None


class Listing(BaseModel):
    portal: str
    portal_id: str
    url: str
    search: str  # label of the SearchSpec that found it (first one wins)
    title: str = ""
    street: str = ""
    postal_code: str = ""
    locality: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rent_gross: Optional[float] = None  # CHF / month, utilities included
    rent_net: Optional[float] = None  # CHF / month, utilities excluded
    rooms: Optional[float] = None
    living_space_m2: Optional[float] = None
    floor: Optional[int] = None
    year_built: Optional[int] = None
    is_furnished: bool = False
    has_balcony: Optional[bool] = None
    has_parking: Optional[bool] = None
    has_elevator: Optional[bool] = None
    pets_allowed: Optional[bool] = None
    categories: list[str] = Field(default_factory=list)
    image_url: Optional[str] = None
    # Ids of the same flat on other portals, e.g. {"homegate": "4001234567"} for a
    # Flatfox ad syndicated from homegate. Used by flatprice.dedup.
    alt_ids: dict[str, str] = Field(default_factory=dict)
    # Result page the listing was found on (homegate); stored per observation so
    # data-quality rules can spot a whole poisoned page (flatprice.quality).
    page: Optional[int] = Field(default=None, exclude=True)
    raw: Optional[dict] = Field(default=None, exclude=True)  # portal payload, not persisted by default

    @property
    def key(self) -> str:
        return f"{self.portal}:{self.portal_id}"


class Observation(BaseModel):
    """One sighting of a listing in one scrape run."""

    run_id: int
    listing_key: str
    seen_at: datetime
    rent_gross: Optional[float] = None
    rent_net: Optional[float] = None
