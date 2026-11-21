"""Load and validate config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from flatprice.models import SearchSpec


class ScrapeConfig(BaseModel):
    delay_seconds: tuple[float, float] = (2.0, 4.0)
    headless: bool = False
    max_pages_per_search: int = 50


class TaxProfileConfig(BaseModel):
    canton: str = "TI"
    filing: str = "single"  # single | married
    taxable_income: float = 120_000
    taxable_wealth: float = 200_000
    reference_bfs: int = 5192  # Lugano


class ScoringConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {"value": 0.55, "tax": 0.25, "price_drop": 0.10, "market_time": 0.10}
    )
    min_listings_for_model: int = 30


class QualityConfig(BaseModel):
    """Data-quality rules against perturbed ("poisoned") portal data, see flatprice/quality.py."""

    # Rent factors an anti-bot system has been seen to apply (new / old rent).
    poison_ratios: list[float] = Field(default_factory=lambda: [1.366])
    ratio_tolerance: float = 0.006  # |r / ratio - 1| below this matches (+ CHF 10 rounding slack)
    # Portals whose data is trusted as-is (never flagged; reference for cross-portal checks).
    trusted_portals: list[str] = Field(default_factory=lambda: ["flatfox"])
    xportal_rent_tolerance: float = 0.03
    xportal_max_days: float = 3.0
    # A page whose observations are mostly flagged poisons the rest of it too.
    page_min_flagged: int = 3
    page_flagged_share: float = 0.5
    # A run with more flagged observations than this share is "poisoned": its
    # observations are kept (flagged) but listing fields are not updated, and
    # `flatprice scrape` exits non-zero.
    max_flagged_share: float = 0.30


class Config(BaseModel):
    db_path: Path = Path("data/flatprice.db")
    dist_dir: Path = Path("dist")
    searches: list[SearchSpec] = Field(default_factory=list)
    scrape: ScrapeConfig = Field(default_factory=ScrapeConfig)
    tax_profile: TaxProfileConfig = Field(default_factory=TaxProfileConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f) or {})
