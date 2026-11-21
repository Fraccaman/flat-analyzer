"""Cleaning of the Store frame before modelling.

Ported from old/analyze_flats.py `load()` (rent mode) with a few improvements:
categories normalisation, conservative removal of non-residential ads, and
size sanity driven by the CHF/m2 plausibility check.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

RENT_BOUNDS = (300.0, 20_000.0)  # plausible gross monthly rent, CHF
M2_RATE_BOUNDS = (5.0, 150.0)  # plausible CHF/m2/month
SIZE_BOUNDS = (12.0, 400.0)  # plausible living space, m2
ROOMS_BOUNDS = (0.5, 15.0)

FURNISHED_RE = r"arredat|ammobiliat|furnish|möbliert|moebliert|meubl"
FEATURE_COLS = ["is_furnished", "has_balcony", "has_parking", "has_elevator", "pets_allowed"]

# A listing is dropped only if it HAS categories and EVERY one of them looks
# non-residential (parking slots, garages, shops, offices, ...).
NON_RESIDENTIAL_RE = (
    r"PARK|GARAGE|COMMERCIAL|OFFICE|SHOP|RETAIL|INDUSTR|WAREHOUSE|STORAGE|"
    r"GASTRO|RESTAURANT|HOTEL|PLOT|LAND|AGRICULT|WORKSHOP|HOBBY|ADVERTIS|PRACTICE"
)


def _to_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, float) and np.isnan(v):
        return []
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            parsed = [p for p in s.split(",")]
        v = parsed if isinstance(parsed, list) else [parsed]
    if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(x).strip().upper() for x in v if x is not None and str(x).strip()]
    return [str(v).strip().upper()]


def _to_bool(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    if v is pd.NA:
        return False
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(v)


def is_non_residential(categories: list[str]) -> bool:
    import re

    if not categories:
        return False
    return all(re.search(NON_RESIDENTIAL_RE, c) for c in categories)


def clean(df: pd.DataFrame, active_only: bool = True) -> pd.DataFrame:
    """Return a cleaned copy of the Store frame.

    - keeps only `active` rows (if `active_only` and the column exists)
    - rent_gross falls back to rent_net when missing (`rent_imputed` = True)
    - drops rows with rent_gross outside RENT_BOUNDS
    - living_space_m2 outside SIZE_BOUNDS -> NaN; rooms outside ROOMS_BOUNDS -> NaN
    - chf_per_m2 = rent_gross / m2; if outside M2_RATE_BOUNDS both chf_per_m2
      and living_space_m2 are set to NaN (the size is implausible for the rent)
    - utilities_pct = (gross - net) / gross * 100 where gross > net
    - m2_per_room, chf_per_room
    - feature booleans normalised (None -> False); is_furnished OR title regex
    - categories normalised to an upper-case list; non-residential ads dropped
    """
    df = df.copy()
    if df.empty:
        return df.reset_index(drop=True)

    if active_only and "active" in df.columns:
        df = df[df["active"].map(_to_bool)]

    for col in ["rent_gross", "rent_net", "rooms", "living_space_m2", "latitude", "longitude",
                "year_built", "days_on_market", "price_drop_pct"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Gross rent is the reference price; fall back to net when gross is missing.
    df["rent_imputed"] = df["rent_gross"].isna() & df["rent_net"].notna()
    df.loc[df["rent_imputed"], "rent_gross"] = df.loc[df["rent_imputed"], "rent_net"]
    df = df[df["rent_gross"].between(*RENT_BOUNDS)]

    df.loc[~df["living_space_m2"].between(*SIZE_BOUNDS), "living_space_m2"] = np.nan
    df.loc[~df["rooms"].between(*ROOMS_BOUNDS), "rooms"] = np.nan

    df["chf_per_m2"] = df["rent_gross"] / df["living_space_m2"]
    bad_rate = df["chf_per_m2"].notna() & ~df["chf_per_m2"].between(*M2_RATE_BOUNDS)
    df.loc[bad_rate, ["chf_per_m2", "living_space_m2"]] = np.nan

    both = df["rent_net"].notna() & (df["rent_gross"] > df["rent_net"])
    df["utilities_pct"] = np.nan
    df.loc[both, "utilities_pct"] = (df["rent_gross"] - df["rent_net"]) / df["rent_gross"] * 100

    df["m2_per_room"] = df["living_space_m2"] / df["rooms"]
    df["chf_per_room"] = df["rent_gross"] / df["rooms"]

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].map(_to_bool).astype(bool)
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    df["is_furnished"] = df["is_furnished"] | title.fillna("").astype(str).str.lower().str.contains(
        FURNISHED_RE, regex=True
    )

    cats = df["categories"] if "categories" in df.columns else pd.Series([[]] * len(df), index=df.index)
    df["categories"] = [_to_list(v) for v in cats]
    df = df[~df["categories"].map(is_non_residential)]

    if "price_drop_pct" in df.columns:
        df["price_drop_pct"] = df["price_drop_pct"].clip(lower=0)

    return df.reset_index(drop=True)
