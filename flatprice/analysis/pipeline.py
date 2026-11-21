"""End-to-end analysis: Store frame -> scored listings + summary info."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from flatprice.analysis.clean import _to_bool, clean
from flatprice.analysis.fair_rent import fit_fair_rent
from flatprice.analysis.score import score


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA:
        return None
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _jsonable(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _bfs_int(v) -> int | None:
    try:
        if v is None or pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def compute_tax(df: pd.DataFrame, config) -> tuple[pd.DataFrame, dict]:
    """Add tax_annual, tax_delta_month, tax_multiplier per row (computed once per
    unique municipality_bfs). Unknown/missing bfs -> NaN."""
    from flatprice.tax import engine  # lazy for monkeypatching

    tp = config.tax_profile
    profile = engine.TaxProfile(canton=tp.canton, filing=tp.filing,
                                taxable_income=tp.taxable_income, taxable_wealth=tp.taxable_wealth)
    ref = int(tp.reference_bfs)
    df = df.copy()
    bfs = df["municipality_bfs"] if "municipality_bfs" in df.columns else pd.Series(None, index=df.index)
    bfs_int = bfs.map(_bfs_int)
    cache: dict[int, tuple[float, float, float]] = {}
    for b in sorted({b for b in bfs_int if b is not None}):
        try:
            res = engine.annual_tax(profile, b)
            total = float(res["total"])
            mult = res.get("multiplier", np.nan) if isinstance(res, dict) else np.nan
            delta = float(engine.tax_delta_month(profile, b, ref))
            cache[b] = (total, delta, float(mult) if mult is not None else np.nan)
        except (KeyError, ValueError):
            cache[b] = (np.nan, np.nan, np.nan)
    df["municipality_bfs"] = bfs_int.astype("Int64")
    df["tax_annual"] = bfs_int.map(lambda b: cache.get(b, (np.nan,) * 3)[0] if b is not None else np.nan).astype(float)
    df["tax_delta_month"] = bfs_int.map(lambda b: cache.get(b, (np.nan,) * 3)[1] if b is not None else np.nan).astype(float)
    df["tax_multiplier"] = bfs_int.map(lambda b: cache.get(b, (np.nan,) * 3)[2] if b is not None else np.nan).astype(float)

    ref_total = None
    try:
        ref_total = float(engine.annual_tax(profile, ref)["total"])
    except (KeyError, ValueError):
        pass
    return df, {"reference_bfs": ref, "reference_tax_annual": ref_total,
                "profile": {"canton": tp.canton, "filing": tp.filing,
                            "taxable_income": tp.taxable_income, "taxable_wealth": tp.taxable_wealth}}


def _group_stats(df: pd.DataFrame, by: list[str]) -> list[dict]:
    d = df[df[by[0]].notna()]
    if d.empty:
        return []
    clean_d = d.assign(_deal=d["deal_pct"].where(~d["suspicious"]))
    g = clean_d.groupby(by, dropna=False)
    out = pd.DataFrame({
        "listings": g.size(),
        "median_rent": g["rent_gross"].median(),
        "median_chf_m2": g["chf_per_m2"].median(),
        "median_deal_pct": g["_deal"].median(),
        "multiplier": g["tax_multiplier"].median(),
        "tax_delta_month": g["tax_delta_month"].median(),
    }).reset_index().sort_values("listings", ascending=False)
    for c in ["median_rent", "median_chf_m2", "median_deal_pct", "tax_delta_month"]:
        out[c] = out[c].round(1)
    return _records(out)


def analyze(frame: pd.DataFrame, config, active_only: bool = True) -> tuple[pd.DataFrame, dict]:
    """dedupe -> clean -> geo.assign_municipality -> geo.assign_area -> fit_fair_rent -> tax -> score.

    Cross-portal duplicates are collapsed first (on active rows when
    `active_only`), so one flat counts once in the model and the comparables.
    """
    from flatprice import geo  # lazy for monkeypatching
    from flatprice.dedup import dedupe, summary as dedup_summary

    n_input = int(len(frame))
    if active_only and "active" in frame.columns:
        frame = frame[frame["active"].map(_to_bool)]
    deduped = dedupe(frame)
    dedup_info = dedup_summary(frame, deduped)
    df = clean(deduped, active_only=active_only)
    n_clean = int(len(df))
    if n_clean == 0:
        raise ValueError(
            f"No usable listings after cleaning ({n_input} in input, active_only={active_only})."
        )
    df = geo.assign_municipality(df)
    df = geo.assign_area(df)
    for c in ["municipality_bfs", "municipality", "area", "dist_center_km"]:
        if c not in df.columns:
            df[c] = np.nan
    df, model_info = fit_fair_rent(df, min_listings=config.scoring.min_listings_for_model)
    df, tax_info = compute_tax(df, config)
    df, scoring_info = score(df, config.scoring.weights)

    from flatprice.quality import frame_counts

    info = {
        "quality": frame_counts(df),
        "counts": {
            "input": n_input,
            "duplicates_collapsed": dedup_info["collapsed"],
            "multi_portal": dedup_info["groups"],
            "listings": n_clean,
            "suspicious": int(df["suspicious"].sum()),
            "data_suspect": int(df["data_suspect"].fillna(False).astype(bool).sum()) if "data_suspect" in df else 0,
            "scored": int(df["score"].notna().sum()),
            "model_high": int((df["model_confidence"] == "high").sum()),
            "model_low": int((df["model_confidence"] == "low").sum()),
            "no_fair_rent": int(df["fair_rent"].isna().sum()),
            "with_tax": int(df["tax_delta_month"].notna().sum()),
            "verdicts": {k: int(v) for k, v in df.loc[~df["suspicious"], "verdict"].value_counts().items()},
            "kinds": {k: int(v) for k, v in df["listing_kind"].value_counts().items()},
        },
        "scoring": scoring_info,
        "dedup": dedup_info,
        "model": model_info,
        "tax": tax_info,
        "weights": dict(config.scoring.weights),
        "stats": {
            "municipalities": _group_stats(df, ["municipality", "municipality_bfs"]),
            "areas": _group_stats(df, ["area"]),
        },
    }
    return df, info
