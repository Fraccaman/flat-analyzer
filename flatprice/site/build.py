"""Build the static GitHub Pages site: dist/{index.html, *.js, style.css, data.json, ...}."""

from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"
GEOJSON_SRC = REPO_ROOT / "data" / "geo" / "ti_municipalities.geojson"

LISTING_COLUMNS = [
    "key", "portal", "url", "title", "street", "locality", "latitude", "longitude",
    "rent_gross", "rent_net", "rooms", "living_space_m2", "chf_per_m2", "floor", "year_built",
    "is_furnished", "has_balcony", "has_parking", "has_elevator", "municipality_bfs",
    "municipality", "area", "fair_rent", "deal_pct", "model_confidence", "suspicious",
    "first_seen", "last_seen", "days_on_market", "price_drop_pct", "tax_annual",
    "tax_delta_month", "effective_rent", "effective_deal_pct", "comp_value", "comp_tax",
    "comp_price_drop", "comp_market_time", "score", "image_url",
    # fair-rent uncertainty and shrinkage (see flatprice/analysis/score.py)
    "listing_kind", "fair_method", "fair_sigma_log", "shrink_k", "n_comps", "comps", "deal_pct_adj",
    "deal_pct_adj_eff", "deal_z", "p_below_fair", "saving_month", "saving_adj_month", "verdict",
    # cross-portal duplicates (see flatprice/dedup.py)
    "sources", "n_sources",
    # data quality (see flatprice/quality.py)
    "data_suspect", "data_flag", "data_corrected", "n_flagged_obs",
]
INT_COLUMNS = {"floor", "year_built", "municipality_bfs", "days_on_market", "n_comps", "n_sources", "n_flagged_obs"}
PRECISE_COLUMNS = {"fair_sigma_log": 6, "shrink_k": 6, "deal_z": 4, "p_below_fair": 4}
BOOL_COLUMNS = {"is_furnished", "has_balcony", "has_parking", "has_elevator", "suspicious", "data_suspect"}
COORD_COLUMNS = {"latitude", "longitude"}
DEFAULT_TAX_PROFILE = {
    "canton": "TI", "filing": "single", "taxable_income": 120000,
    "taxable_wealth": 200000, "reference_bfs": 5192,
}
DEFAULT_WEIGHTS = {"value": 0.55, "tax": 0.25, "price_drop": 0.10, "market_time": 0.10}


def _clean(v: Any, digits: int = 4) -> Any:
    """Recursively convert to JSON-safe python types (NaN/inf -> None)."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x, digits) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_clean(x, digits) for x in v]
    if isinstance(v, np.ndarray):
        return [_clean(x, digits) for x in v.tolist()]
    if isinstance(v, pd.DataFrame):
        return _clean(v.reset_index().to_dict(orient="records"), digits)
    if isinstance(v, pd.Series):
        return _clean(v.to_dict(), digits)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return None
        r = round(f, digits)
        return int(r) if r.is_integer() and abs(r) < 2**53 else r
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v if isinstance(v, str) else str(v)


def _cell(col: str, v: Any) -> Any:
    digits = (6 if col in COORD_COLUMNS else PRECISE_COLUMNS.get(col)
              or (4 if col.startswith("comp_") or col == "tax_delta_month" else 2))
    v = _clean(v, digits)
    if v is None:
        return None
    if col in INT_COLUMNS and isinstance(v, (int, float)):
        return int(round(v))
    if col in BOOL_COLUMNS:
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes"}
        return bool(v)
    return v


def listings_records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    if "key" not in df.columns and {"portal", "portal_id"} <= set(df.columns):
        df["key"] = df["portal"].astype(str) + ":" + df["portal_id"].astype(str)
    if "key" not in df.columns:
        df["key"] = [str(i) for i in range(len(df))]
    if "chf_per_m2" not in df.columns and {"rent_gross", "living_space_m2"} <= set(df.columns):
        area = pd.to_numeric(df["living_space_m2"], errors="coerce")
        df["chf_per_m2"] = pd.to_numeric(df["rent_gross"], errors="coerce") / area.where(area > 0)
    cols = [c for c in LISTING_COLUMNS if c in df.columns]
    out = []
    for row in df[cols].itertuples(index=False, name=None):
        rec = {c: _cell(c, v) for c, v in zip(cols, row)}
        rec["key"] = str(rec["key"])
        out.append(rec)
    return out


def _tax_tables(canton: str) -> dict:
    try:
        from flatprice.tax.engine import export_tables  # lazy: built by another module
    except ImportError as e:
        log.warning("tax engine unavailable (%s); writing empty tax tables", e)
        return {}
    try:
        try:
            return export_tables((canton.upper(),))
        except TypeError:
            return export_tables()
    except Exception as e:  # noqa: BLE001 - site must still build
        log.warning("export_tables failed (%s); writing empty tax tables", e)
        return {}


def _defaults(config: Any) -> dict:
    tp, w = dict(DEFAULT_TAX_PROFILE), dict(DEFAULT_WEIGHTS)
    if config is not None:
        prof = getattr(config, "tax_profile", None)
        if prof is not None:
            tp.update(prof.model_dump() if hasattr(prof, "model_dump") else dict(prof))
        scoring = getattr(config, "scoring", None)
        if scoring is not None and getattr(scoring, "weights", None):
            w.update(scoring.weights)
    return {"tax_profile": tp, "weights": w}


def build_site(
    df: pd.DataFrame,
    info: dict | None,
    config: Any,
    dist_dir: str | Path,
    *,
    tax_tables: dict | None = None,
    geojson_path: str | Path | None = None,
) -> Path:
    """Write the static site to `dist_dir` and return its path.

    `info` is the analysis output: either the stats dict itself or a dict with a
    "stats" key (and optionally "generated_at"). `tax_tables` overrides
    flatprice.tax.engine.export_tables() (used by tests/demo).
    """
    dist = Path(dist_dir)
    dist.mkdir(parents=True, exist_ok=True)
    info = dict(info or {})
    if "stats" in info:
        stats = dict(info["stats"] or {})
        for src, dst in (("counts", "counts"), ("model", "model_info"), ("tax", "tax_info"),
                         ("quality", "quality")):
            if src in info:
                stats.setdefault(dst, info[src])
    else:
        stats = info
    generated_at = info.get("generated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")

    defaults = _defaults(config)
    if tax_tables is None:
        tax_tables = _tax_tables(defaults["tax_profile"].get("canton", "TI"))

    payload = {
        "generated_at": _clean(generated_at),
        "defaults": _clean(defaults),
        "tax": _clean(tax_tables or {}, 6),
        "scoring": _clean(info.get("scoring") or {}, 6),
        "stats": _clean({k: v for k, v in stats.items() if k != "generated_at"}, 3),
        "listings": listings_records(df if df is not None else pd.DataFrame()),
    }
    (dist / "data.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )

    if WEB_DIR.is_dir():
        for src in WEB_DIR.iterdir():
            if src.name.startswith("."):
                continue
            if src.is_dir():
                shutil.copytree(src, dist / src.name, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dist / src.name)
    else:
        log.warning("web/ directory not found at %s", WEB_DIR)

    geo = Path(geojson_path) if geojson_path else GEOJSON_SRC
    if geo.is_file():
        shutil.copy2(geo, dist / "municipalities.geojson")
    else:
        log.info("no municipality geojson at %s; map polygons disabled", geo)

    (dist / ".nojekyll").write_text("", encoding="utf-8")
    log.info("site built: %s (%d listings)", dist, len(payload["listings"]))
    return dist
