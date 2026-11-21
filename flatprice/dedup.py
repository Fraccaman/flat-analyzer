"""Collapse cross-portal duplicates: the same physical flat listed on several portals.

Two rules link rows of the Store frame (one row per portal listing):

(a) ids: a row's ``alt_ids`` (e.g. Flatfox ``{"homegate": "<smg_id>"}``) names
    another row's key ``<portal>:<portal_id>``.
(b) heuristic, for cross-portal pairs that rule (a) does not settle: distance
    < 40 m AND same rooms AND m² within ±5% (or both missing) AND gross rent
    within ±3%. A pair is never linked by (b) if an alt_id contradicts it (the
    row says it is a *different* listing on the other portal).

Linked rows form groups (union-find). Each group keeps one canonical row:
active first, then the most complete data, then portal priority (homegate
first: longest history), then key, so the choice is deterministic. Missing
descriptive fields of the canonical row are filled from its duplicates.

Prices are not cross-filled, except from a trusted portal (Flatfox, the API
serving the original syndicated data): when the canonical row is from another
portal and disagrees with the Flatfox copy (gross rent more than 3% apart,
different rooms, or its latest data flagged by flatprice.quality), the
canonical row takes Flatfox's rent, rooms, m² and title, and gets
``data_corrected = "flatfox"`` and ``data_suspect = False``. The canonical row gets:

- ``sources``: [{portal, key, url, rent_gross, active[, flag]}] for every member,
  canonical first then by rent; rent_gross is the price the ad currently shows
  (``rent_observed`` when present) and ``flag`` its data-quality flag, if any;
- ``n_sources``: number of members;
- ``first_seen``: the earliest, ``last_seen``: the latest;
- ``days_on_market``: the maximum; ``price_drop_pct``: the canonical row's;
- ``active``: any member active.

Same-portal duplicates are left alone (fair_rent.dup_groups keeps them in one
CV fold and out of each other's comparables).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

MAX_DIST_M = 40.0
M2_TOL = 0.05
RENT_TOL = 0.03
PORTAL_PRIORITY = {"homegate": 0, "flatfox": 1}
TRUSTED_PORTALS = ("flatfox",)  # see flatprice.quality
# Taken from the trusted copy when the canonical row disagrees with it.
TRUSTED_FIELDS = ["rent_gross", "rent_net", "rooms", "living_space_m2", "title"]

# Fields counted for "more complete data" and filled from duplicates when missing.
COMPLETENESS_FIELDS = [
    "rent_gross", "rent_net", "rooms", "living_space_m2", "floor", "year_built",
    "latitude", "longitude", "street", "image_url", "title",
    "has_balcony", "has_parking", "has_elevator", "pets_allowed",
]
FILL_FIELDS = [
    "rooms", "living_space_m2", "floor", "year_built", "latitude", "longitude",
    "street", "postal_code", "image_url", "has_balcony", "has_parking", "has_elevator",
    "pets_allowed",
]


def _missing(v: Any) -> bool:
    if v is None or v is pd.NA:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _to_bool(v: Any) -> bool:
    if _missing(v):
        return False
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return bool(v)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _keys(df: pd.DataFrame) -> pd.Series:
    if "key" in df.columns:
        return df["key"].astype(str)
    return df["portal"].astype(str) + ":" + df["portal_id"].astype(str)


def _alt_ids(v: Any) -> dict:
    if isinstance(v, dict):
        return {str(k): str(x) for k, x in v.items() if not _missing(x)}
    if isinstance(v, str) and v.strip():
        import json

        try:
            d = json.loads(v)
        except ValueError:
            return {}
        return _alt_ids(d) if isinstance(d, dict) else {}
    return {}


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _within(a: float, b: np.ndarray, tol: float) -> np.ndarray:
    """|a - b| <= tol * max(a, b), elementwise; False where either is missing."""
    with np.errstate(invalid="ignore"):
        return np.abs(a - b) <= tol * np.maximum(a, b)


def link_pairs(df: pd.DataFrame) -> list[tuple[int, int, str]]:
    """(i, j, rule) positional pairs of rows that are the same flat; rule is "id" or "geo"."""
    n = len(df)
    if n < 2:
        return []
    keys = _keys(df).tolist()
    pos = {k: i for i, k in enumerate(keys)}
    portals = df["portal"].astype(str).str.lower().tolist()
    pids = df["portal_id"].astype(str).tolist() if "portal_id" in df.columns else [k.split(":", 1)[-1] for k in keys]
    alts = [_alt_ids(v) for v in df["alt_ids"]] if "alt_ids" in df.columns else [{} for _ in range(n)]

    pairs: list[tuple[int, int, str]] = []
    id_linked: set[tuple[int, int]] = set()
    for i, alt in enumerate(alts):
        for portal, pid in alt.items():
            j = pos.get(f"{portal.lower()}:{pid}")
            if j is not None and j != i:
                pairs.append((min(i, j), max(i, j), "id"))
                id_linked.add((min(i, j), max(i, j)))

    lat, lon = _num(df, "latitude"), _num(df, "longitude")
    rooms, m2 = _num(df, "rooms"), _num(df, "living_space_m2")
    rent = _num(df, "rent_gross")
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(rooms) & np.isfinite(rent)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return pairs
    portal_arr = np.array(portals, dtype=object)
    ky = 111_195.0  # metres per degree latitude
    kx = ky * math.cos(math.radians(float(np.nanmean(lat[idx]))))
    for a_pos, i in enumerate(idx):
        cand = idx[a_pos + 1:]
        if not len(cand):
            break
        cand = cand[portal_arr[cand] != portals[i]]  # cross-portal only
        if not len(cand):
            continue
        dx = (lon[cand] - lon[i]) * kx
        dy = (lat[cand] - lat[i]) * ky
        close = np.hypot(dx, dy) < MAX_DIST_M
        same_rooms = rooms[cand] == rooms[i]
        if np.isnan(m2[i]):
            m2_ok = np.isnan(m2[cand])
        else:
            m2_ok = _within(m2[i], m2[cand], M2_TOL)
        rent_ok = _within(rent[i], rent[cand], RENT_TOL)
        for j in cand[close & same_rooms & m2_ok & rent_ok]:
            j = int(j)
            pair = (int(min(i, j)), int(max(i, j)))
            if pair in id_linked:
                continue
            # contradicted: one row names a different listing on the other's portal
            if alts[i].get(portals[j], pids[j]) != pids[j] or alts[j].get(portals[i], pids[i]) != pids[i]:
                continue
            pairs.append((pair[0], pair[1], "geo"))
    return pairs


def _completeness(row: pd.Series) -> int:
    return sum(0 if _missing(row.get(f)) else 1 for f in COMPLETENESS_FIELDS if f in row.index)


def disagrees(df: pd.DataFrame, i: int, j: int) -> bool:
    """Row i vs trusted row j: data flagged / suspect, gross rent beyond RENT_TOL, or rooms differ."""
    if not _missing(df.at[i, "data_flag"] if "data_flag" in df.columns else None):
        return True
    if "data_suspect" in df.columns and _to_bool(df.at[i, "data_suspect"]):
        return True
    a, b = _num(df, "rent_gross")[[i, j]]
    if np.isfinite(a) and np.isfinite(b) and not _within(a, np.array([b]), RENT_TOL)[0]:
        return True
    ra, rb = _num(df, "rooms")[[i, j]]
    return bool(np.isfinite(ra) and np.isfinite(rb) and ra != rb)


def _correct_from_trusted(df: pd.DataFrame, c: int, members: list[int], portals: pd.Series) -> None:
    """Canonical row c takes the trusted copy's price/rooms/m²/title when they disagree.

    If c is itself the trusted copy, it is only labelled when another member disagrees."""
    if portals.iat[c] in TRUSTED_PORTALS:
        if any(portals.iat[i] not in TRUSTED_PORTALS and disagrees(df, i, c) for i in members):
            df.at[c, "data_corrected"] = "flatfox"
        return
    trusted = [i for i in members if portals.iat[i] in TRUSTED_PORTALS
               and not _missing(df.at[i, "rent_gross"] if "rent_gross" in df.columns else None)]
    if not trusted:
        return
    t = trusted[0]
    if not disagrees(df, c, t):
        return
    for f in TRUSTED_FIELDS:
        if f not in df.columns:
            continue
        v = df.at[t, f]
        if f in ("rent_gross", "rent_net") or not _missing(v):
            df.at[c, f] = v
    df.at[c, "data_corrected"] = "flatfox"
    if "data_suspect" in df.columns:
        df.at[c, "data_suspect"] = False


def dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return `frame` with cross-portal duplicates collapsed (see module doc).

    Adds ``sources``, ``n_sources``, ``dup_rule`` ("id", "geo", "id+geo" or None)
    to every row. Row order follows the canonical rows' original order.
    """
    df = frame.reset_index(drop=True).copy()
    n = len(df)
    keys = _keys(df)
    if "key" not in df.columns:
        df["key"] = keys
    portals = df["portal"].astype(str).str.lower() if "portal" in df.columns else pd.Series("", index=df.index)
    active = df["active"].map(_to_bool) if "active" in df.columns else pd.Series(True, index=df.index)

    uf = _UnionFind(n)
    rules: dict[int, set] = {}
    for i, j, rule in link_pairs(df):
        uf.union(i, j)
        rules.setdefault(i, set()).add(rule)
        rules.setdefault(j, set()).add(rule)
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)

    def source(i: int) -> dict:
        r = df.iloc[i]
        rent = r.get("rent_observed")
        if _missing(rent):
            rent = r.get("rent_gross")
        if _missing(rent):
            rent = r.get("rent_net")
        out = {
            "portal": str(r.get("portal") or ""),
            "key": str(keys.iat[i]),
            "url": str(r.get("url") or ""),
            "rent_gross": None if _missing(rent) else float(rent),
            "active": bool(active.iat[i]),
        }
        if not _missing(r.get("data_flag")):
            out["flag"] = str(r.get("data_flag"))
        return out

    sources_raw = [source(i) for i in range(n)]  # before any correction below
    if "data_corrected" not in df.columns:
        df["data_corrected"] = None
    df["data_corrected"] = df["data_corrected"].astype(object)
    if "data_suspect" in df.columns:
        df["data_suspect"] = df["data_suspect"].map(_to_bool).astype(bool)

    keep = np.ones(n, dtype=bool)
    sources: list = [None] * n
    n_sources = np.ones(n, dtype=np.int64)
    dup_rule: list = [None] * n
    for root in np.unique(roots):
        members = np.flatnonzero(roots == root).tolist()
        if len(members) == 1:
            sources[members[0]] = [sources_raw[members[0]]]
            continue
        ranked = sorted(members, key=lambda i: (
            not active.iat[i],
            -_completeness(df.iloc[i]),
            PORTAL_PRIORITY.get(portals.iat[i], len(PORTAL_PRIORITY)),
            keys.iat[i],
        ))
        c = ranked[0]
        for i in ranked[1:]:
            keep[i] = False
        for f in FILL_FIELDS:
            if f in df.columns and _missing(df.at[c, f]):
                for i in ranked[1:]:
                    if not _missing(df.at[i, f]):
                        df.at[c, f] = df.at[i, f]
                        break
        _correct_from_trusted(df, c, ranked, portals)
        others = sorted(ranked[1:], key=lambda i: (sources_raw[i]["rent_gross"] is None,
                                                   sources_raw[i]["rent_gross"] or 0, keys.iat[i]))
        sources[c] = [sources_raw[c]] + [sources_raw[i] for i in others]
        n_sources[c] = len(members)
        group_rules = set().union(*(rules.get(i, set()) for i in members))
        dup_rule[c] = "+".join(r for r in ("id", "geo") if r in group_rules)
        if "first_seen" in df.columns:
            vals = [df.at[i, "first_seen"] for i in members if not _missing(df.at[i, "first_seen"])]
            if vals:
                df.at[c, "first_seen"] = min(vals, key=lambda v: pd.Timestamp(v))
        if "last_seen" in df.columns:
            vals = [df.at[i, "last_seen"] for i in members if not _missing(df.at[i, "last_seen"])]
            if vals:
                df.at[c, "last_seen"] = max(vals, key=lambda v: pd.Timestamp(v))
        if "days_on_market" in df.columns:
            d = pd.to_numeric(df.loc[members, "days_on_market"], errors="coerce")
            if d.notna().any():
                df.at[c, "days_on_market"] = float(d.max())
        if "active" in df.columns:
            df.at[c, "active"] = bool(active.iloc[members].any())

    df["sources"] = sources
    df["n_sources"] = n_sources
    df["dup_rule"] = dup_rule
    return df[keep].reset_index(drop=True)


def summary(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Counts for logging/site stats."""
    multi = after[after["n_sources"] > 1] if "n_sources" in after.columns else after.iloc[0:0]
    rules = multi["dup_rule"].value_counts().to_dict() if len(multi) else {}
    return {
        "input": int(len(before)),
        "output": int(len(after)),
        "collapsed": int(len(before) - len(after)),
        "groups": int(len(multi)),
        "groups_by_rule": {str(k): int(v) for k, v in rules.items()},
    }
