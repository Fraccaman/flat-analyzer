"""Data-quality rules against perturbed ("poisoned") portal data.

Homegate sits behind DataDome. In Nov 2026 a whole result page (Lugano page 7)
and a whole search (Sorengo) came back with every listing perturbed: gross rent
x1.366 (rounded to CHF 10), one more room, m2 inflated by 12-36%, the room count
in the title bumped, and the net rent dropped. Flatfox, which syndicates the same
ads through its public API, still had the original values. The working
hypothesis is that the anti-bot system serves plausible fake data to clients it
suspects are bots.

Every observation gets a flag (``observations.flag``): NULL = trusted, else a
comma-separated list of reasons:

- ``ratio_1366`` (one per configured ratio): rent vs the previous trusted
  observation of the same listing is ~ratio. On the *inverse* ratio the earlier,
  higher observation(s) are flagged instead: the perturbation has only ever
  inflated prices, so the drop back is the real price.
- ``rooms_jump``: rooms changed by exactly +-1 together with a rent change
  (same direction logic as the ratio).
- ``net_dropped``: the net rent disappeared while the rent went up (seen on
  every poisoned listing; catches a changed factor).
- ``xportal_mismatch``: a trusted portal's copy of the same flat (Flatfox
  ``smg_id`` -> homegate id) observed within ``xportal_max_days`` has a rent
  more than ``xportal_rent_tolerance`` apart, or different rooms.
- ``page_poisoned``: an observation with nothing to compare against (new
  listing, no twin) on a result page where most comparable observations are
  flagged.

Trusted portals (Flatfox) are never flagged. `apply` recomputes every flag from
scratch, stores ``runs.n_flagged`` and returns a report; it runs after each
scrape and via `flatprice quality`. `Store.record` uses `record_flag` so a
flagged observation never overwrites the listing's fields. `Store.load_frame`
ignores flagged observations for current values.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Optional

import pandas as pd

from flatprice.config import QualityConfig

RENT_CHANGE = 0.005  # relative rent change that counts as "changed"
ROUNDING_CHF = 5.0  # perturbed rents are rounded to CHF 10: +-5 slack on the ratio
REASON_LABELS = {
    "rooms_jump": "rooms +-1 with a rent change",
    "net_dropped": "net rent vanished as the rent rose",
    "xportal_mismatch": "differs from the flatfox copy",
    "page_poisoned": "new listing on a poisoned result page",
}


def config_or_default(cfg: Optional[QualityConfig]) -> QualityConfig:
    return cfg if cfg is not None else QualityConfig()


def ratio_name(ratio: float) -> str:
    return f"ratio_{int(round(ratio * 1000))}"


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _rent(o: dict) -> Optional[float]:
    g = _num(o.get("rent_gross"))
    return g if g is not None else _num(o.get("rent_net"))


def _split(flag: Optional[str]) -> set[str]:
    return {r for r in (flag or "").split(",") if r}


def join_reasons(reasons: set[str]) -> Optional[str]:
    return ",".join(sorted(reasons)) or None


def snapshot(listing) -> dict:
    """The per-observation values the rules look at, from a Listing."""
    return {"rent_gross": listing.rent_gross, "rent_net": listing.rent_net, "rooms": listing.rooms}


# --------------------------------------------------------------------------- #
# Pure rules
# --------------------------------------------------------------------------- #

def match_ratio(prev_rent: Optional[float], cur_rent: Optional[float],
                cfg: Optional[QualityConfig] = None) -> Optional[tuple[str, int]]:
    """(reason, +1) if cur/prev ~ a poison ratio, (reason, -1) if ~ its inverse, else None.

    Tolerance: ``ratio_tolerance`` plus CHF 5 of rounding on the larger rent.
    """
    cfg = config_or_default(cfg)
    if not prev_rent or not cur_rent or prev_rent <= 0 or cur_rent <= 0:
        return None
    r = cur_rent / prev_rent
    tol = cfg.ratio_tolerance + ROUNDING_CHF / max(prev_rent, cur_rent)
    for ratio in cfg.poison_ratios:
        if ratio <= 0:
            continue
        if abs(r / ratio - 1) < tol:
            return ratio_name(ratio), 1
        if abs(r * ratio - 1) < tol:
            return ratio_name(ratio), -1
    return None


def _close(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and abs(a - b) <= RENT_CHANGE * max(a, b)


def temporal_flags(obs: list[dict], cfg: Optional[QualityConfig] = None) -> list[set[str]]:
    """Temporal reasons for one listing's observations in chronological order.

    Each item has rent_gross, rent_net, rooms and optionally ``fixed`` (reasons
    from other rules: such an observation is never trusted and never clears an
    earlier one via the inverse ratio). Returns one set of temporal reasons per
    observation.
    """
    cfg = config_or_default(cfg)
    reasons: list[set[str]] = [set() for _ in obs]

    def trusted(k: int) -> bool:
        return not obs[k].get("fixed") and not reasons[k]

    def prev_trusted(i: int) -> Optional[int]:
        for k in range(i - 1, -1, -1):
            if trusted(k):
                return k
        return None

    for i, o in enumerate(obs):
        fixed = bool(o.get("fixed"))
        j = prev_trusted(i)
        if j is None:
            continue
        p = obs[j]
        pr, cr = _rent(p), _rent(o)
        changed = pr is not None and cr is not None and abs(cr / pr - 1) > RENT_CHANGE
        up: set[str] = set()
        down: set[str] = set()
        m = match_ratio(pr, cr, cfg)
        if m:
            (up if m[1] > 0 else down).add(m[0])
        pro, cro = _num(p.get("rooms")), _num(o.get("rooms"))
        if changed and pro is not None and cro is not None and abs(abs(cro - pro) - 1) < 1e-9:
            (up if cro > pro else down).add("rooms_jump")
        if changed and cr > pr and _num(p.get("rent_net")) is not None and _num(o.get("rent_net")) is None:
            up.add("net_dropped")
        if up:
            reasons[i] |= up
        elif down and not fixed:
            # The earlier, inflated observation(s) at that rent are the poisoned ones.
            ref, k = pr, j
            while k is not None and _close(_rent(obs[k]), ref):
                reasons[k] |= down
                k = prev_trusted(k)
    return reasons


def record_flag(history: list[dict], snap: dict, cfg: Optional[QualityConfig] = None) -> Optional[str]:
    """Temporal flag for a new observation `snap` given the listing's earlier
    observations (dicts with rent_gross, rent_net, rooms, flag)."""
    obs = [{**h, "fixed": _split(h.get("flag"))} for h in history] + [dict(snap)]
    return join_reasons(temporal_flags(obs, cfg)[-1])


def xportal_mismatch(checked: dict, reference: dict, cfg: Optional[QualityConfig] = None) -> bool:
    """True if two copies of the same flat disagree: rent (gross vs gross, else
    net vs net) more than the tolerance apart, or both have rooms and they differ."""
    cfg = config_or_default(cfg)
    for field in ("rent_gross", "rent_net"):
        a, b = _num(checked.get(field)), _num(reference.get(field))
        if a is not None and b is not None and a > 0 and b > 0:
            if abs(a - b) > cfg.xportal_rent_tolerance * min(a, b):
                return True
            break
    ra, rb = _num(checked.get("rooms")), _num(reference.get("rooms"))
    return ra is not None and rb is not None and abs(ra - rb) > 1e-9


# --------------------------------------------------------------------------- #
# Store-level step
# --------------------------------------------------------------------------- #

def _alt_ids(s: Any) -> dict:
    if isinstance(s, dict):
        return s
    try:
        d = json.loads(s or "{}")
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def compute_flags(obs: pd.DataFrame, cfg: Optional[QualityConfig] = None) -> dict[int, set[str]]:
    """Reasons per observation id (``rid``) for an observation frame with columns
    rid, key, run_id, seen_at, rent_gross, rent_net, rooms, page, portal, search, alt_ids
    and optionally run_status (seed observations carry the import time, not the
    time the data was scraped, so they are skipped by the cross-portal rule)."""
    cfg = config_or_default(cfg)
    trusted_portals = {p.lower() for p in cfg.trusted_portals}
    obs = obs.sort_values(["key", "seen_at", "run_id"]).reset_index(drop=True)
    obs["_ts"] = pd.to_datetime(obs["seen_at"], utc=True, format="ISO8601")
    is_ref = obs["portal"].astype(str).str.lower().isin(trusted_portals)
    reasons: dict[int, set[str]] = {int(r): set() for r in obs["rid"]}
    basis: set[int] = set()  # observations that had something trusted to compare with

    # 1. cross-portal: trusted-portal listings name their twin via alt_ids
    ref_obs = obs[is_ref]
    twin_obs: dict[str, list[dict]] = defaultdict(list)
    alts = ref_obs.drop_duplicates("key")[["key", "alt_ids"]]
    by_key = {k: g for k, g in ref_obs.groupby("key")}
    for key, alt in alts.itertuples(index=False):
        for portal, pid in _alt_ids(alt).items():
            if pid:
                twin_obs[f"{str(portal).lower()}:{pid}"].extend(by_key[key].to_dict("records"))
    max_dt = pd.Timedelta(days=cfg.xportal_max_days)
    seed = obs["run_status"].eq("seed") if "run_status" in obs.columns else pd.Series(False, index=obs.index)
    for o in obs[~is_ref & ~seed & obs["key"].isin(twin_obs.keys())].to_dict("records"):
        near = [(abs(t["_ts"] - o["_ts"]), t) for t in twin_obs[o["key"]] if abs(t["_ts"] - o["_ts"]) <= max_dt]
        if not near:
            continue
        ref = min(near, key=lambda x: x[0])[1]
        basis.add(int(o["rid"]))
        if xportal_mismatch(o, ref, cfg):
            reasons[int(o["rid"])].add("xportal_mismatch")

    # 2. temporal, per listing of a checked portal
    for _, g in obs[~is_ref].groupby("key", sort=False):
        rows = g.to_dict("records")
        items = [{**r, "fixed": set(reasons[int(r["rid"])])} for r in rows]
        t = temporal_flags(items, cfg)
        for i, r in enumerate(rows):
            reasons[int(r["rid"])] |= t[i]
        for i, r in enumerate(rows):
            if _rent(r) is not None and any(
                    not reasons[int(rows[k]["rid"])] and _rent(rows[k]) is not None for k in range(i)):
                basis.add(int(r["rid"]))

    # 3. page: unverifiable observations on a mostly-flagged result page
    paged = obs[~is_ref & obs["page"].notna()]
    for _, g in paged.groupby(["run_id", "search", "page"]):
        rids = [int(r) for r in g["rid"]]
        flagged = sum(1 for r in rids if reasons[r])
        if flagged >= cfg.page_min_flagged and flagged / len(rids) >= cfg.page_flagged_share:
            for r in rids:
                if not reasons[r] and r not in basis:
                    reasons[r].add("page_poisoned")
    return reasons


def apply(store, cfg: Optional[QualityConfig] = None) -> dict:
    """Recompute every observation's flag, store them and ``runs.n_flagged``.

    Returns a report: {n_obs, n_flagged, n_listings_flagged, changed,
    by_reason: {reason: n}, by_run: {run_id: {status, portals, n, flagged, share}}}.
    """
    cfg = config_or_default(cfg)
    conn = store.conn
    obs = pd.read_sql_query(
        "SELECT o.rowid AS rid, o.run_id, o.listing_key AS key, o.seen_at, o.rent_gross, o.rent_net, "
        "o.rooms, o.page, o.flag AS old_flag, l.portal, l.search, l.alt_ids, r.status AS run_status "
        "FROM observations o JOIN listings l ON l.key = o.listing_key JOIN runs r ON r.id = o.run_id",
        conn,
    )
    reasons = compute_flags(obs, cfg) if len(obs) else {}
    flags = {rid: join_reasons(r) for rid, r in reasons.items()}
    old = {int(r): (f if isinstance(f, str) else None) for r, f in zip(obs["rid"], obs["old_flag"])}
    changed = [(f, rid) for rid, f in flags.items() if old.get(rid) != f]
    with conn:
        conn.executemany("UPDATE observations SET flag = ? WHERE rowid = ?", changed)
    obs["flag"] = obs["rid"].astype(int).map(flags)
    flagged = obs["flag"].notna()
    store.set_run_flag_counts(flagged.groupby(obs["run_id"]).sum().astype(int).to_dict())

    runs = {int(r[0]): r[1] for r in conn.execute("SELECT id, status FROM runs")}
    by_run = {}
    for run_id, g in obs.groupby("run_id"):
        n, k = len(g), int(g["flag"].notna().sum())
        by_run[int(run_id)] = {"status": runs.get(int(run_id)), "portals": sorted(set(g["portal"])),
                               "n": n, "flagged": k, "share": k / n if n else 0.0}
    by_reason: Counter = Counter()
    for f in obs.loc[flagged, "flag"]:
        by_reason.update(_split(f))
    return {
        "n_obs": int(len(obs)),
        "n_flagged": int(flagged.sum()),
        "n_listings_flagged": int(obs.loc[flagged, "key"].nunique()),
        "changed": len(changed),
        "by_reason": dict(sorted(by_reason.items())),
        "by_run": by_run,
    }


def run_health(report: dict, run_id: int, cfg: Optional[QualityConfig] = None) -> dict:
    """{n, flagged, share, poisoned} for one run of an `apply` report."""
    cfg = config_or_default(cfg)
    r = report.get("by_run", {}).get(int(run_id), {"n": 0, "flagged": 0, "share": 0.0})
    return {"n": r["n"], "flagged": r["flagged"], "share": r["share"],
            "poisoned": r["share"] > cfg.max_flagged_share}


def frame_counts(df: pd.DataFrame) -> dict:
    """Correction counts of an analysed (deduplicated) frame."""
    def col(c, default=None):
        return df[c] if c in df.columns else pd.Series(default, index=df.index)

    corrected = col("data_corrected")
    return {
        "flagged_listings": int(col("data_flag").notna().sum()),
        "corrected_history": int((corrected == "history").sum()),
        "corrected_flatfox": int((corrected == "flatfox").sum()),
        "rent_from_history": int((corrected == "history_rent").sum()),
        "data_suspect": int(col("data_suspect", False).fillna(False).astype(bool).sum()),
    }


def format_report(report: dict, counts: Optional[dict] = None, cfg: Optional[QualityConfig] = None) -> str:
    cfg = config_or_default(cfg)
    lines = [f"observations: {report['n_obs']}, flagged: {report['n_flagged']} "
             f"({report['n_listings_flagged']} listings), flags changed now: {report['changed']}"]
    lines.append("by rule:")
    for reason, n in report["by_reason"].items():
        label = REASON_LABELS.get(reason, "rent x poison ratio (or the earlier side of its inverse)")
        lines.append(f"  {reason:<18} {n:>5}  {label}")
    lines.append("by run:")
    for run_id, r in sorted(report["by_run"].items()):
        warn = "  <- POISONED" if r["share"] > cfg.max_flagged_share else ""
        lines.append(f"  run {run_id:<3} {str(r['status']):<8} {','.join(r['portals']):<18} "
                     f"{r['flagged']:>4}/{r['n']:<5} {r['share'] * 100:5.1f}%{warn}")
    if counts:
        lines.append(f"listings (active, after dedup): {counts['flagged_listings']} with a flagged latest "
                     f"price; corrected from history: {counts['corrected_history']}, from flatfox: "
                     f"{counts['corrected_flatfox']}; still data_suspect: {counts['data_suspect']} "
                     f"(of which {counts['rent_from_history']} have the rent restored from history "
                     f"but unverified rooms/m2)")
    return "\n".join(lines)
