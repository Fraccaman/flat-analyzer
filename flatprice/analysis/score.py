"""Deal score and verdict. THIS IS MIRRORED IN JS (web/score.js, web/deal.js) — keep in sync.

Inputs per listing: rent_gross, fair_rent, fair_sigma_log (sd of the error of
log fair_rent, from fair_rent.py), tax_delta_month (CHF/month, negative =
saving vs reference municipality; NaN = unknown), price_drop_pct,
days_on_market, suspicious, listing_kind.

1. Effective rent
     tax_delta          = tax_delta_month, NaN -> 0      (unknown = same as reference)
     effective_rent     = rent_gross + tax_delta
     effective_deal_pct = (effective_rent / fair_rent - 1) * 100
     log_ratio          = ln(rent_gross / fair_rent)         (plain rent)
     log_ratio_eff      = ln(effective_rent / fair_rent)     (incl. your tax)

2. Empirical-Bayes shrinkage (sigma = fair_sigma_log)
     tau2     = max(var(log_ratio) - mean(sigma^2), TAU2_FLOOR)
                over non-suspicious regular listings with finite log_ratio
                and sigma (population variance). Computed once, shipped.
     shrink_k = tau2 / (tau2 + sigma^2)                     (shipped per row)
     deal_pct_adj     = (exp(log_ratio     * shrink_k) - 1) * 100
     deal_pct_adj_eff = (exp(log_ratio_eff * shrink_k) - 1) * 100
     deal_z           = log_ratio_eff / sigma
     p_below_fair     = Phi(-deal_z)    (Phi: normal CDF via the A&S 7.1.26 erf,
                                         identical code in JS)
     saving_month     = fair_rent - effective_rent
     saving_adj_month = fair_rent * (1 - exp(log_ratio_eff * shrink_k))

3. Verdict (from deal_pct_adj_eff = d, p = p_below_fair)
     d <= -20 -> "great" if p >= 0.8 else "good"
     -20 < d <= -8 -> "good";  -8 < d < 8 -> "fair";  8 <= d <= 20 -> "pricey";
     d > 20 -> "over";  d NaN -> None.  Suspicious listings get a verdict too
     (the UI flags them); they are never scored.

4. Raw component signals (higher = better)
     value       = -log_ratio * shrink_k     (shrunk deal on PLAIN rent: the tax
                                              effect is the separate tax component,
                                              so it is not double counted)
     tax         = -tax_delta                (NaN tax_delta_month -> 0)
     price_drop  = price_drop_pct            (NaN -> 0)
     market_time = min(days_on_market, 90)   (NaN stays NaN)

5. Percentile of each component, 0..100.
   Reference set S_k = values of signal k over listings that are NOT
   suspicious and whose signal k is not NaN; n = |S_k|.
   For a non-suspicious listing with non-NaN signal x:
     less  = count(v in S_k with v <  x)
     equal = count(v in S_k with v == x)      (includes the listing itself)
     comp_k = 100 * (less + (equal - 1) / 2) / (n - 1)
   If every value in S_k is equal (incl. n <= 1) the component carries no
   information: comp_k = NaN for every listing, so it drops out of the
   weighted mean below and the other weights are renormalised.
   NaN signal -> comp_k = NaN. Suspicious listings -> all comp_k = NaN.

6. Score
     K = components k with weight w_k > 0 (weights from config, unknown keys
         ignored) and comp_k not NaN for this listing
     score = sum_{k in K} w_k * comp_k / sum_{k in K} w_k
     score = floor(score * 10 + 0.5) / 10      (= JS Math.round(score*10)/10)
   K empty or listing suspicious -> score = NaN (null in JSON).

7. Order: score descending; NaN scores (incl. all suspicious) last.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

COMPONENTS = ("value", "tax", "price_drop", "market_time")
MARKET_TIME_CAP = 90.0
TAU2_FLOOR = 0.0025  # (5%)^2
GREAT_MIN_P = 0.8


def norm_cdf(x):
    """Standard normal CDF, Abramowitz & Stegun 7.1.26 erf (|err| < 1.5e-7).
    Same operations as web/deal.js normCdf so both give identical numbers."""
    x = np.asarray(x, dtype=float)
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return np.where(x >= 0, 0.5 * (1.0 + erf), 0.5 * (1.0 - erf))


def verdict_for(pct: float, p_below: float | None = None) -> str | None:
    """Verdict id for a (shrunk, effective) deal % and chance below fair rent."""
    if pct is None or not math.isfinite(pct):
        return None
    if pct <= -20.0:
        return "great" if p_below is not None and math.isfinite(p_below) and p_below >= GREAT_MIN_P else "good"
    if pct <= -8.0:
        return "good"
    if pct < 8.0:
        return "fair"
    if pct <= 20.0:
        return "pricey"
    return "over"


def percentile_midrank(values: pd.Series, reference_mask: pd.Series) -> pd.Series:
    """comp = 100 * (rank_avg - 1) / (n - 1) against the reference set (see module doc).
    Rows outside the reference mask get NaN; n <= 1 gives 50."""
    values = pd.to_numeric(values, errors="coerce").astype(float)
    ref_mask = reference_mask & values.notna()
    ref = np.sort(values[ref_mask].to_numpy())
    out = pd.Series(np.nan, index=values.index)
    n = len(ref)
    if n == 0:
        return out
    x = values[ref_mask].to_numpy()
    if n == 1:
        out[ref_mask] = 50.0
        return out
    less = np.searchsorted(ref, x, side="left")
    equal = np.searchsorted(ref, x, side="right") - less
    out[ref_mask] = 100.0 * (less + (equal - 1) / 2.0) / (n - 1)
    return out


def component(values: pd.Series, reference_mask: pd.Series) -> pd.Series:
    """percentile_midrank, but NaN everywhere when the reference values are all equal."""
    values = pd.to_numeric(values, errors="coerce").astype(float)
    ref = values[reference_mask & values.notna()]
    if ref.empty or ref.min() == ref.max():
        return pd.Series(np.nan, index=values.index)
    return percentile_midrank(values, reference_mask)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index)


def estimate_tau2(log_ratio: pd.Series, sigma: pd.Series, mask: pd.Series) -> float:
    ok = mask & log_ratio.notna() & sigma.notna() & np.isfinite(log_ratio) & np.isfinite(sigma)
    if ok.sum() < 2:
        return TAU2_FLOOR
    v = float(np.var(log_ratio[ok].to_numpy()))
    return max(v - float(np.mean(sigma[ok].to_numpy() ** 2)), TAU2_FLOOR)


def raw_signals(df: pd.DataFrame) -> pd.DataFrame:
    tax_delta = _col(df, "tax_delta_month").fillna(0.0)
    if "log_ratio" in df.columns and "shrink_k" in df.columns:
        value = -_col(df, "log_ratio") * _col(df, "shrink_k")
    else:
        value = -_col(df, "deal_pct")
    return pd.DataFrame({
        "value": value,
        "tax": -tax_delta,
        "price_drop": _col(df, "price_drop_pct").fillna(0.0),
        "market_time": _col(df, "days_on_market").clip(upper=MARKET_TIME_CAP),
    }, index=df.index)


def js_round1(x: pd.Series) -> pd.Series:
    return np.floor(x * 10 + 0.5) / 10


def deal_fields(df: pd.DataFrame, tau2: float) -> pd.DataFrame:
    """Effective rent, shrinkage, z, chance below fair, savings and verdict (section 1-3)."""
    df = df.copy()
    tax_delta = _col(df, "tax_delta_month").fillna(0.0)
    fair = _col(df, "fair_rent").where(lambda s: s > 0)
    sigma = _col(df, "fair_sigma_log").where(lambda s: s > 0)
    rent = _col(df, "rent_gross")
    df["effective_rent"] = rent + tax_delta
    df["effective_deal_pct"] = (df["effective_rent"] / fair - 1) * 100
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(rent / fair)
        lr_eff = np.log(df["effective_rent"].where(lambda s: s > 0) / fair)
    k = tau2 / (tau2 + sigma ** 2)
    df["log_ratio"] = lr
    df["shrink_k"] = k
    df["deal_pct_adj"] = (np.exp(lr * k) - 1) * 100
    df["deal_pct_adj_eff"] = (np.exp(lr_eff * k) - 1) * 100
    df["deal_z"] = lr_eff / sigma
    p = pd.Series(norm_cdf(-df["deal_z"].to_numpy(float)), index=df.index)
    df["p_below_fair"] = p.where(df["deal_z"].notna())
    df["saving_month"] = fair - df["effective_rent"]
    df["saving_adj_month"] = fair * (1 - np.exp(lr_eff * k))
    df["verdict"] = [verdict_for(d, q) for d, q in zip(df["deal_pct_adj_eff"], df["p_below_fair"])]
    return df


def score(df: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """Add the deal fields (see module doc), comp_*, score; sort by score desc (NaN last).
    Returns (df, scoring_info) with scoring_info = {tau2, tau2_floor, components_used}."""
    df = df.copy()
    if "suspicious" not in df.columns:
        df["suspicious"] = False
    susp = df["suspicious"].fillna(False).astype(bool)
    if "fair_sigma_log" not in df.columns:
        df["fair_sigma_log"] = np.nan
    kind = df["listing_kind"] if "listing_kind" in df.columns else pd.Series("regular", index=df.index)

    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(_col(df, "rent_gross") / _col(df, "fair_rent").where(lambda s: s > 0))
    tau2 = estimate_tau2(lr, _col(df, "fair_sigma_log"), ~susp & (kind == "regular"))
    df = deal_fields(df, tau2)

    sig = raw_signals(df)
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    used = []
    for k in COMPONENTS:
        comp = component(sig[k], ~susp)
        df[f"comp_{k}"] = comp
        if comp.notna().any():
            used.append(k)
        w = float(weights.get(k, 0.0) or 0.0)
        if w > 0:
            ok = comp.notna()
            num[ok] += w * comp[ok]
            den[ok] += w
    raw = num / den.where(den > 0)
    raw[susp] = np.nan
    df["score"] = js_round1(raw)
    df = df.sort_values("score", ascending=False, na_position="last", kind="mergesort").reset_index(drop=True)
    return df, {"tau2": tau2, "tau2_floor": TAU2_FLOOR, "great_min_p": GREAT_MIN_P, "components_used": used}
