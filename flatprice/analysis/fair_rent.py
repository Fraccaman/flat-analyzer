"""Fair-rent model: hedonic regression + comparables, out-of-fold.

Hedonic model (ridge regression on log(rent_gross); Huber-robust IRLS is
available via ModelSpec(robust=True) but did not lower the CV error here, see
scripts/eval_model.py):

  log(rent) ~ 1 + log(m2) + rooms + clip(dist_center_km, 0, 8)
              + log1p(clip(dist_lake_km, 0, 3))
              + furnished + balcony + parking + elevator
              + title flags: lake view, terrace/attic, renovated/new, garden
              + floor flags: ground floor, high floor (>= 4 or attic)
              + utilities-known flag (net and gross both given)
              + year_built bucket dummies (baseline "unknown")
              + area effects: one dummy per area (areas with < MIN_AREA_LISTINGS
                pooled into "other"), ridge-shrunk towards the grand mean with
                AREA_LAMBDA (all other coefficients get the tiny RIDGE_LAMBDA)

Two models: "high" confidence (rows with m2, formula above) and the "low"
confidence fallback (same formula without log(m2), fit on all rows with rooms)
for rows lacking m2.

Honesty: every listing's hedonic estimate is OUT-OF-FOLD (5-fold CV with a
fixed seed, duplicate listings kept in the same fold), so deal_pct is not the
optimistic in-sample residual. Listings that are not regular flats
(listing_kind "takeover" / "short_term", from the title) are excluded from
training; they still get a fair rent from the model fit on regular flats.

Comparables: for each listing the up-to-COMPS_K nearest regular listings in a
scaled space (lat/lon in km, log m2 / rooms), within COMPS_MAX_DIST, excluding
itself, its duplicates and provisionally suspicious rows. Estimate =
median_j [ log rent_j + b * (size_i - size_j) ] with b the hedonic log-m2
(or rooms) coefficient. It is blended with the hedonic estimate in log space
with inverse-variance weights from each method's CV error (only when it lowers
the CV error of that confidence class; see scripts/eval_model.py).

Uncertainty: fair_sigma_log is the estimated sd of the error of log(fair_rent)
for the method / confidence class used. A listing's CV residual r = y - yhat
mixes two things: its own deviation from the market (the "true deal", variance
tau^2, incl. unobserved quality) and the error of the fair-rent estimate
(sigma^2). Hedonic and comparables estimate the same fair rent with roughly
independent errors, so (Grubbs' method-comparison estimator)

    tau^2     = (MSE_h + MSE_c - mean((yhat_h - yhat_c)^2)) / 2
    sigma_m^2 = CV MSE of method m - tau^2           (floored at SIGMA_FLOOR^2)

If the decomposition is unavailable, sigma = CV rmse (conservative).

If fewer than `min_listings` rows are usable, fair rent falls back to medians
(median CHF/m2 per area * m2, else median CHF/room * rooms), sigma = DEFAULT_SIGMA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_AREA_LISTINGS = 5
DIST_CLIP = (0.0, 8.0)
LAKE_CLIP = (0.0, 3.0)
RIDGE_LAMBDA = 1e-3
AREA_LAMBDA = 10.0
HUBER_C = 1.345
CV_FOLDS = 5
CV_SEED = 20261120
COMPS_K = 8
COMPS_MIN = 3
COMPS_MAX_DIST = 2.0
COMPS_SCALE = {"km": 1.0, "log_m2": 0.2, "rooms": 1.0}  # one unit of comps distance
SIGMA_FLOOR = 0.05
DEFAULT_SIGMA = 0.40
SCAM_DEAL_PCT = -40.0
SCAM_MAX_RENT = 1_200.0
NO_COORDS_SCAM_DEAL_PCT = -30.0
YEAR_BUCKETS = ["<1970", "1970-1999", "2000-2014", "2015+"]  # baseline: "unknown"

# Listing kinds (title regex, lower-cased). Takeover wins over short-term.
TAKEOVER_RE = r"subentr|nachmieter|nachfolgemieter|take ?over"
SHORT_TERM_RE = (
    r"temporane|breve periodo|affitt[oi] brev|short[- ]?term|short stays?|befristet|kurzzeit|"
    r"settiman|weekly|flexible rental|ferienwohnung|vacanz|holiday"
)
# Title feature flags (Italian / German / English).
LAKE_RE = r"vista lago|\blago\b|seeblick|seesicht|lake ?view|\blake\b|am see\b|vom see\b"
TERRACE_RE = r"terrazz|terrace|terrasse|attico|attica|penthouse|attika|dachwohnung"
ATTIC_RE = r"attico|attica|penthouse|attika|dachwohnung"
RENOVATED_RE = (
    r"ristrutturat|rinnovat|\bnuov[oai]\b|nuovissim|nuova costruzione|\bnew\b|neubau|renoviert|"
    r"renovated|saniert|erstbezug"
)
GARDEN_RE = r"giardin|garten|garden"
GROUND_RE = r"piano terra|pianterreno|erdgeschoss|ground floor"
ATTIC_CATEGORIES = {"ATTIC_FLAT", "ROOF_FLAT", "TERRACE_FLAT"}
HIGH_FLOOR = 4

BASE_FLAGS = [("is_furnished", "furnished"), ("has_balcony", "balcony"),
              ("has_parking", "parking"), ("has_elevator", "elevator")]
NEW_FLAGS = ["f_lake_view", "f_terrace", "f_renovated", "f_garden", "f_ground", "f_high", "f_utilities"]


@dataclass(frozen=True)
class ModelSpec:
    """Model configuration (the defaults are the shipped model; eval varies them)."""

    features: str = "new"  # "old" (pre-2026-11 feature set) | "new"
    robust: bool = False  # Huber IRLS instead of least squares (no CV gain on our data)
    area_lambda: float = AREA_LAMBDA  # ridge on area dummies ("new" only)
    exclude_kinds: bool = True  # train on regular listings only
    comps: bool = True  # blend with comparables where it lowers CV error
    blend: str = "cv"  # "cv": inverse CV-MSE weights | "grubbs": inverse error variance


OLD_SPEC = ModelSpec(features="old", robust=False, area_lambda=RIDGE_LAMBDA, exclude_kinds=False, comps=False)


def year_bucket(y) -> str:
    if y is None or pd.isna(y) or y < 1800 or y > 2100:
        return "unknown"
    if y < 1970:
        return "<1970"
    if y < 2000:
        return "1970-1999"
    if y < 2015:
        return "2000-2014"
    return "2015+"


def _title(df: pd.DataFrame) -> pd.Series:
    t = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    # "Molino Nuovo" is a Lugano quarter, not a "new" flat
    return t.fillna("").astype(str).str.lower().str.replace("molino nuovo", "molino", regex=False)


def listing_kind(df: pd.DataFrame) -> pd.Series:
    """"takeover" (lease takeover ad), "short_term" (temporary/furnished stays) or "regular"."""
    t = _title(df)
    kind = pd.Series("regular", index=df.index, dtype=object)
    kind[t.str.contains(SHORT_TERM_RE, regex=True)] = "short_term"
    kind[t.str.contains(TAKEOVER_RE, regex=True)] = "takeover"
    return kind


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "living_space_m2": np.nan, "rooms": np.nan, "dist_center_km": np.nan, "dist_lake_km": np.nan,
        "year_built": np.nan, "latitude": np.nan, "longitude": np.nan, "area": None, "floor": np.nan,
        "rent_net": np.nan,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    for c in ["living_space_m2", "rooms", "dist_center_km", "dist_lake_km", "year_built", "latitude",
              "longitude", "floor", "rent_net", "rent_gross"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c, _ in BASE_FLAGS:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].fillna(False).astype(bool)
    if "chf_per_m2" not in df.columns:
        df["chf_per_m2"] = df["rent_gross"] / df["living_space_m2"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add listing_kind and the f_* model flags (expects _ensure_cols output)."""
    df = df.copy()
    t = _title(df)
    cats = df["categories"] if "categories" in df.columns else pd.Series([[]] * len(df), index=df.index)
    attic_cat = cats.map(lambda c: bool(ATTIC_CATEGORIES & set(c)) if isinstance(c, (list, tuple, set)) else False)
    floor = df["floor"]
    df["listing_kind"] = listing_kind(df)
    df["f_lake_view"] = t.str.contains(LAKE_RE, regex=True)
    df["f_terrace"] = t.str.contains(TERRACE_RE, regex=True)
    df["f_renovated"] = t.str.contains(RENOVATED_RE, regex=True)
    df["f_garden"] = t.str.contains(GARDEN_RE, regex=True)
    df["f_ground"] = (floor <= 0).fillna(False) | t.str.contains(GROUND_RE, regex=True)
    df["f_high"] = (floor >= HIGH_FLOOR).fillna(False) | t.str.contains(ATTIC_RE, regex=True) | attic_cat
    df["f_utilities"] = (df["rent_net"].notna() & (df["rent_gross"] > df["rent_net"])).astype(bool)
    for c in NEW_FLAGS:
        df[c] = df[c].astype(bool)
    return df


def dup_groups(df: pd.DataFrame) -> np.ndarray:
    """Group id per row; rows with the same coords, rooms, m2 and rent are one flat
    listed more than once (kept in the same CV fold, never each other's comparable)."""
    key = pd.DataFrame({
        "la": df["latitude"].round(4), "lo": df["longitude"].round(4), "r": df["rooms"],
        "m": df["living_space_m2"], "p": df["rent_gross"].round(-1),
    }).apply(lambda r: "|".join(map(str, r)), axis=1)
    no_coords = (df["latitude"].isna() | df["longitude"].isna()).to_numpy()
    codes = pd.factorize(key)[0].astype(np.int64)
    codes[no_coords] = -1 - np.arange(int(no_coords.sum()))  # no coords: never grouped
    return codes


def cv_folds(groups: np.ndarray, k: int = CV_FOLDS, seed: int = CV_SEED) -> np.ndarray:
    """Fold index per row (group-aware, fixed seed)."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = dict(zip(uniq[perm], np.arange(len(uniq)) % k))
    return np.array([fold_of[g] for g in groups], dtype=np.int64)


def _area_levels(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    area = df["area"].fillna("other").astype(str)
    counts = area.value_counts()
    small = counts[counts < MIN_AREA_LISTINGS].index
    area = area.where(~area.isin(small), "other")
    levels = list(area.value_counts().index)  # most common first = baseline (old spec)
    return area, levels


@dataclass
class _Ctx:
    areas: list[str]
    dist_fill: float
    lake_fill: float
    rooms_per_m2: float


def _context(df: pd.DataFrame) -> _Ctx:
    med = lambda s: float(s.median()) if s.notna().any() else 0.0  # noqa: E731
    rpm = (df["rooms"] / df["living_space_m2"]).median()
    _, areas = _area_levels(df)
    return _Ctx(areas, med(df["dist_center_km"]), med(df["dist_lake_km"]),
                float(rpm) if pd.notna(rpm) else 1 / 30)


def _design(m: pd.DataFrame, ctx: _Ctx, with_m2: bool, spec: ModelSpec) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Design matrix, column names and per-column ridge penalty."""
    cols: list[np.ndarray] = [np.ones(len(m))]
    names = ["intercept"]
    pens = [0.0]

    def add(name, values, pen=RIDGE_LAMBDA):
        cols.append(np.asarray(values, dtype=float))
        names.append(name)
        pens.append(pen)

    if with_m2:
        add("log_m2", np.log(m["living_space_m2"].to_numpy(float)))
    rooms = m["rooms"]
    if with_m2:
        rooms = rooms.fillna(m["living_space_m2"] * ctx.rooms_per_m2)
    add("rooms", rooms.to_numpy(float))
    add("dist_center_km", m["dist_center_km"].fillna(ctx.dist_fill).clip(*DIST_CLIP).to_numpy(float))
    if spec.features == "new":
        add("log1p_dist_lake_km", np.log1p(m["dist_lake_km"].fillna(ctx.lake_fill).clip(*LAKE_CLIP).to_numpy(float)))
    for c, n in BASE_FLAGS:
        add(n, m[c].astype(float).to_numpy())
    if spec.features == "new":
        for c in NEW_FLAGS:
            add(c[2:], m[c].astype(float).to_numpy())
    yb = m["year_built"].map(year_bucket)
    for b in YEAR_BUCKETS:
        add(f"year_{b}", (yb == b).astype(float).to_numpy())
    if spec.features == "new":
        for a in ctx.areas:  # every level, shrunk towards the grand mean
            add(f"area_{a}", (m["_area_fe"] == a).astype(float).to_numpy(), spec.area_lambda)
    else:
        for a in ctx.areas[1:]:  # areas[0] is the baseline
            add(f"area_{a}", (m["_area_fe"] == a).astype(float).to_numpy())
    return np.column_stack(cols), names, np.array(pens)


def _fit(X: np.ndarray, y: np.ndarray, pen: np.ndarray, robust: bool, c: float = HUBER_C,
         max_iter: int = 50) -> np.ndarray:
    """Weighted ridge; with robust=True, Huber IRLS (scale = 1.4826 * MAD of residuals)."""
    w = np.ones(len(y))
    P = np.diag(pen)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter if robust else 1):
        Xw = X * w[:, None]
        beta, *_ = np.linalg.lstsq(X.T @ Xw + P, Xw.T @ y, rcond=None)
        if not robust:
            break
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        if not s > 1e-9:
            s = float(np.std(r)) or 1.0
        w_new = np.minimum(1.0, c * s / np.maximum(np.abs(r), 1e-12))
        if np.max(np.abs(w_new - w)) < 1e-6:
            w = w_new
            break
        w = w_new
    return beta


def hedonic_oof(df: pd.DataFrame, train: np.ndarray, predict: np.ndarray, with_m2: bool, spec: ModelSpec,
                ctx: _Ctx, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Log predictions for rows in `predict`: out-of-fold for training rows (fit
    without that row's fold), full-fit for the others. Returns (pred, beta_full, names)."""
    X, names, pen = _design(df, ctx, with_m2, spec)
    y = np.log(df["rent_gross"].to_numpy(float))
    pred = np.full(len(df), np.nan)
    beta = _fit(X[train], y[train], pen, spec.robust)
    out = predict & ~train
    pred[out] = X[out] @ beta
    for f in np.unique(folds[train]):
        tr = train & (folds != f)
        te = predict & train & (folds == f)
        if te.any() and tr.sum() > X.shape[1] // 4:
            pred[te] = X[te] @ _fit(X[tr], y[tr], pen, spec.robust)
        elif te.any():
            pred[te] = X[te] @ beta
    return pred, beta, names


def comps_estimate(df: pd.DataFrame, pool: np.ndarray, target: np.ndarray, with_m2: bool, size_coef: float,
                   groups: np.ndarray, k: int = COMPS_K, max_dist: float = COMPS_MAX_DIST,
                   min_n: int = COMPS_MIN) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Comparables estimate of log fair rent for `target` rows (see module doc).
    Returns (log_pred (NaN if < min_n comps), n_comps, keys of the 5 nearest)."""
    n = len(df)
    pred = np.full(n, np.nan)
    ncomp = np.zeros(n, dtype=np.int64)
    keys: list[list[str]] = [[] for _ in range(n)]
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    kx = math.cos(math.radians(46.0)) * 111.32
    feats = [lon * kx / COMPS_SCALE["km"], lat * 110.95 / COMPS_SCALE["km"]]
    if with_m2:
        size = np.log(df["living_space_m2"].to_numpy(float))
        feats.append(size / COMPS_SCALE["log_m2"])
        feats.append(df["rooms"].fillna(df["living_space_m2"] / 30).to_numpy(float) / COMPS_SCALE["rooms"])
    else:
        size = df["rooms"].to_numpy(float)
        feats.append(size / COMPS_SCALE["rooms"])
    Z = np.column_stack(feats)
    ok = np.isfinite(Z).all(axis=1)
    pool = pool & ok
    tgt = np.flatnonzero(target & ok)
    pidx = np.flatnonzero(pool)
    if not len(tgt) or len(pidx) < min_n:
        return pred, ncomp, keys
    y = np.log(df["rent_gross"].to_numpy(float))
    key_col = df["key"].astype(str).to_numpy() if "key" in df.columns else np.array([str(i) for i in range(n)])
    Zp = Z[pidx]
    for i in tgt:
        d = np.sqrt(((Zp - Z[i]) ** 2).sum(axis=1))
        d[groups[pidx] == groups[i]] = np.inf  # itself and its duplicates
        order = np.argsort(d, kind="stable")[:k]
        order = order[d[order] <= max_dist]
        ncomp[i] = len(order)
        if len(order) < min_n:
            continue
        j = pidx[order]
        pred[i] = float(np.median(y[j] + size_coef * (size[i] - size[j])))
        keys[i] = [str(x) for x in key_col[j[:5]]]
    return pred, ncomp, keys


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(pred)
    if not ok.any():
        return {"n": 0, "rmse_log": None, "mae_pct": None, "r2": None}
    r = y[ok] - pred[ok]
    ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return {
        "n": int(ok.sum()),
        "rmse_log": float(np.sqrt(np.mean(r ** 2))),
        "mae_pct": float(np.mean(np.abs(np.exp(-r) - 1)) * 100),  # |fair/rent - 1|
        "r2": float(1 - np.sum(r ** 2) / ss_tot) if ss_tot > 0 else None,
    }


def _median_fallback(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Fair rent from medians for rows in mask: area median CHF/m2 * m2, else
    global median CHF/m2 * m2, else median CHF/room * rooms."""
    out = pd.Series(np.nan, index=df.index)
    area = df["area"].fillna("other").astype(str)
    rate_area = df.groupby(area)["chf_per_m2"].median()
    rate_global = df["chf_per_m2"].median()
    rate = area.map(rate_area).fillna(rate_global)
    by_m2 = rate * df["living_space_m2"]
    per_room = (df["rent_gross"] / df["rooms"]).median()
    by_room = per_room * df["rooms"]
    out[mask] = by_m2[mask].fillna(by_room[mask])
    return out


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, _Ctx, np.ndarray, np.ndarray]:
    """Columns, features, area levels, duplicate groups and CV folds."""
    df = add_features(_ensure_cols(df)).reset_index(drop=True)
    df["_area_fe"], _ = _area_levels(df)
    ctx = _context(df)
    groups = dup_groups(df)
    return df, ctx, groups, cv_folds(groups)


def _provisional_suspicious(df: pd.DataFrame, log_pred: np.ndarray) -> np.ndarray:
    deal = (df["rent_gross"].to_numpy(float) / np.exp(log_pred) - 1) * 100
    no_coords = (df["latitude"].isna() | df["longitude"].isna()).to_numpy()
    with np.errstate(invalid="ignore"):
        s = ((deal < SCAM_DEAL_PCT) & (df["rent_gross"].to_numpy(float) < SCAM_MAX_RENT)) | (
            no_coords & (deal < NO_COORDS_SCAM_DEAL_PCT))
    return np.nan_to_num(s, nan=False).astype(bool)


def grubbs(y: np.ndarray, h: np.ndarray, c: np.ndarray) -> dict | None:
    """Method-comparison split of CV error into listing-level variance (tau2) and
    each method's own error variance. None if not identifiable."""
    ok = np.isfinite(y) & np.isfinite(h) & np.isfinite(c)
    if ok.sum() < 20:
        return None
    mse_h = float(np.mean((y[ok] - h[ok]) ** 2))
    mse_c = float(np.mean((y[ok] - c[ok]) ** 2))
    msd = float(np.mean((h[ok] - c[ok]) ** 2))
    tau2 = (mse_h + mse_c - msd) / 2
    if not tau2 > 0:
        return None
    return {"n": int(ok.sum()), "tau2": tau2, "mse_h": mse_h, "mse_c": mse_c,
            "var_h": max(mse_h - tau2, SIGMA_FLOOR ** 2), "var_c": max(mse_c - tau2, SIGMA_FLOOR ** 2)}


def estimate_class(df: pd.DataFrame, ctx: _Ctx, groups: np.ndarray, folds: np.ndarray, spec: ModelSpec,
                   with_m2: bool, train: np.ndarray, target: np.ndarray, eval_mask: np.ndarray,
                   tau2_override: float | None = None) -> dict:
    """Fit one confidence class. `train`: rows the model learns from; `target`:
    rows that get this class's fair rent; `eval_mask`: rows whose CV error
    defines the class sigma. Returns log preds, sigma, comps and diagnostics."""
    y = np.log(df["rent_gross"].to_numpy(float))
    predict = train | target
    h, beta, names = hedonic_oof(df, train, predict, with_m2, spec, ctx, folds)
    coefs = dict(zip(names, beta.tolist()))
    out: dict = {"coefs": coefs, "hedonic": h, "final": h.copy(), "method": np.where(np.isfinite(h), "hedonic", None),
                 "n_comps": np.zeros(len(df), dtype=np.int64), "comps": [[] for _ in range(len(df))],
                 "cv": {"hedonic": _metrics(y[eval_mask], h[eval_mask])}, "n_train": int(train.sum())}
    sig_h = out["cv"]["hedonic"]["rmse_log"] or DEFAULT_SIGMA
    out["sigma"] = {"hedonic": sig_h}
    out["tau2"] = None

    # Comparables (also used, when not blended, to split the CV error into tau2 + sigma).
    pool = train & ~_provisional_suspicious(df, h)
    size_coef = coefs["log_m2"] if with_m2 else coefs["rooms"]
    c, ncomp, keys = comps_estimate(df, pool, predict, with_m2, size_coef, groups)
    out["n_comps"], out["comps"] = ncomp, keys
    both = eval_mask & np.isfinite(c) & np.isfinite(h)
    g = grubbs(y[both], h[both], c[both])
    out["grubbs"] = g
    out["cv"]["comps"] = _metrics(y[both], c[both])
    out["cv"]["hedonic_on_comps_rows"] = _metrics(y[both], h[both])
    # listing-level variance: from this class, unless given (the small no-m2 class
    # borrows it from the m2 class, where the decomposition is well determined)
    tau2 = tau2_override if tau2_override is not None else (g["tau2"] if g else None)
    out["tau2"] = tau2

    def err_sigma(rmse):  # CV rmse -> error sd of the fair-rent estimate
        if tau2 is None or rmse is None:
            return rmse
        return float(np.sqrt(max(rmse ** 2 - tau2, SIGMA_FLOOR ** 2)))

    out["sigma"]["hedonic"] = err_sigma(sig_h)
    if g and spec.comps:
        if spec.blend == "grubbs":
            wh, wc = 1 / g["var_h"], 1 / g["var_c"]
        else:
            wh, wc = 1 / g["mse_h"], 1 / g["mse_c"]
        w = wh / (wh + wc)
        blend = np.where(np.isfinite(c) & np.isfinite(h), w * h + (1 - w) * c, h)
        m_blend = _metrics(y[both], blend[both])
        out["cv"]["blend"] = m_blend
        out["blend_weight_hedonic"] = float(w)
        # keep comparables only if they lower the CV error on the rows they cover
        if m_blend["rmse_log"] < out["cv"]["hedonic_on_comps_rows"]["rmse_log"]:
            use = np.isfinite(c) & np.isfinite(h)
            out["final"] = blend
            out["method"] = np.where(use, "blend", out["method"])
            out["sigma"]["blend"] = err_sigma(m_blend["rmse_log"])
    out["cv"]["final"] = _metrics(y[eval_mask], out["final"][eval_mask])
    return out


def fit_fair_rent(df: pd.DataFrame, min_listings: int = 30, spec: ModelSpec | None = None) -> tuple[pd.DataFrame, dict]:
    """Add fair_rent, deal_pct, model_confidence ("high"/"low"/None), suspicious,
    listing_kind, fair_sigma_log, fair_method, n_comps, comps.

    Rows with ``data_suspect`` are kept out of training and comparables and are
    marked ``suspicious`` (not scored), but still get a fair-rent estimate.

    Returns (df, model_info); model_info keeps the old keys (method, r2, n,
    rmse_log, coefs, fallback) with r2 / rmse_log now out-of-fold, plus cv,
    sigma and tau2 diagnostics.
    """
    spec = spec or ModelSpec()
    df, ctx, groups, folds = prepare(df)
    n = len(df)
    rent_ok = df["rent_gross"].notna().to_numpy()
    has_m2 = df["living_space_m2"].notna().to_numpy() & rent_ok
    has_rooms = df["rooms"].notna().to_numpy() & rent_ok
    regular = (df["listing_kind"] == "regular").to_numpy() if spec.exclude_kinds else np.ones(n, bool)
    # Rows whose data looks perturbed (flatprice.quality) never train the model or
    # serve as comparables (the comparables pool is a subset of the training rows).
    data_suspect = (df["data_suspect"].fillna(False).astype(bool).to_numpy()
                    if "data_suspect" in df.columns else np.zeros(n, bool))
    regular = regular & ~data_suspect

    logp = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    method = np.full(n, None, dtype=object)
    conf = np.full(n, None, dtype=object)
    ncomp = np.zeros(n, dtype=np.int64)
    comps: list[list[str]] = [[] for _ in range(n)]
    info: dict = {"method": "huber" if spec.robust else "ols", "r2": None, "n": 0, "rmse_log": None, "coefs": {},
                  "fallback": {"r2": None, "n": 0, "rmse_log": None, "coefs": {}},
                  "cv_folds": CV_FOLDS, "spec": spec.__dict__.copy(), "sigma": {}, "tau2": {}}

    def apply(res, target, cls):
        nonlocal ncomp
        fin = res["final"]
        t = target & np.isfinite(fin)
        logp[t] = fin[t]
        method[t] = res["method"][t]
        conf[t] = cls
        for i in np.flatnonzero(t):
            m = res["method"][i]
            sigma[i] = res["sigma"].get(m, res["sigma"]["hedonic"])
            ncomp[i] = res["n_comps"][i]
            comps[i] = res["comps"][i]

    def summary(res):
        m = res["cv"]["final"]
        return {"r2": m["r2"], "n": res["n_train"], "rmse_log": m["rmse_log"], "mae_pct": m["mae_pct"],
                "coefs": res["coefs"], "cv": res["cv"], "sigma": res["sigma"], "tau2": res["tau2"],
                "blend_weight_hedonic": res.get("blend_weight_hedonic")}

    train_hi = has_m2 & regular
    if train_hi.sum() >= min_listings:
        res = estimate_class(df, ctx, groups, folds, spec, True, train_hi, has_m2, train_hi)
        apply(res, has_m2, "high")
        s = summary(res)
        info.update({k: s[k] for k in ("r2", "n", "rmse_log", "coefs")})
        info["high"] = s
        info["sigma"]["high"], info["tau2"]["high"] = res["sigma"], res["tau2"]
        need = ~has_m2 & has_rooms
        train_lo = has_rooms & regular
        if need.any() and train_lo.sum() >= min_listings:
            # sigma from the rows the fallback actually serves (no m2) when there are enough
            ev = train_lo & ~has_m2 if (train_lo & ~has_m2).sum() >= 30 else train_lo
            res = estimate_class(df, ctx, groups, folds, spec, False, train_lo, need, ev,
                                 tau2_override=info["tau2"]["high"])
            apply(res, need, "low")
            info["fallback"] = summary(res)
            info["fallback"]["n"] = int(train_lo.sum())
            info["sigma"]["low"], info["tau2"]["low"] = res["sigma"], res["tau2"]
    else:
        info["method"] = "median"
        info["n"] = int(has_m2.sum())
    rest = ~np.isfinite(logp) & (has_m2 | has_rooms)
    if rest.any():
        med = _median_fallback(df, pd.Series(rest, index=df.index))
        v = med.to_numpy(float)
        ok = rest & np.isfinite(v) & (v > 0)
        logp[ok] = np.log(v[ok])
        sigma[ok] = DEFAULT_SIGMA
        method[ok] = "median"
        conf[ok] = "low"

    df["fair_rent"] = np.round(np.exp(logp), 0)
    df["fair_sigma_log"] = sigma
    df["fair_method"] = method
    df["model_confidence"] = conf
    df["n_comps"] = ncomp
    df["comps"] = comps
    df["deal_pct"] = (df["rent_gross"] / df["fair_rent"] - 1) * 100

    no_coords = df["latitude"].isna() | df["longitude"].isna()
    deal = df["deal_pct"]
    df["suspicious"] = (
        ((deal < SCAM_DEAL_PCT) & (df["rent_gross"] < SCAM_MAX_RENT))
        | (no_coords & (deal < NO_COORDS_SCAM_DEAL_PCT))
    ).fillna(False).astype(bool) | data_suspect

    info["kinds"] = df["listing_kind"].value_counts().to_dict()
    df = df.drop(columns=["_area_fe"])
    return df, info

