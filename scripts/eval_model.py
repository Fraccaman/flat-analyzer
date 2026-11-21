"""Cross-validated comparison of fair-rent model variants.

    uv run python scripts/eval_model.py [--config config.yaml]

All numbers are 5-fold out-of-fold (fixed seed, duplicate listings kept in
one fold), measured on the same rows for every variant: regular listings
(no lease takeovers / short-term stays) with m2 for the full model, and
regular listings without m2 for the no-m2 fallback model.

  rmse_log  root mean squared error of log(rent) - log(fair)
  mae_pct   mean |fair / rent - 1|, in %
  r2        1 - SSE/SST on log rent

OLD = the previous method (OLS, old feature set, area dummies with a tiny
ridge, trained on every listing incl. takeovers) run through the same CV, so
its error is honest rather than in-sample.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from flatprice.analysis.clean import clean
from flatprice.analysis.fair_rent import OLD_SPEC, RIDGE_LAMBDA, ModelSpec, estimate_class, prepare
from flatprice.config import load_config
from flatprice.store import Store


def load_frame(config_path: str):
    from flatprice import geo

    cfg = load_config(config_path)
    store = Store(cfg.db_path)
    try:
        frame = store.load_frame()
    finally:
        store.close()
    df = geo.assign_area(geo.assign_municipality(clean(frame)))
    return df, cfg


def run_variant(df, spec: ModelSpec, min_listings: int = 30) -> dict:
    d, ctx, groups, folds = prepare(df)
    rent_ok = d["rent_gross"].notna().to_numpy()
    has_m2 = d["living_space_m2"].notna().to_numpy() & rent_ok
    has_rooms = d["rooms"].notna().to_numpy() & rent_ok
    regular = (d["listing_kind"] == "regular").to_numpy()
    train_ok = regular if spec.exclude_kinds else np.ones(len(d), bool)
    hi = estimate_class(d, ctx, groups, folds, spec, True, has_m2 & train_ok, has_m2, has_m2 & regular)
    lo = estimate_class(d, ctx, groups, folds, spec, False, has_rooms & train_ok, ~has_m2 & has_rooms,
                        ~has_m2 & has_rooms & regular, tau2_override=hi["tau2"])
    return {"high": hi, "low": lo}


def fmt(m: dict | None) -> str:
    if not m or not m.get("n"):
        return f"{'–':>8} {'–':>8} {'–':>6} {'–':>4}"
    return f"{m['rmse_log']:8.4f} {m['mae_pct']:7.2f}% {m['r2']:6.3f} {m['n']:4d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    df, _ = load_frame(args.config)

    base = ModelSpec(features="old", robust=False, area_lambda=RIDGE_LAMBDA, exclude_kinds=True, comps=False)
    new = replace(base, features="new")
    variants: list[tuple[str, ModelSpec]] = [
        ("OLD (OLS, old features, all kinds)", OLD_SPEC),
        ("OLS, old features, regular only", base),
        ("  + Huber robust", replace(base, robust=True)),
        ("OLS + new features", new),
        ("  + Huber robust", replace(new, robust=True)),
    ]
    for lam in (1.0, 3.0, 10.0, 30.0):
        variants.append((f"OLS + new feat. + area shrink {lam:g}", replace(new, area_lambda=lam)))
    variants += [
        ("+ comps blend, inv CV-MSE (SHIPPED)", ModelSpec()),
        ("+ comps blend, inv error variance", ModelSpec(blend="grubbs")),
        ("SHIPPED but Huber robust", ModelSpec(robust=True)),
    ]
    assert ModelSpec().blend == "cv" and not ModelSpec().robust

    hdr = f"{'variant':38s} | {'full model (m2)':>29s} | {'fallback (no m2)':>29s}"
    sub = f"{'':38s} | {'rmse_log':>8} {'MAE':>8} {'R2':>6} {'n':>4} | {'rmse_log':>8} {'MAE':>8} {'R2':>6} {'n':>4}"
    print(hdr)
    print(sub)
    print("-" * len(sub))
    results = {}
    for name, spec in variants:
        r = run_variant(df, spec)
        results[name] = r
        print(f"{name:38s} | {fmt(r['high']['cv']['final'])} | {fmt(r['low']['cv']['final'])}")

    print("\nComparables diagnostics (shipped spec, rows where comps exist):")
    r = results["+ comps blend, inv CV-MSE (SHIPPED)"]
    for cls in ("high", "low"):
        cv = r[cls]["cv"]
        print(f"  {cls:4s} hedonic {fmt(cv.get('hedonic_on_comps_rows'))} | comps {fmt(cv.get('comps'))}"
              f" | blend {fmt(cv.get('blend'))}")
        g = r[cls].get("grubbs")
        if g:
            print(f"       this class alone: tau={np.sqrt(g['tau2']):.3f}  hedonic err sd={np.sqrt(g['var_h']):.3f}"
                  f"  comps err sd={np.sqrt(g['var_c']):.3f}  w_hedonic={r[cls].get('blend_weight_hedonic'):.3f}")
        print(f"       listing-level sd used: tau={np.sqrt(r[cls]['tau2']):.3f}"
              + (" (borrowed from the m2 class)" if cls == "low" else ""))
        print(f"       fair_sigma_log by method: { {k: round(v, 4) for k, v in r[cls]['sigma'].items()} }")
        meth = r[cls]["method"]
        print(f"       methods used: { {m: int((meth == m).sum()) for m in ('blend', 'hedonic')} }")


if __name__ == "__main__":
    main()
