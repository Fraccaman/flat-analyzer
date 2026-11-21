from __future__ import annotations

import json
import math
import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from flatprice.analysis.clean import clean
from flatprice.analysis.fair_rent import fit_fair_rent, listing_kind
from flatprice.analysis.score import (TAU2_FLOOR, deal_fields, estimate_tau2, norm_cdf, percentile_midrank, score,
                                      verdict_for)
from flatprice.config import Config

AREAS = ["Centro", "Besso", "Molino", "Paradiso"]


def synthetic(n=200, seed=0, cheap=(0, 1, 2), expensive=(3, 4, 5)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m2 = rng.uniform(30, 150, n).round(0)
    rent = 40 * m2 ** 0.8 * np.exp(rng.normal(0, 0.05, n))
    rent[list(cheap)] *= 0.7
    rent[list(expensive)] *= 1.4
    return pd.DataFrame({
        "portal": "homegate",
        "portal_id": [str(i) for i in range(n)],
        "key": [f"homegate:{i}" for i in range(n)],
        "url": [f"https://x/{i}" for i in range(n)],
        "search": "lugano",
        "title": ["Appartamento"] * n,
        "latitude": 46.0 + rng.normal(0, 0.01, n),
        "longitude": 8.95 + rng.normal(0, 0.01, n),
        "rent_gross": rent.round(0),
        "rent_net": (rent * 0.85).round(0),
        "rooms": (m2 / 30).round(0) + 0.5,
        "living_space_m2": m2,
        "year_built": rng.choice([None, 1960, 1985, 2005, 2020], n),
        "is_furnished": False,
        "has_balcony": rng.choice([True, False, None], n),
        "has_parking": None,
        "has_elevator": rng.choice([True, False], n),
        "pets_allowed": None,
        "categories": [json.dumps(["APARTMENT"])] * n,
        "first_seen": "2026-10-01", "last_seen": "2026-11-20",
        "days_on_market": rng.uniform(0, 120, n),
        "price_drop_pct": rng.choice([0.0, 0.0, 5.0], n),
        "active": True,
        "n_observations": 3,
        "area": [AREAS[i % 4] for i in range(n)],
        "dist_center_km": rng.uniform(0, 5, n),
    })


# ---------------------------------------------------------------- clean
def test_clean_bounds_and_normalization():
    df = synthetic(10)
    df.loc[0, "rent_gross"] = 100  # below bounds -> dropped
    df.loc[1, "active"] = False  # inactive -> dropped
    df.loc[2, "living_space_m2"] = 5  # out of size bounds -> NaN
    df.loc[3, "living_space_m2"] = 399
    df.loc[3, "rent_gross"] = 400  # ~1 CHF/m2 -> implausible -> m2 NaN
    df.loc[4, "title"] = "Bilocale ARREDATO"
    cats = list(df["categories"])
    cats[5] = json.dumps(["PARKING_SPACE"])  # non-residential -> dropped
    cats[6] = None
    cats[7] = ["APARTMENT", "GARAGE"]  # mixed -> kept
    df["categories"] = pd.Series(cats, dtype=object)
    out = clean(df)
    keys = set(out["key"])
    assert "homegate:0" not in keys and "homegate:1" not in keys and "homegate:5" not in keys
    assert {"homegate:6", "homegate:7"} <= keys
    o = out.set_index("key")
    assert np.isnan(o.loc["homegate:2", "living_space_m2"])
    assert np.isnan(o.loc["homegate:3", "chf_per_m2"]) and np.isnan(o.loc["homegate:3", "living_space_m2"])
    assert o.loc["homegate:4", "is_furnished"]
    assert o.loc["homegate:6", "categories"] == []
    assert o["has_parking"].dtype == bool and not o["has_parking"].any()
    assert (o["utilities_pct"].dropna().between(14, 16)).all()
    assert len(clean(df, active_only=False)) == len(out) + 1


# ---------------------------------------------------------------- fair rent
def test_fair_rent_recovers_planted_deals():
    df, info = fit_fair_rent(synthetic(200))
    assert info["method"] == "ols" and info["n"] == 200
    assert info["r2"] > 0.8  # out-of-fold
    assert 0.6 < info["coefs"]["log_m2"] < 1.0
    for i in (0, 1, 2):
        assert df.loc[i, "deal_pct"] < -15
    for i in (3, 4, 5):
        assert df.loc[i, "deal_pct"] > 15
    assert (df["model_confidence"] == "high").all()
    assert not df["suspicious"].any()
    assert (df["fair_sigma_log"] > 0).all() and (df["fair_sigma_log"] < 0.3).all()
    assert (df["listing_kind"] == "regular").all()


def test_fair_rent_is_out_of_fold():
    """A planted outlier cannot pull its own fair rent: the OOF estimate ignores it."""
    d = synthetic(200)
    d.loc[7, "rent_gross"] *= 3
    df, _ = fit_fair_rent(d)
    base, _ = fit_fair_rent(synthetic(200))
    assert abs(df.loc[7, "fair_rent"] / base.loc[7, "fair_rent"] - 1) < 0.05
    assert df.loc[7, "deal_pct"] > 150


def test_comparables_exclude_self_and_duplicates():
    d = synthetic(200)
    d.loc[199] = d.loc[0]  # same flat listed twice
    d.loc[199, ["key", "portal_id"]] = ["homegate:dup", "dup"]
    df, info = fit_fair_rent(d)
    assert (df["n_comps"] > 0).any()
    for i, row in df.iterrows():
        assert row["key"] not in row["comps"]
        assert len(row["comps"]) <= 5
    assert "homegate:dup" not in df.loc[0, "comps"] and "homegate:0" not in df.loc[199, "comps"]
    assert set(df["fair_method"].dropna()) <= {"blend", "hedonic"}
    assert info["high"]["cv"]["final"]["rmse_log"] > 0


def test_listing_kind_and_training_exclusion():
    titles = pd.DataFrame({"title": ["Cercasi SUBENTRANTE 3.5 locali", "Nachmieter gesucht", "Affitto temporaneo",
                                     "short-term rental lake view", "Appartamento vista lago", None]})
    assert listing_kind(titles).tolist() == ["takeover", "takeover", "short_term", "short_term", "regular", "regular"]
    d = synthetic(200)
    d.loc[:9, "title"] = "Cercasi subentrante"
    d.loc[:9, "rent_gross"] *= 5  # absurd prices on takeover ads must not move the model
    df, info = fit_fair_rent(d)
    assert (df.loc[:9, "listing_kind"] == "takeover").all()
    assert info["n"] == 190 and info["kinds"]["takeover"] == 10
    assert df.loc[:9, "fair_rent"].notna().all()  # still priced
    assert info["r2"] > 0.8


def test_fair_rent_fallback_no_m2_and_suspicious():
    d = synthetic(200)
    d.loc[10:14, "living_space_m2"] = np.nan
    d.loc[20, ["living_space_m2", "rooms"]] = [120.0, 4.5]
    d.loc[20, "rent_gross"] = 700.0  # ~50% below fair and < 1200 -> scam
    df, info = fit_fair_rent(d)
    assert (df.loc[10:14, "model_confidence"] == "low").all()
    assert df.loc[10:14, "fair_rent"].notna().all()
    assert info["fallback"]["n"] == 200
    # the no-m2 fallback is less certain than the full model
    assert df.loc[10:14, "fair_sigma_log"].min() > df.loc[df.model_confidence == "high", "fair_sigma_log"].max()
    assert df.loc[20, "suspicious"]


def test_fair_rent_median_fallback_small_sample():
    df, info = fit_fair_rent(synthetic(20), min_listings=30)
    assert info["method"] == "median"
    assert df["fair_rent"].notna().all()
    assert (df["model_confidence"] == "low").all()


# ---------------------------------------------------------------- percentile
def test_percentile_midrank_ties():
    v = pd.Series([10.0, 20.0, 20.0, 30.0, np.nan])
    p = percentile_midrank(v, pd.Series(True, index=v.index))
    # n=4; ranks 1, 2.5, 2.5, 4 -> 0, 50, 50, 100
    assert p.tolist()[:4] == [0.0, 50.0, 50.0, 100.0]
    assert np.isnan(p[4])
    assert percentile_midrank(pd.Series([5.0, 5.0, 5.0]), pd.Series(True, index=range(3))).tolist() == [50.0] * 3
    assert percentile_midrank(pd.Series([7.0]), pd.Series([True])).tolist() == [50.0]
    # excluded rows are not in the reference set and get NaN
    p2 = percentile_midrank(pd.Series([1.0, 2.0, 100.0]), pd.Series([True, True, False]))
    assert p2.tolist()[:2] == [0.0, 100.0] and np.isnan(p2[2])


# ---------------------------------------------------------------- score
def _scored_frame():
    return pd.DataFrame({
        "key": list("abcde"),
        "rent_gross": [1000.0, 1500, 2000, 2500, 400],
        "fair_rent": [1500.0, 1500, 1500, 1500, 1500],
        "tax_delta_month": [0.0, -50, np.nan, 20, 0],
        "price_drop_pct": [0.0, 0, 5, 0, 0],
        "days_on_market": [10.0, 200, 30, np.nan, 5],
        "suspicious": [False, False, False, False, True],
        "fair_sigma_log": 0.1,
    }).assign(deal_pct=lambda d: (d.rent_gross / d.fair_rent - 1) * 100)


def test_score_monotonic_in_value_and_suspicious_last():
    out, info = score(_scored_frame(), {"value": 1.0})
    assert info["tau2"] > TAU2_FLOOR
    assert out["key"].tolist() == ["a", "b", "c", "d", "e"]
    assert out["score"].tolist()[:4] == [100.0, 66.7, 33.3, 0.0]
    assert np.isnan(out["score"].iloc[-1]) and out["suspicious"].iloc[-1]
    assert out[out.suspicious][[c for c in out if c.startswith("comp_")]].isna().all().all()


def test_score_formula_weights_and_missing():
    w = {"value": 0.55, "tax": 0.25, "price_drop": 0.10, "market_time": 0.10, "bogus": 9}
    out = score(_scored_frame(), w)[0].set_index("key")
    # effective rent: NaN tax -> 0
    assert out.loc["c", "effective_rent"] == 2000
    assert out.loc["b", "effective_deal_pct"] == pytest.approx((1450 / 1500 - 1) * 100)
    # tax signals (non-susp): a=0, b=50, c=0, d=-20 -> comps 50, 100, 50, 0
    assert out.loc[["a", "b", "c", "d"], "comp_tax"].tolist() == [50.0, 100.0, 50.0, 0.0]
    # market_time capped at 90: a=10, b=90, c=30, d=NaN -> 0, 100, 50, NaN
    assert out.loc["b", "comp_market_time"] == 100.0 and np.isnan(out.loc["d", "comp_market_time"])
    # d: market_time missing -> its weight dropped
    d = out.loc["d"]
    exp = (0.55 * d.comp_value + 0.25 * d.comp_tax + 0.10 * d.comp_price_drop) / 0.90
    assert d.score == np.floor(exp * 10 + 0.5) / 10
    a = out.loc["a"]
    exp_a = (0.55 * a.comp_value + 0.25 * a.comp_tax + 0.10 * a.comp_price_drop + 0.10 * a.comp_market_time)
    assert a.score == np.floor(exp_a * 10 + 0.5) / 10
    # value signal = shrunk deal on plain rent (not effective rent)
    assert a.comp_value == 100.0
    k = a.shrink_k
    assert out.loc["b", "deal_pct_adj"] == pytest.approx(0.0)
    assert a.deal_pct_adj == pytest.approx((np.exp(np.log(1000 / 1500) * k) - 1) * 100)


def test_zero_variance_component_is_dropped():
    f = _scored_frame().assign(price_drop_pct=0.0, days_on_market=0.0)
    out, info = score(f, {"value": 0.5, "price_drop": 0.25, "market_time": 0.25})
    assert out["comp_price_drop"].isna().all() and out["comp_market_time"].isna().all()
    assert info["components_used"] == ["value", "tax"]
    ok = ~out["suspicious"]
    assert (out.loc[ok, "score"] == np.floor(out.loc[ok, "comp_value"] * 10 + 0.5) / 10).all()


def test_shrinkage_tau2_and_verdicts():
    lr = pd.Series([-0.3, 0.0, 0.3, 0.1])
    sig = pd.Series([0.1, 0.1, 0.2, 0.2])
    assert estimate_tau2(lr, sig, pd.Series(True, index=lr.index)) == pytest.approx(np.var(lr) - np.mean(sig ** 2))
    assert estimate_tau2(lr * 0, sig, pd.Series(True, index=lr.index)) == TAU2_FLOOR
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-8)
    for x in (-2.5, -1.0, 0.3, 1.96):
        assert norm_cdf(x) == pytest.approx(0.5 * (1 + math.erf(x / math.sqrt(2))), abs=2e-7)
    assert verdict_for(-25, 0.85) == "great" and verdict_for(-25, 0.7) == "good" and verdict_for(-25) == "good"
    assert [verdict_for(p, 0.99) for p in (-20, -8, -7.9, 7.99, 8, 20, 20.01)] == \
        ["great", "good", "fair", "fair", "pricey", "pricey", "over"]
    assert verdict_for(float("nan")) is None


def test_parity_fixture():
    """Same inputs and expected values as tests/js/parity.test.mjs."""
    f = pd.DataFrame({
        "key": list("ABCDEF"),
        "rent_gross": [1500.0, 2000, 2600, 1300, 1550, 1580],
        "tax_delta_month": [-60.0, 0, 25, np.nan, -20, 0],
        "fair_rent": [2000.0, 2000, 2000, 1800, 2000, 2000],
        "fair_sigma_log": [0.11, 0.25, 0.11, 0.25, 0.4, 0.4],
        "days_on_market": [10.0, 40, 95, np.nan, 5, 20],
        "price_drop_pct": 0.0, "suspicious": False, "listing_kind": "regular",
    })
    d = deal_fields(f, 0.05).set_index("key")
    assert d.loc["A", "shrink_k"] == pytest.approx(0.8051529790660226, abs=1e-12)
    assert d.loc["A", "deal_pct_adj_eff"] == pytest.approx(-23.24073023814429, abs=1e-9)
    assert d.loc["A", "p_below_fair"] == pytest.approx(0.9985885179338168, abs=1e-12)
    assert d.loc["D", "saving_adj_month"] == pytest.approx(242.38705340034795, abs=1e-9)
    assert d.loc["E", "deal_z"] == pytest.approx(-0.6696986128890029, abs=1e-12)
    assert d["verdict"].tolist() == ["great", "fair", "over", "good", "fair", "fair"]
    s, info = score(f, {"value": 0.55, "tax": 0.25, "price_drop": 0.10, "market_time": 0.10})
    assert info["tau2"] == TAU2_FLOOR
    assert dict(zip(s["key"], s["score"])) == {"A": 91.7, "D": 67.5, "E": 58.9, "F": 41.1, "B": 31.7, "C": 11.1}
    assert s["comp_price_drop"].isna().all()


# ---------------------------------------------------------------- pipeline
@dataclass
class FakeProfile:
    canton: str
    filing: str
    taxable_income: float
    taxable_wealth: float


TAX = {5192: 10000.0, 5210: 8800.0}  # Lugano ref, Paradiso cheaper


@pytest.fixture
def fake_modules(monkeypatch):
    geo = types.ModuleType("flatprice.geo")

    def assign_municipality(df):
        df = df.copy()
        df["municipality_bfs"] = [5192 if i % 3 else (5210 if i % 2 else 9999) for i in range(len(df))]
        df["municipality"] = df["municipality_bfs"].map({5192: "Lugano", 5210: "Paradiso", 9999: "Nowhere"})
        return df

    def assign_area(df):
        return df  # synthetic frame already has area + dist_center_km

    geo.assign_municipality = assign_municipality
    geo.assign_area = assign_area

    engine = types.ModuleType("flatprice.tax.engine")
    engine.TaxProfile = FakeProfile

    def annual_tax(profile, bfs):
        return {"total": TAX[bfs], "multiplier": 0.7 if bfs == 5192 else 0.6}

    def tax_delta_month(profile, bfs, reference_bfs):
        return (TAX[bfs] - TAX[reference_bfs]) / 12

    engine.annual_tax = annual_tax
    engine.tax_delta_month = tax_delta_month

    import flatprice
    import flatprice.tax
    monkeypatch.setitem(sys.modules, "flatprice.geo", geo)
    monkeypatch.setattr(flatprice, "geo", geo, raising=False)
    monkeypatch.setitem(sys.modules, "flatprice.tax.engine", engine)
    monkeypatch.setattr(flatprice.tax, "engine", engine, raising=False)


def test_pipeline(fake_modules):
    frame = synthetic(200)
    frame.loc[199, "active"] = False
    df, info = analyze_frame(frame)
    assert len(df) == 199
    for c in ["fair_rent", "deal_pct", "suspicious", "tax_delta_month", "tax_annual", "effective_rent",
              "comp_value", "comp_tax", "comp_price_drop", "comp_market_time", "score", "municipality",
              "fair_sigma_log", "shrink_k", "deal_pct_adj_eff", "p_below_fair", "verdict", "listing_kind"]:
        assert c in df.columns
    para = df[df.municipality == "Paradiso"]
    assert (para["tax_delta_month"] == -100).all()
    assert df[df.municipality == "Nowhere"]["tax_delta_month"].isna().all()
    assert df[df.municipality == "Lugano"]["tax_delta_month"].eq(0).all()
    assert df["score"].iloc[0] >= df["score"].dropna().iloc[-1]
    assert info["counts"]["listings"] == 199 and info["counts"]["input"] == 200
    assert info["model"]["r2"] > 0.8
    assert info["scoring"]["tau2"] >= info["scoring"]["tau2_floor"]
    assert sum(info["counts"]["verdicts"].values()) == int((~df.suspicious & df.verdict.notna()).sum())
    munis = {m["municipality"]: m for m in info["stats"]["municipalities"]}
    assert munis["Paradiso"]["tax_delta_month"] == -100 and munis["Nowhere"]["tax_delta_month"] is None
    assert munis["Lugano"]["multiplier"] == 0.7
    assert {a["area"] for a in info["stats"]["areas"]} == set(AREAS)
    json.dumps(info)  # serializable


def analyze_frame(frame):
    from flatprice.analysis.pipeline import analyze
    return analyze(frame, Config())
