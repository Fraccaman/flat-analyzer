"""Build a demo site into dist/ from old/report/flats_scored.csv (or synthetic data).

    uv run python scripts/demo_site.py [--out dist-demo] [--serve]

Days on market / price drops are synthetic. If flatprice.tax.engine is not
available yet, clearly-labelled DEMO tax tables are used.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flatprice.config import Config, load_config  # noqa: E402
from flatprice.site.build import build_site  # noqa: E402

OLD_CSV = ROOT / "old" / "report" / "flats_scored.csv"
DEMO_BFS = {"lugano": 5192, "massagno": 5196, "paradiso": 5210, "sorengo": 5225, "collina d'oro": 5236}

DEMO_TAX = {
    "federal": {
        "year": 2025, "sources": ["DEMO values - not real"],
        "income": {"single": [[0, 0], [15000, 0.0077], [33000, 0.0088], [43000, 0.0264], [57000, 0.0297],
                              [75000, 0.0594], [81000, 0.066], [107000, 0.088], [139000, 0.11]],
                   "married": [[0, 0], [29000, 0.01], [52000, 0.02], [60000, 0.03], [78000, 0.04],
                               [93000, 0.05], [106000, 0.06], [118000, 0.07], [128000, 0.08], [137000, 0.09]]},
        "flat_top": {"single": {"threshold": 755200, "rate": 0.115}, "married": {"threshold": 895900, "rate": 0.115}},
    },
    "cantons": {"TI": {
        "canton": "TI", "year": 2025, "sources": ["DEMO values - not real"],
        "cantonal_coefficient": 0.97,
        "income": {"single": [[0, 0], [12000, 0.02], [30000, 0.05], [60000, 0.08], [100000, 0.11], [200000, 0.13]],
                   "married": [[0, 0], [20000, 0.015], [50000, 0.04], [90000, 0.07], [150000, 0.10], [300000, 0.12]]},
        "wealth": {"single": [[0, 0.001], [500000, 0.002], [1000000, 0.0025]],
                   "married": [[0, 0.001], [500000, 0.002], [1000000, 0.0025]]},
        "wealth_exemption": {"single": 50000, "married": 100000}, "wealth_exemption_type": "deduction",
        "municipalities": [
            {"bfs": 5192, "name": "Lugano", "multiplier": 0.80},
            {"bfs": 5196, "name": "Massagno", "multiplier": 0.75},
            {"bfs": 5210, "name": "Paradiso", "multiplier": 0.65},
            {"bfs": 5225, "name": "Sorengo", "multiplier": 0.60},
            {"bfs": 5236, "name": "Collina d'Oro", "multiplier": 0.70},
        ],
    }},
}


def _progressive(x, br):
    if x <= 0 or not br:
        return 0.0
    tax = 0.0
    for i, (lo, rate) in enumerate(br):
        hi = br[i + 1][0] if i + 1 < len(br) else float("inf")
        tax += max(0.0, min(x, hi) - lo) * rate
    return tax


def demo_tax_delta(tables, profile) -> dict[int, float]:
    """Python mirror of web/tax.js (demo only; production uses flatprice.tax)."""
    fed_t, c = tables["federal"], tables["cantons"]["TI"]
    f, x, w = profile["filing"], profile["taxable_income"], profile["taxable_wealth"]
    top = fed_t.get("flat_top", {}).get(f)
    fed = x * top["rate"] if top and x > top["threshold"] else _progressive(x, fed_t["income"][f])
    base = _progressive(x, c["income"][f]) + _progressive(max(0, w - c["wealth_exemption"][f]), c["wealth"][f])
    tot = {m["bfs"]: fed + base * (c["cantonal_coefficient"] + m["multiplier"]) for m in c["municipalities"]}
    ref = tot[profile["reference_bfs"]]
    return {b: (t - ref) / 12 for b, t in tot.items()}


def load_frame(rng) -> pd.DataFrame:
    if OLD_CSV.is_file():
        old = pd.read_csv(OLD_CSV)
        df = pd.DataFrame({
            "key": "homegate:" + old["id"].astype(str), "portal": "homegate",
            "url": old["url"], "title": old["title"], "street": old["street"], "locality": old["locality"],
            "latitude": old["latitude"], "longitude": old["longitude"],
            "rent_gross": old["rent_gross"].fillna(old["price"]), "rent_net": old["rent_net"],
            "rooms": old["rooms"], "living_space_m2": old["living_space_m2"], "chf_per_m2": old["chf_per_m2"],
            "floor": old["floor"], "year_built": old["year_built"], "is_furnished": old["is_furnished"].fillna(False),
            "has_balcony": old["has_balcony"], "has_parking": old["has_parking"], "has_elevator": old["has_elevator"],
            "area": old["area"], "fair_rent": old["fair_price"], "deal_pct": old["deal_pct"],
            "suspicious": old["suspicious"].fillna(False).astype(bool),
        })
        df["municipality"] = df["locality"].fillna("").str.strip()
        df.loc[df["area"].eq("Paradiso"), "municipality"] = "Paradiso"
    else:  # synthetic
        n = 300
        df = pd.DataFrame({
            "key": [f"demo:{i}" for i in range(n)], "portal": "demo", "url": "https://example.com/",
            "title": [f"Demo flat {i}" for i in range(n)], "street": "Via Demo",
            "latitude": 46.0 + rng.normal(0, 0.01, n), "longitude": 8.95 + rng.normal(0, 0.01, n),
            "rooms": rng.choice([1.5, 2.5, 3.5, 4.5], n), "living_space_m2": rng.integers(30, 130, n).astype(float),
            "municipality": rng.choice(["Lugano", "Paradiso", "Massagno", "Sorengo"], n), "suspicious": False,
        })
        df["fair_rent"] = 400 + df["living_space_m2"] * 22
        df["rent_gross"] = (df["fair_rent"] * rng.normal(1, 0.12, n)).round(-1)
        df["deal_pct"] = (df["rent_gross"] / df["fair_rent"] - 1) * 100
        df["chf_per_m2"] = df["rent_gross"] / df["living_space_m2"]
        df["area"] = df["municipality"]
    df["municipality_bfs"] = df["municipality"].str.lower().map(DEMO_BFS)
    n = len(df)
    now = datetime.now(timezone.utc)
    days = rng.integers(0, 120, n)
    df["days_on_market"] = days
    df["first_seen"] = [now - timedelta(days=int(d)) for d in days]
    df["last_seen"] = now
    df["price_drop_pct"] = np.where(rng.random(n) < 0.12, rng.uniform(2, 12, n).round(1), 0.0)
    df["model_confidence"] = "high"
    df["fair_sigma_log"] = 0.15  # nominal fair-rent uncertainty for the demo
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--serve", action="store_true", help="serve on http://localhost:8000/")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg_path = ROOT / "config.yaml"
    config = load_config(cfg_path) if cfg_path.is_file() else Config()
    profile = config.tax_profile.model_dump()
    rng = np.random.default_rng(7)
    df = load_frame(rng)

    try:  # real tariffs when the tax engine + data are available
        from flatprice.tax.engine import TaxProfile, export_tables, tax_delta_month
        tables = export_tables(("TI",))
        tp = TaxProfile(canton="TI", filing=profile["filing"], taxable_income=profile["taxable_income"],
                        taxable_wealth=profile["taxable_wealth"])
        deltas = {}
        for m in tables["cantons"]["TI"]["municipalities"]:
            deltas[int(m["bfs"])] = tax_delta_month(tp, int(m["bfs"]), int(profile["reference_bfs"]))
    except Exception as e:  # noqa: BLE001
        logging.warning("using DEMO tax tables (%s)", e)
        tables, deltas = DEMO_TAX, demo_tax_delta(DEMO_TAX, profile)
    df["tax_delta_month"] = df["municipality_bfs"].map(deltas)
    df["tax_annual"] = np.nan

    from flatprice.analysis.score import score
    df, _ = score(df, config.scoring.weights)

    stats_df = df[~df["suspicious"]].groupby("municipality").agg(
        listings=("key", "size"), median_rent=("rent_gross", "median"),
        median_chf_m2=("chf_per_m2", "median"), median_deal_pct=("deal_pct", "median"),
        tax_delta_month=("tax_delta_month", "median"),
    ).reset_index().sort_values("listings", ascending=False).round(1)
    info = {"stats": {"municipalities": stats_df.to_dict(orient="records")}, "model": {"source": "old CSV demo"}}

    dist = build_site(df, info, config, args.out, tax_tables=tables)
    print(f"demo site: {dist}  ({len(df)} listings)")
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(dist))
        print("serving http://localhost:8000/")
        http.server.ThreadingHTTPServer(("127.0.0.1", 8000), handler).serve_forever()


if __name__ == "__main__":
    main()
