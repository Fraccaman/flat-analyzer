import json
import math

import numpy as np
import pandas as pd

from flatprice.config import Config
from flatprice.site.build import build_site


def _df():
    return pd.DataFrame({
        "portal": ["homegate", "homegate", "flatfox"],
        "portal_id": ["1", "2", "x"],
        "url": ["https://a/1", "https://a/2", "https://b/x"],
        "title": ["Nice flat", "Other", None],
        "latitude": [46.0012345678, 46.01, np.nan],
        "longitude": [8.95, 8.96, np.nan],
        "rent_gross": [1500.0, 2000.123456, np.nan],
        "living_space_m2": [50.0, 80.0, 0.0],
        "rooms": [2.5, 3.5, np.nan],
        "floor": [1.0, np.nan, 3.0],
        "is_furnished": [False, True, False],
        "suspicious": [False, False, True],
        "municipality_bfs": [5192.0, np.nan, 5210.0],
        "first_seen": pd.to_datetime(["2026-11-01", "2026-11-10", None]),
        "deal_pct": [-5.0, np.inf, np.nan],
        "score": [80.25, 40.0, np.nan],
        "unrelated_column": [1, 2, 3],
        "fair_sigma_log": [0.1112345678, 0.25, np.nan],
        "p_below_fair": [0.987654321, 0.5, np.nan],
        "n_comps": [8.0, 0.0, 0.0],
        "comps": [["homegate:2", "flatfox:x"], [], []],
        "listing_kind": ["regular", "takeover", "regular"],
    })


def test_build_site(tmp_path):
    geo = tmp_path / "geo.geojson"
    geo.write_text('{"type":"FeatureCollection","features":[]}')
    info = {"stats": {"municipalities": [{"municipality": "Lugano", "median_rent": np.float64(1800.0), "x": np.nan}]},
            "model": {"r2": 0.71234}, "generated_at": "2026-11-20T06:00:00+00:00",
            "scoring": {"tau2": 0.0470778, "tau2_floor": 0.0025}}
    tax = {"federal": {"income": {"single": [[0, 0.0]]}}, "cantons": {"TI": {"municipalities": []}}}
    dist = build_site(_df(), info, Config(), tmp_path / "dist", tax_tables=tax, geojson_path=geo)

    raw = (dist / "data.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    data = json.loads(raw)
    assert data["generated_at"] == "2026-11-20T06:00:00+00:00"
    assert data["defaults"]["tax_profile"]["reference_bfs"] == 5192
    assert math.isclose(sum(data["defaults"]["weights"].values()), 1.0)
    assert data["tax"] == tax
    assert data["stats"]["municipalities"][0]["x"] is None
    assert data["stats"]["model_info"]["r2"] == 0.712

    a, b, c = data["listings"]
    assert a["key"] == "homegate:1" and c["key"] == "flatfox:x"
    assert "unrelated_column" not in a
    assert a["latitude"] == 46.001235  # coords keep 6 decimals
    assert b["rent_gross"] == 2000.12
    assert a["chf_per_m2"] == 30  # derived when missing
    assert c["chf_per_m2"] is None  # 0 m2 -> null
    assert c["latitude"] is None and c["rent_gross"] is None and c["title"] is None
    assert b["deal_pct"] is None  # inf -> null
    assert a["floor"] == 1 and isinstance(a["floor"], int) and b["floor"] is None
    assert a["municipality_bfs"] == 5192 and b["municipality_bfs"] is None
    assert a["is_furnished"] is False and c["suspicious"] is True
    assert a["first_seen"].startswith("2026-11-01") and c["first_seen"] is None
    assert a["fair_sigma_log"] == 0.111235 and a["p_below_fair"] == 0.9877 and c["fair_sigma_log"] is None
    assert a["n_comps"] == 8 and a["comps"] == ["homegate:2", "flatfox:x"] and b["listing_kind"] == "takeover"
    assert data["scoring"]["tau2"] == 0.047078

    for f in ["index.html", "app.js", "tax.js", "score.js", "deal.js", "style.css", ".nojekyll", "municipalities.geojson"]:
        assert (dist / f).is_file(), f


def test_build_site_minimal(tmp_path, monkeypatch):
    import flatprice.site.build as b
    monkeypatch.setattr(b, "GEOJSON_SRC", tmp_path / "missing.geojson")
    dist = build_site(pd.DataFrame({"rent_gross": [1000]}), None, None, tmp_path / "d")
    data = json.loads((dist / "data.json").read_text())
    assert data["listings"] == [{"key": "0", "rent_gross": 1000}]
    assert isinstance(data["tax"], dict) and data["stats"] == {}
    assert not (dist / "municipalities.geojson").exists()
