import json
from pathlib import Path

import pytest

from flatprice.tax import engine
from flatprice.tax.engine import TaxProfile, annual_tax, progressive, tax_delta_month

GEOJSON = Path(__file__).resolve().parent.parent / "data/geo/ti_municipalities.geojson"


def test_progressive_basic():
    br = [[0, 0.0], [100, 0.1], [200, 0.2]]
    assert progressive(0, br) == 0
    assert progressive(-5, br) == 0
    assert progressive(100, br) == 0
    assert progressive(150, br) == pytest.approx(5)
    assert progressive(200, br) == pytest.approx(10)
    assert progressive(300, br) == pytest.approx(30)


def test_progressive_base_override():
    # 3rd element = official tax at the threshold, replaces the running sum
    br = [[0, 0.0], [100, 0.01], [200, 0.1, 50]]
    assert progressive(199, br) == pytest.approx(0.99)
    assert progressive(200, br) == pytest.approx(50)
    assert progressive(210, br) == pytest.approx(51)


def test_schema():
    fed = engine.load_federal()
    ti = engine.load_canton("TI")
    assert fed["year"] == ti["year"] >= 2025
    for f in ("single", "married"):
        for br in (fed["income"][f], ti["income"][f], ti["wealth"][f]):
            assert br[0][0] == 0
            assert [b[0] for b in br] == sorted(b[0] for b in br)
        assert f in fed["flat_top"] and f in ti["wealth_exemption"]
    bfs = [m["bfs"] for m in ti["municipalities"]]
    assert len(bfs) == len(set(bfs)) >= 100
    assert 5192 in bfs and 5210 in bfs


# Reference values from the official ESTV calculator (swisstaxcalculator.estv.admin.ch,
# API_calculateSimpleTaxes, tax year 2026, taxable income/wealth, no church tax).
# (filing, bfs, income, wealth, federal, cantonal_income, communal_income, cantonal_wealth, communal_wealth)
ESTV_2026 = [
    ("single", 5192, 120_000, 500_000, 4249, 11321, 8717, 910, 701),
    ("married", 5192, 120_000, 500_000, 2929, 9050, 6969, 910, 701),
    ("single", 5210, 120_000, 500_000, 4249, 11321, 6566, 910, 528),
    ("married", 5210, 120_000, 500_000, 2929, 9050, 5249, 910, 528),
    ("single", 5192, 1_000_000, 150_000, 115000, 133590, 102864, 0, 0),
    ("single", 5192, 500_000, 199_000, 52503, 63590, 48964, 0, 0),
    ("married", 5192, 1_000_000, 3_000_000, 115000, 134493, 103559, 7500, 5775),
    ("married", 5192, 50_000, 100_000, 203, 1364, 1050, 0, 0),
    ("single", 5192, 60_000, 0, 671, 3980, 3065, 0, 0),
    ("single", 5192, 55_800, 0, 554, 3575, None, 0, 0),
    ("single", 5192, 66_000, 0, 850, 4688, None, 0, 0),
    ("married", 5192, 500_000, 0, 50880, 62794, None, 0, 0),
    ("married", 5192, 80_000, 200_000, 965, 4157, 3201, 200, 154),
    ("single", 5002, 200_000, 1_000_000, 12903, 21785, 20260, 2310, 2148),
    ("married", 5002, 150_000, 400_000, 5408, 12754, 11862, 660, 614),
]


@pytest.mark.parametrize("filing,bfs,inc,wealth,fed,ci,cc,cw,ccw", ESTV_2026)
def test_matches_estv_calculator(filing, bfs, inc, wealth, fed, ci, cc, cw, ccw):
    if engine.load_canton("TI")["year"] != 2026:
        pytest.skip("reference values are for tax year 2026")
    t = annual_tax(TaxProfile("TI", filing, inc, wealth), bfs)
    # ESTV rounds each component to the franc
    assert t["federal"] == pytest.approx(fed, abs=1.01)
    assert t["cantonal_income"] == pytest.approx(ci, abs=1.01)
    if cc is not None:
        assert t["communal_income"] == pytest.approx(cc, abs=1.01)
    assert t["cantonal_wealth"] == pytest.approx(cw, abs=1.01)
    assert t["communal_wealth"] == pytest.approx(ccw, abs=1.01)


def test_annual_tax_keys_and_total():
    t = annual_tax(TaxProfile("TI", "single", 120_000, 200_000), 5192)
    for k in ("federal", "cantonal_income", "communal_income", "cantonal_wealth", "communal_wealth", "total", "multiplier"):
        assert k in t
    parts = sum(t[k] for k in ("federal", "cantonal_income", "communal_income", "cantonal_wealth", "communal_wealth"))
    assert t["total"] == pytest.approx(parts + t.get("personal", 0))
    assert t["multiplier"] == pytest.approx(0.77)


def test_paradiso_cheaper_than_lugano():
    p = TaxProfile("TI", "single", 120_000, 200_000)
    assert annual_tax(p, 5210)["total"] < annual_tax(p, 5192)["total"]
    d = tax_delta_month(p, 5210, 5192)
    assert d < 0
    # 19 points of multiplier on ~11.5k base tax -> ~180 CHF/month
    assert -260 < d < -120
    assert tax_delta_month(p, 5192, 5192) == 0


@pytest.mark.parametrize("inc", [30_000, 60_000, 120_000, 250_000, 500_000])
def test_married_not_more_than_single(inc):
    s = annual_tax(TaxProfile("TI", "single", inc, 0), 5192)
    m = annual_tax(TaxProfile("TI", "married", inc, 0), 5192)
    assert m["federal"] <= s["federal"] + 1e-9
    assert m["cantonal_income"] <= s["cantonal_income"] + 1e-9
    assert m["total"] <= s["total"]


@pytest.mark.parametrize("inc", [800_000, 2_000_000])
def test_married_vs_single_high_income(inc):
    # Official TI quirk: above ~760k the married cantonal tariff is slightly
    # higher than the single one (ESTV: 1M -> 134'493 vs 133'590). Within 1%.
    s = annual_tax(TaxProfile("TI", "single", inc, 0), 5192)
    m = annual_tax(TaxProfile("TI", "married", inc, 0), 5192)
    assert m["federal"] <= s["federal"] + 1e-9
    assert m["total"] <= s["total"] * 1.01


def test_sanity_range():
    t = annual_tax(TaxProfile("TI", "single", 120_000, 200_000), 5192)
    assert 20_000 < t["total"] < 30_000


def test_unknown_bfs():
    with pytest.raises(KeyError):
        annual_tax(TaxProfile("TI", "single", 100_000, 0), 99999)


def test_bad_filing():
    with pytest.raises(ValueError):
        TaxProfile("TI", "widowed", 1, 1)


def test_export_tables():
    t = engine.export_tables()
    assert set(t) == {"federal", "cantons"}
    assert t["cantons"]["TI"]["canton"] == "TI"
    json.dumps(t)  # serialisable


def test_every_geojson_municipality_has_multiplier():
    fc = json.loads(GEOJSON.read_text(encoding="utf-8"))
    mult = {m["bfs"] for m in engine.municipalities("TI")}
    geo = {f["properties"]["bfs"] for f in fc["features"]}
    assert geo - mult == set()
    assert mult - geo == set()
