import numpy as np
import pandas as pd
import pytest

from flatprice.dedup import dedupe, link_pairs, summary


def row(portal, pid, rent=2000.0, lat=46.0, lon=8.95, rooms=3.5, m2=80.0, **kw):
    base = {
        "key": f"{portal}:{pid}", "portal": portal, "portal_id": str(pid),
        "url": f"https://{portal}.example/{pid}", "search": portal, "title": "Appartamento",
        "street": "Via X 1", "latitude": lat, "longitude": lon, "rent_gross": rent,
        "rent_net": None, "rooms": rooms, "living_space_m2": m2, "floor": None, "year_built": None,
        "image_url": None, "has_balcony": None, "has_parking": None, "has_elevator": None,
        "pets_allowed": None, "alt_ids": {}, "active": True,
        "first_seen": "2026-11-10T06:00:00+00:00", "last_seen": "2026-11-20T06:00:00+00:00",
        "days_on_market": 10.0, "price_drop_pct": 0.0,
    }
    base.update(kw)
    return base


def frame(*rows):
    return pd.DataFrame(list(rows))


# --------------------------------------------------------------------------- rule (a): ids


def test_alt_id_collapses_to_homegate():
    df = frame(
        row("homegate", "400123", rent=2000, first_seen="2026-11-01T06:00:00+00:00", days_on_market=19,
            price_drop_pct=5.0),
        # far away, no rooms/m2 and a 2.5% cheaper rent: only the id links them (and
        # the rents agree within 3%, so flatfox does not override the price)
        row("flatfox", "86000001", rent=1950, lat=46.01, rooms=None, m2=None, alt_ids={"homegate": "400123"},
            first_seen="2026-11-15T06:00:00+00:00", days_on_market=25, price_drop_pct=0.0),
        row("homegate", "999", rent=3000, lat=46.02),
    )
    out = dedupe(df).set_index("key")
    assert len(out) == 2
    c = out.loc["homegate:400123"]
    assert c["n_sources"] == 2 and c["dup_rule"] == "id"
    assert [s["portal"] for s in c["sources"]] == ["homegate", "flatfox"]
    assert c["sources"][1] == {"portal": "flatfox", "key": "flatfox:86000001",
                               "url": "https://flatfox.example/86000001", "rent_gross": 1950.0, "active": True}
    assert c["first_seen"] == "2026-11-01T06:00:00+00:00"
    assert c["days_on_market"] == 25  # max over the group
    assert c["price_drop_pct"] == 5.0  # canonical row's
    assert c["rent_gross"] == 2000  # canonical price kept: the copies agree
    assert c["data_corrected"] is None
    assert out.loc["homegate:999", "n_sources"] == 1
    assert out.loc["homegate:999", "sources"][0]["key"] == "homegate:999"
    assert pd.isna(out.loc["homegate:999", "dup_rule"])


def test_alt_id_to_unknown_listing_is_ignored():
    df = frame(row("homegate", "1"), row("flatfox", "2", lat=46.01, alt_ids={"homegate": "nope"}))
    assert len(dedupe(df)) == 2


def test_more_complete_row_wins_and_fills_gaps():
    hg = row("homegate", "1", m2=None, image_url=None)
    ff = row("flatfox", "2", m2=80.0, floor=3, year_built=2010, image_url="https://img/x.jpg",
             has_balcony=True, alt_ids={"homegate": "1"})
    out = dedupe(frame(hg, ff))
    assert len(out) == 1
    c = out.iloc[0]
    assert c["key"] == "flatfox:2"  # more complete
    assert c["sources"][0]["portal"] == "flatfox"


def test_tie_prefers_homegate_and_fills_missing_fields():
    hg = row("homegate", "1", image_url=None)
    ff = row("flatfox", "2", image_url=None, floor=2, alt_ids={"homegate": "1"})
    hg["year_built"] = 1990  # same completeness: each has one field the other lacks
    out = dedupe(frame(ff, hg))  # input order must not matter
    c = out.iloc[0]
    assert c["key"] == "homegate:1"
    assert c["floor"] == 2  # filled from the flatfox copy


def test_active_row_preferred():
    hg = row("homegate", "1", active=False, floor=1, year_built=2000)
    ff = row("flatfox", "2", alt_ids={"homegate": "1"})
    out = dedupe(frame(hg, ff))
    assert out.iloc[0]["key"] == "flatfox:2" and bool(out.iloc[0]["active"])
    assert [s["active"] for s in out.iloc[0]["sources"]] == [True, False]


# --------------------------------------------------------------------------- rule (b): heuristic


def test_geo_heuristic_links_close_similar_listings():
    # ~20 m apart, same rooms, m2 +3%, rent +2%
    df = frame(row("homegate", "1", rent=2000, m2=80), row("flatfox", "2", rent=2040, m2=82.4, lat=46.00018))
    out = dedupe(df)
    assert len(out) == 1 and out.iloc[0]["dup_rule"] == "geo"
    assert out.iloc[0]["key"] == "homegate:1"


@pytest.mark.parametrize("change", [
    {"lat": 46.0005},             # ~55 m away
    {"rooms": 4.5},               # different rooms
    {"m2": 90.0},                 # m2 +12.5%
    {"m2": None},                 # one m2 missing
    {"rent": 2100.0},             # rent +5%
    {"rent": None},               # rent missing
    {"alt_ids": {"homegate": "7"}},  # says it is a different homegate listing
])
def test_geo_heuristic_rejects(change):
    df = frame(row("homegate", "1"), row("flatfox", "2", **change))
    assert len(dedupe(df)) == 2


def test_geo_both_m2_missing_is_ok():
    df = frame(row("homegate", "1", m2=None), row("flatfox", "2", m2=None))
    assert len(dedupe(df)) == 1


def test_geo_never_links_same_portal():
    df = frame(row("homegate", "1"), row("homegate", "2"))
    out = dedupe(df)
    assert len(out) == 2 and (out["n_sources"] == 1).all()


def test_link_pairs_rules():
    df = frame(row("homegate", "1"), row("flatfox", "2", alt_ids={"homegate": "1"}), row("flatfox", "3"))
    pairs = link_pairs(df)
    assert (0, 1, "id") in pairs and (0, 2, "geo") in pairs
    assert not any(p[:2] == (0, 1) and p[2] == "geo" for p in pairs)  # id pair not double-counted
    out = dedupe(df)
    assert len(out) == 1 and out.iloc[0]["n_sources"] == 3 and out.iloc[0]["dup_rule"] == "id+geo"
    # sources after the canonical one are sorted by rent
    assert [s["key"] for s in out.iloc[0]["sources"]][0] == "homegate:1"


def test_summary_and_frame_without_optional_columns():
    df = pd.DataFrame({"portal": ["homegate", "flatfox"], "portal_id": ["1", "2"],
                       "latitude": [46.0, 46.0], "longitude": [8.95, 8.95],
                       "rooms": [3.5, 3.5], "living_space_m2": [np.nan, np.nan],
                       "rent_gross": [1000.0, 1000.0]})
    out = dedupe(df)
    assert len(out) == 1 and out.iloc[0]["key"] == "homegate:1"
    s = summary(df, out)
    assert s == {"input": 2, "output": 1, "collapsed": 1, "groups": 1, "groups_by_rule": {"geo": 1}}
    assert dedupe(df.iloc[0:0]).empty


# --------------------------------------------------------------------------- pipeline


def test_pipeline_counts_duplicate_once():
    from tests.test_analysis import synthetic

    from flatprice.analysis.pipeline import analyze
    from flatprice.config import Config

    d = synthetic(120)
    dup = d.loc[[0]].copy()
    dup["portal"], dup["portal_id"], dup["key"] = "flatfox", "f0", "flatfox:f0"
    dup["alt_ids"] = [{"homegate": "0"}]
    d["alt_ids"] = [{} for _ in range(len(d))]
    d = pd.concat([d, dup], ignore_index=True)
    df, info = analyze(d, Config())
    assert len(df) == 120 and "flatfox:f0" not in set(df["key"])
    row0 = df.set_index("key").loc["homegate:0"]
    assert row0["n_sources"] == 2 and [s["portal"] for s in row0["sources"]] == ["homegate", "flatfox"]
    assert info["counts"]["duplicates_collapsed"] == 1 and info["dedup"]["groups"] == 1
    assert all("flatfox:f0" not in c for c in df["comps"])
