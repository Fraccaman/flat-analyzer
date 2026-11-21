import sqlite3
from contextlib import contextmanager

import pandas as pd
import pytest

from flatprice import quality
from flatprice.config import Config, QualityConfig, ScrapeConfig
from flatprice.dedup import dedupe
from flatprice.models import Listing, SearchSpec
from flatprice.portals import registry
from flatprice.scrape import run_scrape
from flatprice.store import _MIGRATIONS, SCHEMA_VERSION, Store

T1, T2, T3 = "2026-11-20T06:00:00+00:00", "2026-11-21T06:00:00+00:00", "2026-11-22T06:00:00+00:00"


def hg(pid, rent, rooms=3.5, m2=65.0, net="same", title=None, page=None, **kw):
    return Listing(portal="homegate", portal_id=str(pid), url=f"https://hg/{pid}", search="lugano",
                   rent_gross=rent, rent_net=rent - 200 if net == "same" else net, rooms=rooms,
                   living_space_m2=m2, title=title or f"Appartamento di {rooms} locali",
                   categories=["APARTMENT"], page=page, latitude=46.0, longitude=8.95, **kw)


def ff(pid, rent, smg, rooms=3.5, m2=65.0):
    return Listing(portal="flatfox", portal_id=str(pid), url=f"https://ff/{pid}", search="flatfox-lugano",
                   rent_gross=rent, rooms=rooms, living_space_m2=m2, title=f"Appartamento di {rooms} locali",
                   alt_ids={"homegate": str(smg)}, latitude=46.0, longitude=8.95)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "q.db")
    yield s
    s.close()


def run(store, listings, ts, status="ok", **kw):
    r = store.start_run(started_at=ts)
    store.record(r, listings, seen_at=ts, **kw)
    store.finish_run(r, status, len(listings), [], finished_at=ts)
    return r


def flags(store):
    return {(r[0], r[1]): r[2] for r in store.conn.execute(
        "SELECT run_id, listing_key, flag FROM observations")}


# ----------------------------------------------------------------- pure rules

def test_ratio_rule_and_inverse():
    assert quality.match_ratio(1750, 2390) == ("ratio_1366", 1)
    assert quality.match_ratio(2390, 1750) == ("ratio_1366", -1)
    assert quality.match_ratio(1780, 2420) == ("ratio_1366", 1)  # 1.3596, within tolerance
    assert quality.match_ratio(1750, 1800) is None
    assert quality.match_ratio(1750, 2500) is None  # x1.43
    assert quality.match_ratio(None, 2390) is None and quality.match_ratio(0, 2390) is None


def test_configurable_ratios():
    cfg = QualityConfig(poison_ratios=[1.5, 1.366])
    assert quality.match_ratio(1000, 1500, cfg) == ("ratio_1500", 1)
    assert quality.match_ratio(1750, 2390, cfg) == ("ratio_1366", 1)
    only = QualityConfig(poison_ratios=[1.5])
    assert quality.match_ratio(1750, 2390, only) is None


def test_temporal_ratio_flags_current_and_keeps_comparing_with_last_trusted():
    obs = [{"rent_gross": 1750, "rent_net": 1550, "rooms": None},
           {"rent_gross": 2390, "rent_net": None, "rooms": None},
           {"rent_gross": 1750, "rent_net": 1550, "rooms": None},  # back to normal: vs obs 0, ok
           {"rent_gross": 2390, "rent_net": 2190, "rooms": None}]
    r = quality.temporal_flags(obs)
    assert r[0] == set() and r[2] == set()
    assert r[1] == {"ratio_1366", "net_dropped"}
    assert r[3] == {"ratio_1366"}


def test_temporal_inverse_ratio_flags_the_earlier_inflated_observations():
    # first two sightings already poisoned, then the real price shows up
    obs = [{"rent_gross": 2390}, {"rent_gross": 2390}, {"rent_gross": 1750}, {"rent_gross": 1700}]
    r = quality.temporal_flags(obs)
    assert r == [{"ratio_1366"}, {"ratio_1366"}, set(), set()]


def test_rooms_jump_rule():
    base = {"rent_gross": 1750, "rent_net": None, "rooms": 3.5}
    assert quality.temporal_flags([base, {"rent_gross": 1900, "rooms": 4.5}])[1] == {"rooms_jump"}
    # rooms +1 without a rent change, or +2, is not the pattern
    assert quality.temporal_flags([base, {"rent_gross": 1750, "rooms": 4.5}])[1] == set()
    assert quality.temporal_flags([base, {"rent_gross": 1900, "rooms": 5.5}])[1] == set()
    # rooms -1 with a rent drop: the earlier observation was the inflated one
    assert quality.temporal_flags([{"rent_gross": 1900, "rooms": 4.5}, base])[0] == {"rooms_jump"}


def test_xportal_mismatch_rule():
    a = {"rent_gross": 2390, "rooms": 4.5}
    assert quality.xportal_mismatch(a, {"rent_gross": 1750, "rooms": 3.5})
    assert quality.xportal_mismatch({"rent_gross": 1750, "rooms": 4.5}, {"rent_gross": 1750, "rooms": 3.5})
    assert not quality.xportal_mismatch({"rent_gross": 1790, "rooms": 3.5}, {"rent_gross": 1750, "rooms": 3.5})
    assert not quality.xportal_mismatch({"rent_gross": 1750, "rooms": None}, {"rent_gross": 1750, "rooms": 3.5})
    # net vs net when a gross is missing
    assert quality.xportal_mismatch({"rent_gross": None, "rent_net": 2000}, {"rent_gross": None, "rent_net": 1500})


# ----------------------------------------------------------------- store level

def test_flagged_observation_does_not_overwrite_listing_fields(store):
    run(store, [hg(1, 1750, rooms=3.5, m2=65, title="Bel 3.5 locali")], T1)
    run(store, [hg(1, 2390, rooms=4.5, m2=73, net=None, title="Bel 4.5 locali")], T2)
    assert flags(store)[(2, "homegate:1")] == "net_dropped,ratio_1366,rooms_jump"
    row = store.conn.execute("SELECT rent_gross, rooms, living_space_m2, title FROM listings").fetchone()
    assert tuple(row) == (1750, 3.5, 65, "Bel 3.5 locali")
    # the observation keeps its own (poisoned) snapshot
    obs = store.conn.execute("SELECT rooms, living_space_m2, title FROM observations WHERE run_id = 2").fetchone()
    assert tuple(obs) == (4.5, 73, "Bel 4.5 locali")
    # a clean later observation updates the listing again
    run(store, [hg(1, 1700, rooms=3.5, m2=65, title="Bel 3.5 locali, ribasso")], T3)
    assert flags(store)[(3, "homegate:1")] is None
    assert store.conn.execute("SELECT rent_gross, title FROM listings").fetchone()[1] == "Bel 3.5 locali, ribasso"


def test_load_frame_uses_latest_trusted_values(store):
    run(store, [hg(1, 2000), hg(2, 1500)], T1)
    run(store, [hg(1, 1750), hg(2, 1500)], T2)
    # run 3: listing 1 poisoned (bypassing record-time protection, as before v3)
    run(store, [hg(1, 2390, rooms=4.5, m2=73, net=None), hg(2, 1500)], T3, update_listings=False)
    store.conn.execute("UPDATE listings SET rent_gross = 2390, rooms = 4.5 WHERE key = 'homegate:1'")
    store.conn.commit()
    quality.apply(store)
    df = store.load_frame().set_index("key")
    a = df.loc["homegate:1"]
    assert a["rent_gross"] == 1750 and a["rent_net"] == 1550
    assert a["rooms"] == 3.5 and a["living_space_m2"] == 65  # from the run-2 snapshot
    assert a["rent_observed"] == 2390
    assert a["data_flag"] == "net_dropped,ratio_1366,rooms_jump"
    assert a["n_flagged_obs"] == 1 and a["n_observations"] == 3
    assert a["first_rent"] == 2000 and a["min_rent"] == 1750
    assert a["price_drop_pct"] == pytest.approx(12.5)
    assert a["days_on_market"] == pytest.approx(1.0)  # last trusted sighting is run 2
    assert not a["data_suspect"] and a["data_corrected"] == "history"
    b = df.loc["homegate:2"]
    assert b["data_flag"] is None and b["data_corrected"] is None and b["days_on_market"] == pytest.approx(2.0)


def test_only_flagged_data_is_data_suspect(store):
    # new listing on a page where everything else is poisoned, no flatfox twin
    run(store, [hg(i, 1750, page=7) for i in range(5)], T1)
    poisoned = [hg(i, 2390, rooms=4.5, net=None, page=7) for i in range(5)] + [hg(99, 2460, page=7)]
    run(store, poisoned, T2)
    report = quality.apply(store)
    assert flags(store)[(2, "homegate:99")] == "page_poisoned"
    assert report["by_reason"]["page_poisoned"] == 1 and report["by_run"][2]["flagged"] == 6
    df = store.load_frame().set_index("key")
    assert df.loc["homegate:99", "data_suspect"]
    assert df.loc["homegate:0", "rent_gross"] == 1750 and not df.loc["homegate:0", "data_suspect"]
    assert store.conn.execute("SELECT n_flagged FROM runs WHERE id = 2").fetchone()[0] == 6


def test_cross_portal_rule_flags_homegate_only(store):
    run(store, [hg(4003590268, 2390, rooms=4.5, m2=73), hg(5, 1500)], T1)  # first sighting: no history
    run(store, [ff(86, 1750, 4003590268), ff(87, 1500, 5)], T2)
    report = quality.apply(store)
    f = flags(store)
    assert f[(1, "homegate:4003590268")] == "xportal_mismatch"
    assert f[(1, "homegate:5")] is None
    assert f[(2, "flatfox:86")] is None  # the trusted portal is never flagged
    assert report["by_reason"] == {"xportal_mismatch": 1}
    # outside the time window nothing is compared
    quality.apply(store, QualityConfig(xportal_max_days=0.5))
    assert flags(store)[(1, "homegate:4003590268")] is None


def test_quality_is_idempotent_and_configurable(store):
    run(store, [hg(1, 1000)], T1)
    run(store, [hg(1, 1500, net=None)], T2, quality_config=QualityConfig(poison_ratios=[1.5]))
    assert flags(store)[(2, "homegate:1")] == "net_dropped,ratio_1500"
    r1 = quality.apply(store, QualityConfig(poison_ratios=[1.5]))
    r2 = quality.apply(store, QualityConfig(poison_ratios=[1.5]))
    assert r1["n_flagged"] == r2["n_flagged"] == 1 and r2["changed"] == 0
    quality.apply(store, QualityConfig(poison_ratios=[1.366]))
    assert flags(store)[(2, "homegate:1")] == "net_dropped"


# ----------------------------------------------------------------- migration

def test_migrates_v2_to_v3_with_backfill(tmp_path):
    path = tmp_path / "v2.db"
    conn = sqlite3.connect(path)
    for v in (1, 2):
        conn.executescript(_MIGRATIONS[v])
    conn.execute("PRAGMA user_version = 2")
    for rid, ts in ((1, T1), (2, T2), (3, T3)):
        conn.execute("INSERT INTO runs (id, started_at, status) VALUES (?, ?, 'ok')", (rid, ts))
    # listing row holds the (poisoned) latest scrape, as v2 always overwrote it
    conn.execute(
        "INSERT INTO listings (key, portal, portal_id, url, search, title, rent_gross, rooms, living_space_m2, "
        "first_seen, last_seen) VALUES ('homegate:1', 'homegate', '1', 'u', 'lugano', '4.5 locali', 2390, 4.5, 73, ?, ?)",
        (T1, T2))
    conn.execute(
        "INSERT INTO listings (key, portal, portal_id, url, search, title, rent_gross, rooms, alt_ids, "
        "first_seen, last_seen) VALUES ('flatfox:9', 'flatfox', '9', 'u', 'flatfox-lugano', '3.5 locali', 1750, 3.5, "
        "'{\"homegate\": \"1\"}', ?, ?)", (T3, T3))
    conn.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?)", [
        (1, "homegate:1", T1, 1750, 1550), (2, "homegate:1", T2, 2390, None), (3, "flatfox:9", T3, 1750, None)])
    conn.commit()
    conn.close()

    s = Store(path)
    assert s.schema_version == SCHEMA_VERSION == 3
    obs_cols = {r[1] for r in s.conn.execute("PRAGMA table_info(observations)")}
    assert {"rooms", "living_space_m2", "title", "page", "flag"} <= obs_cols
    assert "n_flagged" in {r[1] for r in s.conn.execute("PRAGMA table_info(runs)")}
    rows = {r[0]: tuple(r[1:]) for r in s.conn.execute(
        "SELECT run_id, rooms, title, page, flag FROM observations WHERE listing_key = 'homegate:1'")}
    assert rows[1] == (None, None, 1, None)  # history kept, no snapshot for older sightings
    assert rows[2] == (4.5, "4.5 locali", 1, "net_dropped,ratio_1366,xportal_mismatch")  # backfilled
    assert s.conn.execute("SELECT n_flagged FROM runs WHERE id = 2").fetchone()[0] == 1
    df = s.load_frame().set_index("key")
    a = df.loc["homegate:1"]
    # rent restored from history, but rooms/m2 only known from the flagged sighting
    assert a["rent_gross"] == 1750 and a["data_suspect"] and a["data_corrected"] == "history_rent"
    # ... and the flatfox twin fixes the rest at dedup time
    d = dedupe(df.reset_index())
    assert len(d) == 1  # the pair collapses (flatfox canonical here: the only active row)
    c = d.iloc[0]
    assert c["rent_gross"] == 1750 and c["rooms"] == 3.5 and c["title"] == "3.5 locali"
    assert not c["data_suspect"] and c["data_corrected"] == "flatfox"
    assert {x["key"]: x.get("flag") for x in c["sources"]} == {
        "flatfox:9": None, "homegate:1": "net_dropped,ratio_1366,xportal_mismatch"}
    s.close()
    Store(path).close()  # reopening is a no-op


# ----------------------------------------------------------------- dedup

def test_dedup_uses_flatfox_values_on_mismatch():
    base = {"url": "u", "search": "s", "latitude": 46.0, "longitude": 8.95, "active": True,
            "first_seen": T1, "last_seen": T2, "days_on_market": 1.0, "price_drop_pct": 0.0}
    df = pd.DataFrame([
        {**base, "key": "homegate:1", "portal": "homegate", "portal_id": "1", "rent_gross": 2390.0,
         "rent_net": None, "rooms": 4.5, "living_space_m2": 73.0, "title": "4.5 locali", "alt_ids": {},
         "rent_observed": 2390.0, "data_flag": "ratio_1366", "data_suspect": True},
        {**base, "key": "flatfox:9", "portal": "flatfox", "portal_id": "9", "rent_gross": 1750.0,
         "rent_net": 1550.0, "rooms": 3.5, "living_space_m2": 65.0, "title": "3.5 locali",
         "alt_ids": {"homegate": "1"}, "rent_observed": 1750.0, "data_flag": None, "data_suspect": False},
        # agreeing pair: untouched
        {**base, "key": "homegate:2", "portal": "homegate", "portal_id": "2", "rent_gross": 1500.0,
         "rent_net": None, "rooms": 2.5, "living_space_m2": 50.0, "title": "x", "alt_ids": {},
         "rent_observed": 1500.0, "data_flag": None, "data_suspect": False, "latitude": 46.1},
        {**base, "key": "flatfox:8", "portal": "flatfox", "portal_id": "8", "rent_gross": 1490.0,
         "rent_net": None, "rooms": 2.5, "living_space_m2": 50.0, "title": "y", "alt_ids": {"homegate": "2"},
         "rent_observed": 1490.0, "data_flag": None, "data_suspect": False, "latitude": 46.1},
    ])
    out = dedupe(df).set_index("key")
    assert len(out) == 2
    # homegate lacks rent_net here, so the (more complete) flatfox row is canonical
    assert "flatfox:9" in out.index
    c = out.loc["flatfox:9"]
    assert c["data_corrected"] == "flatfox"
    # ... and when homegate is canonical (e.g. more complete), it takes flatfox's values
    df.loc[0, ["rent_net", "street"]] = [2190.0, "Via X"]
    out = dedupe(df.assign(street=df.get("street"))).set_index("key")
    c = out.loc["homegate:1"]
    assert (c["rent_gross"], c["rent_net"], c["rooms"], c["living_space_m2"], c["title"]) == (
        1750.0, 1550.0, 3.5, 65.0, "3.5 locali")
    assert c["data_corrected"] == "flatfox" and not c["data_suspect"]
    # both ads stay in sources with their own price; the homegate one carries its flag
    assert [(s["portal"], s["rent_gross"], s.get("flag")) for s in c["sources"]] == [
        ("homegate", 2390.0, "ratio_1366"), ("flatfox", 1750.0, None)]
    ok = out.loc["homegate:2"]
    assert ok["rent_gross"] == 1500.0 and ok["data_corrected"] is None
    assert all("flag" not in s for s in ok["sources"])


def test_data_suspect_rows_are_kept_out_of_training():
    from tests.test_analysis import synthetic
    from flatprice.analysis.fair_rent import fit_fair_rent

    d = synthetic(200)
    d["data_suspect"] = False
    d.loc[:9, "data_suspect"] = True
    d.loc[:9, "rent_gross"] *= 5  # poisoned prices must not move the model
    df, info = fit_fair_rent(d)
    assert info["n"] == 190
    assert df.loc[:9, "suspicious"].all() and df.loc[:9, "fair_rent"].notna().all()
    suspect = set(df.loc[:9, "key"])
    assert not any(suspect & set(c) for c in df["comps"])


# ----------------------------------------------------------------- scrape

def test_poisoned_run_keeps_observations_but_not_listing_fields(tmp_path, monkeypatch):
    rents = {"clean": 1750, "poisoned": 2390}

    class Fake:
        name = "fake"
        needs_browser = False
        mode = "clean"

        def scrape(self, page, spec, cfg):
            for i in range(4):
                l = hg(i, rents[Fake.mode], rooms=3.5 if Fake.mode == "clean" else 4.5,
                       net="same" if Fake.mode == "clean" else None)
                yield l.model_copy(update={"search": spec.label})

    monkeypatch.setitem(registry.PORTALS, "fake", Fake)
    cfg = Config(db_path=tmp_path / "db.sqlite", scrape=ScrapeConfig(delay_seconds=(0, 0)),
                 searches=[SearchSpec(label="lugano", portal="fake")])
    first = run_scrape(cfg, open_browser=lambda c: None)
    assert first["quality"]["flagged"] == 0 and first["quality"]["updated"] == 4
    Fake.mode = "poisoned"
    second = run_scrape(cfg, open_browser=lambda c: None)
    q = second["quality"]
    assert q["flagged"] == 4 and q["poisoned"] and q["updated"] == 0
    with Store(cfg.db_path) as s:
        assert {r[0] for r in s.conn.execute("SELECT rooms FROM listings")} == {3.5}
        assert s.conn.execute("SELECT COUNT(*) FROM observations WHERE flag IS NOT NULL").fetchone()[0] == 4
        assert s.conn.execute("SELECT n_flagged FROM runs WHERE id = ?", (second["run_id"],)).fetchone()[0] == 4

    from flatprice import cli

    monkeypatch.setattr("flatprice.scrape.run_scrape", lambda c: second)
    assert cli.cmd_scrape(cfg, type("A", (), {"portal": None, "search": None})()) == 2
