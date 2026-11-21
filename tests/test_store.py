import csv
from contextlib import contextmanager
from pathlib import Path

import pytest

from flatprice.config import Config, ScrapeConfig
from flatprice.models import Listing, SearchSpec
from flatprice.portals.base import BlockedError
from flatprice.portals import registry
from flatprice.scrape import run_scrape
from flatprice.store import SCHEMA_VERSION, Store

OLD_CSV = Path(__file__).resolve().parent / "fixtures" / "homegate_listings_sample.csv"


def mk(pid, rent, search="lugano", **kw):
    return Listing(portal="homegate", portal_id=str(pid), url=f"https://x/{pid}",
                   search=search, rent_gross=rent, categories=["APARTMENT", "FLAT"], **kw)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sub" / "t.db")
    yield s
    s.close()


def test_schema_created(store, tmp_path):
    assert (tmp_path / "sub" / "t.db").exists()
    assert store.schema_version == SCHEMA_VERSION
    # reopening is idempotent
    Store(tmp_path / "sub" / "t.db").close()


def test_two_runs_history(store):
    r1 = store.start_run(started_at="2026-11-01T06:00:00+00:00")
    store.record(r1, [mk(1, 2000, rooms=3.5, has_balcony=True), mk(2, 1500), mk(3, 1800)],
                 seen_at="2026-11-01T06:00:00+00:00")
    store.record_search(r1, "lugano", "ok", 3)
    store.finish_run(r1, "ok", 3, [])

    r2 = store.start_run(started_at="2026-11-11T06:00:00+00:00")
    # 1 dropped its price, 2 went up, 3 disappeared, 4 is new
    store.record(r2, [mk(1, 1800, rooms=3.5, has_balcony=True), mk(2, 1600), mk(4, 900)],
                 seen_at="2026-11-11T06:00:00+00:00")
    store.record_search(r2, "lugano", "ok", 3)
    store.finish_run(r2, "ok", 3, [])

    df = store.load_frame().set_index("portal_id")
    assert len(df) == 4
    a = df.loc["1"]
    assert a["rent_gross"] == 1800
    assert a["first_rent"] == 2000
    assert a["min_rent"] == 1800
    assert a["price_drop_pct"] == pytest.approx(10.0)
    assert a["days_on_market"] == pytest.approx(10.0)
    assert a["n_observations"] == 2
    assert a["has_balcony"] is True
    assert a["categories"] == ["APARTMENT", "FLAT"]
    assert a["active"]
    assert df.loc["2", "price_drop_pct"] == 0.0  # price increase is not a drop
    assert not df.loc["3", "active"]
    assert df.loc["3", "days_on_market"] == 0.0
    assert df.loc["4", "active"] and df.loc["4", "n_observations"] == 1
    assert df.loc["4", "first_seen"].startswith("2026-11-11")


def test_active_ignores_failed_runs_and_failed_labels(store):
    r1 = store.start_run()
    store.record(r1, [mk(1, 1000, "lugano"), mk(2, 1000, "paradiso")], seen_at="2026-11-01T00:00:00")
    store.record_search(r1, "lugano", "ok", 1)
    store.record_search(r1, "paradiso", "ok", 1)
    store.finish_run(r1, "ok", 2, [])
    # partial run: paradiso blocked -> its listings stay active
    r2 = store.start_run()
    store.record(r2, [mk(1, 1000, "lugano")], seen_at="2026-11-02T00:00:00")
    store.record_search(r2, "lugano", "ok", 1)
    store.record_search(r2, "paradiso", "blocked", 0, "captcha")
    store.finish_run(r2, "partial", 1, [{"search": "paradiso", "error": "captcha"}])
    # failed run is ignored entirely
    r3 = store.start_run()
    store.finish_run(r3, "failed", 0, [])

    df = store.load_frame().set_index("portal_id")
    assert df.loc["1", "active"]
    assert df.loc["2", "active"]


def test_upsert_keeps_first_search_label(store):
    r1 = store.start_run()
    store.record(r1, [mk(1, 1000, "lugano")])
    r2 = store.start_run()
    store.record(r2, [mk(1, 1000, "paradiso", title="new")])
    df = store.load_frame()
    assert df.loc[0, "search"] == "lugano" and df.loc[0, "title"] == "new"


def test_empty_frame(store):
    df = store.load_frame()
    assert df.empty and "price_drop_pct" in df.columns


def test_import_csv(store, tmp_path):
    with open(OLD_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:5]
    small = tmp_path / "small.csv"
    with open(small, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    run_id = store.import_csv(small, seen_at="2026-10-01T00:00:00+00:00")
    run = store.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "seed" and run["n_listings"] == len({r["id"] for r in rows})

    df = store.load_frame().set_index("portal_id")
    first = rows[0]
    got = df.loc[first["id"]]
    assert got["rent_gross"] == float(first["rent_gross"])
    assert got["search"] == first["search"]
    assert got["first_seen"].startswith("2026-10-01")
    assert got["categories"] == first["categories"].split("|")
    # with no real scrape yet, the latest seed is the active snapshot
    assert df["active"].all()


def test_run_scrape_isolates_failures(tmp_path, monkeypatch):
    class Fake:
        name = "fake"

        def build_url(self, spec):
            return "x"

        def scrape(self, page, spec, cfg):
            if spec.label == "bad":
                yield mk(99, 1)
                raise BlockedError("captcha")
            if spec.label == "boom":
                raise RuntimeError("kaboom")
            yield mk(1, 1000, spec.label)
            yield mk(2, 1200, spec.label)

    monkeypatch.setitem(registry.PORTALS, "fake", Fake)

    @contextmanager
    def fake_browser(cfg):
        yield object()

    cfg = Config(
        db_path=tmp_path / "db.sqlite",
        scrape=ScrapeConfig(delay_seconds=(0, 0)),
        searches=[SearchSpec(label=l, portal="fake") for l in ["a", "bad", "boom", "b"]],
    )
    summary = run_scrape(cfg, open_browser=fake_browser)
    assert summary["status"] == "partial"
    by = {s["label"]: s for s in summary["searches"]}
    assert by["a"]["n"] == 2 and by["b"]["n"] == 0  # dedup: first label wins
    assert by["bad"]["status"] == "blocked" and by["bad"]["n"] == 1
    assert by["boom"]["status"] == "error"
    assert summary["n_listings"] == 3

    s = Store(cfg.db_path)
    df = s.load_frame().set_index("portal_id")
    assert df.loc["1", "search"] == "a"
    s.close()


def test_run_scrape_all_failed(tmp_path, monkeypatch):
    @contextmanager
    def broken_browser(cfg):
        raise RuntimeError("no display")
        yield

    cfg = Config(db_path=tmp_path / "db.sqlite", searches=[SearchSpec(label="a", locations=["x"])])
    summary = run_scrape(cfg, open_browser=broken_browser)
    assert summary["status"] == "failed" and summary["n_listings"] == 0


# --------------------------------------------------------------------------- v2: alt_ids


def test_alt_ids_roundtrip(store):
    r = store.start_run()
    ff = Listing(portal="flatfox", portal_id="86", url="https://flatfox.ch/86/", search="flatfox-lugano",
                 rent_gross=1500, alt_ids={"homegate": "4001"})
    store.record(r, [ff, mk(1, 1000)])
    df = store.load_frame().set_index("key")
    assert df.loc["flatfox:86", "alt_ids"] == {"homegate": "4001"}
    assert df.loc["homegate:1", "alt_ids"] == {}


def test_migrates_v1_database_in_place(tmp_path):
    import sqlite3

    from flatprice.store import _MIGRATIONS

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(_MIGRATIONS[1])
    conn.execute("PRAGMA user_version = 1")
    conn.execute("INSERT INTO runs (id, started_at, status) VALUES (1, '2026-11-01T00:00:00+00:00', 'ok')")
    conn.execute(
        "INSERT INTO listings (key, portal, portal_id, url, search, rent_gross, first_seen, last_seen) "
        "VALUES ('homegate:1', 'homegate', '1', 'u', 'lugano', 1000, "
        "'2026-11-01T00:00:00+00:00', '2026-11-01T00:00:00+00:00')")
    conn.execute("INSERT INTO observations VALUES (1, 'homegate:1', '2026-11-01T00:00:00+00:00', 1000, NULL)")
    conn.commit()
    conn.close()

    s = Store(path)
    assert s.schema_version == SCHEMA_VERSION == 3
    cols = {r[1] for r in s.conn.execute("PRAGMA table_info(listings)")}
    assert "alt_ids" in cols
    df = s.load_frame()
    assert len(df) == 1 and df.loc[0, "alt_ids"] == {} and df.loc[0, "rent_gross"] == 1000
    # old rows keep working with new writes
    r = s.start_run()
    s.record(r, [mk(1, 900, alt_ids={"flatfox": "5"})])
    df = s.load_frame()
    assert df.loc[0, "alt_ids"] == {"flatfox": "5"} and df.loc[0, "price_drop_pct"] == pytest.approx(10.0)
    s.close()
    Store(path).close()  # reopening is a no-op


def test_run_scrape_http_only_never_opens_browser(tmp_path, monkeypatch):
    class HttpOnly:
        name = "httponly"
        needs_browser = False

        def build_url(self, spec):
            return "x"

        def scrape(self, page, spec, cfg):
            assert page is None
            yield Listing(portal="httponly", portal_id="1", url="u", search=spec.label, rent_gross=1000)

    monkeypatch.setitem(registry.PORTALS, "httponly", HttpOnly)
    opened = []

    @contextmanager
    def spy_browser(cfg):
        opened.append(1)
        yield object()

    cfg = Config(db_path=tmp_path / "db.sqlite", searches=[SearchSpec(label="h", portal="httponly")])
    summary = run_scrape(cfg, open_browser=spy_browser)
    assert summary["status"] == "ok" and summary["n_listings"] == 1
    assert opened == []


def test_run_scrape_browser_failure_only_hits_browser_searches(tmp_path, monkeypatch):
    class HttpOnly:
        name = "httponly"
        needs_browser = False

        def build_url(self, spec):
            return "x"

        def scrape(self, page, spec, cfg):
            yield Listing(portal="httponly", portal_id=spec.label, url="u", search=spec.label, rent_gross=1000)

    monkeypatch.setitem(registry.PORTALS, "httponly", HttpOnly)
    attempts = []

    @contextmanager
    def broken_browser(cfg):
        attempts.append(1)
        raise RuntimeError("no display")
        yield

    cfg = Config(db_path=tmp_path / "db.sqlite", searches=[
        SearchSpec(label="hg1", locations=["x"]), SearchSpec(label="ff", portal="httponly"),
        SearchSpec(label="hg2", locations=["y"])])
    summary = run_scrape(cfg, open_browser=broken_browser)
    by = {s["label"]: s for s in summary["searches"]}
    assert summary["status"] == "partial" and summary["n_listings"] == 1
    assert by["ff"]["status"] == "ok"
    assert by["hg1"]["status"] == by["hg2"]["status"] == "error"
    assert attempts == [1]  # launch failure remembered, not retried per search
