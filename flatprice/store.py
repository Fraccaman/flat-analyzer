"""SQLite persistence: runs, listings (latest state) and per-run observations."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from flatprice.models import Listing

SCHEMA_VERSION = 3

# Listing fields persisted as columns (order = column order in load_frame()).
LISTING_FIELDS = [
    "portal", "portal_id", "url", "search", "title", "street", "postal_code", "locality",
    "latitude", "longitude", "rent_gross", "rent_net", "rooms", "living_space_m2", "floor",
    "year_built", "is_furnished", "has_balcony", "has_parking", "has_elevator",
    "pets_allowed", "categories", "image_url", "alt_ids",
]
BOOL_FIELDS = ["is_furnished", "has_balcony", "has_parking", "has_elevator", "pets_allowed"]
# Fields that keep their first-seen value on upsert (the search label that
# first found the listing wins; identity fields never change).
IMMUTABLE_FIELDS = {"portal", "portal_id", "search"}
ACTIVE_STATUSES = ("ok", "partial")

_MIGRATIONS = {
    1: """
    CREATE TABLE runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL DEFAULT 'running',
        n_listings INTEGER,
        errors TEXT
    );
    CREATE TABLE listings (
        key TEXT PRIMARY KEY,
        portal TEXT NOT NULL,
        portal_id TEXT NOT NULL,
        url TEXT NOT NULL,
        search TEXT NOT NULL,
        title TEXT,
        street TEXT,
        postal_code TEXT,
        locality TEXT,
        latitude REAL,
        longitude REAL,
        rent_gross REAL,
        rent_net REAL,
        rooms REAL,
        living_space_m2 REAL,
        floor INTEGER,
        year_built INTEGER,
        is_furnished INTEGER NOT NULL DEFAULT 0,
        has_balcony INTEGER,
        has_parking INTEGER,
        has_elevator INTEGER,
        pets_allowed INTEGER,
        categories TEXT NOT NULL DEFAULT '[]',
        image_url TEXT,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL
    );
    CREATE TABLE observations (
        run_id INTEGER NOT NULL REFERENCES runs(id),
        listing_key TEXT NOT NULL REFERENCES listings(key),
        seen_at TEXT NOT NULL,
        rent_gross REAL,
        rent_net REAL,
        PRIMARY KEY (run_id, listing_key)
    );
    CREATE INDEX observations_by_listing ON observations(listing_key, seen_at);
    CREATE TABLE run_searches (
        run_id INTEGER NOT NULL REFERENCES runs(id),
        search_label TEXT NOT NULL,
        status TEXT NOT NULL,
        n INTEGER,
        error TEXT,
        PRIMARY KEY (run_id, search_label)
    );
    """,
    # v2: cross-portal ids of the same flat (JSON object, see Listing.alt_ids).
    2: """
    ALTER TABLE listings ADD COLUMN alt_ids TEXT NOT NULL DEFAULT '{}';
    """,
    # v3: per-observation snapshot (rooms, m2, title, result page) and a data-quality
    # flag (flatprice/quality.py), plus the number of flagged observations per run.
    # Old observations get a snapshot only for the latest sighting of each listing
    # (the listings row holds exactly those values); `page` is inferred for old
    # homegate observations from insertion order (20 results per page, approximate).
    3: """
    ALTER TABLE observations ADD COLUMN rooms REAL;
    ALTER TABLE observations ADD COLUMN living_space_m2 REAL;
    ALTER TABLE observations ADD COLUMN title TEXT;
    ALTER TABLE observations ADD COLUMN page INTEGER;
    ALTER TABLE observations ADD COLUMN flag TEXT;
    ALTER TABLE runs ADD COLUMN n_flagged INTEGER;
    CREATE TEMP TABLE _latest AS
        SELECT listing_key, MAX(run_id) AS run_id FROM observations GROUP BY listing_key;
    UPDATE observations SET
        rooms = (SELECT l.rooms FROM listings l WHERE l.key = observations.listing_key),
        living_space_m2 = (SELECT l.living_space_m2 FROM listings l WHERE l.key = observations.listing_key),
        title = (SELECT COALESCE(l.title, '') FROM listings l WHERE l.key = observations.listing_key)
    WHERE run_id = (SELECT t.run_id FROM _latest t WHERE t.listing_key = observations.listing_key);
    CREATE TEMP TABLE _pages AS
        SELECT o.rowid AS rid,
               (ROW_NUMBER() OVER (PARTITION BY o.run_id, l.search ORDER BY o.rowid) - 1) / 20 + 1 AS page
        FROM observations o
        JOIN listings l ON l.key = o.listing_key
        JOIN runs r ON r.id = o.run_id
        WHERE l.portal = 'homegate' AND r.status <> 'seed';
    UPDATE observations SET page = (SELECT p.page FROM _pages p WHERE p.rid = observations.rowid)
    WHERE rowid IN (SELECT rid FROM _pages);
    DROP TABLE _latest;
    DROP TABLE _pages;
    """,
}
# Listing fields that are also snapshotted per observation.
SNAPSHOT_FIELDS = ["rooms", "living_space_m2", "title"]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iso(ts: datetime | str | None) -> str:
    if ts is None:
        return _now()
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _to_db(field: str, value):
    if field == "categories":
        return json.dumps(list(value or []), ensure_ascii=False)
    if field == "alt_ids":
        return json.dumps(dict(value or {}), ensure_ascii=False, sort_keys=True)
    if field in BOOL_FIELDS and value is not None:
        return int(bool(value))
    return value


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    # -- schema ------------------------------------------------------------ #

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        for v in sorted(_MIGRATIONS):
            if v > version:
                try:
                    self.conn.executescript(
                        f"BEGIN;\n{_MIGRATIONS[v]}\nPRAGMA user_version = {v};\nCOMMIT;")
                except Exception:
                    if self.conn.in_transaction:
                        self.conn.rollback()
                    raise
        if version < 3 <= self.schema_version:
            has_obs = self.conn.execute("SELECT 1 FROM observations LIMIT 1").fetchone()
            if has_obs:
                # Backfill the quality flags of the existing history (default rules).
                from flatprice import quality

                quality.apply(self)

    @property
    def schema_version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    # -- runs -------------------------------------------------------------- #

    def start_run(self, started_at: datetime | str | None = None, status: str = "running") -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO runs (started_at, status) VALUES (?, ?)", (_iso(started_at), status)
            )
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, n_listings: int,
                   errors: Optional[Iterable] = None,
                   finished_at: datetime | str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET finished_at = ?, status = ?, n_listings = ?, errors = ? WHERE id = ?",
                (_iso(finished_at), status, n_listings,
                 json.dumps(list(errors or []), ensure_ascii=False, default=str), run_id),
            )

    def record_search(self, run_id: int, label: str, status: str, n: int,
                      error: Optional[str] = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO run_searches (run_id, search_label, status, n, error) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, label, status, n, error),
            )

    # -- listings ---------------------------------------------------------- #

    def record(self, run_id: int, listings: list[Listing],
               seen_at: datetime | str | None = None, *, update_listings: bool = True,
               quality_config=None) -> int:
        """Upsert listings and add one observation each for `run_id`.

        Each observation stores a snapshot (rent, rooms, m2, title, page) and is
        checked against the listing's earlier observations with the temporal
        data-quality rules (flatprice.quality). A flagged observation never
        overwrites the listing's mutable fields: the last trusted values stay.
        With `update_listings=False` existing listings only get `last_seen`
        updated (see `update_listing_fields`); new listings are always inserted.
        """
        from flatprice import quality

        cfg = quality.config_or_default(quality_config)
        trusted = {p.lower() for p in cfg.trusted_portals}
        seen = _iso(seen_at)
        cols = ["key", *LISTING_FIELDS, "first_seen", "last_seen"]
        mutable = [f for f in LISTING_FIELDS if f not in IMMUTABLE_FIELDS]
        insert = (
            f"INSERT INTO listings ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            f"ON CONFLICT(key) DO UPDATE SET "
            "first_seen = MIN(listings.first_seen, excluded.first_seen)"
            ", last_seen = MAX(listings.last_seen, excluded.last_seen)"
        )
        upsert = insert + "".join(f", {f} = excluded.{f}" for f in mutable)
        obs = (
            "INSERT OR REPLACE INTO observations (run_id, listing_key, seen_at, rent_gross, rent_net, "
            "rooms, living_space_m2, title, page, flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        with self.conn:
            for l in listings:
                flag = None
                if l.portal.lower() not in trusted:
                    history = self.conn.execute(
                        "SELECT rent_gross, rent_net, rooms, flag FROM observations "
                        "WHERE listing_key = ? AND run_id <> ? ORDER BY seen_at, run_id",
                        (l.key, run_id),
                    ).fetchall()
                    flag = quality.record_flag([dict(r) for r in history], quality.snapshot(l), cfg)
                values = [l.key, *(_to_db(f, getattr(l, f)) for f in LISTING_FIELDS), seen, seen]
                self.conn.execute(upsert if update_listings and flag is None else insert, values)
                self.conn.execute(obs, (run_id, l.key, seen, l.rent_gross, l.rent_net, l.rooms,
                                        l.living_space_m2, l.title or "", l.page, flag))
        return len(listings)

    def update_listing_fields(self, run_id: int, listings: list[Listing]) -> int:
        """Write the mutable fields of `listings` whose observation in `run_id` is
        not flagged (used after a deferred `record(update_listings=False)`)."""
        mutable = [f for f in LISTING_FIELDS if f not in IMMUTABLE_FIELDS]
        sql = f"UPDATE listings SET {', '.join(f'{f} = ?' for f in mutable)} WHERE key = ?"
        n = 0
        with self.conn:
            for l in listings:
                row = self.conn.execute(
                    "SELECT flag FROM observations WHERE run_id = ? AND listing_key = ?", (run_id, l.key)
                ).fetchone()
                if row is None or row[0] is not None:
                    continue
                self.conn.execute(sql, [*(_to_db(f, getattr(l, f)) for f in mutable), l.key])
                n += 1
        return n

    def set_run_flag_counts(self, counts: dict[int, int]) -> None:
        with self.conn:
            self.conn.execute("UPDATE runs SET n_flagged = 0")
            self.conn.executemany("UPDATE runs SET n_flagged = ? WHERE id = ?",
                                  [(int(n), int(r)) for r, n in counts.items()])

    def load_frame(self) -> pd.DataFrame:
        """One row per listing with history-derived metrics.

        Flagged observations (flatprice.quality) are ignored for current values:
        rent comes from the latest unflagged observation, rooms / m2 / title
        from the latest unflagged snapshot; `first_rent`, `min_rent`,
        `price_drop_pct` and `days_on_market` use unflagged observations only.
        Extra columns: `rent_observed` (latest raw rent, flagged or not),
        `data_flag` (flag of the latest observation), `n_flagged_obs`,
        `data_suspect` (latest observation flagged and no trusted rent or
        description to fall back on) and `data_corrected` ("history" when the
        flagged values were replaced by older trusted ones, "history_rent" when
        only the rent could be: rooms / m2 then stay unverified and the row is
        data_suspect).
        """
        listings = pd.read_sql_query("SELECT * FROM listings", self.conn)
        derived_cols = ["first_rent", "min_rent", "n_observations", "days_on_market",
                        "price_drop_pct", "active", "rent_observed", "data_flag", "n_flagged_obs",
                        "data_suspect", "data_corrected"]
        if listings.empty:
            return pd.DataFrame(columns=["key", *LISTING_FIELDS, "first_seen", "last_seen", *derived_cols])

        obs = pd.read_sql_query(
            "SELECT listing_key AS key, run_id, seen_at, rent_gross, rent_net, rooms, living_space_m2, "
            "title, flag FROM observations ORDER BY listing_key, seen_at, run_id",
            self.conn,
        )
        obs["ok"] = obs["flag"].isna()
        g = obs.groupby("key", sort=False)
        last = g.tail(1).set_index("key")
        stats = pd.DataFrame({
            "n_observations": g.size(),
            "n_flagged_obs": (~obs["ok"]).groupby(obs["key"], sort=False).sum(),
            "data_flag": last["flag"],
            "rent_observed": last["rent_gross"].fillna(last["rent_net"]),
        })
        trusted = obs[obs["ok"]]
        tg = trusted.groupby("key", sort=False)
        last_t = tg.tail(1).set_index("key")
        priced = trusted[trusted["rent_gross"].notna()].groupby("key", sort=False)["rent_gross"]
        stats["first_rent"] = priced.first()
        stats["min_rent"] = priced.min()
        stats["_t_gross"] = last_t["rent_gross"]
        stats["_t_net"] = last_t["rent_net"]
        stats["_t_first"] = tg["seen_at"].first()
        stats["_t_last"] = last_t["seen_at"]
        snap = trusted[trusted["title"].notna()].groupby("key", sort=False).tail(1).set_index("key")
        for f in SNAPSHOT_FIELDS:
            stats[f"_s_{f}"] = snap[f]
        stats["_has_snap"] = stats.index.isin(snap.index)
        stats["_has_trusted"] = stats.index.isin(last_t.index)
        df = listings.merge(stats.rename_axis("key").reset_index(), on="key", how="left")
        df["n_observations"] = df["n_observations"].fillna(0).astype(int)
        df["n_flagged_obs"] = df["n_flagged_obs"].fillna(0).astype(int)
        df["data_flag"] = df["data_flag"].astype(object).where(df["data_flag"].notna(), None)
        has_trusted = df["_has_trusted"].fillna(False).astype(bool)
        has_snap = df["_has_snap"].fillna(False).astype(bool)
        latest_flagged = df["data_flag"].notna()

        # Current values = latest trusted observation / snapshot.
        df.loc[has_trusted, "rent_gross"] = df.loc[has_trusted, "_t_gross"]
        df.loc[has_trusted, "rent_net"] = df.loc[has_trusted, "_t_net"]
        for f in SNAPSHOT_FIELDS:
            df.loc[has_snap, f] = df.loc[has_snap, f"_s_{f}"]
        # Without a trusted snapshot the listings row is trustworthy only if it was
        # written from an unflagged latest observation (pre-v3 history).
        descr_trusted = has_snap | ~latest_flagged
        df["data_suspect"] = (latest_flagged & ~(has_trusted & descr_trusted)).astype(bool)
        df["data_corrected"] = pd.Series(np.select(
            [latest_flagged & ~df["data_suspect"], latest_flagged & has_trusted],
            ["history", "history_rent"], default=None), index=df.index, dtype=object)

        # Active: seen in the latest ok/partial run that covered its search label
        # (falls back to the latest ok/partial run overall).
        active_keys = self._active_keys()
        df["active"] = df["key"].isin(active_keys)

        for f in BOOL_FIELDS:
            df[f] = df[f].map(lambda v: None if pd.isna(v) else bool(v)).astype(object)
        df["is_furnished"] = df["is_furnished"].fillna(False).astype(bool)
        df["categories"] = df["categories"].map(lambda s: json.loads(s) if s else [])
        df["alt_ids"] = df["alt_ids"].map(lambda s: json.loads(s) if s else {})

        first = pd.to_datetime(df["_t_first"].where(has_trusted, df["first_seen"]), utc=True, format="ISO8601")
        last_seen = pd.to_datetime(df["_t_last"].where(has_trusted, df["last_seen"]), utc=True, format="ISO8601")
        df["days_on_market"] = (last_seen - first).dt.total_seconds() / 86400.0

        drop = (df["first_rent"] - df["rent_gross"]) / df["first_rent"] * 100.0
        df["price_drop_pct"] = drop.where(drop > 0, 0.0).fillna(0.0).astype(float)
        return df.drop(columns=[c for c in df.columns if c.startswith("_")])

    def _active_keys(self) -> set[str]:
        statuses = ",".join("?" * len(ACTIVE_STATUSES))
        rows = self.conn.execute(
            f"""
            WITH good AS (
                SELECT id FROM runs WHERE status IN ({statuses})
            ),
            latest_per_label AS (
                SELECT rs.search_label AS label, MAX(rs.run_id) AS run_id
                FROM run_searches rs JOIN good g ON g.id = rs.run_id
                WHERE rs.status = 'ok'
                GROUP BY rs.search_label
            )
            SELECT o.listing_key FROM observations o
            JOIN listings l ON l.key = o.listing_key
            JOIN latest_per_label lp ON lp.label = l.search AND lp.run_id = o.run_id
            """,
            ACTIVE_STATUSES,
        ).fetchall()
        keys = {r[0] for r in rows}
        # Anything seen in the latest ok/partial run is active too (covers
        # listings picked up by a different label than the one stored, and
        # labels without per-search records).
        latest = self.conn.execute(
            f"SELECT MAX(id) FROM runs WHERE status IN ({statuses})", ACTIVE_STATUSES
        ).fetchone()[0]
        if latest is None:
            # No real scrape yet (only seed imports): the latest seed is the
            # best available snapshot of what is on the market.
            latest = self.conn.execute(
                "SELECT MAX(id) FROM runs WHERE status = 'seed'"
            ).fetchone()[0]
        if latest is not None:
            keys |= {r[0] for r in self.conn.execute(
                "SELECT listing_key FROM observations WHERE run_id = ?", (latest,)
            )}
        return keys

    # -- maintenance ------------------------------------------------------- #

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- seeding ----------------------------------------------------------- #

    def import_csv(self, path: str | Path, portal: str = "homegate",
                   seen_at: datetime | str | None = None) -> int:
        """Seed a run (status "seed") from the old scrape_homegate.py CSV format.

        Only monthly rentals are imported (yearly/weekly converted to monthly).
        Returns the new run id.
        """
        from flatprice.portals.homegate import FURNISHED_RE, INTERVAL_TO_MONTHLY

        def num(v):
            try:
                return float(v) if v not in (None, "") else None
            except ValueError:
                return None

        def boolean(v):
            if v in (None, ""):
                return None
            return str(v).strip().lower() in ("true", "1", "yes")

        listings: list[Listing] = []
        seen_ids: set[str] = set()
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("offer_type") not in (None, "", "RENT"):
                    continue
                interval = row.get("rent_interval") or ""
                if interval not in INTERVAL_TO_MONTHLY:
                    continue
                factor = INTERVAL_TO_MONTHLY[interval]
                pid = str(row.get("id") or "").strip()
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                cats = [c for c in (row.get("categories") or "").split("|") if c]
                gross, net = num(row.get("rent_gross")), num(row.get("rent_net"))
                floor, year = num(row.get("floor")), num(row.get("year_built"))
                title = row.get("title") or ""
                listings.append(Listing(
                    portal=portal,
                    portal_id=pid,
                    url=row.get("url") or "",
                    search=row.get("search") or "seed",
                    title=title,
                    street=row.get("street") or "",
                    postal_code=row.get("postal_code") or "",
                    locality=row.get("locality") or "",
                    latitude=num(row.get("latitude")),
                    longitude=num(row.get("longitude")),
                    rent_gross=round(gross * factor, 2) if gross is not None else None,
                    rent_net=round(net * factor, 2) if net is not None else None,
                    rooms=num(row.get("rooms")),
                    living_space_m2=num(row.get("living_space_m2")),
                    floor=int(floor) if floor is not None else None,
                    year_built=int(year) if year is not None else None,
                    is_furnished=bool(boolean(row.get("is_furnished")))
                    or "FURNISHED_FLAT" in cats or bool(FURNISHED_RE.search(title)),
                    has_balcony=boolean(row.get("has_balcony")),
                    has_parking=boolean(row.get("has_parking")),
                    has_elevator=boolean(row.get("has_elevator")),
                    pets_allowed=boolean(row.get("pets_allowed")),
                    categories=cats,
                ))
        run_id = self.start_run(started_at=seen_at, status="seed")
        self.record(run_id, listings, seen_at=seen_at)
        self.finish_run(run_id, "seed", len(listings), [], finished_at=seen_at)
        return run_id
