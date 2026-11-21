"""Build data/geo/ti_municipalities.geojson (Ticino municipality polygons, WGS84).

Sources (official, public, no API key):
  * BFS official municipality register -> list of Ticino municipalities + names:
      https://www.agvchapp.bfs.admin.ch/api/communes/snapshot?date=01-01-<year>
  * swisstopo swissBOUNDARIES3D via the geo.admin.ch REST API, one request per
    municipality (current year, WGS84):
      https://api3.geo.admin.ch/rest/services/api/MapServer/
          ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill/<bfs>?geometryFormat=geojson&sr=4326
    (since 2023 the Swiss share of the lakes is assigned to the municipalities,
    so lakeside points fall inside a municipality.)
  * Lakes (for dist_lake_km): BAFU VEC25 lakes layer, features 17 (Lago di
    Lugano) and 18 (Lago Maggiore): .../MapServer/ch.bafu.vec25-seen/<id>

Output: GeoJSON FeatureCollection; features = municipalities with properties
{"bfs": int, "name": str}; plus a foreign member "lakes": [{"id", "name",
"geometry"}] (RFC 7946 allows foreign members; feature consumers ignore it).
Geometries are simplified (SIMPLIFY_DEG) and rounded to 5 decimals (~1 m).

Usage:  uv run python scripts/fetch_geo.py [--year 2026] [--out data/geo/ti_municipalities.geojson]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.request

from shapely.geometry import mapping, shape

BFS_SNAPSHOT = "https://www.agvchapp.bfs.admin.ch/api/communes/snapshot?date=01-01-{year}"
API = "https://api3.geo.admin.ch/rest/services/api/MapServer/{layer}/{id}?geometryFormat=geojson&sr=4326"
MUNI_LAYER = "ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill"
LAKE_LAYER = "ch.bafu.vec25-seen"
LAKES = {17: "Lago di Lugano", 18: "Lago Maggiore"}
SIMPLIFY_DEG = 0.00008  # ~6-9 m


def _get(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flatprice-geo/1.0"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def ticino_municipalities(year: int) -> dict[int, str]:
    rows = list(csv.DictReader(io.StringIO(_get(BFS_SNAPSHOT.format(year=year)).decode("utf-8"))))
    ti = next(r for r in rows if r["Level"] == "1" and r["ShortName"] == "TI")
    districts = {r["HistoricalCode"] for r in rows if r["Level"] == "2" and r["Parent"] == ti["HistoricalCode"]}
    return {int(r["BfsCode"]): r["Name"] for r in rows if r["Level"] == "3" and r["Parent"] in districts}


def _clean(geom_json: dict):
    g = shape(geom_json)
    if g.has_z:
        from shapely import force_2d

        g = force_2d(g)
    return g.buffer(0).simplify(SIMPLIFY_DEG, preserve_topology=True)


def _round(geom, nd: int = 5) -> dict:
    def r(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], nd), round(c[1], nd)]
        return [r(x) for x in c]

    m = mapping(geom)
    return {"type": m["type"], "coordinates": r(m["coordinates"])}


def build(year: int) -> dict:
    names = ticino_municipalities(year)
    feats = []
    for bfs, name in sorted(names.items()):
        f = json.loads(_get(API.format(layer=MUNI_LAYER, id=bfs)))["feature"]
        p = f["properties"]
        if p.get("gde_nr") != bfs or p.get("kanton") != "TI":
            raise ValueError(f"unexpected feature for {bfs}: {p}")
        if p.get("jahr") != year:
            print(f"warning: {bfs} {name}: geometry year {p.get('jahr')} != {year}")
        feats.append(
            {"type": "Feature", "properties": {"bfs": bfs, "name": name}, "geometry": _round(_clean(f["geometry"]))}
        )
    lakes = []
    for lid, lname in LAKES.items():
        f = json.loads(_get(API.format(layer=LAKE_LAYER, id=lid)))["feature"]
        lakes.append({"id": lid, "name": lname, "geometry": _round(_clean(f["geometry"]))})
    print(f"{len(feats)} municipalities, {len(lakes)} lakes")
    return {
        "type": "FeatureCollection",
        "name": f"ti_municipalities_{year}",
        "source": "swisstopo swissBOUNDARIES3D (api3.geo.admin.ch), BFS register, BAFU VEC25 lakes",
        "features": feats,
        "lakes": lakes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--out", default="data/geo/ti_municipalities.geojson")
    a = ap.parse_args()
    fc = build(a.year)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
