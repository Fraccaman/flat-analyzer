"""Geography helpers: municipality assignment (point-in-polygon), Lugano
quarters, distances to Lugano centre and to the nearest lake.

Data: ``data/geo/ti_municipalities.geojson`` (built by ``scripts/fetch_geo.py``),
features with properties ``{"bfs": int, "name": str}`` and a foreign member
``"lakes"`` with Lago di Lugano / Lago Maggiore polygons.
"""

from __future__ import annotations

import json
import math
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import STRtree, points
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GEOJSON = "data/geo/ti_municipalities.geojson"

LUGANO_BFS = 5192
LUGANO_CENTER = (46.0037, 8.9511)  # Piazza della Riforma
EARTH_RADIUS_KM = 6371.0088

# Lugano quarter centroids (from old/analyze_flats.py). "Paradiso" is its own
# municipality (BFS 5210) and is therefore not a Lugano quarter.
LUGANO_QUARTERS = {
    "Centro": (46.0037, 8.9511),
    "Molino Nuovo": (46.0155, 8.9550),
    "Besso": (46.0090, 8.9410),
    "Loreto": (45.9990, 8.9440),
    "Cassarate": (46.0060, 8.9650),
    "Viganello": (46.0120, 8.9680),
    "Pregassona": (46.0230, 8.9700),
    "Breganzona": (46.0080, 8.9250),
    "Castagnola / Brè": (46.0090, 8.9840),
}

# Localities used by portals that are not municipality names (former comuni /
# quarters merged into a current municipality). Keys are normalised.
LOCALITY_ALIASES = {
    "lugano paradiso": 5210,
    "castagnola": 5192, "viganello": 5192, "pregassona": 5192, "breganzona": 5192,
    "cassarate": 5192, "molino nuovo": 5192, "besso": 5192, "loreto": 5192,
    "davesco soragno": 5192, "davesco": 5192, "soragno": 5192, "gandria": 5192,
    "pambio noranco": 5192, "pazzallo": 5192, "carabbia": 5192, "barbengo": 5192,
    "cadro": 5192, "carona": 5192, "sonvico": 5192, "villa luganese": 5192,
    "valcolla": 5192, "bre sopra lugano": 5192, "cureggia": 5192, "figino": 5192,
    "gentilino": 5236, "montagnola": 5236, "agra": 5236, "certenago": 5236,
    "taverne": 5227, "torricella": 5227,
    "rivera": 5238, "bironico": 5238, "camignolo": 5238, "medeglia": 5238,
    "sigirino": 5238, "tesserete": 5226,
    "giubiasco": 5002, "monte carasso": 5002, "sementina": 5002,
    "gudo": 5002, "camorino": 5002, "claro": 5002, "gorduno": 5002,
    "gnosca": 5002, "moleno": 5002, "pianezzo": 5002, "preonzo": 5002,
    "sant antonio": 5002, "magadino": 5398, "vira gambarogno": 5398,
    "ponte tresa": 5239, "croglio": 5239, "monteggio": 5239,
    "sessa": 5239, "lugaggia": 5226, "solduno": 5113, "tenero": 5131,
    "capolago": 5254, "rancate": 5254, "ligornetto": 5254, "genestrerio": 5254,
    "arzo": 5254, "besazio": 5254, "meride": 5254, "tremona": 5254, "salorino": 5254,
    "morbio superiore": 5269, "sagno": 5269, "pedrinate": 5250, "seseglio": 5250,
}


# --------------------------------------------------------------------------- distance


def haversine_km(lat, lon, ref_lat, ref_lon):
    """Great-circle distance in km. Vectorised over numpy arrays / pandas Series."""
    lat1, lon1 = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    lat2, lon2 = np.radians(np.asarray(ref_lat, dtype=float)), np.radians(np.asarray(ref_lon, dtype=float))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return d if np.ndim(d) else float(d)


# --------------------------------------------------------------------------- data


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ROOT / p


def _norm(s: str) -> str:
    s = str(s)
    for ch in "\u2019\u2018\u00b4`":
        s = s.replace(ch, " ")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("'", " ").replace("-", " ").replace("/", " ")
    s = "".join(c if c.isalnum() or c == " " else " " for c in s)
    return " ".join(s.split())


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(_resolve(path), encoding="utf-8") as f:
        return json.load(f)


def load_municipalities_geojson(path: str | Path = DEFAULT_GEOJSON) -> dict:
    """The raw GeoJSON FeatureCollection (dict)."""
    return _load(str(path))


@lru_cache(maxsize=4)
def _index(path: str):
    fc = _load(path)
    geoms = [shape(f["geometry"]) for f in fc["features"]]
    bfs = np.array([int(f["properties"]["bfs"]) for f in fc["features"]])
    names = np.array([f["properties"]["name"] for f in fc["features"]], dtype=object)
    by_name = {_norm(n): int(b) for b, n in zip(bfs, names)}
    for b, n in zip(bfs, names):  # "Campo (Vallemaggia)" -> also "campo"
        if "(" in n:
            by_name.setdefault(_norm(n.split("(")[0]), int(b))
    lakes = fc.get("lakes") or []
    lake_union = unary_union([shape(l["geometry"]) for l in lakes]) if lakes else None
    return STRtree(geoms), geoms, bfs, names, by_name, dict(zip(bfs.tolist(), names.tolist())), lake_union


def _locality_bfs(locality, by_name: dict) -> int | None:
    if locality is None or (isinstance(locality, float) and math.isnan(locality)):
        return None
    n = _norm(locality)
    if not n:
        return None
    for cand in (n, n.removeprefix("lugano ").strip(), n.split(" ")[0]):
        if cand in by_name:
            return by_name[cand]
        if cand in LOCALITY_ALIASES:
            return LOCALITY_ALIASES[cand]
    if n in LOCALITY_ALIASES:
        return LOCALITY_ALIASES[n]
    return None


def municipality_shape(name: str, path: str | Path = DEFAULT_GEOJSON):
    """(bfs, name, shapely geometry) of the municipality called `name`.

    Matching is case/accent/apostrophe/hyphen-insensitive, so URL slugs such as
    "lugano" or "collina-d-oro" work. Raises KeyError if nothing matches.
    """
    _tree, geoms, bfs, names, by_name, _name_of, _ = _index(str(path))
    b = by_name.get(_norm(name))
    if b is None:
        raise KeyError(f"unknown municipality {name!r}")
    i = int(np.flatnonzero(bfs == b)[0])
    return b, str(names[i]), geoms[i]


# --------------------------------------------------------------------------- assignment


def assign_municipality(df: pd.DataFrame, path: str | Path = DEFAULT_GEOJSON) -> pd.DataFrame:
    """Add ``municipality_bfs`` (Int64) and ``municipality`` (name).

    Point-in-polygon on ``latitude``/``longitude``; rows with missing coords or
    outside every polygon fall back to matching ``locality`` against
    municipality names (case/accent-insensitive, plus a few known aliases of
    former comuni / quarters).
    """
    tree, geoms, bfs, names, by_name, name_of, _ = _index(str(path))
    out = df.copy()
    n = len(out)
    res = np.full(n, -1, dtype=np.int64)

    if n and {"latitude", "longitude"} <= set(out.columns):
        lat = pd.to_numeric(out["latitude"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(out["longitude"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        if ok.any():
            idx = np.flatnonzero(ok)
            pts = points(lon[idx], lat[idx])
            pi, gi = tree.query(pts, predicate="intersects")
            # first hit per point (borders: lowest geometry index wins, deterministic)
            seen = np.zeros(len(idx), dtype=bool)
            for p, g in zip(pi, gi):
                if not seen[p]:
                    res[idx[p]] = bfs[g]
                    seen[p] = True

    if "locality" in out.columns:
        loc = out["locality"].to_numpy(dtype=object)
        for i in np.flatnonzero(res < 0):
            b = _locality_bfs(loc[i], by_name)
            if b is not None:
                res[i] = b

    out["municipality_bfs"] = pd.array([int(b) if b >= 0 else pd.NA for b in res], dtype="Int64")
    out["municipality"] = [name_of.get(int(b)) if b >= 0 else None for b in res]
    return out


def assign_area(df: pd.DataFrame, path: str | Path = DEFAULT_GEOJSON) -> pd.DataFrame:
    """Add ``area``, ``dist_center_km`` and ``dist_lake_km``.

    area: Lugano (BFS 5192) -> nearest quarter centroid of ``LUGANO_QUARTERS``
    (or "Lugano" if coords missing); other municipalities -> municipality name.
    Calls :func:`assign_municipality` first if needed.
    """
    out = df if "municipality_bfs" in df.columns else assign_municipality(df, path)
    out = out.copy()
    n = len(out)
    lat = pd.to_numeric(out.get("latitude", pd.Series([np.nan] * n, index=out.index)), errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(out.get("longitude", pd.Series([np.nan] * n, index=out.index)), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)

    area = out["municipality"].astype(object).to_numpy(copy=True)
    is_lugano = (out["municipality_bfs"] == LUGANO_BFS).fillna(False).to_numpy(dtype=bool)
    sel = is_lugano & ok
    if sel.any():
        qnames = np.array(list(LUGANO_QUARTERS), dtype=object)
        qlat = np.array([c[0] for c in LUGANO_QUARTERS.values()])
        qlon = np.array([c[1] for c in LUGANO_QUARTERS.values()])
        d = haversine_km(lat[sel][:, None], lon[sel][:, None], qlat[None, :], qlon[None, :])
        area[sel] = qnames[d.argmin(axis=1)]
    out["area"] = area

    dist = np.full(n, np.nan)
    if ok.any():
        dist[ok] = haversine_km(lat[ok], lon[ok], *LUGANO_CENTER)
    out["dist_center_km"] = dist
    out["dist_lake_km"] = dist_lake_km(lat, lon, path)
    return out


def dist_lake_km(lat, lon, path: str | Path = DEFAULT_GEOJSON) -> np.ndarray:
    """Approximate distance (km) to the nearest shore of Lago di Lugano / Maggiore
    (0 if the point is on the water). NaN where coords are missing."""
    *_, lakes = _index(str(path))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    res = np.full(lat.shape, np.nan)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if lakes is None or not ok.any():
        return res
    # local equirectangular scaling around 46°N: degrees -> km
    kx = math.cos(math.radians(46.0)) * 111.32
    ky = 110.95
    import shapely

    from shapely.affinity import scale

    lakes_km = scale(lakes, xfact=kx, yfact=ky, origin=(0, 0))
    pts = points(lon[ok] * kx, lat[ok] * ky)
    res[ok] = shapely.distance(pts, lakes_km)
    return res
