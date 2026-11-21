"""Flatfox.ch adapter (rentals only, plain HTTP: no browser needed).

Flatfox has a public JSON API (verified Nov 2026, Cloudflare in front but a
browser User-Agent gets through; robots.txt allows `/`):

    GET /api/v1/pin/?west=&east=&south=&north=&offer_type=RENT&max_count=400
        -> [{pk, smg_id, latitude, longitude, price_display, price_unit, ...}]
        Hard cap of 400 pins per call, so the bounding box is split
        recursively (quadtree) until every tile returns fewer than 400.
    GET /api/v1/public-listing/?pk=1&pk=2&...&expand=cover_image&limit=100
        -> {count, next, results: [listing, ...]}

A search's `locations` are municipality names/slugs. Each is mapped to its
polygon in data/geo/ti_municipalities.geojson; the polygon's bbox drives the
pin queries and only pins inside the polygon are kept. Unlike homegate's
"luogo-lugano" search this covers the whole municipality (Pregassona,
Viganello, ...).

The pin endpoint's price/room filters are not reliable (the price filter
targets a different field than `price_display`), so all SearchSpec filters
are applied client-side after normalisation.

`smg_id` is the SMG (homegate / ImmoScout24) listing id of syndicated ads;
it is kept as `Listing.alt_ids["homegate"]` so flatprice.dedup can collapse
the two copies of the same flat.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from flatprice.config import ScrapeConfig
from flatprice.models import Listing, SearchSpec
from flatprice.portals.base import BlockedError, ScrapeError

BASE = "https://flatfox.ch"
PIN_PATH = "/api/v1/pin/"
LISTING_PATH = "/api/v1/public-listing/"
MAX_PINS = 400
DETAIL_BATCH = 100
MAX_DEPTH = 10  # 4**10 tiles: never reached in practice
DEBUG_DIR = Path("data/debug")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

FURNISHED_RE = re.compile(r"arredat|ammobiliat|furnish|möbliert|moebliert|meubl", re.I)
BLOCK_RE = re.compile(r"captcha|datadome|cf-chl|challenge-platform|just a moment|access denied", re.I)

# Flatfox `price_unit` -> factor to CHF/month. Per-m2 units are skipped.
UNIT_TO_MONTHLY = {None: 1.0, "": 1.0, "monthly": 1.0, "yearly": 1 / 12, "weekly": 52 / 12}

# Apartment object types to keep (object_category APARTMENT minus rooms in a flat).
SKIP_OBJECT_TYPES = {"SINGLE_ROOM"}

# Flatfox attribute name -> Listing field.
ATTRIBUTE_FLAGS = {
    "balconygarden": "has_balcony",
    "lift": "has_elevator",
    "parkingspace": "has_parking",
    "garage": "has_parking",
    "petsallowed": "pets_allowed",
}

BBox = tuple[float, float, float, float]  # west, south, east, north


# --------------------------------------------------------------------------- #
# Pure helpers (testable offline)
# --------------------------------------------------------------------------- #

def _num(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v) -> Optional[int]:
    f = _num(v)
    return int(f) if f is not None else None


def _abs_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    return u if u.startswith("http") else BASE + (u if u.startswith("/") else "/" + u)


def is_rent_apartment(item: dict) -> bool:
    """Monthly-priceable apartment rentals only (no rooms, houses, offices, parking)."""
    if item.get("offer_type") not in (None, "RENT"):
        return False
    if item.get("object_category") not in (None, "APARTMENT"):
        return False
    if item.get("object_type") in SKIP_OBJECT_TYPES:
        return False
    return item.get("price_unit") in UNIT_TO_MONTHLY


def normalize(item: dict, search_label: str) -> Listing:
    """Convert one public-listing item into a canonical Listing (CHF/month)."""
    pk = item.get("pk")
    if pk is None:
        raise ValueError("flatfox item without pk")
    unit = item.get("price_unit")
    if unit not in UNIT_TO_MONTHLY:
        raise ValueError(f"flatfox item {pk}: unsupported price_unit {unit!r}")
    factor = UNIT_TO_MONTHLY[unit]

    def monthly(v):
        f = _num(v)
        return round(f * factor, 2) if f is not None else None

    net, charges = _num(item.get("rent_net")), _num(item.get("rent_charges"))
    gross = _num(item.get("rent_gross"))
    if gross is None and net is not None and charges is not None:
        gross = net + charges

    object_type = item.get("object_type") or ""
    title = (item.get("description_title") or item.get("public_title")
             or item.get("short_title") or "").strip()
    attrs = {a.get("name") for a in (item.get("attributes") or []) if isinstance(a, dict)}
    flags: dict[str, Optional[bool]] = {f: None for f in set(ATTRIBUTE_FLAGS.values())}
    for name, field in ATTRIBUTE_FLAGS.items():
        if name in attrs:
            flags[field] = True

    space = _num(item.get("surface_living")) or _num(item.get("livingspace"))
    smg = str(item.get("smg_id") or "").strip()
    cover = item.get("cover_image") if isinstance(item.get("cover_image"), dict) else {}
    zipcode = item.get("zipcode")

    return Listing(
        portal="flatfox",
        portal_id=str(pk),
        url=_abs_url(item.get("url")) or f"{BASE}/{pk}/",
        search=search_label,
        title=title,
        street=str(item.get("street") or ""),
        postal_code="" if zipcode is None else str(zipcode),
        locality=str(item.get("city") or ""),
        latitude=_num(item.get("latitude")),
        longitude=_num(item.get("longitude")),
        rent_gross=monthly(gross),
        rent_net=monthly(net),
        rooms=_num(item.get("number_of_rooms")),
        living_space_m2=space if space else None,
        floor=_int(item.get("floor")),
        year_built=_int(item.get("year_built")),
        is_furnished=bool(item.get("is_furnished")) or object_type == "FURNISHED_FLAT"
        or bool(FURNISHED_RE.search(title)),
        categories=[object_type] if object_type else [],
        image_url=_abs_url(cover.get("url")),
        alt_ids={"homegate": smg} if smg else {},
        raw=item,
        **flags,
    )


def passes_filters(listing: Listing, spec: SearchSpec) -> bool:
    """Client-side SearchSpec filters. A set filter excludes listings lacking the value."""
    rent = listing.rent_gross if listing.rent_gross is not None else listing.rent_net
    checks = [
        (listing.rooms, spec.rooms_min, spec.rooms_max),
        (rent, spec.rent_min, spec.rent_max),
        (listing.living_space_m2, spec.size_min, spec.size_max),
    ]
    for value, lo, hi in checks:
        if lo is None and hi is None:
            continue
        if value is None:
            return False
        if lo is not None and value < lo:
            return False
        if hi is not None and value > hi:
            return False
    return True


def split_bbox(bbox: BBox) -> list[BBox]:
    """Four quadrants of `bbox` (west, south, east, north)."""
    w, s, e, n = bbox
    mx, my = (w + e) / 2, (s + n) / 2
    return [(w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)]


def fetch_pins(get_pins: Callable[[BBox], list], bbox: BBox, geom=None,
               max_pins: int = MAX_PINS, max_depth: int = MAX_DEPTH, _depth: int = 0) -> dict:
    """All pins in `bbox` as {pk: pin}, splitting tiles that hit the API cap.

    `get_pins(bbox)` performs one pin query. Tiles that do not intersect
    `geom` (a shapely geometry, optional) are not queried at all.
    """
    if geom is not None:
        from shapely.geometry import box

        if not geom.intersects(box(*bbox)):
            return {}
    pins = get_pins(bbox)
    if len(pins) >= max_pins and _depth < max_depth:
        out: dict = {}
        for quad in split_bbox(bbox):
            out.update(fetch_pins(get_pins, quad, geom, max_pins, max_depth, _depth + 1))
        return out
    if len(pins) >= max_pins:
        print(f"[flatfox] tile {bbox} still capped at depth {_depth}", file=sys.stderr)
    return {p["pk"]: p for p in pins if isinstance(p, dict) and p.get("pk") is not None}


def pins_in_polygon(pins: Iterable[dict], geom) -> list[dict]:
    """Pins whose coordinates fall inside (or on the border of) `geom`."""
    from shapely import points

    pins = [p for p in pins if _num(p.get("latitude")) is not None and _num(p.get("longitude")) is not None]
    if not pins:
        return []
    pts = points([float(p["longitude"]) for p in pins], [float(p["latitude"]) for p in pins])
    import shapely

    inside = shapely.intersects(geom, pts)
    return [p for p, ok in zip(pins, inside) if ok]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def _proxy_url() -> Optional[str]:
    url = os.environ.get("PROXY_URL")
    if not url:
        return None
    return url if "://" in url else "http://" + url


def save_debug_body(body: str, name: str, directory: Path | str = DEBUG_DIR) -> Optional[Path]:
    """Write a blocked response body to data/debug/ (never raises)."""
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}.html"
        path.write_text(body, encoding="utf-8")
        print(f"[debug] saved {path}", file=sys.stderr)
        return path
    except Exception as exc:  # noqa: BLE001
        print(f"[debug] save failed: {exc}", file=sys.stderr)
        return None


class HttpClient:
    """Polite JSON client: browser UA, timeouts, retry/backoff, block detection."""

    def __init__(self, delay: tuple[float, float] = (0.5, 1.5), retries: int = 4,
                 backoff_s: float = 5.0, timeout_s: float = 30.0, label: str = "flatfox"):
        import httpx

        self.delay = delay
        self.retries = retries
        self.backoff_s = backoff_s
        self.label = label
        self._last = 0.0
        self.n_requests = 0
        self.client = httpx.Client(
            base_url=BASE,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json",
                     "Accept-Language": "it-CH,it;q=0.9,en;q=0.8"},
            timeout=timeout_s,
            proxy=_proxy_url(),
            follow_redirects=True,
        )

    def _pause(self) -> None:
        wait = random.uniform(*self.delay) - (time.monotonic() - self._last)
        if wait > 0 and self._last:
            time.sleep(wait)
        self._last = time.monotonic()

    def get_json(self, path: str, params) -> object:
        import httpx

        last = ""
        for attempt in range(self.retries + 1):
            self._pause()
            self.n_requests += 1
            try:
                r = self.client.get(path, params=params)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                ctype = r.headers.get("content-type", "")
                if r.status_code == 403 or ("html" in ctype and BLOCK_RE.search(r.text[:20000])):
                    save_debug_body(r.text, f"{self.label}-blocked")
                    raise BlockedError(f"flatfox blocked (HTTP {r.status_code}) at {r.url}")
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}"
                    retry_after = _num(r.headers.get("retry-after"))
                    if retry_after and attempt < self.retries:
                        time.sleep(min(retry_after, 120))
                        continue
                elif r.status_code != 200:
                    raise ScrapeError(f"flatfox HTTP {r.status_code} at {r.url}")
                else:
                    try:
                        return r.json()
                    except ValueError as exc:
                        save_debug_body(r.text, f"{self.label}-badjson")
                        raise ScrapeError(f"flatfox returned non-JSON at {r.url}") from exc
            if attempt < self.retries:
                pause = self.backoff_s * 2 ** attempt
                print(f"[{self.label}] {last}, retry {attempt + 1}/{self.retries} in {pause:.0f}s",
                      file=sys.stderr)
                time.sleep(pause)
        raise ScrapeError(f"flatfox request failed after {self.retries + 1} attempts: {last}")

    def close(self) -> None:
        self.client.close()


# --------------------------------------------------------------------------- #
# Portal
# --------------------------------------------------------------------------- #

class Flatfox:
    name = "flatfox"
    needs_browser = False

    def __init__(self, client_factory: Optional[Callable[[], object]] = None,
                 max_pins: int = MAX_PINS):
        self.client_factory = client_factory or HttpClient
        self.max_pins = max_pins

    @staticmethod
    def pin_params(bbox: BBox, max_pins: int = MAX_PINS) -> dict:
        w, s, e, n = bbox
        return {"west": f"{w:.6f}", "east": f"{e:.6f}", "south": f"{s:.6f}", "north": f"{n:.6f}",
                "offer_type": "RENT", "object_category": "APARTMENT", "max_count": str(max_pins)}

    def regions(self, spec: SearchSpec) -> list[tuple[str, object]]:
        from flatprice.geo import municipality_shape

        if not spec.locations:
            raise ScrapeError(f"search {spec.label!r}: flatfox needs `locations` (municipalities)")
        out = []
        for loc in spec.locations:
            try:
                _bfs, name, geom = municipality_shape(loc)
            except KeyError as exc:
                raise ScrapeError(f"search {spec.label!r}: {exc}") from None
            out.append((name, geom))
        return out

    def build_url(self, spec: SearchSpec) -> str:
        from urllib.parse import urlencode

        _name, geom = self.regions(spec)[0]
        w, s, e, n = geom.bounds
        return f"{BASE}{PIN_PATH}?{urlencode(self.pin_params((w, s, e, n), self.max_pins))}"

    def collect_pins(self, client, spec: SearchSpec) -> list[dict]:
        """Pins inside the spec's municipality polygons (deduplicated by pk)."""
        def get_pins(bbox: BBox) -> list:
            data = client.get_json(PIN_PATH, self.pin_params(bbox, self.max_pins))
            if not isinstance(data, list):
                raise ScrapeError(f"unexpected pin response: {type(data).__name__}")
            return data

        kept: dict = {}
        for name, geom in self.regions(spec):
            pins = fetch_pins(get_pins, geom.bounds, geom, self.max_pins)
            inside = pins_in_polygon(pins.values(), geom)
            print(f"[flatfox:{spec.label}] {name}: {len(pins)} pins in bbox, {len(inside)} inside",
                  file=sys.stderr)
            for p in inside:
                kept.setdefault(p["pk"], p)
        return list(kept.values())

    def fetch_details(self, client, pks: list) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(pks), DETAIL_BATCH):
            chunk = pks[i:i + DETAIL_BATCH]
            params = [("pk", str(pk)) for pk in chunk] + [("expand", "cover_image"),
                                                          ("limit", str(DETAIL_BATCH))]
            data = client.get_json(LISTING_PATH, params)
            results = data.get("results") if isinstance(data, dict) else data
            if not isinstance(results, list):
                raise ScrapeError("unexpected public-listing response")
            out.extend(r for r in results if isinstance(r, dict))
        return out

    def scrape(self, page, spec: SearchSpec, cfg: ScrapeConfig) -> Iterator[Listing]:
        client = self.client_factory()
        try:
            pins = self.collect_pins(client, spec)
            items = self.fetch_details(client, [p["pk"] for p in pins])
            kept = skipped = 0
            for item in items:
                if not is_rent_apartment(item):
                    skipped += 1
                    continue
                try:
                    listing = normalize(item, spec.label)
                except Exception as exc:  # noqa: BLE001
                    print(f"[flatfox:{spec.label}] skip malformed item: {exc}", file=sys.stderr)
                    continue
                if not passes_filters(listing, spec):
                    skipped += 1
                    continue
                kept += 1
                yield listing
            print(f"[flatfox:{spec.label}] {len(pins)} pins, {len(items)} details, {kept} kept, "
                  f"{skipped} skipped (type/unit/filters)", file=sys.stderr)
        finally:
            close = getattr(client, "close", None)
            if close:
                close()
