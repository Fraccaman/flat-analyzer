import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon, box

from flatprice.config import ScrapeConfig
from flatprice.models import SearchSpec
from flatprice.portals.base import BlockedError, ScrapeError
from flatprice.portals.flatfox import (
    Flatfox,
    HttpClient,
    fetch_pins,
    is_rent_apartment,
    normalize,
    passes_filters,
    pins_in_polygon,
    split_bbox,
)
from flatprice.portals.registry import get_portal

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def items():
    return json.loads((FIXTURES / "flatfox_listing.json").read_text(encoding="utf-8"))["results"]


@pytest.fixture
def pins():
    return json.loads((FIXTURES / "flatfox_pins.json").read_text(encoding="utf-8"))


def by_type(items, t):
    return next(i for i in items if i["object_type"] == t)


# --------------------------------------------------------------------------- normalize


def test_normalize_syndicated_apartment(items):
    item = next(i for i in items if i["object_type"] == "APARTMENT" and i["smg_id"])
    l = normalize(item, "flatfox-lugano")
    assert l.portal == "flatfox" and l.portal_id == str(item["pk"])
    assert l.key == f"flatfox:{item['pk']}"
    assert l.url == "https://flatfox.ch" + item["url"]
    assert l.search == "flatfox-lugano"
    assert l.rent_gross == item["rent_gross"] and l.rent_net == item["rent_net"]
    assert l.rooms == float(item["number_of_rooms"]) and isinstance(l.rooms, float)
    assert l.living_space_m2 == item["surface_living"]
    assert l.latitude == item["latitude"] and l.longitude == item["longitude"]
    assert l.postal_code == str(item["zipcode"]) and l.locality == item["city"]
    assert l.alt_ids == {"homegate": item["smg_id"]}
    assert l.categories == ["APARTMENT"]
    assert l.has_balcony and l.has_elevator and l.has_parking
    assert l.pets_allowed is None  # attribute absent -> unknown, not False
    assert l.title == item["description_title"]
    assert l.image_url.startswith("https://")  # syndicated ads use absolute CDN urls
    assert not l.is_furnished
    assert l.raw == item


def test_normalize_relative_image_url(items):
    item = {**items[0], "cover_image": {"url": "/media/ff/x.jpg"}}
    assert normalize(item, "x").image_url == "https://flatfox.ch/media/ff/x.jpg"


def test_normalize_furnished_flat(items):
    l = normalize(by_type(items, "FURNISHED_FLAT"), "x")
    assert l.is_furnished and l.categories == ["FURNISHED_FLAT"]


def test_normalize_gross_from_net_plus_charges(items):
    item = next(i for i in items if i["rent_gross"] is None and i["rent_net"])
    l = normalize(item, "x")
    assert l.rent_gross == item["rent_net"] + item["rent_charges"]
    assert l.alt_ids == {}  # no smg_id
    assert l.pets_allowed and l.has_parking  # from petsallowed / garage+parkingspace


def test_normalize_handcrafted_units_and_flags():
    base = {"pk": 1, "url": "/en/flat/x/1/", "offer_type": "RENT", "object_category": "APARTMENT",
            "object_type": "APARTMENT", "number_of_rooms": "3.5", "price_unit": "yearly",
            "rent_net": 24000, "rent_charges": None, "rent_gross": None,
            "description_title": "Bel appartamento ammobiliato", "attributes": [],
            "surface_living": None, "livingspace": 80, "smg_id": "", "floor": None}
    l = normalize(base, "x")
    assert l.rent_net == 2000 and l.rent_gross is None  # no charges -> gross unknown (clean imputes)
    assert l.is_furnished  # title regex
    assert l.living_space_m2 == 80  # livingspace fallback
    assert l.has_balcony is None and l.image_url is None
    with pytest.raises(ValueError):
        normalize({**base, "price_unit": "yearlym2"}, "x")


def test_is_rent_apartment(items):
    assert is_rent_apartment(by_type(items, "APARTMENT"))
    assert is_rent_apartment(by_type(items, "FURNISHED_FLAT"))
    assert not is_rent_apartment(by_type(items, "SINGLE_ROOM"))
    apt = by_type(items, "APARTMENT")
    assert not is_rent_apartment({**apt, "object_category": "HOUSE", "object_type": "SINGLE_HOUSE"})
    assert not is_rent_apartment({**apt, "object_category": "INDUSTRY", "object_type": "OFFICE"})
    assert not is_rent_apartment({**apt, "price_unit": "yearlym2"})
    assert not is_rent_apartment({**apt, "offer_type": "SALE"})


def test_passes_filters(items):
    l = normalize(next(i for i in items if i["object_type"] == "APARTMENT" and i["smg_id"]), "x")
    spec = lambda **kw: SearchSpec(label="x", portal="flatfox", **kw)  # noqa: E731
    assert passes_filters(l, spec())
    assert passes_filters(l, spec(rooms_min=l.rooms, rent_max=l.rent_gross))
    assert not passes_filters(l, spec(rooms_min=l.rooms + 1))
    assert not passes_filters(l, spec(rent_max=l.rent_gross - 1))
    assert not passes_filters(l, spec(size_min=l.living_space_m2 + 1))
    assert not passes_filters(l.model_copy(update={"living_space_m2": None}), spec(size_min=10))


# --------------------------------------------------------------------------- geometry


def test_pins_in_polygon(pins):
    # triangle covering the first pin only
    p0 = pins[0]
    tri = Polygon([(p0["longitude"] - 1e-4, p0["latitude"] - 1e-4),
                   (p0["longitude"] + 1e-4, p0["latitude"] - 1e-4),
                   (p0["longitude"], p0["latitude"] + 1e-4)])
    assert [p["pk"] for p in pins_in_polygon(pins, tri)] == [p0["pk"]]
    assert pins_in_polygon([{"pk": 1, "latitude": None, "longitude": 8.9}], tri) == []


def test_pins_in_real_municipality(pins):
    from flatprice.geo import municipality_shape

    _, name, geom = municipality_shape("Paradiso")
    assert name == "Paradiso"
    inside = pins_in_polygon(pins, geom)
    assert 0 < len(inside) <= len(pins)
    assert all(geom.intersects(box(p["longitude"], p["latitude"], p["longitude"], p["latitude"])) for p in inside)


def test_split_bbox():
    quads = split_bbox((0, 0, 2, 2))
    assert quads == [(0, 0, 1, 1), (1, 0, 2, 1), (0, 1, 1, 2), (1, 1, 2, 2)]


def _grid_pins(n_side=40):
    """n_side x n_side pins evenly spread over (0,0)-(1,1)."""
    return [{"pk": i * n_side + j, "latitude": (i + 0.5) / n_side, "longitude": (j + 0.5) / n_side}
            for i in range(n_side) for j in range(n_side)]


class FakeApi:
    """Answers pin queries like the real API: at most `cap` pins per call."""

    def __init__(self, all_pins, cap=400):
        self.all_pins, self.cap, self.calls = all_pins, cap, []

    def get_pins(self, bbox):
        self.calls.append(bbox)
        w, s, e, n = bbox
        hit = [p for p in self.all_pins if w <= p["longitude"] <= e and s <= p["latitude"] <= n]
        return hit[: self.cap]


def test_quadtree_splits_until_below_cap():
    api = FakeApi(_grid_pins(40))  # 1600 pins, cap 400
    got = fetch_pins(api.get_pins, (0, 0, 1, 1))
    assert len(got) == 1600  # nothing lost to the cap
    assert len(api.calls) == 1 + 4 + 16  # root, 4 quads of exactly 400 (capped), 16 x 100
    leaves = [c for c in api.calls if c[2] - c[0] == 0.25]
    assert len(leaves) == 16
    assert all(len(FakeApi(_grid_pins(40)).get_pins(c)) == 100 for c in leaves)


def test_quadtree_no_split_when_small():
    api = FakeApi(_grid_pins(10))
    assert len(fetch_pins(api.get_pins, (0, 0, 1, 1))) == 100
    assert len(api.calls) == 1


def test_quadtree_skips_tiles_outside_geometry():
    api = FakeApi(_grid_pins(40))
    geom = box(0.0, 0.0, 0.2, 0.2)  # only the south-west quadrant
    got = fetch_pins(api.get_pins, (0, 0, 1, 1), geom)
    # root (capped) -> only the SW quadrant intersects (capped) -> only its SW child intersects
    assert api.calls == [(0, 0, 1, 1), (0, 0, 0.5, 0.5), (0, 0, 0.25, 0.25)]
    assert len(got) == 100
    assert all(p["longitude"] <= 0.5 and p["latitude"] <= 0.5 for p in got.values())


# --------------------------------------------------------------------------- scrape (mock HTTP)


class FakeClient:
    def __init__(self, all_pins, items, cap=400):
        self.api = FakeApi(all_pins, cap)
        self.items = {i["pk"]: i for i in items}
        self.detail_calls = []
        self.closed = False

    def get_json(self, path, params):
        if path.endswith("/pin/"):
            assert params["offer_type"] == "RENT"
            bbox = tuple(float(params[k]) for k in ("west", "south", "east", "north"))
            return self.api.get_pins(bbox)
        pks = [int(v) for k, v in params if k == "pk"]
        assert len(pks) <= 100
        self.detail_calls.append(pks)
        return {"count": len(pks), "next": None,
                "results": [self.items[p] for p in pks if p in self.items]}

    def close(self):
        self.closed = True


def test_scrape_end_to_end_offline(items, monkeypatch):
    from flatprice import geo

    _, _, para = geo.municipality_shape("paradiso")
    x, y = para.representative_point().coords[0]
    # every fixture item sits inside Paradiso; plus 450 filler pins outside it (forces a split)
    placed = [{**i, "latitude": y, "longitude": x} for i in items]
    w, s, e, n = para.bounds
    filler = [{"pk": 10_000 + k, "latitude": n + 0.001, "longitude": w + (e - w) * k / 450}
              for k in range(450)]
    client = FakeClient([{"pk": i["pk"], "latitude": y, "longitude": x} for i in placed]
                        + [{**f} for f in filler], placed)
    portal = Flatfox(client_factory=lambda: client)
    spec = SearchSpec(label="flatfox-paradiso", portal="flatfox", locations=["paradiso"])
    got = list(portal.scrape(None, spec, ScrapeConfig()))
    assert {l.portal_id for l in got} == {str(i["pk"]) for i in items if i["object_type"] != "SINGLE_ROOM"}
    assert all(l.search == "flatfox-paradiso" for l in got)
    assert client.closed
    # only pins inside the polygon are fetched in detail
    assert sorted(p for c in client.detail_calls for p in c) == sorted(i["pk"] for i in items)

    rooms_spec = spec.model_copy(update={"rooms_min": 4})
    got = list(Flatfox(client_factory=lambda: FakeClient(client.api.all_pins, placed)).scrape(None, rooms_spec, ScrapeConfig()))
    assert got and all(l.rooms >= 4 for l in got)


def test_regions_and_errors():
    p = Flatfox()
    assert p.needs_browser is False
    assert [n for n, _ in p.regions(SearchSpec(label="x", locations=["lugano", "Collina-d'Oro"]))] == ["Lugano", "Collina d'Oro"]
    with pytest.raises(ScrapeError):
        p.regions(SearchSpec(label="x", locations=["atlantis"]))
    with pytest.raises(ScrapeError):
        p.regions(SearchSpec(label="x"))
    url = p.build_url(SearchSpec(label="x", locations=["paradiso"]))
    assert url.startswith("https://flatfox.ch/api/v1/pin/?") and "max_count=400" in url


def test_registry():
    assert isinstance(get_portal("flatfox"), Flatfox)
    assert get_portal("homegate").needs_browser is True


# --------------------------------------------------------------------------- HTTP client


class _Resp:
    def __init__(self, status, body, ctype="application/json", headers=None):
        self.status_code, self.text, self.url = status, body, "https://flatfox.ch/x"
        self.headers = {"content-type": ctype, **(headers or {})}

    def json(self):
        return json.loads(self.text)


def _client(monkeypatch, responses, tmp_path):
    import flatprice.portals.flatfox as ff

    monkeypatch.setattr(ff, "DEBUG_DIR", tmp_path)
    monkeypatch.setattr(ff.time, "sleep", lambda s: None)
    c = HttpClient(delay=(0, 0), backoff_s=0)
    seq = iter(responses)
    monkeypatch.setattr(c.client, "get", lambda path, params=None: next(seq))
    return c


def test_http_retries_then_succeeds(monkeypatch, tmp_path):
    c = _client(monkeypatch, [_Resp(429, ""), _Resp(503, ""), _Resp(200, "[1, 2]")], tmp_path)
    assert c.get_json("/api/v1/pin/", {}) == [1, 2]
    assert c.n_requests == 3


def test_http_blocked_saves_body(monkeypatch, tmp_path):
    import flatprice.portals.flatfox as ff

    monkeypatch.setattr(ff, "save_debug_body", lambda body, name, directory=tmp_path: (tmp_path / "b.html").write_text(body))
    c = _client(monkeypatch, [_Resp(200, "<html>captcha-delivery datadome</html>", "text/html")], tmp_path)
    with pytest.raises(BlockedError):
        c.get_json("/api/v1/pin/", {})
    assert "datadome" in (tmp_path / "b.html").read_text()
    c = _client(monkeypatch, [_Resp(403, "nope", "text/plain")], tmp_path)
    with pytest.raises(BlockedError):
        c.get_json("/api/v1/pin/", {})


def test_http_gives_up(monkeypatch, tmp_path):
    c = _client(monkeypatch, [_Resp(500, "")] * 5, tmp_path)
    with pytest.raises(ScrapeError):
        c.get_json("/x", {})


def test_proxy_env(monkeypatch):
    from flatprice.portals.flatfox import _proxy_url

    monkeypatch.delenv("PROXY_URL", raising=False)
    assert _proxy_url() is None
    monkeypatch.setenv("PROXY_URL", "user:pw@host:8080")
    assert _proxy_url() == "http://user:pw@host:8080"
