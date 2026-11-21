import json
from pathlib import Path

import pytest

from flatprice.models import SearchSpec
from flatprice.portals.base import ScrapeError
from flatprice.portals.homegate import (
    Homegate,
    build_page_url,
    extract_result,
    is_rent_listing,
    normalize,
)
from flatprice.portals.registry import get_portal

FIXTURE = Path(__file__).parent / "fixtures" / "homegate_state.json"


@pytest.fixture
def state():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_extract_result(state):
    result = extract_result(state)
    assert result["pageCount"] >= 1
    assert result["resultCount"] >= len(result["listings"]) >= 2


def test_extract_result_bad_shape():
    with pytest.raises(ScrapeError):
        extract_result({"resultList": {}})


def test_normalize_all_fixture_items(state):
    for item in extract_result(state)["listings"]:
        assert is_rent_listing(item)
        l = normalize(item, "paradiso")
        lid = str(item["listing"]["id"])
        assert l.portal == "homegate" and l.portal_id == lid
        assert l.key == f"homegate:{lid}"
        assert l.url == f"https://www.homegate.ch/affittare/{lid}"
        assert l.search == "paradiso"
        assert l.title
        assert l.rent_gross is None or l.rent_gross > 0
        assert isinstance(l.categories, list)
        assert l.raw == item
        if l.image_url is not None:
            assert l.image_url.startswith("http")


def test_normalize_fields(state):
    item = extract_result(state)["listings"][0]
    src = item["listing"]
    l = normalize(item, "x")
    assert l.rooms == src["characteristics"].get("numberOfRooms")
    assert l.latitude == src["address"]["geoCoordinates"]["latitude"]
    assert l.locality == src["address"].get("locality", "")
    assert l.rent_gross == src["prices"]["rent"].get("gross")


def _item(**over):
    listing = {
        "id": "123",
        "offerType": "RENT",
        "categories": ["APARTMENT"],
        "address": {"street": "Via X 1", "postalCode": "6900", "locality": "Lugano"},
        "characteristics": {"numberOfRooms": 2.5, "livingSpace": 60, "floor": "3", "hasGarage": True},
        "prices": {"currency": "CHF", "rent": {"gross": 1200, "net": 1000, "interval": "MONTH"}},
        "localization": {
            "primary": "de",
            "de": {"text": {"title": "Schöne Wohnung"},
                   "attachments": [{"type": "DOCUMENT", "url": "https://d/doc.pdf"},
                                   {"type": "IMAGE", "url": "https://img/1.jpg"}]},
        },
    }
    listing.update(over)
    return {"listing": listing, "listingType": {"type": "STANDARD"}}


def test_normalize_handcrafted():
    l = normalize(_item(), "s")
    assert l.title == "Schöne Wohnung"
    assert l.image_url == "https://img/1.jpg"
    assert l.floor == 3 and l.has_parking is True
    assert l.is_furnished is False
    assert l.latitude is None


def test_furnished_detection():
    assert normalize(_item(categories=["FURNISHED_FLAT"]), "s").is_furnished
    loc = {"primary": "it", "it": {"text": {"title": "Bilocale ARREDATO vista lago"}}}
    assert normalize(_item(localization=loc), "s").is_furnished
    loc = {"primary": "de", "de": {"text": {"title": "Möblierte Wohnung"}}}
    assert normalize(_item(localization=loc), "s").is_furnished


def test_interval_conversion_and_filtering():
    yearly = _item(prices={"rent": {"gross": 24000, "interval": "YEAR"}})
    assert is_rent_listing(yearly)
    assert normalize(yearly, "s").rent_gross == 2000
    weekly = _item(prices={"rent": {"gross": 300, "interval": "WEEK"}})
    assert normalize(weekly, "s").rent_gross == pytest.approx(1300)
    assert not is_rent_listing(_item(prices={"rent": {"gross": 80, "interval": "DAY"}}))
    assert not is_rent_listing(_item(offerType="BUY"))


def test_build_urls():
    hg = Homegate()
    spec = SearchSpec(label="x", locations=["lugano", "Paradiso"], rooms_min=2.5,
                      rooms_max=4, rent_max=2500, size_min=60)
    urls = hg.build_urls(spec)
    assert urls == [
        "https://www.homegate.ch/affittare/appartamento/luogo-lugano/lista-annunci?ac=2.5&ad=4&ah=2500&ak=60",
        "https://www.homegate.ch/affittare/appartamento/luogo-paradiso/lista-annunci?ac=2.5&ad=4&ah=2500&ak=60",
    ]
    assert hg.build_url(spec) == urls[0]
    assert hg.build_urls(SearchSpec(label="x", locations=["a"], url="https://h/q")) == ["https://h/q"]
    with pytest.raises(ScrapeError):
        hg.build_urls(SearchSpec(label="x"))


def test_build_page_url_keeps_filters():
    url = build_page_url("https://h/l?ac=2&ah=1500", 3)
    assert "ac=2" in url and "ah=1500" in url and "ep=3" in url


def test_registry():
    assert isinstance(get_portal("homegate"), Homegate)
    with pytest.raises(ValueError, match="unknown portal"):
        get_portal("nope")


def test_parse_proxy():
    from flatprice.browser import parse_proxy

    assert parse_proxy(None) is None
    assert parse_proxy("user:p%40ss@proxy.ch:8080") == {
        "server": "http://proxy.ch:8080", "username": "user", "password": "p@ss"}
    assert parse_proxy("socks5://h:1080") == {"server": "socks5://h:1080"}
