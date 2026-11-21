"""Homegate.ch adapter (rentals only).

Homegate is behind DataDome: a headful real Chrome passes, headless does not.
Every result page embeds `window.__INITIAL_STATE__`; the search result lives at
`resultList.search.fullSearch.result` with keys `listings`, `pageCount`,
`resultCount`. Pagination uses the `ep=<page>` query parameter.

Filter query parameters (verified live against resultCount, Nov 2026):
    ac / ad  rooms min / max
    ag / ah  rent min / max (CHF/month)
    ak / al  living space min / max (m2)
"""

from __future__ import annotations

import random
import re
import sys
import time
from typing import Iterator, Optional
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse

from flatprice.config import ScrapeConfig
from flatprice.models import Listing, SearchSpec
from flatprice.portals.base import BlockedError, ScrapeError

BASE = "https://www.homegate.ch"
SEARCH_TEMPLATE = BASE + "/affittare/appartamento/luogo-{location}/lista-annunci"
LISTING_TEMPLATE = BASE + "/affittare/{id}"

# SearchSpec field -> homegate query param
FILTER_PARAMS = {
    "rooms_min": "ac",
    "rooms_max": "ad",
    "rent_min": "ag",
    "rent_max": "ah",
    "size_min": "ak",
    "size_max": "al",
}

FURNISHED_RE = re.compile(r"arredat|ammobiliat|furnish|möbliert|moebliert|meubl", re.I)

# Rent interval -> factor to convert to CHF/month.
INTERVAL_TO_MONTHLY = {
    None: 1.0,
    "": 1.0,
    "MONTH": 1.0,
    "MONTHLY": 1.0,
    "YEAR": 1 / 12,
    "YEARLY": 1 / 12,
    "WEEK": 52 / 12,
    "WEEKLY": 52 / 12,
}

STATE_JS = "() => window.__INITIAL_STATE__"
CAPTCHA_JS = """() => !!document.querySelector(
    'iframe[src*="captcha-delivery"], iframe[src*="geo.captcha"], iframe[src*="ct.captcha-delivery"]'
)"""


# --------------------------------------------------------------------------- #
# Pure helpers (testable without a browser)
# --------------------------------------------------------------------------- #

def build_page_url(base_url: str, page: int) -> str:
    parts = urlparse(base_url)
    query = parse_qs(parts.query)
    query["ep"] = [str(page)]
    return urlunparse(parts._replace(query=urlencode(query, doseq=True)))


def _fmt_num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def filter_query(spec: SearchSpec) -> dict[str, str]:
    q: dict[str, str] = {}
    for field, param in FILTER_PARAMS.items():
        v = getattr(spec, field)
        if v is not None:
            q[param] = _fmt_num(v)
    return q


def extract_result(state: dict) -> dict:
    """Pull the search result object out of __INITIAL_STATE__."""
    try:
        result = state["resultList"]["search"]["fullSearch"]["result"]
    except (KeyError, TypeError) as exc:
        raise ScrapeError(f"unexpected __INITIAL_STATE__ shape: missing {exc}") from exc
    if not isinstance(result, dict):
        raise ScrapeError("unexpected __INITIAL_STATE__ shape: result is not an object")
    return result


def _lang_order(localization: dict) -> list[str]:
    order = [localization.get("primary"), "it", "en", "de", "fr"]
    order += [k for k in localization if k != "primary"]
    seen, out = set(), []
    for lang in order:
        if lang and lang not in seen:
            seen.add(lang)
            out.append(lang)
    return out


def pick_title(localization: dict) -> str:
    """Prefer the primary language of the ad, fall back to any available."""
    for lang in _lang_order(localization or {}):
        entry = localization.get(lang)
        if isinstance(entry, dict):
            title = (entry.get("text") or {}).get("title")
            if title:
                return title
    return ""


def pick_image(localization: dict) -> Optional[str]:
    """First IMAGE attachment URL from any language block."""
    for lang in _lang_order(localization or {}):
        entry = localization.get(lang)
        if not isinstance(entry, dict):
            continue
        for att in entry.get("attachments") or []:
            if not isinstance(att, dict):
                continue
            if att.get("type", "IMAGE") == "IMAGE" and att.get("url"):
                return att["url"]
    return None


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


def _str(v) -> str:
    return "" if v is None else str(v)


def is_rent_listing(item: dict) -> bool:
    listing = item.get("listing") or {}
    if listing.get("offerType") not in (None, "RENT"):
        return False
    rent = (listing.get("prices") or {}).get("rent") or {}
    return rent.get("interval") in INTERVAL_TO_MONTHLY


def normalize(item: dict, search_label: str) -> Listing:
    """Convert one `result.listings[]` item into a canonical Listing.

    Rents are converted to CHF/month according to `prices.rent.interval`.
    """
    listing = item.get("listing") or {}
    lid = listing.get("id") or item.get("id")
    if lid is None:
        raise ValueError("homegate item without id")
    address = listing.get("address") or {}
    coords = address.get("geoCoordinates") or {}
    chars = listing.get("characteristics") or {}
    rent = (listing.get("prices") or {}).get("rent") or {}
    categories = [str(c) for c in (listing.get("categories") or [])]
    localization = listing.get("localization") or {}

    factor = INTERVAL_TO_MONTHLY.get(rent.get("interval"), 1.0)

    def monthly(v):
        f = _num(v)
        return round(f * factor, 2) if f is not None else None

    title = pick_title(localization)
    parking = chars.get("hasParking")
    garage = chars.get("hasGarage")
    has_parking = True if (parking or garage) else (False if (parking is False or garage is False) else None)

    return Listing(
        portal="homegate",
        portal_id=str(lid),
        url=LISTING_TEMPLATE.format(id=lid),
        search=search_label,
        title=title,
        street=_str(address.get("street")),
        postal_code=_str(address.get("postalCode")),
        locality=_str(address.get("locality")),
        latitude=_num(coords.get("latitude")),
        longitude=_num(coords.get("longitude")),
        rent_gross=monthly(rent.get("gross")),
        rent_net=monthly(rent.get("net")),
        rooms=_num(chars.get("numberOfRooms")),
        living_space_m2=_num(chars.get("livingSpace")),
        floor=_int(chars.get("floor")),
        year_built=_int(chars.get("yearBuilt")),
        is_furnished="FURNISHED_FLAT" in categories or bool(FURNISHED_RE.search(title)),
        has_balcony=chars.get("hasBalcony"),
        has_parking=has_parking,
        has_elevator=chars.get("hasElevator"),
        pets_allowed=chars.get("arePetsAllowed"),
        categories=categories,
        image_url=pick_image(localization),
        raw=item,
    )


# --------------------------------------------------------------------------- #
# Portal
# --------------------------------------------------------------------------- #

class Homegate:
    name = "homegate"
    needs_browser = True

    def __init__(self, state_timeout_ms: int = 30_000, nav_timeout_ms: int = 60_000,
                 block_retries: int = 3, block_backoff_s: float = 20.0):
        self.state_timeout_ms = state_timeout_ms
        self.nav_timeout_ms = nav_timeout_ms
        self.block_retries = block_retries
        self.block_backoff_s = block_backoff_s

    def build_urls(self, spec: SearchSpec) -> list[str]:
        if spec.url:
            return [spec.url]
        if not spec.locations:
            raise ScrapeError(f"search {spec.label!r}: needs `locations` or `url`")
        query = urlencode(filter_query(spec))
        urls = []
        for loc in spec.locations:
            url = SEARCH_TEMPLATE.format(location=quote(loc.strip().lower()))
            urls.append(f"{url}?{query}" if query else url)
        return urls

    def build_url(self, spec: SearchSpec) -> str:
        return self.build_urls(spec)[0]

    # -- browser side ------------------------------------------------------- #

    def _is_captcha(self, page) -> bool:
        try:
            return bool(page.evaluate(CAPTCHA_JS))
        except Exception:
            return False

    def _load_state(self, page, url: str, label: str) -> dict:
        """Navigate and return __INITIAL_STATE__.

        DataDome sometimes serves a captcha for the first few requests of a
        fresh browser session and then lets it through, so a blocked load is
        retried `block_retries` times with a growing pause before giving up.
        """
        from flatprice.browser import save_debug

        last_exc: Exception | None = None
        captcha = False
        for attempt in range(self.block_retries + 1):
            if attempt:
                pause = self.block_backoff_s * attempt
                print(f"[homegate:{label}] blocked ({'captcha' if captcha else 'no state'}), "
                      f"retry {attempt}/{self.block_retries} in {pause:.0f}s", file=sys.stderr)
                time.sleep(pause)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=self.nav_timeout_ms)
                # Don't bail out on a visible captcha right away: DataDome's
                # interstitial sometimes clears itself and reloads into the page.
                page.wait_for_function(
                    "() => window.__INITIAL_STATE__ !== undefined", timeout=self.state_timeout_ms
                )
                break
            except Exception as exc:
                captcha = self._is_captcha(page)
                last_exc = exc
        else:
            save_debug(page, f"homegate-{label}-blocked")
            reason = "captcha" if captcha else "no __INITIAL_STATE__"
            raise BlockedError(f"homegate blocked ({reason}) at {url}") from last_exc
        state = page.evaluate(STATE_JS)
        if not isinstance(state, dict):
            save_debug(page, f"homegate-{label}-badstate")
            raise ScrapeError(f"__INITIAL_STATE__ is not an object at {url}")
        return state

    def _scrape_url(self, page, base_url: str, spec: SearchSpec, cfg: ScrapeConfig,
                    seen: set[str]) -> Iterator[Listing]:
        current, page_count = 1, 1
        cap = max(1, cfg.max_pages_per_search)
        while current <= min(page_count, cap):
            url = build_page_url(base_url, current)
            print(f"[homegate:{spec.label}] page {current}/{page_count}: {url}", file=sys.stderr)
            state = self._load_state(page, url, spec.label)
            result = extract_result(state)
            page_count = int(result.get("pageCount") or current)
            items = result.get("listings") or []
            kept = 0
            for item in items:
                if not is_rent_listing(item):
                    continue
                try:
                    listing = normalize(item, spec.label)
                except Exception as exc:
                    print(f"[homegate:{spec.label}] skip malformed item: {exc}", file=sys.stderr)
                    continue
                if listing.portal_id in seen:
                    continue
                seen.add(listing.portal_id)
                listing.page = current
                kept += 1
                yield listing
            print(f"[homegate:{spec.label}]   {len(items)} items, {kept} kept "
                  f"({result.get('resultCount')} reported)", file=sys.stderr)
            if page_count > cap and current == cap:
                print(f"[homegate:{spec.label}] page cap {cap} reached "
                      f"({page_count} pages available)", file=sys.stderr)
            current += 1
            if current <= min(page_count, cap):
                time.sleep(random.uniform(*cfg.delay_seconds))

    def scrape(self, page, spec: SearchSpec, cfg: ScrapeConfig) -> Iterator[Listing]:
        seen: set[str] = set()
        for i, url in enumerate(self.build_urls(spec)):
            if i > 0:
                time.sleep(random.uniform(*cfg.delay_seconds))
            yield from self._scrape_url(page, url, spec, cfg, seen)
