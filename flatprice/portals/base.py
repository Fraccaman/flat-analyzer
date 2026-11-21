"""Portal adapter interface.

To add a portal: create `flatprice/portals/<name>.py` with a class implementing
`Portal`, then register it in `flatprice/portals/registry.py`.
"""

from __future__ import annotations

from typing import Iterator, Protocol

from flatprice.config import ScrapeConfig
from flatprice.models import Listing, SearchSpec


class ScrapeError(Exception):
    """A search failed; the run continues with the other searches."""


class BlockedError(ScrapeError):
    """Anti-bot protection (captcha / DataDome) blocked the page."""


class Portal(Protocol):
    name: str
    # True if `scrape` needs the shared Playwright page. HTTP-only portals set
    # False and get `page=None`; the browser is then never launched for them.
    needs_browser: bool

    def build_url(self, spec: SearchSpec) -> str:
        """Translate a SearchSpec to the portal's first result-page URL."""
        ...

    def scrape(self, page, spec: SearchSpec, cfg: ScrapeConfig) -> Iterator[Listing]:
        """Yield every listing for `spec` (`page` is None when needs_browser is False)."""
        ...
