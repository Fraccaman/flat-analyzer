"""Scrape orchestration: all configured searches -> one store run."""

from __future__ import annotations

import sys
import traceback
from contextlib import ExitStack
from typing import Callable, Optional

from flatprice.config import Config
from flatprice.models import Listing
from flatprice.portals.base import BlockedError, ScrapeError
from flatprice.portals.registry import get_portal
from flatprice.store import Store


def _log(msg: str) -> None:
    print(f"[scrape] {msg}", file=sys.stderr, flush=True)


class _LazyBrowser:
    """Opens the shared browser on first use only, so HTTP-only runs never launch Chrome.

    A launch failure is remembered: every later browser search fails with it.
    """

    def __init__(self, open_browser: Callable, cfg, stack: ExitStack):
        self._open, self._cfg, self._stack = open_browser, cfg, stack
        self._page = None
        self._error: Optional[str] = None

    def page(self):
        if self._error:
            raise ScrapeError(f"browser unavailable: {self._error}")
        if self._page is None:
            try:
                self._page = self._stack.enter_context(self._open(self._cfg))
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                _log(f"browser failure: {self._error}")
                traceback.print_exc(file=sys.stderr)
                raise ScrapeError(f"browser unavailable: {self._error}") from exc
        return self._page


def run_scrape(config: Config, open_browser: Optional[Callable] = None) -> dict:
    """Run every search in `config` and persist the results.

    `open_browser` is injectable for tests (defaults to flatprice.browser.open_browser).
    The browser is only opened if a search's portal has `needs_browser = True`.
    Returns {run_id, status, n_listings, searches: [{label, portal, status, n, error}]}.
    status: "ok" (all searches ok), "partial" (some failed), "failed" (zero listings).
    """
    if open_browser is None:
        from flatprice.browser import open_browser

    store = Store(config.db_path)
    run_id = store.start_run()
    searches: list[dict] = []
    errors: list[dict] = []
    seen: set[str] = set()
    recorded: list[Listing] = []
    total = 0
    status = "failed"
    health: dict = {}
    try:
        with ExitStack() as stack:
            browser = _LazyBrowser(open_browser, config.scrape, stack)
            for spec in config.searches:
                entry = {"label": spec.label, "portal": spec.portal,
                         "status": "ok", "n": 0, "error": None}
                batch: list[Listing] = []
                try:
                    portal = get_portal(spec.portal)
                    page = browser.page() if getattr(portal, "needs_browser", True) else None
                    for listing in portal.scrape(page, spec, config.scrape):
                        if listing.key in seen:
                            continue  # first search label wins
                        seen.add(listing.key)
                        batch.append(listing)
                except BlockedError as exc:
                    entry.update(status="blocked", error=str(exc))
                except ScrapeError as exc:
                    entry.update(status="error", error=str(exc))
                except Exception as exc:  # noqa: BLE001 - one search must not abort the rest
                    entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    traceback.print_exc(file=sys.stderr)
                # Keep whatever was scraped before a mid-search failure.
                # Listing fields are only written once the whole run passed the
                # data-quality check (see below); observations are stored now.
                if batch:
                    store.record(run_id, batch, update_listings=False, quality_config=config.quality)
                    recorded.extend(batch)
                entry["n"] = len(batch)
                total += len(batch)
                if entry["error"]:
                    _log(f"{spec.label}: {entry['status']} after {len(batch)} listings: {entry['error']}")
                    errors.append({"search": spec.label, "error": entry["error"]})
                else:
                    _log(f"{spec.label}: {len(batch)} listings")
                store.record_search(run_id, spec.label, entry["status"], entry["n"], entry["error"])
                searches.append(entry)

        if total == 0:
            status = "failed"
        elif errors or len(searches) < len(config.searches):
            status = "partial"
        else:
            status = "ok"
    finally:
        store.finish_run(run_id, status, total, errors)
        try:
            health = _quality_step(store, run_id, recorded, config)
        finally:
            store.close()

    _log(f"run {run_id}: {status}, {total} listings")
    return {"run_id": run_id, "status": status, "n_listings": total, "searches": searches,
            "quality": health}


def _quality_step(store: Store, run_id: int, recorded: list[Listing], config: Config) -> dict:
    """Recompute data-quality flags; update listing fields unless the run is poisoned."""
    from flatprice import quality

    if not recorded:
        return {}
    report = quality.apply(store, config.quality)
    health = quality.run_health(report, run_id, config.quality)
    if health["flagged"]:
        _log(f"WARNING run {run_id}: {health['flagged']}/{health['n']} observations flagged as "
             f"perturbed data ({health['share'] * 100:.1f}%; {report['by_reason']})")
    if health["poisoned"]:
        _log(f"WARNING run {run_id} looks poisoned (> {config.quality.max_flagged_share * 100:.0f}% "
             f"flagged): observations kept, listing fields NOT updated")
        health["updated"] = 0
    else:
        health["updated"] = store.update_listing_fields(run_id, recorded)
    return health
