"""Playwright browser setup shared by all portal adapters.

Portals such as Homegate sit behind DataDome, which blocks headless browsers.
We therefore launch a real, headful Chrome (falling back to the bundled
Chromium) with the most obvious automation fingerprint disabled.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import unquote, urlparse

from flatprice.config import ScrapeConfig

DEBUG_DIR = Path("data/debug")
LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]


def parse_proxy(url: Optional[str]) -> Optional[dict]:
    """Turn `[scheme://][user:pass@]host:port` into a Playwright proxy dict."""
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    parts = urlparse(url)
    if not parts.hostname:
        raise ValueError(f"invalid PROXY_URL: {url!r}")
    server = f"{parts.scheme}://{parts.hostname}"
    if parts.port:
        server += f":{parts.port}"
    proxy: dict = {"server": server}
    if parts.username:
        proxy["username"] = unquote(parts.username)
    if parts.password:
        proxy["password"] = unquote(parts.password)
    return proxy


@contextmanager
def open_browser(cfg: ScrapeConfig) -> Iterator["object"]:
    """Yield a Playwright `Page` in a fresh it-CH browser context."""
    from playwright.sync_api import sync_playwright

    proxy = parse_proxy(os.environ.get("PROXY_URL"))
    with sync_playwright() as p:
        launch_kwargs: dict = {"headless": cfg.headless, "args": LAUNCH_ARGS}
        if proxy:
            launch_kwargs["proxy"] = proxy
        try:
            browser = p.chromium.launch(channel="chrome", **launch_kwargs)
        except Exception as exc:  # no system Chrome installed
            print(f"[browser] system Chrome unavailable ({exc.__class__.__name__}); "
                  "falling back to bundled Chromium", file=sys.stderr)
            browser = p.chromium.launch(**launch_kwargs)
        try:
            context = browser.new_context(
                locale="it-CH",
                timezone_id="Europe/Zurich",
                viewport={"width": 1366, "height": 900},
            )
            page = context.new_page()
            yield page
        finally:
            browser.close()


def save_debug(page, name: str, directory: Path | str = DEBUG_DIR) -> Optional[Path]:
    """Save a screenshot and the HTML of `page` for post-mortem debugging.

    Never raises: debugging must not mask the original error.
    Returns the base path (without extension) or None if saving failed.
    """
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        base = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}"
        try:
            page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
        except Exception as exc:
            print(f"[debug] screenshot failed: {exc}", file=sys.stderr)
        try:
            base.with_suffix(".html").write_text(page.content(), encoding="utf-8")
        except Exception as exc:
            print(f"[debug] html dump failed: {exc}", file=sys.stderr)
        print(f"[debug] saved {base}.{{png,html}}", file=sys.stderr)
        return base
    except Exception as exc:
        print(f"[debug] save_debug failed: {exc}", file=sys.stderr)
        return None
