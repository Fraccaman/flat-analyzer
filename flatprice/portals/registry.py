"""Portal registry: name -> adapter instance."""

from __future__ import annotations

from typing import Callable

from flatprice.portals.base import Portal
from flatprice.portals.flatfox import Flatfox
from flatprice.portals.homegate import Homegate

PORTALS: dict[str, Callable[[], Portal]] = {
    "homegate": Homegate,
    "flatfox": Flatfox,
}


def get_portal(name: str) -> Portal:
    try:
        factory = PORTALS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown portal {name!r}; available: {', '.join(sorted(PORTALS))}"
        ) from None
    return factory()
