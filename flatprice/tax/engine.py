"""Swiss income + wealth tax engine (federal + cantonal + communal).

All tariff data lives in JSON files under ``flatprice/tax/data/``:
``ch_federal_<year>.json`` and ``<canton>_<year>.json`` (lowercase canton code).
The website ships the same JSON (see :func:`export_tables`) and a JS mirror
implements exactly the formula below. Keep the two in sync.

EXACT FORMULA (f = filing: "single" | "married"; I = taxable income;
W = taxable net wealth; m = municipality multiplier, e.g. 0.77 = 77%)

    progressive(x, brackets):
        # brackets: list of [threshold, rate] or [threshold, rate, base],
        # sorted by threshold, first threshold 0. Each rate applies to the
        # tranche [threshold, next_threshold). The optional 3rd element `base`
        # is the official tax AT that threshold and replaces the running sum
        # (used where the official table is discontinuous).
        if x <= 0: return 0
        tax = 0
        for i, b in enumerate(brackets):
            lo, rate = b[0], b[1]
            if x < lo: break
            if len(b) > 2: tax = b[2]
            hi = brackets[i+1][0] if i+1 < len(brackets) else +inf
            tax += (min(x, hi) - lo) * rate
        return tax

    federal = I * F.flat_top[f].rate              if I >= F.flat_top[f].threshold
              progressive(I, F.income[f])          otherwise

    # optional canton key "married_splitting_divisor" (d, default 1; Ticino: absent):
    #   base_income = progressive(I / d, C.income[f]) * d
    base_income = progressive(I, C.income[f])
    wealth_base = 0                                if W < C.wealth_exemption[f]   (type "threshold")
    #   type "deduction" (optional key wealth_exemption_type, default "threshold"):
    #   wealth_base = progressive(max(0, W - C.wealth_exemption[f]), C.wealth[f])
                  progressive(W, C.wealth[f])      otherwise

    cantonal_income  = base_income * C.cantonal_coefficient
    communal_income  = base_income * m
    cantonal_wealth  = wealth_base * C.cantonal_coefficient
    communal_wealth  = wealth_base * m
    personal         = C.personal_tax[f]           (optional key, default 0)
    total = federal + cantonal_income + communal_income
            + cantonal_wealth + communal_wealth + personal

No rounding is applied (ESTV rounds taxable income down to 100 CHF and each tax
to the franc; the difference is a few CHF at most). Church tax, child
deductions and deductions in general are out of scope: inputs are already
*taxable* income / wealth.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FILINGS = ("single", "married")


# --------------------------------------------------------------------------- data


def _latest(prefix: str, year: int | None) -> Path:
    if year is not None:
        p = DATA_DIR / f"{prefix}_{year}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{4}})\.json$")
    found = sorted(
        (int(m.group(1)), p) for p in DATA_DIR.glob(f"{prefix}_*.json") if (m := pat.match(p.name))
    )
    if not found:
        raise FileNotFoundError(f"no {prefix}_<year>.json in {DATA_DIR}")
    return found[-1][1]


@lru_cache(maxsize=None)
def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_federal(year: int | None = None) -> dict:
    """Federal tariff JSON (latest year available if ``year`` is None)."""
    return _load(_latest("ch_federal", year))


def load_canton(code: str = "TI", year: int | None = None) -> dict:
    """Canton tariff JSON (latest year available if ``year`` is None)."""
    return _load(_latest(code.lower(), year))


def municipalities(canton: str = "TI") -> list[dict]:
    """``[{"bfs": int, "name": str, "multiplier": float}, ...]`` for the canton."""
    return [dict(m) for m in load_canton(canton)["municipalities"]]


@lru_cache(maxsize=None)
def _multipliers(path: Path) -> dict[int, float]:
    return {int(m["bfs"]): float(m["multiplier"]) for m in _load(path)["municipalities"]}


# --------------------------------------------------------------------------- math


def progressive(amount: float, brackets: list) -> float:
    """Sum of marginal tranches; see module docstring for the exact semantics."""
    if amount is None or amount <= 0:
        return 0.0
    tax = 0.0
    n = len(brackets)
    for i, b in enumerate(brackets):
        lo, rate = b[0], b[1]
        if amount < lo:
            break
        if len(b) > 2:
            tax = float(b[2])
        hi = brackets[i + 1][0] if i + 1 < n else math.inf
        tax += (min(amount, hi) - lo) * rate
    return tax


@dataclass
class TaxProfile:
    canton: str = "TI"
    filing: str = "single"  # single | married
    taxable_income: float = 0.0  # CHF/yr, after deductions
    taxable_wealth: float = 0.0  # CHF, net

    def __post_init__(self) -> None:
        if self.filing not in FILINGS:
            raise ValueError(f"filing must be one of {FILINGS}, got {self.filing!r}")


def federal_tax(income: float, filing: str, federal: dict | None = None) -> float:
    fed = federal or load_federal()
    top = fed["flat_top"][filing]
    if income >= top["threshold"]:
        return income * top["rate"]
    return progressive(income, fed["income"][filing])


def _base_taxes(profile: TaxProfile, canton: dict) -> tuple[float, float]:
    f = profile.filing
    d = float(canton.get("married_splitting_divisor", 1) or 1) if f == "married" else 1.0
    base_income = progressive(profile.taxable_income / d, canton["income"][f]) * d

    w = profile.taxable_wealth or 0.0
    exemption = canton.get("wealth_exemption", {}).get(f, 0)
    if canton.get("wealth_exemption_type", "threshold") == "deduction":
        base_wealth = progressive(max(0.0, w - exemption), canton["wealth"][f])
    else:
        base_wealth = 0.0 if w < exemption else progressive(w, canton["wealth"][f])
    return base_income, base_wealth


def annual_tax(profile: TaxProfile, bfs: int) -> dict:
    """Annual tax bill (CHF) for ``profile`` living in municipality ``bfs``.

    Raises KeyError for a BFS number not in the canton's municipality list.
    """
    path = _latest(profile.canton.lower(), None)
    canton = _load(path)
    mult = _multipliers(path)[int(bfs)]  # KeyError if unknown
    coef = float(canton["cantonal_coefficient"])
    base_income, base_wealth = _base_taxes(profile, canton)

    out = {
        "federal": federal_tax(profile.taxable_income or 0.0, profile.filing),
        "cantonal_income": base_income * coef,
        "communal_income": base_income * mult,
        "cantonal_wealth": base_wealth * coef,
        "communal_wealth": base_wealth * mult,
        "personal": float(canton.get("personal_tax", {}).get(profile.filing, 0)),
        "multiplier": mult,
    }
    out["total"] = sum(
        out[k]
        for k in ("federal", "cantonal_income", "communal_income", "cantonal_wealth", "communal_wealth", "personal")
    )
    return out


def tax_delta_month(profile: TaxProfile, bfs: int, reference_bfs: int) -> float:
    """Monthly tax difference of living in ``bfs`` vs ``reference_bfs`` (negative = saving)."""
    return (annual_tax(profile, bfs)["total"] - annual_tax(profile, reference_bfs)["total"]) / 12.0


def export_tables(cantons: tuple[str, ...] = ("TI",)) -> dict:
    """``{"federal": <json>, "cantons": {"TI": <json>, ...}}`` for embedding in the site."""
    return {
        "federal": load_federal(),
        "cantons": {c.upper(): load_canton(c) for c in cantons},
    }
