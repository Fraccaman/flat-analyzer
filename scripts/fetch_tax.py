import json, sys, csv
estv, out = sys.argv[1], sys.argv[2]
Y = 2026
scales = json.load(open(f"{estv}/scales{Y}.json"))["response"]
rates = json.load(open(f"{estv}/rates{Y}.json"))["response"]

def prog(x, br):
    t = 0.0
    for j, b in enumerate(br):
        lo, rt = b[0], b[1]
        if x < lo: break
        if len(b) > 2: t = b[2]
        hi = br[j + 1][0] if j + 1 < len(br) else float("inf")
        t += (min(x, hi) - lo) * rt
    return t

def to_brackets(table, drop_tail=0):
    """Official rates per tranche; where the official table has a discontinuity
    (tax at the next threshold != accumulated tranches, >= 0.5 CHF) the bracket
    gets a 3rd element = official tax at its threshold."""
    rows = table[: len(table) - drop_tail] if drop_tail else table
    br = []
    for i, r in enumerate(rows):
        e = [r["Amount"], round(r["Percent"] / 100, 8)]
        if i > 0 and abs(prog(r["Amount"], br) - r["Taxes"]) >= 0.5:
            e.append(r["Taxes"])
        br.append(e)
    for r in rows:
        assert abs(prog(r["Amount"], br) - r["Taxes"]) < 0.5, (r, prog(r["Amount"], br))
    return br

def pick(target, taxtype, group_pred, canton=None):
    for e in scales:
        if e["Target"] == target and e["TaxType"] == taxtype and group_pred(e["Group"]) and (canton is None or e["Location"]["Canton"] == canton):
            return e
    raise KeyError((target, taxtype))

single = lambda g: g.startswith("LEDIG_ALLEINE") and "VERHEIRATET" not in g
married = lambda g: "VERHEIRATET" in g
src_scales = "https://swisstaxcalculator.estv.admin.ch/ (ESTV, API_exportManyTaxScales, TaxYear 2026)"
src_calc = "https://swisstaxcalculator.estv.admin.ch/ (ESTV, API_calculateSimpleTaxes, used to validate)"

fs, fm = pick("BUND", "EINKOMMENSSTEUER", single)["Table"], pick("BUND", "EINKOMMENSSTEUER", married)["Table"]
fed = {
    "year": Y,
    "sources": [src_scales, "https://www.estv.admin.ch/estv/de/home/direkte-bundessteuer/dbst-steuertarife.html", src_calc],
    "notes": "Federal direct tax (DBG art. 36). Official tariff published as base amount at threshold + rate per 100 CHF above; expressed as marginal tranches [threshold, rate] that reproduce the official amount at every threshold (within rounding). Married tariff (art. 36 al. 2) is also applied to single parents (child deduction from tax not modelled). flat_top: if taxable income >= threshold, tax = income * rate (replaces the brackets).",
    "income": {"single": to_brackets(fs, 2), "married": to_brackets(fm, 2)},
    "flat_top": {"single": {"threshold": fs[-1]["Amount"], "rate": fs[-1]["Percent"] / 100},
                 "married": {"threshold": fm[-1]["Amount"], "rate": fm[-1]["Percent"] / 100}},
}
ts, tm = pick("KANTON", "EINKOMMENSSTEUER", single, "TI")["Table"], pick("KANTON", "EINKOMMENSSTEUER", married, "TI")["Table"]
tw = pick("KANTON", "VERMOEGENSSTEUER", lambda g: True, "TI")["Table"]
wb = to_brackets(tw)
# first official row has 0%: tariff alone yields 0 below 200k and 200 at 200k (cliff)
coefs = {e["IncomeRateCanton"] for e in rates}
assert coefs == {100}, coefs

names = {}
for r in csv.DictReader(open(f"{estv}/../dl/snap2026.csv", encoding="utf-8")):
    if r["Level"] == "3": names[int(r["BfsCode"])] = r["Name"]
munis = []
for e in sorted(rates, key=lambda e: e["Location"]["BfsID"]):
    b = e["Location"]["BfsID"]
    assert e["IncomeRateCity"] == e["FortuneRateCity"], e
    munis.append({"bfs": b, "name": names.get(b, e["Location"]["City"]), "multiplier": round(e["IncomeRateCity"] / 100, 4)})
ti = {
    "canton": "TI",
    "year": Y,
    "sources": [src_scales,
                "https://swisstaxcalculator.estv.admin.ch/ (ESTV, API_exportManySimpleRates TaxGroupID=21, TaxYear 2026: Steuerfuesse / moltiplicatori comunali)",
                src_calc,
                "https://www4.ti.ch/dfe/dc/ (Divisione delle contribuzioni, Canton Ticino)",
                "https://www.agvchapp.bfs.admin.ch/ (BFS official municipality register, snapshot 2026-01-01, for names/BFS numbers)"],
    "notes": "Ticino (LT). Base tariffs = ESTV 2026 official tariffs as marginal tranches [threshold, rate]. The official Ticino tables are discontinuous at a few thresholds (e.g. tax jumps 0.16 -> 20 CHF at 12'600 single; drops at 58'800); there the bracket carries an optional 3rd element [threshold, rate, base] = official tax at that threshold, which REPLACES the accumulated sum before adding the tranche. Verified against the ESTV calculator (exact to the franc). Ticino has separate married ('coniugati') tariffs, no splitting. cantonal_coefficient 2026 = 100% (ESTV IncomeRateCanton=100, confirmed by ESTV calculator). wealth_exemption is a threshold ('Freigrenze', wealth_exemption_type='threshold'): wealth below it is untaxed, otherwise the wealth tariff applies to the WHOLE taxable wealth (ESTV calculator: 199'900 -> 0, 200'000 -> 200 base). The tariff itself already encodes this (0% then base 200 at 200'000), so the exemption check is redundant but kept explicit. personal_tax = 'imposta personale' (fixed CHF per taxpayer, same in every comune). Municipal multiplier applies to both income and wealth base tax. ESTV rounds results to whole CHF and taxable income down to 100 CHF; not modelled.",
    "cantonal_coefficient": 1.0,
    "income": {"single": to_brackets(ts), "married": to_brackets(tm)},
    "wealth": {"single": wb, "married": wb},
    "wealth_exemption": {"single": tw[1]["Amount"], "married": tw[1]["Amount"]},
    "wealth_exemption_type": "threshold",
    "personal_tax": {"single": 20, "married": 40},
    "municipalities": munis,
}
import re
def dump(obj, path):
    txt = json.dumps(obj, indent=2, ensure_ascii=False)
    txt = re.sub(r"\[\s+(-?[\d.e-]+),\s+(-?[\d.e-]+)(?:,\s+(-?[\d.e-]+))?\s+\]", lambda m: "[" + ", ".join(g for g in m.groups() if g) + "]", txt)
    txt = re.sub(r"\{\s+(\"bfs\": \d+),\s+(\"name\": \"[^\"]*\"),\s+(\"multiplier\": [\d.]+)\s+\}", r"{\1, \2, \3}", txt)
    open(path, "w", encoding="utf-8").write(txt + "\n")
dump(fed, f"{out}/ch_federal_{Y}.json")
dump(ti, f"{out}/ti_{Y}.json")
print(len(munis), fed["flat_top"], ti["wealth"]["single"])
