// Swiss income + wealth tax. EXACT mirror of flatprice/tax/engine.py (see its
// module docstring for the formula). Pure, data-driven: all rates come from
// data.json "tax" (= engine.export_tables()).
//
// brackets: [[threshold, rate] | [threshold, rate, base], ...] sorted by
// threshold; the optional `base` is the official tax at that threshold and
// replaces the running sum. The last tranche is open-ended.

export function progressive(x, brackets) {
  if (!(x > 0) || !Array.isArray(brackets)) return 0;
  let tax = 0;
  for (let i = 0; i < brackets.length; i++) {
    const b = brackets[i];
    const lo = b[0];
    if (x < lo) break;
    if (b.length > 2) tax = Number(b[2]);
    const hi = i + 1 < brackets.length ? brackets[i + 1][0] : Infinity;
    tax += (Math.min(x, hi) - lo) * b[1];
  }
  return tax;
}

export function federalTax(income, federal, filing) {
  if (!federal || !federal.income) return 0;
  const top = federal.flat_top && federal.flat_top[filing];
  if (top && income >= top.threshold) return income * top.rate;
  return progressive(income, federal.income[filing]);
}

/** Cantonal "simple tax" (before cantonal coefficient + municipal multiplier). */
export function cantonBase(income, wealth, canton, filing) {
  const d = filing === "married" ? Number(canton.married_splitting_divisor) || 1 : 1;
  const baseIncome = progressive(income / d, canton.income[filing]) * d;
  const exemption = (canton.wealth_exemption && canton.wealth_exemption[filing]) || 0;
  const wb = (canton.wealth || {})[filing];
  const baseWealth =
    (canton.wealth_exemption_type || "threshold") === "deduction"
      ? progressive(Math.max(0, wealth - exemption), wb)
      : wealth < exemption
        ? 0
        : progressive(wealth, wb);
  return { income: baseIncome, wealth: baseWealth };
}

export function municipalityIndex(canton) {
  const m = new Map();
  for (const row of (canton && canton.municipalities) || []) m.set(Number(row.bfs), row);
  return m;
}

/** True when tables contain enough data to compute municipality deltas. */
export function tablesUsable(tables, cantonCode = "TI") {
  const c = tables && tables.cantons && tables.cantons[cantonCode];
  return !!(c && c.income && Array.isArray(c.municipalities) && c.municipalities.length);
}

/**
 * Builds a calculator for one profile. Returns {total(bfs), delta(bfs)}:
 * total = federal + (base_income + base_wealth) * (cantonal_coefficient + multiplier) + personal_tax
 * delta = (total(bfs) - total(reference_bfs)) / 12, null if bfs unknown.
 */
export function makeTaxCalculator(tables, profile) {
  const cantonCode = profile.canton || "TI";
  if (!tablesUsable(tables, cantonCode)) return null;
  const canton = tables.cantons[cantonCode];
  const filing = profile.filing === "married" ? "married" : "single";
  const income = Number(profile.taxable_income) || 0;
  const wealth = Number(profile.taxable_wealth) || 0;
  const fed = federalTax(income, tables.federal, filing);
  const base = cantonBase(income, wealth, canton, filing);
  const baseSum = base.income + base.wealth;
  const coef = Number(canton.cantonal_coefficient) || 0;
  const personal = Number((canton.personal_tax || {})[filing]) || 0;
  const idx = municipalityIndex(canton);

  const total = (bfs) => {
    const row = bfs == null ? undefined : idx.get(Number(bfs));
    if (!row) return null;
    return fed + baseSum * (coef + Number(row.multiplier)) + personal;
  };
  const refTotal = total(profile.reference_bfs);
  const delta = (bfs) => {
    const t = total(bfs);
    if (t == null || refTotal == null) return null;
    return (t - refTotal) / 12;
  };
  return { total, delta, municipalities: idx };
}

/** Multiplier for display as percent (accepts 0.65 or 65). */
export function multiplierPct(m) {
  if (m == null || Number.isNaN(Number(m))) return null;
  m = Number(m);
  return Math.abs(m) <= 5 ? m * 100 : m;
}
