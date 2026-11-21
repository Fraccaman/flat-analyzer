// Deal verdicts and savings. Mirrors flatprice/analysis/score.py (module doc, sections 1-3).
//
// Per listing (sigma = fair_sigma_log, k = shrink_k, both shipped in data.json):
//   log_ratio_eff    = ln(effective_rent / fair_rent)
//   deal_pct_adj_eff = (exp(log_ratio_eff * k) - 1) * 100     shrunk deal % (verdicts use this)
//   deal_z           = log_ratio_eff / sigma
//   p_below_fair     = Phi(-deal_z)                           chance rent is below fair rent
//   saving_month     = fair_rent - effective_rent             raw saving
//   saving_adj_month = fair_rent * (1 - exp(log_ratio_eff * k))   shrunk saving (headline)
// Verdict: d <= -20 -> great if p_below_fair >= 0.8 else good; -20 < d <= -8 good;
//          -8 < d < 8 fair; 8 <= d <= 20 pricey; > 20 over.
// Pure functions only; no DOM.

const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export const GREAT_MIN_P = 0.8;

/** Ordered best -> worst. `max` is the inclusive upper bound of deal % for the bucket. */
export const VERDICTS = [
  { id: "great", label: "Great deal", short: "Great", max: -20, rank: 4 },
  { id: "good", label: "Good deal", short: "Good", max: -8, rank: 3 },
  { id: "fair", label: "Fair price", short: "Fair", max: 8, rank: 2 },
  { id: "pricey", label: "Pricey", short: "Pricey", max: 20, rank: 1 },
  { id: "over", label: "Overpriced", short: "Over", max: Infinity, rank: 0 },
];
export const VERDICT_BY_ID = Object.fromEntries(VERDICTS.map((v) => [v.id, v]));

export const KINDS = {
  takeover: { label: "Takeover", title: "Lease takeover (subentro): the current tenant looks for a successor. Not used to train the model." },
  short_term: { label: "Short-term", title: "Temporary / short-stay rental, usually priced differently. Not used to train the model." },
};

/** Standard normal CDF via Abramowitz & Stegun 7.1.26 erf (same operations as score.py norm_cdf). */
export function normCdf(x) {
  if (!isNum(x)) return null;
  const z = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * z);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-z * z);
  return x >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
}

/** shrink_k for a sigma given the global tau2 (fallback when a row lacks shrink_k). */
export const shrinkK = (sigma, tau2) => (isNum(sigma) && isNum(tau2) && sigma > 0 ? tau2 / (tau2 + sigma * sigma) : null);

/**
 * Viewer-dependent deal fields from effective rent (unrounded), fair_rent,
 * fair_sigma_log and shrink_k. Sets them on `l` in place and returns it.
 */
export function dealFields(l, eff) {
  const fair = l.fair_rent;
  const sigma = l.fair_sigma_log;
  const k = l.shrink_k;
  const ok = isNum(eff) && eff > 0 && isNum(fair) && fair > 0;
  const lr = ok ? Math.log(eff / fair) : null;
  const shrunk = isNum(lr) && isNum(k) ? Math.exp(lr * k) : null;
  l.deal_pct_adj_eff = isNum(shrunk) ? (shrunk - 1) * 100 : null;
  l.deal_z = isNum(lr) && isNum(sigma) && sigma > 0 ? lr / sigma : null;
  l.p_below_fair = isNum(l.deal_z) ? normCdf(-l.deal_z) : null;
  l.saving_adj_month = isNum(shrunk) ? fair * (1 - shrunk) : null;
  return l;
}

/** Verdict for a (shrunk) deal % (negative = cheaper than fair) and chance below fair. null when unknown. */
export function verdictFor(pct, pBelow) {
  if (!isNum(pct)) return null;
  if (pct <= -20) return isNum(pBelow) && pBelow >= GREAT_MIN_P ? VERDICT_BY_ID.great : VERDICT_BY_ID.good;
  if (pct <= -8) return VERDICT_BY_ID.good;
  if (pct < 8) return VERDICT_BY_ID.fair; // "fair" is the open interval (-8, 8)
  if (pct <= 20) return VERDICT_BY_ID.pricey;
  return VERDICT_BY_ID.over;
}

/** The deal % the UI judges by: shrunk, includes the viewer's tax effect. */
export const dealPctOf = (l) => (isNum(l.deal_pct_adj_eff) ? l.deal_pct_adj_eff : null);

/** Raw CHF/month below fair rent (positive = good). null when unknown. */
export function savingMonth(l) {
  if (!isNum(l.fair_rent) || l.fair_rent <= 0) return null;
  const eff = isNum(l.effective_rent) ? l.effective_rent : isNum(l.rent_gross) ? l.rent_gross : null;
  if (!isNum(eff)) return null;
  return Math.round((l.fair_rent - eff) * 100) / 100;
}

/** Model range for the fair rent: fair_rent * exp(-/+ sigma). null when unknown. */
export function fairRange(l) {
  if (!isNum(l.fair_rent) || !isNum(l.fair_sigma_log)) return null;
  return { lo: l.fair_rent * Math.exp(-l.fair_sigma_log), hi: l.fair_rent * Math.exp(l.fair_sigma_log) };
}

/** Set l.saving_month, l.verdict (id|null), l.verdict_rank (number|null) in place. */
export function annotate(listings) {
  for (const l of listings) {
    const v = verdictFor(dealPctOf(l), l.p_below_fair);
    l.saving_month = savingMonth(l);
    l.verdict = v ? v.id : null;
    l.verdict_rank = v ? v.rank : null;
  }
  return listings;
}

/** True when the listing's verdict is at least `minId` (null minId = always true). */
export function meetsVerdict(l, minId) {
  if (!minId) return true;
  const min = VERDICT_BY_ID[minId];
  if (!min) return true;
  return isNum(l.verdict_rank) && l.verdict_rank >= min.rank;
}

/**
 * Geometry of a diverging bar centred at 0 (percent of the full track width).
 * Negative pct grows to the left of the centre, positive to the right.
 */
export function divergingBar(pct, clamp = 40) {
  if (!isNum(pct)) return null;
  const c = Math.max(-clamp, Math.min(clamp, pct));
  const w = (Math.abs(c) / clamp) * 50;
  return { left: c < 0 ? 50 - w : 50, width: w, side: c < 0 ? "left" : c > 0 ? "right" : "none", clamped: Math.abs(pct) > clamp };
}

const isRegular = (l) => !l.listing_kind || l.listing_kind === "regular";

/**
 * Best `n` listings by score: non-suspicious with a score and a positive (shrunk)
 * saving. Low-confidence estimates and takeover / short-term ads only fill
 * remaining slots.
 */
export function topDeals(listings, n = 6) {
  const saving = (l) => (isNum(l.saving_adj_month) ? l.saving_adj_month : l.saving_month);
  const pool = listings.filter((l) => !l.suspicious && isNum(l.score) && isNum(saving(l)) && saving(l) > 0);
  const byScore = (a, b) => b.score - a.score;
  const sure = pool.filter((l) => l.model_confidence !== "low" && isRegular(l)).sort(byScore);
  const rest = pool.filter((l) => l.model_confidence === "low" || !isRegular(l)).sort(byScore);
  return [...sure, ...rest].slice(0, n);
}

/** Median of the finite numbers in `values` (null when none). */
export function median(values) {
  const v = values.filter(isNum).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/** Header summary over the listings currently shown. */
export function summarize(listings) {
  const ok = listings.filter((l) => !l.suspicious);
  const reliable = ok.filter((l) => l.model_confidence !== "low" && isRegular(l));
  const savings = reliable.map((l) => (isNum(l.saving_adj_month) ? l.saving_adj_month : l.saving_month)).filter(isNum);
  return {
    count: listings.length,
    great: ok.filter((l) => l.verdict === "great").length,
    goodOrBetter: ok.filter((l) => isNum(l.verdict_rank) && l.verdict_rank >= VERDICT_BY_ID.good.rank).length,
    bestSaving: savings.length ? Math.max(...savings) : null,
    medianChfM2: median(ok.map((l) => l.chf_per_m2)),
  };
}
