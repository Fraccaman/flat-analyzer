// Deal score, mirrored from the Python analysis.
//
// Mirrors flatprice/analysis/score.py (see its module docstring):
// raw signals: value = -ln(rent_gross / fair_rent) * shrink_k  (shrunk deal on PLAIN rent;
//                      -deal_pct only for legacy rows with no shrink_k field; null stays null)
//              tax = -(tax_delta_month ?? 0), price_drop = price_drop_pct ?? 0,
//              market_time = min(days_on_market, 90) (null stays null)
// comp_k = 100*(avg_rank-1)/(n-1) among NON-suspicious listings with non-null signal
//          (ties -> average rank); null signal / suspicious -> null. If all those
//          signals are equal (incl. n<=1) the component carries no information and is
//          null for everyone, so it drops out of the weighted mean.
// score  = sum(w_k*comp_k)/sum(w_k) over k with w_k>0 and comp_k non-null,
//          Math.round(x*10)/10; no such k or suspicious -> null.
// Deal fields (deal_z, p_below_fair, shrunk deal %, savings) come from deal.js dealFields.

import { dealFields, shrinkK } from "./deal.js";

export const COMPONENTS = ["value", "tax", "price_drop", "market_time"];

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
export const round = (v, d = 1) => (isNum(v) ? Math.round(v * 10 ** d) / 10 ** d : null);

/** Percentiles with average ranks: 100*(avg_rank-1)/(n-1); 50 if n<=1. */
export function percentiles(values) {
  const n = values.length;
  if (n <= 1) return values.map(() => 50);
  const order = values.map((v, i) => i).sort((a, b) => values[a] - values[b]);
  const out = new Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && values[order[j + 1]] === values[order[i]]) j++;
    const avgRank = (i + j) / 2 + 1; // ranks are 1-based
    const pct = (100 * (avgRank - 1)) / (n - 1);
    for (let k = i; k <= j; k++) out[order[k]] = pct;
    i = j + 1;
  }
  return out;
}

export function rawComponent(l, key) {
  switch (key) {
    case "value":
      if (l.shrink_k !== undefined) {
        const ok = isNum(l.shrink_k) && isNum(l.rent_gross) && isNum(l.fair_rent) && l.rent_gross > 0 && l.fair_rent > 0;
        return ok ? -Math.log(l.rent_gross / l.fair_rent) * l.shrink_k : null;
      }
      return isNum(l.deal_pct) ? -l.deal_pct : null; // legacy data without shrinkage
    case "tax":
      return isNum(l.tax_delta_month) ? -l.tax_delta_month : 0;
    case "price_drop":
      return isNum(l.price_drop_pct) ? l.price_drop_pct : 0;
    case "market_time":
      return isNum(l.days_on_market) ? Math.min(l.days_on_market, 90) : null;
    default:
      return null;
  }
}

/** Sets l["comp_"+key] for every listing (null for suspicious). */
export function computeComponent(listings, key) {
  const field = "comp_" + key;
  const pool = [];
  for (const l of listings) {
    if (l.suspicious) {
      l[field] = null;
      continue;
    }
    const r = rawComponent(l, key);
    if (r == null) l[field] = null;
    else pool.push([l, r]);
  }
  const vals = pool.map((p) => p[1]);
  if (!vals.length || vals.every((v) => v === vals[0])) {
    pool.forEach(([l]) => (l[field] = null)); // no variance -> no information
    return;
  }
  const pct = percentiles(vals);
  pool.forEach(([l], i) => (l[field] = pct[i]));
}

export function weightedScore(l, weights) {
  if (l.suspicious) return null;
  let num = 0;
  let den = 0;
  for (const k of COMPONENTS) {
    const w = Number(weights[k]) || 0;
    const c = l["comp_" + k];
    if (!(w > 0) || !isNum(c)) continue;
    num += w * c;
    den += w;
  }
  return den > 0 ? round(num / den, 1) : null;
}

/**
 * Recompute viewer-dependent fields in place.
 * opts.taxDelta: optional (bfs) => CHF/month delta (null = unknown). When absent
 * the precomputed tax_delta_month is kept.
 * opts.tau2: global shrinkage prior variance, used only for rows missing shrink_k.
 * Components other than tax are recomputed only when missing from the data.
 */
export function recompute(listings, weights, opts = {}) {
  const { taxDelta, tau2 } = opts;
  if (taxDelta) {
    for (const l of listings) {
      const d = taxDelta(l.municipality_bfs);
      l.tax_delta_month = d == null ? null : d;
    }
  }
  for (const l of listings) {
    const eff = isNum(l.rent_gross) ? l.rent_gross + (isNum(l.tax_delta_month) ? l.tax_delta_month : 0) : null;
    l.effective_rent = round(eff, 2);
    l.effective_deal_pct = isNum(eff) && isNum(l.fair_rent) && l.fair_rent > 0 ? round((eff / l.fair_rent - 1) * 100, 2) : null;
    if (!isNum(l.shrink_k) && isNum(tau2)) l.shrink_k = shrinkK(l.fair_sigma_log, tau2);
    dealFields(l, eff);
  }
  if (taxDelta || listings.some((l) => !l.suspicious && l.comp_tax === undefined)) computeComponent(listings, "tax");
  for (const k of ["value", "price_drop", "market_time"]) {
    if (listings.some((l) => !l.suspicious && l["comp_" + k] === undefined)) computeComponent(listings, k);
  }
  for (const l of listings) l.score = weightedScore(l, weights);
  return listings;
}
