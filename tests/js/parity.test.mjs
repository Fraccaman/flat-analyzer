// Run: node --test tests/js/
// JS <-> Python parity for the uncertainty-aware deal fields and the score.
// Expected numbers were computed by flatprice/analysis/score.py on the same
// inputs (tests/test_analysis.py::test_parity_fixture pins the Python side).
import { test } from "node:test";
import assert from "node:assert/strict";
import { recompute } from "../../web/score.js";
import { annotate } from "../../web/deal.js";

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

function fixture() {
  // key, rent_gross, tax_delta_month, fair_rent, fair_sigma_log, days_on_market
  return [
    ["A", 1500, -60, 2000, 0.11, 10],
    ["B", 2000, 0, 2000, 0.25, 40],
    ["C", 2600, 25, 2000, 0.11, 95],
    ["D", 1300, null, 1800, 0.25, null],
    ["E", 1550, -20, 2000, 0.4, 5],
    ["F", 1580, 0, 2000, 0.4, 20],
  ].map(([key, rent_gross, tax_delta_month, fair_rent, fair_sigma_log, days_on_market]) => ({
    key, rent_gross, tax_delta_month, fair_rent, fair_sigma_log, days_on_market,
    price_drop_pct: 0, deal_pct: (rent_gross / fair_rent - 1) * 100,
  }));
}
const by = (ls) => Object.fromEntries(ls.map((l) => [l.key, l]));
const W = { value: 0.55, tax: 0.25, price_drop: 0.1, market_time: 0.1 };

// Python: deal_fields(df, tau2=0.05)
const EXPECTED_DEAL = {
  A: { shrink_k: 0.8051529790660226, deal_pct_adj_eff: -23.24073023814429, deal_z: -2.986400608836692, p_below_fair: 0.9985885179338168, saving_adj_month: 464.81460476288584, verdict: "great" },
  B: { shrink_k: 0.4444444444444445, deal_pct_adj_eff: 0.0, deal_z: 0.0, p_below_fair: 0.5000000005, saving_adj_month: 0.0, verdict: "fair" },
  C: { shrink_k: 0.8051529790660226, deal_pct_adj_eff: 24.476684729490206, deal_z: 2.472124686214925, p_below_fair: 0.006715642038393366, saving_adj_month: -489.5336945898041, verdict: "over" },
  D: { shrink_k: 0.4444444444444445, deal_pct_adj_eff: -13.465947411130442, deal_z: -1.301689601738512, p_below_fair: 0.9034886744545872, saving_adj_month: 242.38705340034795, verdict: "good" },
  E: { shrink_k: 0.23809523809523808, deal_pct_adj_eff: -6.17893862591491, deal_z: -0.6696986128890029, p_below_fair: 0.7484750977916568, saving_adj_month: 123.57877251829818, verdict: "fair" },
  F: { shrink_k: 0.23809523809523808, deal_pct_adj_eff: -5.457844888967189, deal_z: -0.5893058338026745, p_below_fair: 0.722171977953864, saving_adj_month: 109.15689777934378, verdict: "fair" },
};

test("deal fields match Python (shrinkage, z, chance below fair, verdict)", () => {
  const m = by(annotate(recompute(fixture(), W, { tau2: 0.05 })));
  for (const [k, e] of Object.entries(EXPECTED_DEAL)) {
    for (const f of ["shrink_k", "deal_pct_adj_eff", "deal_z", "p_below_fair", "saving_adj_month"]) close(m[k][f], e[f]);
    assert.equal(m[k].verdict, e.verdict, k);
  }
});

// Python: score(df, W) -> tau2 estimate hits the floor (0.0025) on this tiny set;
// price_drop is constant -> dropped from every score.
const EXPECTED_SCORE = {
  A: { comp_value: 100, comp_tax: 100, comp_market_time: 25, score: 91.7 },
  B: { comp_value: 20, comp_tax: 40, comp_market_time: 75, score: 31.7 },
  C: { comp_value: 0, comp_tax: 0, comp_market_time: 100, score: 11.1 },
  D: { comp_value: 80, comp_tax: 40, comp_market_time: null, score: 67.5 },
  E: { comp_value: 60, comp_tax: 80, comp_market_time: 0, score: 58.9 },
  F: { comp_value: 40, comp_tax: 40, comp_market_time: 50, score: 41.1 },
};

test("score matches Python (shrunk value signal, zero-variance component dropped)", () => {
  const m = by(recompute(fixture(), W, { tau2: 0.0025 }));
  for (const [k, e] of Object.entries(EXPECTED_SCORE)) {
    assert.equal(m[k].comp_price_drop, null, k);
    for (const f of ["comp_value", "comp_tax", "comp_market_time"]) {
      if (e[f] === null) assert.equal(m[k][f], null, `${k}.${f}`);
      else close(m[k][f], e[f]);
    }
    assert.equal(m[k].score, e.score, k);
  }
});
