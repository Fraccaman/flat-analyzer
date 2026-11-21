// Run: node --test tests/js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { progressive, federalTax, cantonBase, makeTaxCalculator, tablesUsable, multiplierPct } from "../../web/tax.js";
import { percentiles, computeComponent, weightedScore, recompute } from "../../web/score.js";

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test("progressive brackets", () => {
  const br = [[0, 0], [10000, 0.01], [30000, 0.02]];
  close(progressive(50000, br), 20000 * 0.01 + 20000 * 0.02); // 600
  close(progressive(5000, br), 0);
  close(progressive(20000, br), 100);
  close(progressive(-5, br), 0);
  close(progressive(10000, [[0, 0.1]]), 1000);
  close(progressive(10000, []), 0);
  // optional 3rd element = official tax at threshold, replaces running sum
  close(progressive(25000, [[0, 0.01], [20000, 0.05, 300]]), 300 + 5000 * 0.05);
  close(progressive(15000, [[0, 0.01], [20000, 0.05, 300]]), 150);
});

test("federal flat top and married splitting", () => {
  const fed = { income: { single: [[0, 0], [50000, 0.1]] }, flat_top: { single: { threshold: 100000, rate: 0.115 } } };
  close(federalTax(200000, fed, "single"), 23000);
  close(federalTax(100000, fed, "single"), 11500); // threshold inclusive
  close(federalTax(80000, fed, "single"), 3000);
  const canton = { income: { married: [[0, 0], [20000, 0.05]] }, wealth: {}, married_splitting_divisor: 2 };
  close(cantonBase(60000, 0, canton, "married").income, 1000);
  delete canton.married_splitting_divisor;
  close(cantonBase(60000, 0, canton, "married").income, 2000);
});

test("municipality delta", () => {
  const tables = {
    federal: { income: { single: [[0, 0.02]] } },
    cantons: {
      TI: {
        cantonal_coefficient: 1.0,
        income: { single: [[0, 0.1]] },
        wealth: { single: [[0, 0.001]] },
        wealth_exemption: { single: 100000 },
        wealth_exemption_type: "deduction",
        municipalities: [{ bfs: 1, name: "A", multiplier: 0.6 }, { bfs: 2, name: "B", multiplier: 0.8 }],
      },
    },
  };
  assert.ok(tablesUsable(tables));
  assert.ok(!tablesUsable({}));
  const calc = makeTaxCalculator(tables, { filing: "single", taxable_income: 100000, taxable_wealth: 300000, reference_bfs: 1 });
  // base = 10000 + 200 = 10200 ; federal = 2000
  close(calc.total(1), 2000 + 10200 * 1.6);
  close(calc.total(2), 2000 + 10200 * 1.8);
  close(calc.delta(2), 2040 / 12);
  close(calc.delta(1), 0);
  assert.equal(calc.delta(999), null);
  assert.equal(calc.delta(null), null);
  close(multiplierPct(0.65), 65);
  // default exemption type "threshold": below -> 0, at/above -> whole wealth taxed
  const c = tables.cantons.TI;
  delete c.wealth_exemption_type;
  close(cantonBase(0, 99999, c, "single").wealth, 0);
  close(cantonBase(0, 300000, c, "single").wealth, 300);
  // personal tax cancels in the delta but is part of the total
  c.personal_tax = { single: 40 };
  const calc2 = makeTaxCalculator(tables, { filing: "single", taxable_income: 100000, taxable_wealth: 300000, reference_bfs: 1 });
  close(calc2.total(1), 2000 + 10300 * 1.6 + 40);
  close(calc2.delta(2), (10300 * 0.2) / 12);
  close(multiplierPct(65), 65);
});

test("percentiles with average-rank ties", () => {
  assert.deepEqual(percentiles([10, 20, 20, 30]), [0, 50, 50, 100]);
  assert.deepEqual(percentiles([30, 10, 20]), [100, 0, 50]);
  assert.deepEqual(percentiles([5]), [50]);
  assert.deepEqual(percentiles([3, 3, 3]), [50, 50, 50]);
  assert.deepEqual(percentiles([]), []);
});

function sample() {
  return [
    { key: "A", rent_gross: 1000, fair_rent: 1100, deal_pct: -10, tax_delta_month: -50, price_drop_pct: 5, days_on_market: 10 },
    { key: "B", rent_gross: 1000, fair_rent: 1000, deal_pct: 0, tax_delta_month: 0, price_drop_pct: 0, days_on_market: 100 },
    { key: "C", rent_gross: 1100, fair_rent: 1000, deal_pct: 10, tax_delta_month: null, price_drop_pct: null, days_on_market: 30 },
    { key: "D", rent_gross: 100, fair_rent: 1000, deal_pct: -90, tax_delta_month: -500, price_drop_pct: 50, days_on_market: 80, suspicious: true },
    { key: "E", rent_gross: 1200, fair_rent: null, deal_pct: null, tax_delta_month: 50, price_drop_pct: 0, days_on_market: null },
  ];
}
const W = { value: 0.55, tax: 0.25, price_drop: 0.1, market_time: 0.1 };
const by = (ls) => Object.fromEntries(ls.map((l) => [l.key, l]));

test("components and weighted score", () => {
  const ls = recompute(sample(), W);
  const m = by(ls);
  // value: A10 B0 C-10, E null
  assert.deepEqual([m.A.comp_value, m.B.comp_value, m.C.comp_value, m.E.comp_value], [100, 50, 0, null]);
  // tax (null -> 0): E-50 B0 C0 A50
  assert.deepEqual([m.A.comp_tax, m.B.comp_tax, m.C.comp_tax, m.E.comp_tax], [100, 50, 50, 0]);
  // price drop (null -> 0): B,C,E tie at avg rank 2 of 4
  close(m.B.comp_price_drop, 100 / 3);
  close(m.E.comp_price_drop, 100 / 3);
  assert.equal(m.A.comp_price_drop, 100);
  // market time capped at 90: A10 C30 B90, E null
  assert.deepEqual([m.A.comp_market_time, m.C.comp_market_time, m.B.comp_market_time, m.E.comp_market_time], [0, 50, 100, null]);
  assert.equal(m.D.comp_value, null);
  assert.equal(m.A.score, 90);
  assert.equal(m.B.score, 53.3);
  assert.equal(m.C.score, 20.8);
  assert.equal(m.E.score, 9.5); // (0.25*0 + 0.1*33.33)/0.35, null comps drop out
  assert.equal(m.D.score, null);
  close(m.A.effective_rent, 950);
  close(m.A.effective_deal_pct, (950 / 1100 - 1) * 100, 1e-2);
  close(m.C.effective_rent, 1100);
  assert.equal(m.E.effective_deal_pct, null);
});

test("weights change and zero weights", () => {
  const m = by(recompute(sample(), { value: 0, tax: 1, price_drop: 0, market_time: 0 }));
  assert.deepEqual([m.A.score, m.B.score, m.C.score, m.E.score], [100, 50, 50, 0]);
  assert.equal(weightedScore({ comp_value: 10 }, { value: 0, tax: 0 }), null);
});

test("recompute with new tax deltas keeps precomputed other components", () => {
  const ls = sample();
  ls.forEach((l) => (l.comp_value = 42)); // pretend precomputed
  ls[3].comp_value = null;
  const deltas = { A: 30, B: -30, C: null, D: 0, E: 0 };
  ls.forEach((l) => (l.municipality_bfs = l.key));
  const m = by(recompute(ls, W, { taxDelta: (bfs) => deltas[bfs] }));
  assert.equal(m.A.comp_value, 42);
  assert.equal(m.A.tax_delta_month, 30);
  // tax raw: A-30 B30 C0 E0 -> A0, C/E 50, B100
  assert.deepEqual([m.A.comp_tax, m.B.comp_tax, m.C.comp_tax, m.E.comp_tax], [0, 100, 50, 50]);
  close(m.B.effective_rent, 970);
});

test("computeComponent: no variance -> null (drops out of the score)", () => {
  const ls = [{ deal_pct: 3 }];
  computeComponent(ls, "value");
  assert.equal(ls[0].comp_value, null);
  const same = [{ days_on_market: 0 }, { days_on_market: 0 }, { days_on_market: 0, suspicious: true }];
  computeComponent(same, "market_time");
  assert.deepEqual(same.map((l) => l.comp_market_time), [null, null, null]);
  // the other weights are renormalised
  assert.equal(weightedScore({ comp_value: 80, comp_market_time: null }, { value: 0.5, market_time: 0.5 }), 80);
});
