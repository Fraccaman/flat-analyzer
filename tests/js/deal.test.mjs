// Run: node --test tests/js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { verdictFor, savingMonth, annotate, meetsVerdict, divergingBar, topDeals, median, summarize, normCdf, fairRange } from "../../web/deal.js";
import { recompute } from "../../web/score.js";

test("verdict thresholds", () => {
  const id = (p) => verdictFor(p, 0.95)?.id ?? null;
  assert.equal(id(-35), "great");
  assert.equal(id(-20), "great");
  assert.equal(id(-19.99), "good");
  assert.equal(id(-8), "good");
  assert.equal(id(-7.9), "fair");
  assert.equal(id(0), "fair");
  assert.equal(id(7.99), "fair");
  assert.equal(id(8), "pricey");
  assert.equal(id(20), "pricey");
  assert.equal(id(20.01), "over");
  assert.equal(id(300), "over");
  assert.equal(id(null), null);
  assert.equal(id(NaN), null);
});

test("great needs >= 80% chance below fair", () => {
  assert.equal(verdictFor(-30, 0.8).id, "great");
  assert.equal(verdictFor(-30, 0.79).id, "good");
  assert.equal(verdictFor(-30).id, "good"); // unknown confidence
  assert.equal(verdictFor(-10, 0.1).id, "good"); // only "great" is gated
});

test("normal CDF", () => {
  const close = (a, b, eps) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
  close(normCdf(0), 0.5, 1e-8);
  close(normCdf(1.959963985), 0.975, 2e-7);
  close(normCdf(-1), 0.158655254, 2e-7);
  close(normCdf(-1) + normCdf(1), 1, 1e-12);
  assert.equal(normCdf(null), null);
});

test("fair range", () => {
  const r = fairRange({ fair_rent: 2000, fair_sigma_log: 0.1 });
  assert.ok(Math.abs(r.lo - 2000 * Math.exp(-0.1)) < 1e-9 && Math.abs(r.hi - 2000 * Math.exp(0.1)) < 1e-9);
  assert.equal(fairRange({ fair_rent: 2000 }), null);
});

test("saving uses effective rent", () => {
  assert.equal(savingMonth({ fair_rent: 2000, effective_rent: 1700, rent_gross: 1800 }), 300);
  assert.equal(savingMonth({ fair_rent: 2000, effective_rent: null, rent_gross: 2230 }), -230);
  assert.equal(savingMonth({ fair_rent: null, effective_rent: 1700 }), null);
  assert.equal(savingMonth({ fair_rent: 0, effective_rent: 1700 }), null);
});

test("annotate follows tax recompute", () => {
  // shrink_k = 1 (no shrinkage) and a tight sigma keep the arithmetic readable
  const ls = [
    { key: "A", rent_gross: 1000, fair_rent: 1100, deal_pct: -9.09, tax_delta_month: 0, municipality_bfs: 1, fair_sigma_log: 0.05, shrink_k: 1 },
    { key: "B", rent_gross: 1000, fair_rent: 1000, deal_pct: 0, tax_delta_month: 0, municipality_bfs: 2, fair_sigma_log: 0.05, shrink_k: 1 },
  ];
  const W = { value: 1, tax: 1, price_drop: 0, market_time: 0 };
  annotate(recompute(ls, W));
  assert.equal(ls[0].verdict, "good");
  assert.equal(ls[0].saving_month, 100);
  assert.ok(Math.abs(ls[0].saving_adj_month - 100) < 1e-9);
  assert.equal(ls[1].verdict, "fair");
  // a much higher tax in A's municipality flips it to pricey
  annotate(recompute(ls, W, { taxDelta: (bfs) => (bfs === 1 ? 200 : 0) }));
  assert.equal(ls[0].effective_rent, 1200);
  assert.equal(ls[0].saving_month, -100);
  assert.equal(ls[0].verdict, "pricey");
  assert.ok(meetsVerdict(ls[1], "fair"));
  assert.ok(!meetsVerdict(ls[0], "fair"));
  assert.ok(meetsVerdict(ls[0], ""));
  assert.ok(!meetsVerdict({ verdict_rank: null }, "over"));
});

test("diverging bar geometry", () => {
  assert.deepEqual(divergingBar(-20), { left: 25, width: 25, side: "left", clamped: false });
  assert.deepEqual(divergingBar(40), { left: 50, width: 50, side: "right", clamped: false });
  assert.equal(divergingBar(-90).left, 0);
  assert.equal(divergingBar(-90).clamped, true);
  assert.equal(divergingBar(0).width, 0);
  assert.equal(divergingBar(null), null);
});

test("top deals prefer reliable estimates and skip suspicious", () => {
  const ls = [
    { key: "a", score: 90, saving_month: 500, model_confidence: "low" },
    { key: "b", score: 80, saving_month: 100, model_confidence: "high" },
    { key: "c", score: 99, saving_month: 900, suspicious: true },
    { key: "d", score: 70, saving_month: -10, model_confidence: "high" },
    { key: "e", score: 60, saving_month: 50, model_confidence: null },
    { key: "f", score: 95, saving_month: 300, saving_adj_month: 200, model_confidence: "high", listing_kind: "takeover" },
    { key: "g", score: 85, saving_month: 300, saving_adj_month: -5, model_confidence: "high" },
  ];
  // takeover ads rank after confident regular ones; the shrunk saving must be > 0
  assert.deepEqual(topDeals(ls, 6).map((l) => l.key), ["b", "e", "f", "a"]);
  assert.deepEqual(topDeals(ls, 1).map((l) => l.key), ["b"]);
});

test("median and summary", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(median([null]), null);
  const s = summarize([
    { verdict: "great", verdict_rank: 4, saving_month: 400, chf_per_m2: 20, model_confidence: "high" },
    { verdict: "good", verdict_rank: 3, saving_month: 900, chf_per_m2: 30, model_confidence: "low" },
    { verdict: "great", verdict_rank: 4, saving_month: 2000, chf_per_m2: 5, suspicious: true },
  ]);
  assert.deepEqual(s, { count: 3, great: 1, goodOrBetter: 2, bestSaving: 400, medianChfM2: 25 });
});
