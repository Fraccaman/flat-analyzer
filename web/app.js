import { makeTaxCalculator, tablesUsable, multiplierPct } from "./tax.js";
import { COMPONENTS, recompute } from "./score.js";
import { VERDICTS, VERDICT_BY_ID, KINDS, GREAT_MIN_P, annotate, meetsVerdict, divergingBar, topDeals, summarize, fairRange } from "./deal.js";

/* global L */
const $ = (sel) => document.querySelector(sel);
const STORE_KEY = "flatprice.settings.v1";
const THEME_KEY = "flatprice.theme";
const ROW_CAP_STEP = 500;
const WEIGHT_LABELS = { value: "Value", tax: "Tax", price_drop: "Price drop", market_time: "Market time" };

const state = {
  data: null,
  listings: [],
  byKey: new Map(),
  defaults: null,
  profile: null,
  weights: null,
  calc: null,
  sort: { key: "score", dir: "desc" },
  cap: ROW_CAP_STEP,
  excludedMunis: new Set(),
  excludedPortals: new Set(),
  activeKey: null,
  visible: [], // filtered + sorted (table)
};

/* ---------- formatting ---------- */
const nf0 = new Intl.NumberFormat("de-CH", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("de-CH", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const safeUrl = (u) => (typeof u === "string" && /^https?:\/\//i.test(u) ? u : null);
const chf = (v) => (isNum(v) ? nf0.format(v) : "–");
const signed = (v, digits = 0, suffix = "") => {
  if (!isNum(v)) return "–";
  const s = digits ? nf1.format(Math.abs(v)) : nf0.format(Math.abs(v));
  if (Math.abs(v) < (digits ? 0.05 : 0.5)) return "0" + suffix;
  return (v > 0 ? "+" : "−") + s + suffix;
};
const signClass = (v) => (isNum(v) && Math.abs(v) >= 0.5 ? (v < 0 ? "neg" : "pos") : "");
const fmtDate = (iso) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
};
const placeLabel = (l) => [l.municipality || l.locality, l.area && l.area !== l.municipality ? l.area : null].filter(Boolean).join(" · ") || "–";
const muniName = (l) => l.municipality || l.locality || "Unknown";
const lowConf = (l) => l.model_confidence === "low";

/* ---------- verdict presentation ---------- */
// Marker fills (canvas needs literal colours; readable on light and dimmed dark tiles)
const MARKER = { great: "#0b8a5f", good: "#55bf95", fair: "#9aa3ad", pricey: "#f0a04b", over: "#d64a3c", none: "#c3c7cc" };
const vClass = (l) => "v-" + (l.verdict || "none");
const dealPctText = (l) => (l.data_suspect ? "–" : signed(l.deal_pct_adj_eff, 1, "%"));
const pctText = (p) => (!isNum(p) ? "–" : p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : `${nf0.format(Math.round(p * 100))}%`);
const savingOf = (l) => (isNum(l.saving_adj_month) ? l.saving_adj_month : l.saving_month);
function fairPm(l) {
  // half-width of the model range fair * exp(+-sigma), shown as "± CHF"
  const r = fairRange(l);
  return r ? `<span class="pm" title="Model range CHF ${chf(r.lo)}–${chf(r.hi)}">±${chf((r.hi - r.lo) / 2)}</span>` : "";
}
function kindBadge(l) {
  const k = KINDS[l.listing_kind];
  return k ? `<span class="kind" title="${esc(k.title)}">${k.label}</span>` : "";
}
const compsText = (l) => (isNum(l.n_comps) && l.n_comps > 0 && l.fair_method === "blend" ? `${l.n_comps} comps` : null);
const EST_TITLE = "Low model confidence: few comparable flats or missing size, so the fair rent is a rough estimate";
const estMark = (l) => (lowConf(l) ? `<abbr class="est" title="${EST_TITLE}" aria-label="estimate">≈</abbr>` : "");
const suspFlag = `<span class="flag" title="Implausible data (e.g. price far below market). Not scored.">⚠ Too good to be true?</span>`;
// data_suspect: the portal served values that look manipulated (flatprice/quality.py) and
// there is no trusted copy to fall back on. Such rows are also `suspicious` (not scored).
const DATA_TITLE = "The portal served values that look manipulated (anti-bot perturbation) and no trusted copy was found. Not scored.";
const dataFlag = `<span class="flag" title="${DATA_TITLE}">⚠ data may be wrong</span>`;
const flagOf = (l) => (l.data_suspect ? dataFlag : suspFlag);
const MANIP = "price looks manipulated";

/* ---------- portals (cross-portal duplicates, see flatprice/dedup.py) ---------- */
// Every ad of the flat: l.sources from data.json, or the listing itself (older builds).
function sourcesOf(l) {
  if (l._sources) return l._sources;
  const src = Array.isArray(l.sources) && l.sources.length ? l.sources : [{ portal: l.portal, url: l.url, rent_gross: l.rent_gross, key: l.key }];
  l._sources = src.filter((s) => s && s.portal);
  return l._sources;
}
const portalName = (p) => String(p || "ad").toLowerCase();
// {portal, saving} when another portal lists the same flat for less than the cheapest other one.
// Prices flagged as manipulated (s.flag) are ignored.
function cheaperOn(l) {
  const priced = sourcesOf(l).filter((s) => isNum(s.rent_gross) && !s.flag);
  if (priced.length < 2) return null;
  const sorted = [...priced].sort((a, b) => a.rent_gross - b.rent_gross);
  const lo = sorted[0];
  const hi = sorted[sorted.length - 1];
  if (hi.rent_gross - lo.rent_gross < 0.5 || portalName(lo.portal) === portalName(hi.portal)) return null;
  return { portal: portalName(lo.portal), saving: hi.rent_gross - lo.rent_gross, rent: lo.rent_gross };
}
function sourceBadges(l) {
  const cheap = cheaperOn(l);
  return sourcesOf(l)
    .map((s) => {
      const url = safeUrl(s.url);
      const p = portalName(s.portal);
      const isCheap = cheap && p === cheap.portal && s.rent_gross === cheap.rent;
      const title = `Open the ad on ${p}${isNum(s.rent_gross) ? ` · CHF ${chf(s.rent_gross)}/mo` : ""}${isCheap ? ` · cheapest (−CHF ${chf(cheap.saving)})` : ""}${s.flag ? ` · ${MANIP}` : ""}`;
      const cls = `src src-${esc(p.replace(/[^a-z0-9]/g, ""))}${isCheap ? " cheap" : ""}`;
      const label = `${esc(p)}${isCheap ? " ↓" : ""}`;
      return url ? `<a class="${cls}" href="${esc(url)}" target="_blank" rel="noopener" title="${esc(title)}">${label}</a>` : `<span class="${cls}" title="${esc(title)}">${label}</span>`;
    })
    .join("");
}

function badge(l) {
  // the verdict of a row with possibly manipulated rooms/m² would be meaningless
  if (l.data_suspect) return `<span class="badge v-none" title="${DATA_TITLE}">Unverified</span>`;
  const v = l.verdict ? VERDICT_BY_ID[l.verdict] : null;
  return v ? `<span class="badge v-${v.id}">${v.label}</span>` : `<span class="badge v-none">No estimate</span>`;
}
function savingText(l, { year = false } = {}) {
  if (l.data_suspect) return { amt: "–", sub: "data may be wrong" };
  const s = savingOf(l);
  if (!isNum(s)) return { amt: "–", sub: "no fair-rent estimate" };
  const yr = year ? ` · CHF ${chf(Math.abs(s) * 12)}/yr` : "";
  if (Math.abs(s) < 0.5) return { amt: "CHF 0", sub: "/mo, at fair rent" + yr };
  return { amt: `CHF ${chf(Math.abs(s))}`, sub: (s > 0 ? "/mo below fair" : "/mo above fair") + yr };
}
function savingTitle(l) {
  const s = savingOf(l);
  if (!isNum(s)) return "";
  const raw = l.saving_month;
  const r = fairRange(l);
  return [
    `Likely saving CHF ${chf(Math.abs(s))}/mo ${s >= 0 ? "below" : "above"} fair rent (CHF ${chf(Math.abs(s) * 12)}/yr, incl. your tax Δ), shrunk for model uncertainty`,
    isNum(raw) ? `Raw: CHF ${chf(Math.abs(raw))}/mo ${raw >= 0 ? "below" : "above"} the fair-rent point estimate` : null,
    r ? `Fair rent model range: CHF ${chf(r.lo)}–${chf(r.hi)}` : null,
    compsText(l) ? `Based on the model + ${l.n_comps} comparable listings` : "Based on the hedonic model only",
  ]
    .filter(Boolean)
    .join("\n");
}
function dealBar(l) {
  const g = divergingBar(l.deal_pct_adj_eff);
  if (!g) return `<span class="dbar" aria-hidden="true"><span class="mid"></span></span>`;
  const label = `${nf0.format(Math.abs(l.deal_pct_adj_eff))}% ${l.deal_pct_adj_eff < 0 ? "below" : "above"} fair rent`;
  return (
    `<span class="dbar" role="img" aria-label="${label}" title="${label}">` +
    `<span class="fill v-${l.verdict} ${g.side}${g.clamped ? " clamped" : ""}" style="left:${g.left}%;width:${g.width}%"></span>` +
    `<span class="mid"></span></span>`
  );
}
function scoreRing(l) {
  if (l.suspicious || !isNum(l.score)) return `<span class="score-ring none" title="Not scored"><span>–</span></span>`;
  return `<span class="score-ring" style="--p:${l.score}" title="Deal score ${nf1.format(l.score)} / 100"><span>${nf0.format(l.score)}</span></span>`;
}
const HOUSE_SVG =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 11.5 12 4l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>';
function photo(l, cls, extra = "") {
  const img = safeUrl(l.image_url);
  if (!img) return `<span class="ph empty ${cls} ${vClass(l)}">${HOUSE_SVG}${extra}</span>`;
  return `<span class="ph ${cls} ${vClass(l)}"><img src="${esc(img)}" alt="" loading="lazy" decoding="async" referrerpolicy="no-referrer">${extra}</span>`;
}
// Broken images fall back to the placeholder (event delegation; error doesn't bubble)
document.addEventListener(
  "error",
  (e) => {
    const t = e.target;
    if (t instanceof HTMLImageElement && t.parentElement?.classList.contains("ph")) {
      t.parentElement.classList.add("empty");
      t.insertAdjacentHTML("beforebegin", HOUSE_SVG);
      t.remove();
    }
  },
  true,
);

/* ---------- settings persistence ---------- */
function loadStored() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) || "null");
  } catch {
    return null;
  }
}
function saveStored() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ profile: state.profile, weights: state.weights }));
  } catch {
    /* storage unavailable */
  }
}
function parseHash() {
  const out = {};
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const p = params.get("p");
  if (p) {
    const [filing, income, wealth, ref] = p.split(",");
    out.profile = {};
    if (filing === "single" || filing === "married") out.profile.filing = filing;
    if (isFinite(+income) && income !== "") out.profile.taxable_income = +income;
    if (isFinite(+wealth) && wealth !== "") out.profile.taxable_wealth = +wealth;
    if (isFinite(+ref) && ref !== "") out.profile.reference_bfs = +ref;
  }
  const w = params.get("w");
  if (w) {
    const vals = w.split(",").map(Number);
    out.weights = {};
    COMPONENTS.forEach((k, i) => {
      if (isFinite(vals[i]) && vals[i] >= 0) out.weights[k] = vals[i];
    });
  }
  return out;
}
function writeHash() {
  const p = state.profile;
  const hash = `p=${p.filing},${p.taxable_income},${p.taxable_wealth},${p.reference_bfs}&w=${COMPONENTS.map((k) => state.weights[k]).join(",")}`;
  history.replaceState(null, "", "#" + hash);
}
const sameProfile = (a, b) =>
  a.filing === b.filing && +a.taxable_income === +b.taxable_income && +a.taxable_wealth === +b.taxable_wealth && +a.reference_bfs === +b.reference_bfs;

/* ---------- scoring ---------- */
function rescore() {
  const tablesOk = tablesUsable(state.data.tax, state.profile.canton || "TI");
  const useDefaultTax = sameProfile(state.profile, state.defaults.tax_profile) || !tablesOk;
  state.calc = tablesOk ? makeTaxCalculator(state.data.tax, state.profile) : null;
  if (useDefaultTax) for (const l of state.listings) l.tax_delta_month = l._tdm0;
  const tau2 = state.data.scoring?.tau2;
  recompute(state.listings, state.weights, useDefaultTax ? { tau2 } : { taxDelta: state.calc.delta, tau2 });
  annotate(state.listings); // verdict + saving follow effective rent
  $("#tax-note").textContent = tablesOk
    ? "Tax Δ = your annual income + wealth tax in that municipality minus the reference one, per month. Effective rent = rent + tax Δ; verdicts and savings use it."
    : "Tax tables unavailable in this build — showing precomputed tax deltas only.";
}

/* ---------- filters ---------- */
function readFilters() {
  const f = new FormData($("#filters"));
  const num = (k) => (f.get(k) === "" || f.get(k) == null ? null : Number(f.get(k)));
  return {
    q: String(f.get("q") || "").trim().toLowerCase(),
    verdict_min: String(f.get("verdict_min") || ""),
    rent_min: num("rent_min"),
    rent_max: num("rent_max"),
    rooms_min: num("rooms_min"),
    m2_min: num("m2_min"),
    furnished: f.get("furnished"),
    hide_suspicious: f.get("hide_suspicious") === "on",
    in_view: f.get("in_view") === "on",
  };
}
function passes(l, f) {
  // data_suspect rows stay visible (with their "data may be wrong" badge)
  if (f.hide_suspicious && l.suspicious && !l.data_suspect) return false;
  if (!meetsVerdict(l, f.verdict_min)) return false;
  if (f.rent_min != null && !(l.rent_gross >= f.rent_min)) return false;
  if (f.rent_max != null && !(l.rent_gross <= f.rent_max)) return false;
  if (f.rooms_min != null && !(l.rooms >= f.rooms_min)) return false;
  if (f.m2_min != null && !(l.living_space_m2 >= f.m2_min)) return false;
  if (f.furnished === "yes" && !l.is_furnished) return false;
  if (f.furnished === "no" && l.is_furnished) return false;
  if (state.excludedMunis.has(muniName(l))) return false;
  if (state.excludedPortals.size && sourcesOf(l).every((s) => state.excludedPortals.has(portalName(s.portal)))) return false;
  if (f.q && !l._search.includes(f.q)) return false;
  return true;
}

/* ---------- sorting ---------- */
const SORTS = {
  score: { label: "Best score", dir: "desc" },
  saving_adj_month: { label: "Biggest saving (CHF/mo)", dir: "desc" },
  deal_pct_adj_eff: { label: "Best deal %", dir: "asc" },
  p_below_fair: { label: "Most likely below fair", dir: "desc" },
  rent_gross: { label: "Lowest rent", dir: "asc" },
  effective_rent: { label: "Lowest effective rent", dir: "asc" },
  rooms: { label: "Most rooms", dir: "desc" },
  living_space_m2: { label: "Largest m²", dir: "desc" },
  chf_per_m2: { label: "Lowest CHF/m²", dir: "asc" },
  tax_delta_month: { label: "Lowest tax Δ", dir: "asc" },
  days_on_market: { label: "Longest listed", dir: "desc" },
  price_drop_pct: { label: "Biggest price drop", dir: "desc" },
  location: { label: "Municipality A–Z", dir: "asc", get: (l) => placeLabel(l).toLowerCase() },
  portal: { label: "Portal", dir: "asc", get: (l) => (l.portal || "").toLowerCase() },
};

function sortList(list) {
  const s = SORTS[state.sort.key] || SORTS.score;
  const get = s.get || ((l) => l[state.sort.key]);
  const dir = state.sort.dir === "asc" ? 1 : -1;
  const keyed = list.map((l) => [l, get(l)]);
  keyed.sort((a, b) => {
    const va = a[1];
    const vb = b[1];
    const na = va == null || (typeof va === "number" && !Number.isFinite(va));
    const nb = vb == null || (typeof vb === "number" && !Number.isFinite(vb));
    if (na || nb) return na === nb ? 0 : na ? 1 : -1; // nulls (incl. suspicious score) last
    return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
  });
  return keyed.map((k) => k[0]);
}
function setSort(key, toggle = false) {
  if (!SORTS[key]) return;
  if (toggle && state.sort.key === key) state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
  else state.sort = { key, dir: SORTS[key].dir };
  $("#sort-select").value = key;
  buildHeader();
  renderTable();
}

/* ---------- table ---------- */
const COLUMNS = [
  { key: "photo", label: "Photo", sr: true, cls: "c-photo", fmt: (l) => photo(l, "thumb") },
  {
    key: "deal_pct_adj_eff",
    label: "Deal",
    title: "Effective rent (rent + your tax Δ) vs model fair rent, shrunk towards 0 by the model's uncertainty. Negative = cheaper.",
    cls: "txt c-deal",
    fmt: (l) =>
      `<div class="deal-top">${badge(l)}<span class="deal-pct v-${l.verdict || "none"}-t">${estMark(l)} ${dealPctText(l)}</span></div>` +
      dealBar(l) +
      (isNum(l.p_below_fair) && !l.data_suspect ? `<span class="sub" title="Chance the rent is below the true fair rent, given the model's uncertainty">${pctText(l.p_below_fair)} chance below fair${compsText(l) ? " · " + compsText(l) : ""}</span>` : "") +
      (l.suspicious || kindBadge(l) ? `<div class="deal-flags">${kindBadge(l)}${l.suspicious ? flagOf(l) : ""}</div>` : ""),
  },
  {
    key: "saving_adj_month",
    label: "Saving / mo",
    title: "Likely saving vs fair rent, CHF per month (effective rent, shrunk for model uncertainty; positive = you pay less than fair)",
    cls: "c-saving",
    fmt: (l) => {
      const t = savingText(l);
      return `<span class="amt v-${l.verdict || "none"}-t">${t.amt}</span><span class="sub">${t.sub}</span>`;
    },
    tdTitle: savingTitle,
  },
  {
    key: "rent_gross",
    label: "Rent",
    title: "Gross monthly rent; model fair rent ± its typical error below",
    cls: "c-rent",
    fmt: (l) => `<span class="amt">${chf(l.rent_gross)}</span><span class="sub">fair ${lowConf(l) ? "≈ " : ""}${chf(l.fair_rent)} ${fairPm(l)}</span>`,
  },
  { key: "score", label: "Score", title: "Deal score 0–100 (value, tax, price drop, market time)", cls: "c-score", fmt: scoreRing },
  {
    key: "rooms",
    label: "Size",
    title: "Rooms; m² and CHF/m² below",
    cls: "c-size",
    fmt: (l) =>
      `${isNum(l.rooms) ? l.rooms + " rm" : "–"}<span class="sub">${isNum(l.living_space_m2) ? chf(l.living_space_m2) + " m²" : "– m²"}${isNum(l.chf_per_m2) ? " · " + nf1.format(l.chf_per_m2) + "/m²" : ""}</span>`,
  },
  {
    key: "tax_delta_month",
    label: "Tax Δ / mo",
    title: "Your monthly tax vs the reference municipality; effective rent below",
    cls: "c-tax",
    fmt: (l) => `<span class="${signClass(l.tax_delta_month)}">${signed(l.tax_delta_month)}</span><span class="sub">eff. ${chf(l.effective_rent)}</span>`,
  },
  {
    key: "location",
    label: "Place",
    cls: "txt c-place",
    fmt: (l) => {
      const url = safeUrl(l.url);
      const bits = [
        isNum(l.rooms) ? `${l.rooms} rooms` : null,
        isNum(l.living_space_m2) ? `${chf(l.living_space_m2)} m²` : null,
        isNum(l.tax_delta_month) && Math.abs(l.tax_delta_month) >= 0.5 ? `tax ${signed(l.tax_delta_month)}/mo` : null,
      ].filter(Boolean);
      return (
        `<span class="t">${esc(placeLabel(l))}</span><span class="s">${esc(l.title || l.street || "")}</span>` +
        `<span class="m s">${bits.join(" · ")}${url ? ` <span class="srcs">${sourceBadges(l)}</span>` : ""}</span>`
      );
    },
    tdTitle: (l) => `${placeLabel(l)} — ${l.title || ""}`,
  },
  {
    key: "days_on_market",
    label: "Listed",
    title: "Days listed; rent reduction since first seen below",
    cls: "c-listed",
    fmt: (l) =>
      `${isNum(l.days_on_market) ? nf0.format(l.days_on_market) + " d" : "–"}` +
      (isNum(l.price_drop_pct) && l.price_drop_pct > 0 ? `<span class="sub neg">−${nf1.format(l.price_drop_pct)}%</span>` : ""),
  },
  {
    key: "portal",
    label: "Ads",
    title: "Portals listing this flat (the same flat on several portals is shown once); ↓ = cheapest",
    cls: "txt c-ad",
    fmt: (l) => `<span class="srcs">${sourceBadges(l)}</span>`,
  },
];

function buildHeader() {
  const tr = $("#thead-row");
  tr.textContent = "";
  for (const c of COLUMNS) {
    const th = document.createElement("th");
    th.scope = "col";
    if (c.cls) th.className = c.cls;
    if (!SORTS[c.key]) {
      th.innerHTML = c.sr ? `<span class="sr-only">${esc(c.label)}</span>` : `<button type="button" tabindex="-1" style="cursor:default">${esc(c.label)}</button>`;
    } else {
      th.innerHTML = `<button type="button" data-sort="${c.key}"${c.title ? ` title="${esc(c.title)}"` : ""}>${esc(c.label)}</button>`;
      if (state.sort.key === c.key) th.setAttribute("aria-sort", state.sort.dir === "asc" ? "ascending" : "descending");
    }
    tr.appendChild(th);
  }
}
function buildSortSelect() {
  $("#sort-select").innerHTML = Object.entries(SORTS)
    .map(([k, s]) => `<option value="${k}">${esc(s.label)}</option>`)
    .join("");
  $("#sort-select").value = state.sort.key;
}

function rowHtml(l) {
  const cls = [vClass(l), l.suspicious ? "suspicious" : "", l.key === state.activeKey ? "active" : ""].filter(Boolean).join(" ");
  return (
    `<tr data-key="${esc(l.key)}" class="${cls}">` +
    COLUMNS.map((c) => {
      const t = c.tdTitle ? c.tdTitle(l) : "";
      return `<td class="${c.cls || ""}"${t ? ` title="${esc(t)}"` : ""}>${c.fmt(l)}</td>`;
    }).join("") +
    "</tr>"
  );
}

function renderTable() {
  const f = readFilters();
  const bounds = f.in_view && map ? map.getBounds() : null;
  let list = state.listings.filter((l) => passes(l, f));
  updateMarkers(list);
  if (bounds) list = list.filter((l) => l._latlng && bounds.contains(l._latlng));
  state.visible = sortList(list);
  const rows = state.visible.slice(0, state.cap);
  const tpl = document.createElement("template");
  tpl.innerHTML = rows.map(rowHtml).join("");
  const tbody = $("#tbody");
  tbody.textContent = "";
  tbody.appendChild(tpl.content);
  const more = $("#more");
  more.hidden = state.visible.length <= state.cap;
  more.textContent = `Show more (${nf0.format(state.visible.length - state.cap)} remaining)`;
  $("#count").textContent = `${nf0.format(state.visible.length)} of ${nf0.format(state.listings.length)} listings`;
  renderStats(state.visible);
  renderTopDeals(state.visible);
}

/* ---------- header stats & top deals ---------- */
function renderStats(list) {
  const s = summarize(list);
  $("#st-count").textContent = nf0.format(s.count);
  $("#st-great").innerHTML = `${nf0.format(s.great)} <small>· ${nf0.format(s.goodOrBetter)} good+</small>`;
  $("#st-best").innerHTML = isNum(s.bestSaving) && s.bestSaving > 0 ? `CHF ${chf(s.bestSaving)}<small>/mo</small>` : "–";
  $("#st-best").parentElement.title = "Largest monthly saving vs fair rent among non-suspicious listings with a confident estimate";
  $("#st-m2").textContent = isNum(s.medianChfM2) ? nf1.format(s.medianChfM2) : "–";
}

function cardHtml(l) {
  const t = savingText(l);
  const size = [isNum(l.rooms) ? `${l.rooms} rooms` : null, isNum(l.living_space_m2) ? `${chf(l.living_space_m2)} m²` : null].filter(Boolean).join(" · ");
  const v = l.verdict ? VERDICT_BY_ID[l.verdict].label : "No estimate";
  return (
    `<button type="button" class="card${l.key === state.activeKey ? " active" : ""}" data-key="${esc(l.key)}" ` +
    `aria-label="${esc(`${v}, ${t.amt} ${t.sub}, rent CHF ${chf(l.rent_gross)}, ${placeLabel(l)}. Show on map`)}">` +
    photo(l, "", badge(l) + scoreRing(l)) +
    `<span class="body">` +
    `<span class="save v-${l.verdict || "none"}-t" title="${esc(savingTitle(l))}">${t.amt}<small>${t.sub}</small></span>` +
    `<span class="rent"><b>CHF ${chf(l.rent_gross)}</b>/mo · fair ${estMark(l)} ${chf(l.fair_rent)} ${fairPm(l)}</span>` +
    `<span class="rent">${esc(size || "size n/a")}</span>` +
    `<span class="where">${esc(placeLabel(l))}${kindBadge(l) ? " " + kindBadge(l) : ""}${l.data_suspect ? " " + dataFlag : ""}</span>` +
    `</span></button>`
  );
}
function renderTopDeals(list) {
  const top = topDeals(list, 6);
  $("#top-deals").hidden = !top.length;
  $("#cards").innerHTML = top.map(cardHtml).join("");
}

/* ---------- map ---------- */
let map = null;
let markerLayer = null;
let muniLayer = null;
let renderer = null;
let activeMarker = null;

function initMap() {
  if (typeof L === "undefined") {
    $("#map").innerHTML = '<p class="hint" style="padding:16px">Map library failed to load.</p>';
    return;
  }
  map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([46.0, 8.95], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  map.createPane("munis").style.zIndex = 350;
  renderer = L.canvas({ padding: 0.5 });
  markerLayer = L.layerGroup().addTo(map);
  const pts = [];
  for (const l of state.listings) {
    if (!isNum(l.latitude) || !isNum(l.longitude)) continue;
    l._latlng = L.latLng(l.latitude, l.longitude);
    pts.push(l._latlng);
    const m = L.circleMarker(l._latlng, { renderer, radius: 6, weight: 1, color: "#ffffff", fillOpacity: 0.92 });
    m.bindPopup(() => popupHtml(l), { maxWidth: 300, minWidth: 280, autoPanPadding: [20, 20] });
    m.bindTooltip(() => `${esc(l.verdict ? VERDICT_BY_ID[l.verdict].label : "No estimate")} · CHF ${chf(l.rent_gross)}`, { direction: "top", offset: [0, -6] });
    m.on("click", () => selectListing(l.key, { fromMap: true }));
    m.on("popupclose", () => {
      if (activeMarker === m) styleMarker(l, false);
    });
    l._marker = m;
  }
  if (pts.length) map.fitBounds(L.latLngBounds(pts), { padding: [20, 20], maxZoom: 15 });
  let t = null;
  map.on("moveend", () => {
    if (!$("#filters").elements.in_view.checked) return;
    clearTimeout(t);
    t = setTimeout(renderTable, 100);
  });
  renderLegend();
}

function markerRadius(l) {
  const s = isNum(savingOf(l)) ? savingOf(l) : 0;
  return 5 + Math.max(0, Math.min(5, s / 150)); // 5px .. 10px at >= CHF 750/mo saving
}
function styleMarker(l, active) {
  if (!l._marker) return;
  const fill = l.suspicious ? MARKER.none : MARKER[l.verdict || "none"];
  const r = markerRadius(l);
  l._marker.setStyle({
    fillColor: fill,
    color: active ? "#1d4ed8" : "rgba(255,255,255,.95)",
    weight: active ? 3.5 : 1.2,
    radius: active ? r + 4 : r,
  });
  if (active) l._marker.bringToFront();
}

function updateMarkers(filtered) {
  if (!markerLayer) return;
  const show = new Set(filtered.map((l) => l.key));
  for (const l of state.listings) {
    if (!l._marker) continue;
    styleMarker(l, l.key === state.activeKey);
    const has = markerLayer.hasLayer(l._marker);
    if (show.has(l.key) && !has) markerLayer.addLayer(l._marker);
    else if (!show.has(l.key) && has) markerLayer.removeLayer(l._marker);
  }
  // draw the best deals on top
  filtered
    .filter((l) => l._marker && !l.suspicious && (l.verdict === "great" || l.verdict === "good"))
    .sort((a, b) => (savingOf(a) ?? 0) - (savingOf(b) ?? 0))
    .forEach((l) => l._marker.bringToFront());
  const active = state.activeKey && state.byKey.get(state.activeKey);
  if (active && active._marker && show.has(active.key)) active._marker.bringToFront();
}

function renderLegend() {
  const range = { great: `≤ −20%, ≥${GREAT_MIN_P * 100}% sure`, good: "−20 … −8%", fair: "±8%", pricey: "+8 … +20%", over: "> +20%" };
  $("#legend").innerHTML =
    `<div class="lt">Deal vs fair rent</div><ul>` +
    VERDICTS.map((v) => `<li><span class="dot" style="background:${MARKER[v.id]}"></span>${v.label} <span class="hint">${range[v.id]}</span></li>`).join("") +
    `<li><span class="dot" style="background:${MARKER.none}"></span>No estimate / flagged</li></ul>` +
    `<div class="lh">Deal % shrunk for model uncertainty · bigger dot = bigger saving</div>`;
}

const TAX_LO = [219, 234, 254];
const TAX_HI = [30, 64, 175];
function taxColor(t) {
  const c = TAX_LO.map((x, k) => Math.round(x + (TAX_HI[k] - x) * Math.max(0, Math.min(1, t))));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function muniTooltip(feature) {
  const p = feature.properties || {};
  const row = state.calc ? state.calc.municipalities.get(Number(p.bfs)) : taxMuniIndex().get(Number(p.bfs));
  let s = `<strong>${esc(p.name || row?.name || p.bfs)}</strong>`;
  if (row) {
    s += ` — tax multiplier ${nf0.format(multiplierPct(row.multiplier))}%`;
    if (state.calc) {
      const d = state.calc.delta(p.bfs);
      s += `<br>Your tax Δ/month vs reference: ${isNum(d) ? signed(d) + " CHF" : "–"}`;
    }
  }
  return s;
}
function taxMuniIndex() {
  const c = state.data.tax?.cantons?.[state.profile.canton || "TI"];
  return new Map(((c && c.municipalities) || []).map((r) => [Number(r.bfs), r]));
}
async function loadMunicipalities() {
  if (!map) return;
  let gj;
  try {
    const r = await fetch("municipalities.geojson");
    if (!r.ok) return;
    gj = await r.json();
  } catch {
    return;
  }
  const idx = taxMuniIndex();
  const mults = [...idx.values()].map((r) => Number(r.multiplier)).filter(isNum);
  const lo = Math.min(...mults);
  const hi = Math.max(...mults);
  muniLayer = L.geoJSON(gj, {
    pane: "munis",
    style: (f) => {
      const row = idx.get(Number(f.properties?.bfs));
      const t = row && hi > lo ? (Number(row.multiplier) - lo) / (hi - lo) : 0.5;
      return { color: "#64748b", weight: 0.7, fillColor: row ? taxColor(t) : "#9aa0a6", fillOpacity: row ? 0.2 : 0.05 };
    },
    onEachFeature: (f, layer) => layer.bindTooltip(() => muniTooltip(f), { sticky: true }),
  });
  if ($("#filters").elements.show_munis.checked) muniLayer.addTo(map);
}

function popupHtml(l) {
  const url = safeUrl(l.url);
  const t = savingText(l, { year: true });
  const r = fairRange(l);
  const vals = [l.rent_gross, l.effective_rent, l.fair_rent, r?.hi].filter(isNum);
  const max = vals.length ? Math.max(...vals) : 1;
  const pc = (v) => (v / max) * 100;
  const cmpRow = (label, v, color, extra = "", band = "") =>
    `<div class="r"><span class="l">${label}</span><span class="track"><span style="width:${isNum(v) ? Math.max(2, pc(v)) : 0}%;background:${color}"></span>${band}</span><span class="v">${chf(v)}${extra}</span></div>`;
  const band = r ? `<span class="band" title="Model range CHF ${chf(r.lo)}–${chf(r.hi)}" style="left:${pc(r.lo)}%;width:${pc(r.hi) - pc(r.lo)}%"></span>` : "";
  const vColor = l.verdict ? `var(--${l.verdict})` : "var(--faint)";
  const compLinks = (l.comps || [])
    .filter((k) => state.byKey.has(k))
    .map((k, i) => `<a href="#" data-comp="${esc(k)}" title="${esc(state.byKey.get(k).title || k)}">${i + 1}</a>`)
    .join(" ");
  const kv = [
    ["Deal (eff., adjusted)", `${estMark(l)} ${dealPctText(l)}`],
    ["Deal (eff., raw)", signed(l.effective_deal_pct, 1, "%")],
    ["Deal (rent only, raw)", signed(l.deal_pct, 1, "%")],
    ["Chance below fair", pctText(l.p_below_fair)],
    ["Fair rent range", r ? `${chf(r.lo)}–${chf(r.hi)}` : "–"],
    ["Comparables", compsText(l) ? `${l.n_comps}${compLinks ? ` <span class="comps">${compLinks}</span>` : ""}` : "model only"],
    ["Tax Δ / month", isNum(l.tax_delta_month) ? `${signed(l.tax_delta_month)} CHF` : "–"],
    ["Rooms · m²", `${isNum(l.rooms) ? l.rooms : "–"} · ${chf(l.living_space_m2)}`],
    ["CHF/m²", isNum(l.chf_per_m2) ? nf1.format(l.chf_per_m2) : "–"],
    ["Where", esc(placeLabel(l))],
    ["Days listed", isNum(l.days_on_market) ? nf0.format(l.days_on_market) : "–"],
  ];
  if (isNum(l.price_drop_pct) && l.price_drop_pct > 0) kv.push(["Price drop", `−${nf1.format(l.price_drop_pct)}%`]);
  const bars = l.data_suspect
    ? `<p class="flag" style="margin:8px 0;white-space:normal">⚠ Data may be wrong: the portal served values that look manipulated — not scored.</p>`
    : l.suspicious
    ? `<p class="flag" style="margin:8px 0">⚠ Too good to be true? Flagged as suspicious — not scored.</p>`
    : `<div class="sh"><span>Score breakdown</span><span>${isNum(l.score) ? nf1.format(l.score) : "–"} / 100</span></div><div class="bars">` +
      COMPONENTS.map((k) => {
        const v = l["comp_" + k];
        const w = isNum(v) ? Math.max(0, Math.min(100, v)) : 0;
        return `<span>${WEIGHT_LABELS[k]}</span><span class="bar"><span style="width:${w}%"></span></span><span class="v">${isNum(v) ? nf0.format(v) : "–"}</span>`;
      }).join("") +
      `</div>`;
  return (
    `<div class="popup">` +
    photo(l, "") +
    `<div class="pb">` +
    `<div class="row">${badge(l)}${kindBadge(l)}${lowConf(l) ? `<span class="flag" title="${EST_TITLE}">≈ estimate</span>` : ""}${l.data_suspect ? dataFlag : ""}</div>` +
    `<h3>${esc(l.title || l.street || "Listing")}</h3>` +
    (l.street ? `<div class="addr">${esc(l.street)}${l.locality ? ", " + esc(l.locality) : ""}</div>` : "") +
    `<p class="save v-${l.verdict || "none"}-t">${t.amt}<small> ${t.sub}</small></p>` +
    `<div class="cmp">` +
    cmpRow("Rent", l.rent_gross, "var(--faint)") +
    cmpRow("Effective", l.effective_rent, vColor) +
    cmpRow(`Fair${lowConf(l) ? " ≈" : ""}`, l.fair_rent, "var(--accent)", r ? ` <small class="pm">±${chf((r.hi - r.lo) / 2)}</small>` : "", band) +
    `</div>` +
    popupSources(l) +
    `<dl class="kv">${kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>` +
    bars +
    (url ? `<a class="go" href="${esc(url)}" target="_blank" rel="noopener">Open ad on ${esc(l.portal || "portal")} ↗</a>` : "") +
    `</div></div>`
  );
}

const CORRECTED = {
  flatfox: "Rent, rooms and m² taken from flatfox: the homegate copy looks manipulated.",
  history: "Rent and size from an earlier sighting: the latest homegate data looks manipulated.",
  history_rent: "Rent from an earlier sighting; rooms and m² could not be verified.",
};
function popupSources(l) {
  const src = sourcesOf(l);
  const flagged = src.some((s) => s.flag);
  if (src.length < 2 && !flagged) return "";
  const cheap = cheaperOn(l);
  const rows = src
    .map((s) => {
      const url = safeUrl(s.url);
      const p = esc(portalName(s.portal));
      const name = url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${p} ↗</a>` : p;
      const manip = s.flag ? ` <small class="manip" title="Flag: ${esc(s.flag)}">${MANIP}</small>` : "";
      return `<li>${name}<span>${isNum(s.rent_gross) ? `CHF ${chf(s.rent_gross)}` : "–"}${manip}</span></li>`;
    })
    .join("");
  const note = CORRECTED[l.data_corrected];
  return (
    `<div class="sh"><span>${src.length > 1 ? `Listed on ${src.length} portals` : "Ad"}</span></div>` +
    `<ul class="sources">${rows}</ul>` +
    (cheap ? `<p class="cheaper">Cheaper on ${esc(cheap.portal)} by CHF ${chf(cheap.saving)}/mo</p>` : "") +
    (note ? `<p class="corrected">${esc(note)}</p>` : "")
  );
}

/* ---------- selection ---------- */
function selectListing(key, { fromMap = false } = {}) {
  const prev = state.activeKey ? state.byKey.get(state.activeKey) : null;
  if (prev) styleMarker(prev, false);
  state.activeKey = key;
  const l = state.byKey.get(key);
  if (!l) return;
  styleMarker(l, true);
  activeMarker = l._marker || null;
  document.querySelectorAll("#tbody tr.active, #cards .card.active").forEach((el) => el.classList.remove("active"));
  document.querySelector(`#cards .card[data-key="${CSS.escape(key)}"]`)?.classList.add("active");
  if (fromMap) {
    const idx = state.visible.indexOf(l);
    if (idx >= state.cap) {
      state.cap = Math.ceil((idx + 1) / ROW_CAP_STEP) * ROW_CAP_STEP;
      renderTable();
    }
    const tr = document.querySelector(`#tbody tr[data-key="${CSS.escape(key)}"]`);
    if (tr) {
      tr.classList.add("active");
      if (!window.matchMedia("(max-width: 860px)").matches) tr.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  } else {
    const tr = document.querySelector(`#tbody tr[data-key="${CSS.escape(key)}"]`);
    if (tr) tr.classList.add("active");
    if (map && l._marker) {
      if (!markerLayer.hasLayer(l._marker)) markerLayer.addLayer(l._marker);
      if (window.matchMedia("(max-width: 860px)").matches) $("#map").scrollIntoView({ behavior: "smooth" });
      map.once("moveend", () => l._marker.openPopup());
      map.flyTo(l._latlng, Math.max(map.getZoom(), 15), { duration: 0.6 });
    }
  }
}

/* ---------- settings UI ---------- */
function fillSettingsForm() {
  const f = $("#settings").elements;
  f.filing.value = state.profile.filing;
  f.taxable_income.value = state.profile.taxable_income;
  f.taxable_wealth.value = state.profile.taxable_wealth;
  f.reference_bfs.value = String(state.profile.reference_bfs);
  for (const k of COMPONENTS) {
    f["w_" + k].value = state.weights[k];
    f["w_" + k].nextElementSibling.textContent = Number(state.weights[k]).toFixed(2);
  }
}
function buildReferenceSelect() {
  const sel = $("#settings").elements.reference_bfs;
  const opts = new Map();
  for (const r of taxMuniIndex().values()) opts.set(Number(r.bfs), r.name);
  if (!opts.size) for (const l of state.listings) if (l.municipality_bfs != null) opts.set(Number(l.municipality_bfs), l.municipality || String(l.municipality_bfs));
  if (!opts.has(Number(state.profile.reference_bfs))) opts.set(Number(state.profile.reference_bfs), `BFS ${state.profile.reference_bfs}`);
  sel.innerHTML = [...opts.entries()]
    .sort((a, b) => String(a[1]).localeCompare(String(b[1])))
    .map(([bfs, name]) => `<option value="${bfs}">${esc(name)}</option>`)
    .join("");
}
function onSettingsChange() {
  const f = $("#settings").elements;
  state.profile = {
    ...state.profile,
    filing: f.filing.value,
    taxable_income: Math.max(0, Number(f.taxable_income.value) || 0),
    taxable_wealth: Math.max(0, Number(f.taxable_wealth.value) || 0),
    reference_bfs: Number(f.reference_bfs.value),
  };
  for (const k of COMPONENTS) {
    state.weights[k] = Number(f["w_" + k].value);
    f["w_" + k].nextElementSibling.textContent = state.weights[k].toFixed(2);
  }
  saveStored();
  writeHash();
  rescore();
  renderTable();
  if (activeMarker && activeMarker.isPopupOpen()) activeMarker.setPopupContent(popupHtml(state.byKey.get(state.activeKey)));
}

function buildMuniFilter() {
  const counts = new Map();
  for (const l of state.listings) counts.set(muniName(l), (counts.get(muniName(l)) || 0) + 1);
  const names = [...counts.keys()].sort((a, b) => a.localeCompare(b));
  $("#muni-list").innerHTML = names
    .map((n) => `<label><input type="checkbox" value="${esc(n)}" checked> ${esc(n)} <span class="hint">${counts.get(n)}</span></label>`)
    .join("");
}

function buildPortalFilter() {
  const counts = new Map();
  for (const l of state.listings) for (const p of new Set(sourcesOf(l).map((s) => portalName(s.portal)))) counts.set(p, (counts.get(p) || 0) + 1);
  const box = $("#portal-filter");
  if (!box) return;
  box.hidden = counts.size < 2;
  $("#portal-list").innerHTML = [...counts.keys()]
    .sort()
    .map((p) => `<label><input type="checkbox" value="${esc(p)}" checked> ${esc(p)} <span class="hint">${counts.get(p)}</span></label>`)
    .join("");
}

function renderMarket(stats) {
  const m = stats && stats.municipalities;
  if (!m) return;
  let rows = Array.isArray(m) ? m : Object.entries(m).map(([k, v]) => (v && typeof v === "object" ? { name: k, ...v } : { name: k, value: v }));
  rows = rows.filter((r) => r && typeof r === "object").slice(0, 40);
  if (!rows.length) return;
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))].filter((c) => !/bfs/i.test(c) || rows.length < 2).slice(0, 8);
  const fmt = (v) => (isNum(v) ? (Math.abs(v) >= 100 ? nf0.format(v) : nf1.format(v)) : esc(v ?? "–"));
  $("#market").innerHTML =
    `<table><thead><tr>${cols.map((c) => `<th class="txt">${esc(c.replace(/_/g, " "))}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${cols.map((c) => `<td class="${isNum(r[c]) ? "" : "txt"}">${fmt(r[c])}</td>`).join("")}</tr>`).join("") +
    `</tbody></table>`;
  $("#market-panel").hidden = false;
}

function renderMeta() {
  const d = state.data;
  const susp = state.listings.filter((l) => l.suspicious).length;
  const portals = new Set(state.listings.flatMap((l) => sourcesOf(l).map((s) => portalName(s.portal))));
  const multi = state.listings.filter((l) => sourcesOf(l).length > 1).length;
  $("#meta").textContent =
    `Updated ${fmtDate(d.generated_at)} · ${nf0.format(state.listings.length)} listings` +
    (portals.size ? ` from ${portals.size} portal${portals.size > 1 ? "s" : ""}` : "") +
    (multi ? ` (${nf0.format(multi)} on several)` : "") +
    (susp ? ` · ${susp} flagged` : "") +
    qualityNote(d.stats && d.stats.quality);
}
// Data-quality note: prices corrected because homegate served manipulated values.
function qualityNote(q) {
  if (!q) return "";
  const fixed = (q.corrected_flatfox || 0) + (q.corrected_history || 0);
  const parts = [];
  if (fixed) parts.push(`${nf0.format(fixed)} homegate price${fixed > 1 ? "s" : ""} corrected${q.corrected_flatfox ? " using flatfox" : ""}`);
  if (q.data_suspect) parts.push(`${nf0.format(q.data_suspect)} with data that may be wrong`);
  return parts.length ? ` · ${parts.join(" · ")}` : "";
}

/* ---------- theme ---------- */
const THEMES = ["auto", "light", "dark"];
function currentTheme() {
  const t = document.documentElement.dataset.theme;
  return t === "light" || t === "dark" ? t : "auto";
}
function applyTheme(t) {
  if (t === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
  try {
    t === "auto" ? localStorage.removeItem(THEME_KEY) : localStorage.setItem(THEME_KEY, t);
  } catch {
    /* storage unavailable */
  }
  const b = $("#theme");
  b.title = `Colour theme: ${t} (click to change)`;
  b.setAttribute("aria-label", `Colour theme: ${t}`);
}

/* ---------- wiring ---------- */
function debounce(fn, ms) {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

function wire() {
  const refilter = debounce(() => {
    state.cap = ROW_CAP_STEP;
    renderTable();
  }, 150);
  $("#filters").addEventListener("input", (e) => {
    if (e.target.id === "sort-select") return;
    if (e.target.name === "show_munis") {
      if (muniLayer) e.target.checked ? muniLayer.addTo(map) : muniLayer.remove();
      return;
    }
    if (e.target.closest("#muni-list")) {
      e.target.checked ? state.excludedMunis.delete(e.target.value) : state.excludedMunis.add(e.target.value);
    }
    if (e.target.closest("#portal-list")) {
      e.target.checked ? state.excludedPortals.delete(e.target.value) : state.excludedPortals.add(e.target.value);
    }
    refilter();
  });
  $("#filters").addEventListener("submit", (e) => e.preventDefault());
  $("#sort-select").addEventListener("change", (e) => setSort(e.target.value));
  const setAll = (on) => {
    document.querySelectorAll("#muni-list input").forEach((i) => {
      i.checked = on;
      on ? state.excludedMunis.delete(i.value) : state.excludedMunis.add(i.value);
    });
    refilter();
  };
  $("#munis-all").addEventListener("click", () => setAll(true));
  $("#munis-none").addEventListener("click", () => setAll(false));

  $("#settings").addEventListener("input", debounce(onSettingsChange, 120));
  $("#settings").addEventListener("submit", (e) => e.preventDefault());
  $("#reset-settings").addEventListener("click", () => {
    state.profile = { ...state.defaults.tax_profile };
    state.weights = { ...state.defaults.weights };
    fillSettingsForm();
    onSettingsChange();
  });
  $("#share").addEventListener("click", async (e) => {
    writeHash();
    try {
      await navigator.clipboard.writeText(window.location.href);
      e.target.textContent = "Link copied";
    } catch {
      e.target.textContent = "Copy from address bar";
    }
    setTimeout(() => (e.target.textContent = "Copy share link"), 2000);
  });

  $("#thead-row").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-sort]");
    if (b) setSort(b.dataset.sort, true);
  });
  $("#tbody").addEventListener("click", (e) => {
    if (e.target.closest("a")) return;
    const tr = e.target.closest("tr[data-key]");
    if (tr) selectListing(tr.dataset.key);
  });
  $("#cards").addEventListener("click", (e) => {
    const c = e.target.closest(".card[data-key]");
    if (c) selectListing(c.dataset.key);
  });
  $("#more").addEventListener("click", () => {
    state.cap += ROW_CAP_STEP;
    renderTable();
  });
  $("#theme").addEventListener("click", () => applyTheme(THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length]));
  // comparable links inside map popups
  $("#map").addEventListener("click", (e) => {
    const a = e.target.closest("a[data-comp]");
    if (!a) return;
    e.preventDefault();
    if (activeMarker) activeMarker.closePopup();
    selectListing(a.dataset.comp);
  });
}

const DEFAULTS = {
  tax_profile: { canton: "TI", filing: "single", taxable_income: 120000, taxable_wealth: 200000, reference_bfs: 5192 },
  weights: { value: 0.55, tax: 0.25, price_drop: 0.1, market_time: 0.1 },
};

async function main() {
  applyTheme(currentTheme());
  let data;
  try {
    const r = await fetch("data.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    $("#meta").textContent = `Could not load data.json (${err.message}).`;
    return;
  }
  state.data = data;
  data.tax = data.tax || {};
  state.defaults = {
    tax_profile: { ...DEFAULTS.tax_profile, ...(data.defaults?.tax_profile || {}) },
    weights: { ...DEFAULTS.weights, ...(data.defaults?.weights || {}) },
  };
  state.listings = (data.listings || []).filter((l) => l && l.key != null);
  for (const l of state.listings) {
    l.key = String(l.key);
    l._tdm0 = l.tax_delta_month ?? null;
    l._search = `${l.title || ""} ${l.street || ""} ${l.locality || ""}`.toLowerCase();
    state.byKey.set(l.key, l);
  }
  const stored = loadStored() || {};
  const fromHash = parseHash();
  state.profile = { ...state.defaults.tax_profile, ...(stored.profile || {}), ...(fromHash.profile || {}) };
  state.weights = { ...state.defaults.weights, ...(stored.weights || {}), ...(fromHash.weights || {}) };

  renderMeta();
  buildReferenceSelect();
  fillSettingsForm();
  buildMuniFilter();
  buildPortalFilter();
  buildSortSelect();
  buildHeader();
  rescore();
  initMap();
  renderTable();
  renderMarket(data.stats);
  wire();
  loadMunicipalities();
}

main();
