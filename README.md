# flat-price

Daily rental-deal finder for Ticino. It scrapes flats from rental portals (homegate.ch and flatfox.ch), keeps their history in SQLite, scores each flat on how good a deal it is, and publishes an interactive static site (a map and a sortable list) to GitHub Pages. The score includes the tax effect of the municipality you would live in.

## How it works

```
config.yaml searches ─► portals/<portal>.py ─► SQLite (runs, listings, observations)
                                                   │
                     dedup (same flat, many portals) ◄┤
                     geo (municipality, area) ◄────┤
                     tax engine (TI multipliers) ◄─┤
                     hedonic fair-rent model ◄─────┤
                     score (0–100) ◄───────────────┘
                                                   │
                                     dist/ (data.json + web app) ─► GitHub Pages
```

- **Portals**: *homegate* scrapes the result pages with a headful Chrome (DataDome). *flatfox* uses Flatfox's public JSON API over plain HTTP: municipality polygons from `data/geo/` are queried as map pins (the bbox is split as a quadtree because the API returns at most 400 pins per call), pins outside the polygon are dropped, and details are fetched 100 at a time. Because it searches the whole municipality polygon, `flatfox-lugano` also covers quarters such as Pregassona, Viganello and Breganzona that homegate's `luogo-lugano` search misses. The browser is only launched if a configured search needs it.
- **Cross-portal duplicates** (`flatprice/dedup.py`): one flat is often on several portals (Flatfox syndicates many homegate ads). Before cleaning and modelling, listings are collapsed if (a) a listing's `alt_ids` names another stored listing (Flatfox's `smg_id` is the homegate id), or (b) for cross-portal pairs without ids: < 40 m apart, same rooms, m² within ±5% (or both missing) and gross rent within ±3%. The kept row is the active one with the most complete data (ties: homegate, then key). It carries `sources` (every ad with its portal, link and rent), `n_sources`, the earliest `first_seen` and the longest `days_on_market`. The site shows one badge per portal, lists every ad with its price in the popup ("cheaper on flatfox"), and has a portal filter.
- **Data quality** (`flatprice/quality.py`): homegate's anti-bot system (DataDome) has been seen serving perturbed data to a client it suspects is a bot. On 2026-11-21 one whole result page (Lugano page 7) and one whole search (Sorengo) came back with every listing changed: gross rent ×1.366 (rounded to CHF 10), one more room, m² inflated by 12–36%, the room count in the title bumped ("3.5 locali" → "4.5 locali"), and the net rent dropped. Flatfox's copies of the same ads kept the original values. Every observation now stores a snapshot (rent, rooms, m², title, result page) and a `flag` (NULL = trusted, else the reasons):
  - `ratio_1366`: rent vs the previous trusted observation ≈ a configured ratio (`quality.poison_ratios`, ±0.6% plus CHF 5 of rounding). On the inverse ratio the *earlier*, higher observations are flagged, because the perturbation has only ever inflated prices.
  - `rooms_jump`: rooms ±1 together with a rent change. `net_dropped`: the net rent disappeared while the rent rose.
  - `xportal_mismatch`: the Flatfox copy (`smg_id` = homegate id) observed within 3 days differs by more than 3% in rent, or in rooms. Flatfox is trusted and never flagged.
  - `page_poisoned`: a new listing with nothing to compare against, on a result page whose other observations are mostly flagged.

  Rules run after every scrape and on demand with `uv run flatprice quality` (`--show` lists the affected listings), which recomputes all flags, stores `runs.n_flagged` and prints counts by rule and by run. A flagged observation never overwrites the listing's fields. If more than `quality.max_flagged_share` (30%) of a run is flagged, the run's observations are kept (flagged), no listing fields are updated, and `flatprice scrape`/`run` exit with code 2, so the site is not rebuilt. `load_frame` takes current values from the latest trusted observation and computes `first_rent`, price drop and days on market from trusted observations only. It also adds `rent_observed`, `data_flag`, `n_flagged_obs`, `data_corrected` and `data_suspect`. In dedup a disagreeing homegate copy takes Flatfox's rent, rooms, m² and title (`data_corrected = "flatfox"`); both ads stay in `sources` with their own prices, and the flagged one is marked "price looks manipulated" (it is ignored for "cheaper on"). Rows whose data is only known from flagged observations (no trusted rent or size, no Flatfox twin) are `data_suspect`. They are kept out of training, comparables and scoring, but still listed with a "⚠ data may be wrong" badge and no verdict. The header shows how many prices were corrected. Do not run homegate scrapes back to back: the poisoning appeared on the second scrape in under 24 hours.
- **Fair rent** (`flatprice/analysis/fair_rent.py`): a log-linear regression with these inputs: size, rooms, distance to the centre and to the lake, furnished/balcony/parking/elevator, title keywords (lake view, terrace/attic, renovated/new, garden), ground or high floor, utilities stated, age bucket, and area effects shrunk towards the overall mean. It is blended with a **comparables** estimate: the median size-adjusted rent of the 8 most similar nearby listings. Every listing's estimate is **out-of-fold** (5-fold CV), so a listing never helps price itself. Lease takeovers (*subentrante*) and short-term stays are tagged as `listing_kind`, kept out of training, and labelled in the UI. Listings without m² use a fallback model and are marked `≈` (low confidence).
- **Uncertainty**: `fair_sigma_log` is the typical error of log(fair rent) for that kind of estimate. It is about 0.11 for blended estimates with m² and about 0.25 without m². To get it, the CV error is split into a listing-level part τ² (the "true deal" spread) and an estimation-error part, by comparing the regression with the comparables (Grubbs' method-comparison estimator). The UI shows it as `fair ± CHF` and as a model range of fair × e^±σ.
- **Deal %**: the raw `deal_pct` = rent / fair − 1. Verdicts use the **shrunk** deal (empirical Bayes): `k = τ² / (τ² + σ²)` and `deal_pct_adj = exp(k · ln(rent / fair)) − 1`, where τ² = max(var(ln(rent / fair)) − mean(σ²), 0.0025). Uncertain estimates shrink towards 0, and confident ones barely move. `p_below_fair = Φ(−ln(rent / fair) / σ)` is the chance the rent is below the true fair rent.
- **Verdict** (on the shrunk deal including your tax, `deal_pct_adj_eff`): Great ≤ −20% *and* at least 80% chance below fair (otherwise Good); Good −20…−8%; Fair ±8%; Pricey +8…+20%; Overpriced > +20%. The headline saving is the shrunk one, fair × (1 − e^{k · ln(eff / fair)}). The tooltip shows the raw saving, the model range and the number of comparables.
- **Tax**: in Ticino, income and wealth tax equal the cantonal base tax × (cantonal coefficient + the municipality's *moltiplicatore*). `tax_delta_month` is your monthly tax difference compared with a reference municipality (Lugano by default). `effective_rent` = rent + tax delta.
- **History**: days on market and price drops since the flat was first seen.
- **Score**: a weighted mix of the components' percentiles. *Value* is the shrunk deal on the plain rent; the tax effect has its own *Tax* component, so it is not counted twice. A component that is the same for every listing (for example, price drops before any history exists) is dropped and the other weights are renormalised. The site recomputes the tax, effective rent, shrunk deal, chance below fair, verdict and score in the browser when you change your tax profile or the weights. `web/score.js` and `web/deal.js` mirror `flatprice/analysis/score.py`, and `tests/js/parity.test.mjs` checks that they agree.
- **Model check**: `uv run python scripts/eval_model.py` prints the cross-validated error of the old method and of each model variant.

## Usage

```bash
uv sync
uv run playwright install chrome        # if Google Chrome is not installed
uv run flatprice seed old/homegate_listings.csv   # optional: import old data
uv run flatprice scrape                 # opens a visible Chrome window (DataDome) for homegate
uv run flatprice scrape --portal flatfox   # only Flatfox searches: plain HTTP, no browser
uv run flatprice analyze --top 20
uv run flatprice quality                # recompute data-quality flags and report (perturbed homegate data)
uv run flatprice build && uv run flatprice serve   # http://127.0.0.1:8000
uv run flatprice run                    # scrape + build + vacuum (daily job)
```

Edit `config.yaml` to change the searches, the default tax profile, or the scoring weights.

## Daily GitHub Action

`.github/workflows/daily.yml` runs at 05:00 UTC, and you can also start it by hand.

1. Restores `flatprice.db` from the orphan `data` branch. The first run creates that branch.
2. Runs `xvfb-run uv run flatprice run`, which uses headful Chrome on a virtual display.
3. Commits the database back to `data` and deploys `dist/` to GitHub Pages.

Setup: Settings → Pages → Source: **GitHub Actions**.

If any search fails, the run continues with the others. If zero listings are scraped, the job fails and the current site stays up. Screenshots of blocked pages are uploaded as the `debug` artifact.

**If DataDome blocks GitHub's IPs**, you have two options:
- Add a `PROXY_URL` secret (`http://user:pass@host:port`, ideally a residential proxy).
- Set the repo variable `RUNNER=self-hosted` and register a self-hosted runner (any machine with Chrome).

## Adding a portal

1. Create `flatprice/portals/<name>.py` with a class that implements `Portal` (`flatprice/portals/base.py`):
   - `needs_browser`: `True` to get the shared Playwright page, `False` for HTTP-only adapters (they get `page=None`; see `flatfox.py`).
   - `build_url(spec)` translates a `SearchSpec` into the portal's search URL.
   - `scrape(page, spec, cfg)` paginates and yields canonical `Listing` objects.
   - Raise `BlockedError` when anti-bot protection blocks the page.
   - If the portal knows the flat's id on another portal, set `Listing.alt_ids = {"<portal>": "<id>"}` so dedup can link them.
2. Register the class in `flatprice/portals/registry.py`.
3. Add searches with `portal: <name>` in `config.yaml`.
4. Add a normalizer test with a saved payload fixture in `tests/fixtures/`.

Everything after the store is portal-agnostic.

## Adding a canton

Add `flatprice/tax/data/<canton>_<year>.json`, using the same schema as `ti_*.json`, and the municipality polygons to `data/geo/`. The JS mirror (`web/tax.js`) reads the same JSON.

## Layout

| Path | Role |
|---|---|
| `flatprice/portals/` | portal adapters (homegate, flatfox) + registry |
| `flatprice/dedup.py` | collapses the same flat listed on several portals |
| `flatprice/browser.py` | shared headful Chrome session, proxy, debug captures |
| `flatprice/store.py` | SQLite history (schema v3: per-observation snapshot + quality flag) |
| `flatprice/quality.py` | data-quality rules against perturbed portal data |
| `flatprice/geo.py` | municipality (point-in-polygon), Lugano quarters, distances |
| `flatprice/tax/` | tax engine + tariff/multiplier JSON |
| `flatprice/analysis/` | cleaning, fair-rent model, scoring, pipeline |
| `flatprice/site/build.py` | writes `dist/` |
| `web/` | static app (Leaflet + vanilla JS) |
| `old/` | original one-off scripts and data |
