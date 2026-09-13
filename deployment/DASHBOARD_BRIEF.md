# Dashboard rebuild — handover brief

**Task:** rebuild the interactive Streamlit dashboard so it is **100% based on the final consolidated
EDA notebook** and retells the *same story* from that notebook — just more interactive. Deployable
publicly on Streamlit Cloud.

---

## 0. Hard rules (read first)

1. **Single source of truth:** `notebooks/Team5_Phase1_IT5006_AY2627Sem1.ipynb`
   (`C:\Users\KLi\Team5_IT5006_Ecommerce_Analytics_AY2627Sem1\notebooks\Team5_Phase1_IT5006_AY2627Sem1.ipynb`).
   The dashboard mirrors that notebook's sections, story, and numbers.
2. **Do NOT invent or recompute statistics.** Every number shown in the dashboard must already appear
   in the notebook (a cell output or markdown). If a number isn't in the notebook, don't put it in the
   dashboard. (Precomputing *aggregates for charts* from the raw data is fine — but the headline
   figures/conclusions must match the notebook.)
3. **Ship aggregates, not raw data.** The raw dataset is 121 MB — never commit it or load it in the
   deployed app. Precompute small CSV/JSON aggregates (pattern already in `gen_dashboard_data.py`) into
   `deployment/data/`, and have the app read those.
4. **This is an EDA dashboard** — it tells the data-understanding story. No trained/predictive models
   are required (the notebook itself is EDA-only). See the "Fair-freight estimator" decision in §6 below.
5. **Commit identity** (if committing): author `likeyou-ky` <82462672+likeyou-ky@users.noreply.github.com>,
   never credit Claude. Nothing is committed yet — the user decides when to push.

---

## 1. Data

- Raw Olist CSVs (9 tables): `C:\Users\KLi\OneDrive - Revenue Management Solutions, LLC\Documents\ClaudeProjects\IT5006_Project-Data\Olist_CSV`
- The notebook's loader reads `OLIST_DATA_DIR` first, then Colab/`data/` fallbacks. To run the
  notebook or the aggregate generator locally, set:
  `OLIST_DATA_DIR="C:/Users/.../IT5006_Project-Data/Olist_CSV"`.
- Analytical tables the notebook builds (reuse the exact logic in notebook §3):
  - geolocation cleaned = Brazil bounding box, then **median** lat/long per zip prefix.
  - `item` = order-item grain with product weight/volume, `volumetric_weight_kg = L*W*H/5000`,
    seller & customer coords, Haversine `distance_km`, `freight_ratio = freight/price`.
  - `order` = order grain (items+payments aggregated, delivery timing, review score, `is_late`).

## 2. Existing assets in `deployment/` — reuse and REALIGN (don't start blind)

- `app.py` — current Streamlit app (Olist theme, 7 sections, plotly). **Built before the final merge**,
  so its §6 and §8 reflect old content. Realign every section to the FINAL notebook (see story map §5).
- `gen_dashboard_data.py` — aggregate generator. Re-run and update so aggregates match the final
  notebook's numbers/sections.
- `data/` — precomputed aggregates + `brazil_states.geojson` (2.1 MB; choropleth keys on
  `properties.sigla` = 2-letter state code).
- `requirements.txt` — `streamlit, pandas, numpy, plotly` (no statsmodels — see gotchas).
- `.streamlit/config.toml` (repo root) — the locked Olist theme.

## 3. Olist theme (from designsystem.olist.io tokens) — keep it

- Primary `#0a4ee4`, chart blue `#2766ec`, coral `#ed6e5a`, green `#779e3d`, slate `#3e3e3d`.
- Warm neutrals: background `#fcfbf8`, surface `#f2f0e8`, text `#10100f`.
- Semantic: success `#779e3d`, warning `#f0a028`, error `#e64e36`.
- `.streamlit/config.toml` sets these; plotly uses a transparent-background template + Olist colorway
  so charts sit on the warm canvas.

## 4. Story to mirror (section map from the final notebook, with section owners)

Keep the dashboard's flow aligned to this. Owners: §1-4 Zi Qing, §5 Huixin, §6 Trixie, §7 Keyou,
§8-9 Pei Shi, §10 shared.

- **§1 Dataset overview** — 9 tables, join keys, row counts.
- **§2 Data quality** — missing values; §2.1 handling decisions (review text majority-empty -> use
  score; delivery dates missing only for non-delivered; **do NOT use city-name columns** — accented vs
  plain spellings split one city; key on zip + centroids).
- **§4 Temporal** — growth through 2017-2018, Black Friday Nov 2017; ~2-year window (late 2016-Oct 2018).
- **§5 Delivery (Huixin)** — 97% delivered, **6.8% late vs estimate**, 91.9% early, median 10 days
  delivery / ~12 days early (padded estimates). Lead time = mostly the **carrier->customer leg (~9 of
  ~12 days)**. Strong state gradient: **SP ~7 days/<6% late vs remote AM/RR/AP ~24-25 days**; late rate
  up to **21.4% (AL)**. Key nuance: **longest-transit != highest-late** states.
- **§6 Customer/seller/product (Trixie)** — repeat rate ~3% (rules out repeat/CLV target); seller
  revenue Pareto-concentrated (top 10% ~67%); item volume concentrated in a few categories; payments
  credit-card dominated, instalments track basket size.
- **§7 Freight economics (Keyou) — the spine.**
  - Freight ~**14% of GMV**, median **23% of item price**, 46% of items >25% of price; right-skewed
    (median ~R$16, tail to R$410).
  - Priced off the parcel: corr **weight 0.61, volume 0.59, distance 0.39** vs only **0.41 price**.
    volumetric_weight = L*W*H/5000 (5000 = carrier convention; ref DHL). Chargeable weight =
    max(actual, volumetric) ~0.61 (actual weight usually binding for Olist's small parcels).
  - **How deterministic is freight?** near-identical parcels (same weight/volume/distance band) still
    vary ~**1.7x (p10->p90)**; only ~**53%** of freight variation aligns with the band -> the residual
    is seller-driven pricing/deals = the overcharge signal.
  - **Regressive**: near-flat fee = ~60% burden on cheap items, <5% on dear ones. Category burden
    (electronics ~60% vs watches ~12% at the SAME ~R$15 freight) is a **price** story, not a cost one.
- **§8 Reviews (Pei Shi)** — review-score distribution (bimodal 5-star/1-star), over time, correlation.
  Review **text is ~59% empty** (use the numeric score, not NLP). Satisfaction tracks **delivery time,
  not freight**.
- **§9 Candidate problems** — **A: fair-freight pricing + overcharge detection** (regression predicts
  fair freight from distance/weight/volume/category; classification flags residual overcharges) —
  signal from §7. **B: delivery-time / late-delivery** — signal from §5. Each has a scoping checklist.
- **§10 Recommended direction** — Candidate A primary, B companion; limitations (freight is revenue not
  cost; great-circle distance; completed-orders only; ~2-year window).

## 5. Interactivity to deliver (mirror the notebook, but interactive)

- **Overview / key distributions** — freight value, freight ratio, delivery days, review score, each
  with a one-line conclusion tying to problem selection.
- **Temporal** — series toggle (orders/revenue/late-rate), day-of-week x hour heatmap, year overlay.
- **Geography** — **three choropleth maps at once** (median delivery days, freight, distance), plus a
  **customer-state selector -> best seller states** shown across all three metrics + a lane table.
- **Freight economics** — freight-vs-driver scatter with **axis toggle + category filter** (use a
  binned-median trend line, NOT lowess); price-band regressivity; category-burden; the determinism
  (near-identical-parcel spread) result. Label each chart's scope (filtered vs all-items).
- **Reviews** — score distribution + the "text mostly empty" and "delivery not freight drives
  satisfaction" points.
- **Candidate problems** — the two candidates + scoping checklists (verbatim from §9).

## 6. Decisions already made

- **Fair-freight estimator: DROP IT.** The previous app had an interactive slider that *predicts* a
  fair freight from a tiny linear model. The user has decided to remove it — the dashboard is **purely
  EDA storytelling, no prediction**. Delete that section from `app.py`, and drop the model coefficients
  (`freight_model.json`) and any statsmodels/sklearn dependency from the generator and requirements.

## 7. Technical gotchas (learned the hard way)

- Streamlit: use `width="stretch"`, NOT the deprecated `use_container_width=True`.
- Matplotlib titles/labels: **two `$` triggers mathtext** and crashes (`ParseException`). Use "BRL" or a
  single `$`. Same for Streamlit/Jupyter markdown (two `$` -> KaTeX). Prefer "R$" once, or "reais".
- Do NOT use lowess/OLS trendlines on large scatters (statsmodels; hangs on ~9k points). Use a
  manual binned-median line. Keeps requirements light (no statsmodels).
- Verify visually: `streamlit run deployment/app.py --server.headless true --server.port 8600
  --server.address 127.0.0.1`, then open in the browser and screenshot each section; also check the
  process logs for errors.
- Keep `deployment/data/` small; `.gitignore` already un-ignores `deployment/data/*.csv` and `*.geojson`.

## 8. Deploy (Streamlit Cloud, public)

1. Commit the repo (as `likeyou-ky`) incl. `deployment/` and its `data/` aggregates.
2. share.streamlit.io -> New app -> pick repo/branch -> **Main file path = `deployment/app.py`**.
3. Put the public URL in the report. (More in `deployment/README.md`.)
