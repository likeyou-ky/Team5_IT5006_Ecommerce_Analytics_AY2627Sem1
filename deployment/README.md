# deployment/

Interactive Streamlit dashboard for the consolidated EDA.

It is an interactive retelling of `notebooks/00_Consolidated_EDA_final.ipynb` — the eight pages follow
that notebook's sections in order, and every headline number is read from `data/facts.json`, which the
generator produces straight from the notebook's own logic (with `assert`s that fail if any number
drifts). EDA only — **no prediction model**.

The app reads only the **small precomputed aggregates** in `deployment/data/` (~0.8 MB of CSV/JSON plus
the 2.1 MB `brazil_states.geojson`), not the 121 MB raw dataset, so it deploys on Streamlit Cloud with
nothing extra to host.

## Files
- `app.py` — the Streamlit application (8 pages, all interactive).
- `gen_dashboard_data.py` — regenerates the aggregates from the raw Olist CSVs (no sklearn/statsmodels).
- `data/` — precomputed aggregates, `facts.json` (headline numbers), and `brazil_states.geojson` for the maps.
- `requirements.txt` — dependencies for Streamlit Cloud (`streamlit, pandas, numpy, plotly`).

## Run locally
```bash
pip install -r deployment/requirements.txt
streamlit run deployment/app.py
```

## Regenerate the aggregates (only if the analysis changes)
```bash
OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/gen_dashboard_data.py
```

## Deploy on Streamlit Cloud (public)
1. Push the repo to GitHub (the `data/` aggregates here are committed on purpose — they are small).
2. Go to https://share.streamlit.io → **New app** → pick this repo/branch.
3. Set **Main file path** to `deployment/app.py` and deploy. The public URL goes in the report.

## Pages (mirror the notebook)
1. Overview & data quality (§1–§2) — tables, join keys, missing values, the city-name trap, four framing distributions
2. Temporal patterns (§4) — monthly series toggle, year overlay, day × hour heatmap
3. Delivery performance (§5) — late/early split, lead-time decomposition, the state gradient
4. Geography — three choropleths + a customer-state → best-seller-state drill-down
5. Customers, sellers & products (§6) — repeat rate, seller Pareto, category volume/revenue, payments
6. Freight economics (§7, the spine) — drivers, chargeable weight, determinism spread, regressivity, category burden
7. Reviews & satisfaction (§8) — score distribution, review-vs-delivery-vs-freight
8. Problem selection (§9–§10) — Candidate A (primary) / B (companion) with verbatim scoping checklists

The previous "fair-freight estimator" page and its model file were removed — the dashboard is purely EDA.
