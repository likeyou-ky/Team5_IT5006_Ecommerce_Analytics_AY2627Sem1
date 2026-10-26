# Team 5 — IT5006 E-Commerce Analytics (AY2026/27 Semester 1)

This repository holds our coursework for the IT5006 project. We work with the Olist Brazilian E-Commerce dataset and study the logistics of an order with two problems:
a **regression** task (what freight *should* an item be charged, from weight, size, distance and category) and a
**classification** task (will a delivered order arrive later than its promised date, `is_late`, a 6.8% minority class).

**Repository:** https://github.com/likeyou-ky/Team5_IT5006_Ecommerce_Analytics_AY2627Sem1

**Live EDA dashboard (Phase 1):** https://team5-olist-eda-phase1.streamlit.app/

## Repository structure

```
project-root/
├── data/            raw Olist CSVs (not committed; see data/README.md)
├── notebooks/
│   ├── Team5_Phase1_IT5006_AY2627Sem1.ipynb   consolidated EDA — the Phase 1 deliverable
│   ├── Team5_Phase2_IT5006_AY2627Sem1.ipynb   modelling and evaluation — the Phase 2 deliverable
│   └── draft/individual/                       each member's working notebooks
├── src/             reusable modules: features.py, models.py, pipeline.py, plots.py, olist_theme.py
├── deployment/
│   ├── app.py, gen_dashboard_data.py, data/   Phase 1 EDA dashboard
│   └── model_app/                              Phase 3 model app: app.py, build_models.py, models/, data/
├── docs/            reports and slides, kept local (only docs/README.md is versioned)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data

We use the Olist dataset from Canvas (Week 1). It comes as nine CSV tables: orders, order items,
payments, reviews, customers, sellers, products, geolocation, and the product-category
translation. The raw files are not committed to the repository because they are large and are
distributed through Canvas; see `data/README.md` for where to put them.

Two things are worth keeping in mind. Customer-level analysis should use `customer_unique_id`
rather than `customer_id`, because `customer_id` is assigned per order. When building models we
avoid using fields that are only known once an order has completed (for example, delivery dates)
as predictors, so that information from the outcome does not leak into the features.

## Notebooks

`notebooks/Team5_Phase1_IT5006_AY2627Sem1.ipynb` is the **consolidated EDA and the Phase 1
deliverable**. It covers the dataset and data quality, temporal patterns, delivery performance,
the customer / seller / product distributions, freight economics (the central theme), reviews, and
the two candidate problems. Throughout, each chart is read for two things: what it implies for our
choice of problem, and whether the underlying data is usable.

Each member's original working notebooks are kept under `notebooks/draft/individual/` for
reference; the consolidated notebook is the version to read and grade.

To run on Google Colab, place the CSVs in Google Drive under
`MyDrive/IT5006_Project-Data/Olist_CSV/`, open the notebook, run the cell that mounts Drive, then
run the rest. To run locally, put the CSVs in `data/` (see `data/README.md`), install the
requirements above, and launch `jupyter notebook`. The notebook detects Colab vs local and reads
the data accordingly.

## Phase 2 — modelling

`notebooks/Team5_Phase2_IT5006_AY2627Sem1.ipynb` is the Phase 2 deliverable. Two problems:

| | Problem 1 — regression | Problem 2 — classification |
|---|---|---|
| Question | fair freight for an order item | will the order arrive after its promised date? |
| Target | `log1p(freight_value)` (112,650 items) | `is_late` (96,476 delivered orders, 6.8% late) |
| Stakeholder | logistics-pricing / finance analyst | operations and customer-experience team |
| Validation | 80/20 hold-out grouped by `order_id`; robustness splits by product and by time | time-ordered: latest 20% of orders as test, 30-day embargo, 5-fold forward-chaining CV |

Three model families across both problems: Linear (LinearRegression/Ridge, Logistic), Tree-based (DecisionTree
baseline, RandomForest) and Ensemble (Stacking for Problem 1, soft Voting for Problem 2). All preprocessing sits inside
scikit-learn Pipelines; the seed is `RANDOM_STATE = 42`. Feature groups are chosen on the training data only, by their
paired fold-by-fold gain (forward selection for Problem 2). Problem 2 features use only information available at
checkout; precision, recall and F1 are reported for the riskiest 10% of orders (an assumed review capacity).

To reproduce it, put the CSVs in `data/` (or set `OLIST_DATA_DIR`) and run the notebook top to bottom (about 15
minutes on 12 cores). It calls the reusable functions in `src/`.

`notebooks/Team5_Phase3_IT5006_AY2627Sem1.ipynb` responds to the Phase 2 feedback. It re-evaluates Problem 1 on a
**time-ordered split** (train on the earliest 80% of items, test on the latest 20%; feature selection, tuning and model
choice on 5-fold forward-chaining CV), reports training, CV and test scores side by side, compares models with a
paired fold-by-fold rule, and lists every candidate of the searches of both problems. On later periods the listing
and timing features no longer help and are dropped; the final model stays a RandomForest, now trained and tested in
time order. Problem 2 is unchanged. It runs in about 15 minutes.

## Dashboard

An interactive Streamlit dashboard retells the consolidated EDA, section by section, and is
deployed publicly:

**https://team5-olist-eda-phase1.streamlit.app/**

It reads only the small precomputed aggregates in `deployment/data/` (not the 121 MB raw dataset),
so it deploys on Streamlit Cloud with nothing extra to host. To run it locally:

```bash
pip install -r requirements.txt
streamlit run deployment/app.py
```

To regenerate the aggregates from the raw CSVs (only if the analysis changes):

```bash
OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/gen_dashboard_data.py
```

See `deployment/README.md` for more detail.

## Phase 3 — model app

`deployment/model_app/` deploys the final models as a Streamlit app organised around the two stakeholders:

- **Logistics-pricing / finance analyst** (Problem 1, RandomForest from the Phase 3 notebook, with Ridge as a
  second opinion): a monthly
  *Freight audit* (sellers ranked by flag rate, items flagged by both models first, held-out items or an uploaded CSV),
  *Check a charge* (one charge, or a quote range for a new listing) and a *Tariff review* (where a simple published
  tariff misprices freight by distance and weight).
- **Operations and customer-experience team** (Problem 2, plain LogisticRegression): a *Daily risk queue* (the day's
  riskiest orders with a suggested low-cost follow-up, the promise that would have kept each off the list, and the
  real outcomes on request), a *Promise check* for new orders at checkout, and *Monitoring* of promise changes and
  model skill week by week.

The Olist data end in 2018, so the app scores held-out historical records (the Problem 1 test items, 24 May – 3 Sep
2018, and the Problem 2 test window, 26 May – 29 Aug 2018) as if they were new orders. To run it locally:

```bash
pip install -r deployment/model_app/requirements.txt
streamlit run deployment/model_app/app.py
```

`deployment/model_app/build_models.py` rebuilds the model files and data from the raw CSVs with the code in `src/`
and stops if a result drifts from the notebooks. See `deployment/model_app/README.md` for the pages, files and
Streamlit Cloud settings.

## Tools

The analysis and modelling use Python with pandas, NumPy, scikit-learn and matplotlib / seaborn.
The dashboard and the model app use Streamlit and Plotly. `requirements.txt` gives minimum versions; the model app's
own `requirements.txt` pins the libraries that load its saved models.

## Reproducibility

Random seeds are recorded in the notebooks and scripts (`RANDOM_STATE = 42`). The analytical
tables are built once and reused, delivery analysis is restricted to delivered orders, and skewed
monetary variables are log-transformed for modelling. Model training uses scikit-learn pipelines
with a held-out test set, so preprocessing does not see the test data during fitting.
