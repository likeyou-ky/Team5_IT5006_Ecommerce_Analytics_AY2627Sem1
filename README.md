# Team 5 — IT5006 E-Commerce Analytics (AY2026/27 Semester 1)

This repository holds our coursework for the IT5006 project. We work with the Olist Brazilian
E-Commerce dataset and study the economics of shipping: what determines the freight charged on an
order, and how well that cost can be predicted. We frame this as a regression task (estimating a
fair freight value from a parcel's weight, size and shipping distance) and a classification task
(flagging orders whose freight is abnormally high relative to that prediction).

**Repository:** https://github.com/likeyou-ky/Team5_IT5006_Ecommerce_Analytics_AY2627Sem1

**Live dashboard:** https://team5-olist-eda.streamlit.app/

## Repository structure

```
project-root/
├── data/            raw Olist CSVs (not committed; see data/README.md)
├── notebooks/
│   ├── Team5_Phase1_IT5006_AY2627Sem1.ipynb   consolidated EDA — the Phase 1 deliverable
│   └── draft/individual/                       each member's working notebooks
├── src/             reusable Python modules (e.g. olist_theme.py plotting theme)
├── deployment/      Streamlit dashboard (app.py, gen_dashboard_data.py, data/ aggregates)
├── docs/            reports, EDA summaries and figures
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

## Dashboard

An interactive Streamlit dashboard retells the consolidated EDA, section by section, and is
deployed publicly:

**https://team5-olist-eda.streamlit.app/**

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

## Tools

The analysis and modelling use Python with pandas, NumPy, scikit-learn and matplotlib / seaborn.
The dashboard uses Streamlit and Plotly. All dependencies are pinned in `requirements.txt`.

## Reproducibility

Random seeds are recorded in the notebooks and scripts (`RANDOM_STATE = 42`). The analytical
tables are built once and reused, delivery analysis is restricted to delivered orders, and skewed
monetary variables are log-transformed for modelling. Model training uses scikit-learn pipelines
with a held-out test set, so preprocessing does not see the test data during fitting.
