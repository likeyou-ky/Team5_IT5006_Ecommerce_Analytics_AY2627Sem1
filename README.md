# Team 5 — IT5006 E-Commerce Analytics (AY2026/27 Semester 1)

This repository holds our coursework for the IT5006 project. We are working with the Olist
Brazilian E-Commerce dataset and looking at the economics of shipping: what determines the
freight charged on an order, and how well that cost can be predicted. We intend to frame this as
a classification task (flagging orders where freight is high relative to the item price) and a
regression task (estimating a reasonable freight value from a parcel's weight, size and shipping
distance). The exact problem definition may still change as we scope the project.

## Repository structure

```
project-root/
├── data/          raw Olist CSVs (not committed; see data/README.md)
├── notebooks/     Jupyter/Colab notebooks for EDA, feature engineering and modelling
├── src/           reusable Python modules (data loading, feature builds, model utilities)
├── deployment/    Streamlit application for the final phase
├── docs/          reports, EDA summaries and figures
├── requirements.txt
└── README.md
```

## Getting the data

We use the Olist dataset from Canvas (Week 1). It comes as nine CSV tables: orders, order items,
payments, reviews, customers, sellers, products, geolocation, and the product-category
translation. The raw files are not committed to the repository because they are large and are
distributed through Canvas; see `data/README.md` for where to put them.

Two things are worth keeping in mind when working with the data. Customer-level analysis should
use `customer_unique_id` rather than `customer_id`, because `customer_id` is assigned per order.
When building models we avoid using fields that are only known once an order has completed (for
example, delivery dates) as predictors, so that information from the outcome does not leak into
the features.

## Running the notebooks

On Google Colab, place the CSVs in Google Drive under `MyDrive/IT5006_Project-Data/Olist_CSV/`,
open a notebook from the `notebooks/` folder, run the cell that mounts Drive, and then run the
remaining cells.

To run locally:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Put the CSVs in the `data/` folder (see `data/README.md`) and open a notebook. The notebooks
check whether they are running on Colab or locally and read the data accordingly.

## Notebooks

`notebooks/EDA_freight_economics_keyou.ipynb` is the exploratory analysis. It profiles each table
for data-quality issues, joins the tables into a single order-item table that later work builds
on, and examines how freight relates to price, weight, volume, distance and product category. New
notebooks are named with the author's name or a number so they are easy to tell apart.

## Tools

The analysis and modelling use Python with pandas, NumPy, scikit-learn and matplotlib/seaborn.
The deployed application in the final phase uses Streamlit.

## Reproducibility

Random seeds are recorded in the notebooks and scripts. Model training uses scikit-learn
pipelines with a held-out test set, so that preprocessing does not see the test data during
fitting.
