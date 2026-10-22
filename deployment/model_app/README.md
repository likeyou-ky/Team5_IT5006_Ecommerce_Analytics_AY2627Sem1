# deployment/model_app/

The Phase 3 proof-of-concept: a Streamlit app that deploys the two final models of
`notebooks/Team5_Phase2_IT5006_AY2627Sem1.ipynb`. The Phase 1 EDA dashboard (`deployment/app.py`) is a separate app
and is unchanged.

| Tool | Model | User |
|---|---|---|
| Fair-freight checker | RandomForest regressor (200 trees), Problem 1 | logistics-pricing / finance analyst |
| Late-delivery risk queue | plain LogisticRegression, Problem 2 | operations and customer-experience team |

The Olist data end in 2018, so the app scores **held-out historical records as if they were new**: the Problem 1
test items and the Problem 2 test window (26 May – 29 Aug 2018), none of which the models saw in training. It also
scores listings and orders that you type in, and CSV files of items that you upload.

## Pages
1. **Home**: what each tool does and how to use it.
2. **Fair-freight checker**
   - *Check a held-out item*: a random test item (or one you look up by order ID), its fair freight and a verdict.
   - *Price a new listing*: fair freight for the product and route you enter; add the charged freight to check it.
   - *Audit a batch*: scores all held-out items or an uploaded CSV, flags unusual freight, shows the sellers behind
     it and downloads the review list. A template CSV is provided.
3. **Late-delivery risk queue**
   - *Daily review queue*: one purchase day's orders ranked by risk, with the riskiest share (the review capacity,
     10% by default) marked for checking. *Reveal what happened* adds the real outcomes and how many late orders the
     list caught.
   - *Score a new order*: risk of one order you describe, its rank among test-window orders and what drives it.
4. **Model card**: training data, validation, test results and limits of both models.

An item is **unusual** when its residual (log scale) is in the top 5% of the test set, the rule of the Phase 2
unusual-freight audit. The queue flags the riskiest share of each day's orders with the notebook's top-k rule.

## Files
- `app.py`: the app.
- `build_models.py`: trains both models with the code in `src/`, writes `models/` and `data/`, and stops if a split
  or test score drifts from the Phase 2 notebook.
- `models/freight_forest.joblib` (48 MB) and `models/late_logit.joblib`: the trained pipelines.
- `data/heldout_items.csv`, `data/heldout_orders.csv`: the held-out records the app scores.
- `data/zip_centroids.csv`: zip-code prefix → location and state, for listings and orders you type in.
- `data/facts.json`: every headline number the app shows.
- `requirements.txt`: dependencies, with the libraries that load the models pinned.

## Run locally
From the repository root:
```bash
pip install -r deployment/model_app/requirements.txt
streamlit run deployment/model_app/app.py
```

## Rebuild the models (only if the code in `src/` changes)
```bash
OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/model_app/build_models.py
```
About 2 minutes. If you change the library versions, update the pins in `requirements.txt` to match.

## Deploy on Streamlit Cloud
1. Push the repository to GitHub.
2. On https://share.streamlit.io choose **Create app**, pick this repository and branch, and set the main file path
   to `deployment/model_app/app.py`.
3. Under **Advanced settings** choose Python 3.12 or 3.13, then deploy. No secrets are needed.
