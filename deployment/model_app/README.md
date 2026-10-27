# deployment/model_app/

The Phase 3 proof-of-concept: a Streamlit app that deploys the final models as decision tools for the two
stakeholders of our problems: Problem 1 from `notebooks/Team5_Phase3_IT5006_AY2627Sem1.ipynb` (re-evaluated on a
time-ordered split after the Phase 2 feedback) and Problem 2 from `notebooks/Team5_Phase2_IT5006_AY2627Sem1.ipynb`. The
Phase 1 EDA dashboard (`deployment/app.py`) is a separate app and is unchanged.

**Live (testing):** https://team5-olist-phase3-test1.streamlit.app; it moves to https://team5-olist-phase3.streamlit.app once the design is final. Streamlit Cloud builds it from
branch `Phase3_Test`, main file `deployment/model_app/app.py`.

The Olist data end in 2018, so the app scores **held-out historical records as if they were new**: the Problem 1
test items (the latest 20% of items, 24 May – 3 Sep 2018) and the Problem 2 test window (26 May – 29 Aug 2018), none of which the models saw in training. It also
scores listings and orders that you type in, and CSV files of items that you upload.

## Pages, by stakeholder

**Logistics-pricing / finance analyst** (Problem 1, fair freight: RandomForest on a time-ordered split, with
Ridge as a second opinion)

| Page | When | Decision it supports |
|---|---|---|
| Freight audit | monthly | which sellers to audit (ranked by flag rate, with a minimum number of items) and which items to review (flagged by both models first); download both lists |
| Check a charge | ad hoc | is one held-out charge fair; the fair freight and quote range for a new listing |
| Tariff review | quarterly | where a simple published tariff, or what sellers charged, sits above or below the fair freight, by distance and weight |

An item is **unusual** when its residual (log scale) is in the top 5% of the test set: overall, as in the
notebooks, or within each price band, as the report recommends. The rule is chosen at the top of the analyst's pages.

**Operations and customer-experience team** (Problem 2, late delivery: plain LogisticRegression)

| Page | When | Decision it supports |
|---|---|---|
| Daily risk queue | daily | which of the day's orders to follow up, with a suggested low-cost action for each (from its biggest risk driver), the expected yield, a download, and the promise that would have kept the order off the list; *Reveal what happened* shows the outcomes |
| Promise check | at checkout | the risk of a new order and the shortest promise that keeps it below the review line |
| Monitoring | weekly | whether a change in the promise rules calls for a model review (weekly promise, late rate and catch rate; monthly lift) |

**Model card**: training data, validation, test results and limits of every model.

## Files
- `app.py`: the app.
- `build_models.py`: trains the models with the code in `src/`, writes `models/` and `data/`, and stops if a split,
  a test score, the monthly lift or an unusual-freight count drifts from the notebooks.
- `models/freight_forest.joblib`, `models/freight_ridge.joblib`, `models/late_logit.joblib`: the trained pipelines.
- `data/heldout_items.csv`, `data/heldout_orders.csv`: the held-out records the app scores.
- `data/zip_centroids.csv`: zip-code prefix → location and state, for listings and orders you type in.
- `data/facts.json`: every headline number and cut-off the app uses.
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
3. Under **Advanced settings** choose Python 3.12 or 3.13, then deploy. No secrets are needed, and the app has no
   API endpoints: everything happens in the browser pages above.
