# deployment/model_app/

The Phase 3 model app: a Streamlit app that turns the final models into decision tools for the two stakeholders of
our problems: Problem 1 from `notebooks/Team5_Phase3_IT5006_AY2627Sem1.ipynb` (re-evaluated on a time-ordered split
after the Phase 2 feedback) and Problem 2 from `notebooks/Team5_Phase2_IT5006_AY2627Sem1.ipynb`. The Phase 1 EDA
dashboard (`deployment/app.py`) is a separate app and is unchanged.

**Live (testing):** https://team5-olist-phase3-test1.streamlit.app; it moves to https://team5-olist-phase3.streamlit.app once the design is final. Streamlit Cloud builds it from
branch `Phase3_Test`, main file `deployment/model_app/app.py`.

## Architecture: an API and its frontend

The app holds **no model files**. The models run in a separate service, the model API in `deployment/api/` (FastAPI
on Render, documented in `deployment/api/README.md`), and this app calls it over HTTP:

| What the app shows | How it gets the scores |
|---|---|
| *Check a charge* (a new listing), *Promise check* | one call per form: `POST /freight/predict`, `POST /late/predict` |
| *Freight audit* with an uploaded CSV | `POST /freight/batch`, 2,000 rows per call; rows the API rejects are skipped and listed |
| the promise what-if curves (*Daily risk queue*, *Promise check*) | `POST /late/batch` with the order under every promise from 3 to 60 days |
| *Freight audit*, *Check a charge* (held-out item), *Tariff review*, *Daily risk queue*, *Monitoring* | `data/scored_items.csv` and `data/scored_orders.csv`: the scores the API returned for every held-out record in a batch run (`deployment/api/batch_score.py`, which checks them against the build) |
| *Model service* | `GET /`, `GET /model-info`, `GET /logs/summary` |

The API address comes from `st.secrets["API_URL"]`, then the `API_URL` environment variable, and defaults to
https://team5-olist-api.onrender.com. On Render's free tier the API sleeps after 15 minutes without traffic: the
app then shows *Waking the API (30–60 s)*, waits and retries. The sidebar shows whether the API is online, and every
form result shows the API's notes (how it placed the seller and customer) and drift warnings (inputs unlike the
training data).

The Olist data end in 2018, so the app scores **held-out historical records as if they were new**: the Problem 1
test items (the latest 20% of items, 24 May – 3 Sep 2018) and the Problem 2 test window (26 May – 29 Aug 2018), none
of which the models saw in training. It also scores listings and orders that you type in, and CSV files of items
that you upload.

## Pages, by stakeholder

**Logistics-pricing / finance analyst** (Problem 1, fair freight: RandomForest on a time-ordered split, with
Ridge as a second opinion)

| Page | When | Decision it supports |
|---|---|---|
| Freight audit | monthly | which sellers to audit (ranked by flag rate, with a minimum number of items) and which items to review (flagged by both models first); download both lists |
| Check a charge | ad hoc | is one held-out charge fair; the fair freight and quote range for a new listing |
| Tariff review | quarterly | where a simple published tariff, or what sellers charged, sits above or below the fair freight, by distance and weight |

An item is **unusual** when its residual (log scale) is in the top 5% of the test set. Two rules are offered at the
top of the analyst's pages: **Top 5% overall (evaluated)**, the rule the notebooks evaluate and the headline audit
numbers use, and **Top 5% within each price band (recommended)**, the rule the report recommends so expensive items
do not crowd the review list.

**Operations and customer-experience team** (Problem 2, late delivery: plain LogisticRegression)

| Page | When | Decision it supports |
|---|---|---|
| Daily risk queue | daily | which of the day's orders to follow up, with a suggested low-cost action for each (from its biggest risk driver), the expected yield, a download, and the promise that would have kept the order off the list; *Reveal what happened* shows the outcomes |
| Promise check | at checkout | the risk of a new order and the shortest promise that keeps it below the review line |
| Monitoring | weekly | whether a change in the promise rules calls for a model review (weekly promise, late rate and catch rate; monthly lift) |

**About:** the *Model card* (training data, validation, test results and limits of every model) and *Model service*
(the API's status, `/model-info` and the live monitoring totals of `/logs/summary`).

## Files
- `app.py`: the app (the API's frontend).
- `build_models.py`: trains the models with the code in `src/`, writes the API's files (`deployment/api/models/`,
  `deployment/api/data/`) and this app's `data/`, and stops if a split, a test score, the monthly lift or an
  unusual-freight count drifts from the notebooks.
- `data/heldout_items.csv`, `data/heldout_orders.csv`: the held-out records (inputs and, for orders, outcomes).
- `data/scored_items.csv`, `data/scored_orders.csv`: the API's scores for them, written by
  `deployment/api/batch_score.py`.
- `data/facts.json`: every headline number the app shows.
- `requirements.txt`: dependencies (no scikit-learn: the app loads no model).

## Run locally
From the repository root, start the API, then the app:
```bash
pip install -r deployment/api/requirements.txt -r deployment/model_app/requirements.txt
uvicorn main:app --app-dir deployment/api --port 8000
API_URL=http://localhost:8000 streamlit run deployment/model_app/app.py
```
Without `API_URL` the app calls the API on Render.

## Rebuild (only if the code in `src/` changes)
```bash
OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/model_app/build_models.py
python deployment/api/batch_score.py --api http://localhost:8000      # with the rebuilt API running
```
The build takes about a minute; the batch run about 20 seconds locally and a few minutes on Render's free tier. If
you change the library versions, update the pins in `deployment/api/requirements.txt` to match.

## Deploy on Streamlit Cloud
1. Deploy the API first (see `deployment/api/README.md`).
2. On https://share.streamlit.io choose **Create app**, pick this repository and branch, and set the main file path
   to `deployment/model_app/app.py`; choose Python 3.12 or 3.13 under **Advanced settings**.
3. If the API is not at https://team5-olist-api.onrender.com, add `API_URL = "https://<your-api>.onrender.com"` under
   **Settings → Secrets**.
