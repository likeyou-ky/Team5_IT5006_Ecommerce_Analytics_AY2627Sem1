# -*- coding: utf-8 -*-
"""
Bulk scoring through the model API: score every held-out record and write the files the Streamlit app reads.

The app's population pages (Freight audit, Tariff review, Daily risk queue, Monitoring) show thousands of records, so
they read scores computed in advance instead of calling the API on every page view. This script computes them the
way any batch client would: it posts the held-out items and orders (deployment/model_app/data/heldout_*.csv) to
/freight/batch and /late/batch in chunks, against a local uvicorn or the Render service, and writes

  deployment/model_app/data/scored_items.csv    fair freight from the forest and from Ridge, and the unusual flags
                                                under both rules (forest and Ridge)
  deployment/model_app/data/scored_orders.csv   risk score, percentile, suggested follow-up and each input's
                                                contribution to the log-odds

It then checks the API against the pipelines run locally: the forest to within FLAG_TOL on the log scale (its
parallel predictions vary in the last bits), Ridge and the logistic model to 1e-9, and every unusual-freight and
review-queue count in facts.json, which build_models.py checked against the notebooks.

Usage (from the repository root):
    uvicorn main:app --app-dir deployment/api --port 8000        # in another terminal
    python deployment/api/batch_score.py --api http://localhost:8000
    python deployment/api/batch_score.py --api https://team5-olist-api.onrender.com
"""
import argparse, json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
APP_DATA = HERE.parent / "model_app" / "data"
sys.path.insert(0, str(HERE.parents[1] / "src"))
from models import flag_top_share               # the queue rule of the notebooks and build_models.py

DEFAULT_API = "https://team5-olist-api.onrender.com"
ITEM_FIELDS = ["category", "price", "weight_kg", "volume_cm3", "distance_km", "n_items", "n_sellers", "seller_state",
               "customer_state", "freight_value"]
ORDER_FIELDS = ["promised_days", "price_total", "freight_total", "distance_km", "n_items", "n_sellers", "seller_state",
                "customer_state"]


def wait_for_api(api, limit=180):
    """Ping / until the service answers: a free Render service sleeps after 15 minutes and takes 30-60 s to wake."""
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            r = requests.get(f"{api}/", timeout=60)
            if r.ok:
                print(f"API up after {time.time() - t0:.0f} s: {r.json()['service']} {r.json()['version']}")
                return
        except requests.RequestException:
            pass
        print("  waiting for the API to wake up...")
        time.sleep(5)
    raise SystemExit(f"The API at {api} did not answer within {limit} s.")


def post(api, path, payload, tries=3):
    for k in range(tries):
        try:
            r = requests.post(f"{api}{path}", json=payload, timeout=300)
            if r.status_code == 422:
                raise SystemExit(f"{path} rejected the batch: {r.json()['errors'][:5]}")
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if k == tries - 1:
                raise
            print(f"  {path} failed ({e}); retrying")
            time.sleep(5)


def rows(df, fields, ids):
    """JSON rows. A missing distance (zip not in the Olist table) is sent as null, so the models impute it exactly
    as they did in training; leaving the field out would make the API derive it from the state centres instead."""
    out = df[fields].astype(object).where(df[fields].notna(), None).to_dict("records")
    for r, i in zip(out, ids):
        r["id"] = i
    return out


def score(api, path, key, records, chunk):
    preds, t0 = [], time.time()
    for s in range(0, len(records), chunk):
        res = post(api, path, {key: records[s:s + chunk]})
        preds += res["predictions"]
        print(f"  {path}: {min(s + chunk, len(records)):,} of {len(records):,} "
              f"({res['latency_ms'] / 1000:.1f} s in the API, {res['n_with_drift']} with drift warnings)")
    print(f"  {len(preds):,} scored in {time.time() - t0:.0f} s")
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default {DEFAULT_API})")
    ap.add_argument("--chunk", type=int, default=2000, help="rows per request (the API takes at most 5,000)")
    ap.add_argument("--out", default=str(APP_DATA), help="folder for scored_items.csv and scored_orders.csv")
    a = ap.parse_args()
    api, out = a.api.rstrip("/"), Path(a.out)
    wait_for_api(api)

    # ------------------------------------------------------------------ score through the API
    items = pd.read_csv(APP_DATA / "heldout_items.csv")
    orders = pd.read_csv(APP_DATA / "heldout_orders.csv")
    item_ids = (items.order_id + ":" + items.order_item_id.astype(str)).tolist()
    print(f"Scoring {len(items):,} held-out items and {len(orders):,} held-out orders through {api}")
    p_items = score(api, "/freight/batch", "items", rows(items, ITEM_FIELDS, item_ids), a.chunk)
    p_orders = score(api, "/late/batch", "orders", rows(orders, ORDER_FIELDS, orders.order_id.tolist()), a.chunk)
    assert [p["id"] for p in p_items] == item_ids and [p["id"] for p in p_orders] == orders.order_id.tolist()

    s_items = items[["order_id", "order_item_id"]].assign(
        expected=[p["fair_freight"] for p in p_items], expected_ridge=[p["fair_freight_ridge"] for p in p_items],
        audit_band=[p["check"]["audit_band"] for p in p_items],
        **{f"unusual{'_ridge' if m == 'ridge' else ''}_{rule}": [p["check"]["rules"][rule][m] for p in p_items]
           for rule in ("overall", "band") for m in ("forest", "ridge")},
        drift_warnings=[len(p["drift_warnings"]) for p in p_items])
    drivers = pd.DataFrame([p["drivers"] for p in p_orders]).add_prefix("drv_")
    s_orders = pd.concat([orders[["order_id"]].assign(
        risk=[p["risk"] for p in p_orders], percentile=[p["percentile"] for p in p_orders],
        action=[p["suggested_follow_up"] for p in p_orders]), drivers], axis=1).assign(
        drift_warnings=[len(p["drift_warnings"]) for p in p_orders])

    # ------------------------------------------------------------------ check against the local pipelines and facts.json
    meta = json.loads((HERE / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    facts = json.loads((APP_DATA / "facts.json").read_text(encoding="utf-8"))
    tol = meta["freight"]["flag"]["tol"]
    load = lambda f: joblib.load(HERE / "models" / f)
    forest, ridge, logit = load("freight_forest.joblib"), load("freight_ridge.joblib"), load("late_logit.joblib")
    fair = lambda m: np.clip(np.expm1(m.predict(items[meta["freight"]["columns"]])), 0, None)
    X_b = orders[meta["late"]["columns"]]
    prep, coef = logit.named_steps["prep"], logit.named_steps["m"].coef_.ravel()
    contrib = (prep.transform(X_b) - np.array(meta["late"]["reference"]["values"])) * coef
    names = [n.removeprefix("missingindicator_") for n in prep.get_feature_names_out()]
    names = [next((c for c in ("seller_state", "customer_state") if n.startswith(c + "_")), n) for n in names]
    local_drv = pd.DataFrame(contrib, columns=names).T.groupby(level=0).sum().T
    gap = {
        "forest, log scale": np.abs(np.log1p(s_items.expected) - np.log1p(fair(forest))).max(),
        "ridge, log scale": np.abs(np.log1p(s_items.expected_ridge) - np.log1p(fair(ridge))).max(),
        "logistic risk": np.abs(s_orders.risk - logit.predict_proba(X_b)[:, 1]).max(),
        "driver contributions": np.abs(drivers.rename(columns=lambda c: c[4:])[local_drv.columns].values
                                       - local_drv.values).max(),
    }
    fl, p2 = facts["p1"]["flag"], facts["p2"]
    y = orders.clf_target.values
    daily = np.zeros(len(orders), dtype=int)
    for _, idx in orders.groupby(pd.to_datetime(orders.order_purchase_timestamp).dt.normalize()).indices.items():
        daily[idx] = flag_top_share(s_orders.risk.values[idx], p2["share"])
    whole = flag_top_share(s_orders.risk.values, p2["share"])
    counts = {
        **{f"unusual items, forest, {r}": (int(s_items[f"unusual_{r}"].sum()), fl["n_unusual"][r]) for r in fl["n_unusual"]},
        **{f"Ridge agrees, {r}": (round((s_items[f"unusual_{r}"] & s_items[f"unusual_ridge_{r}"]).sum()
                                        / s_items[f"unusual_{r}"].sum(), 3), fl["both_share"][r]) for r in fl["both_share"]},
        "daily list, orders flagged": (int(daily.sum()), p2["daily"]["flagged"]),
        "daily list, late orders caught": (int((daily & y).sum()), p2["daily"]["late_caught"]),
        "whole window, late orders caught": (int((whole & y).sum()), p2["late_caught"]),
    }
    print("\nAPI vs the pipelines run locally (largest difference):")
    for k, v in gap.items():
        print(f"  {k:<24} {v:.2e}")
    print("API scores vs facts.json (computed, facts):")
    for k, (c, f) in counts.items():
        print(f"  {k:<34} {c} vs {f}{'' if c == f else '   <-- MISMATCH'}")
    bad = [k for k, v in gap.items() if not v <= tol] + [k for k, (c, f) in counts.items() if c != f]
    if bad:
        raise SystemExit(f"The API's scores differ from the build: {bad}. Nothing was written.")

    out.mkdir(parents=True, exist_ok=True)
    s_items.to_csv(out / "scored_items.csv", index=False)       # full precision, so the app shows the API's numbers
    s_orders.to_csv(out / "scored_orders.csv", index=False)
    print(f"\nAll checks pass. Wrote {out / 'scored_items.csv'} and {out / 'scored_orders.csv'}.")


if __name__ == "__main__":
    main()
