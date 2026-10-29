# -*- coding: utf-8 -*-
"""Tests for the model API, with FastAPI's TestClient (no server needed).

Run from deployment/api:  python -m pytest -q
"""
import json, os, tempfile
from pathlib import Path

os.environ["API_LOG_DIR"] = tempfile.mkdtemp()        # keep test traffic out of the service's own log

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)
HERE = Path(__file__).resolve().parent
HELDOUT = HERE.parent / "model_app" / "data"
ITEM = dict(category="bed_bath_table", price=120.0, weight_kg=1.5, volume_cm3=9000, seller_state="SP",
            customer_state="RJ", seller_zip="01310", customer_zip="20040", freight_value=35.0)
ORDER = dict(promised_days=12, price_total=150.0, freight_total=25.0, seller_state="SP", customer_state="BA")


def post(path, payload, **params):
    return client.post(path, json=payload, params=params)


# ------------------------------------------------------------------ service endpoints
def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy" and r.json()["models_loaded"] is True
    assert client.head("/").status_code == 200


def test_model_info_reports_the_tested_models():
    info = client.get("/model-info").json()
    assert info["models"]["freight_forest"]["test"]["MAE_rs"] == pytest.approx(5.398, abs=0.02)
    assert info["models"]["freight_ridge"]["params"]["alpha"] == 10
    assert info["models"]["late_logit"]["test"]["Lift"] == pytest.approx(2.0, abs=0.01)
    assert info["validation"]["max_batch"] == main.MAX_BATCH
    assert set(info["freight"]["unusual_rules"]) == {"overall", "band"}


def test_docs_are_generated():
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/freight/predict", "/freight/batch", "/late/predict", "/late/batch", "/logs/summary"} <= set(paths)


# ------------------------------------------------------------------ Problem 1: fair freight
def test_freight_predict_returns_quote_and_both_rules():
    p = post("/freight/predict", ITEM).json()
    assert p["fair_freight"] > 0 and p["fair_freight_ridge"] > 0
    assert p["quote_low"] == pytest.approx(max(0, p["fair_freight"] - p["band_mae"]))
    assert p["quote_high"] == pytest.approx(p["fair_freight"] + p["band_mae"])
    assert p["price_band"] == "100-200"
    assert set(p["check"]["rules"]) == {"overall", "band"}
    assert p["check"]["rules"]["band"]["label"] == "Top 5% within each price band (recommended)"
    assert p["check"]["rules"]["overall"]["label"] == "Top 5% overall (evaluated)"
    assert p["request_id"].startswith("req_") and p["latency_ms"] >= 0


def test_no_charge_no_check():
    p = post("/freight/predict", {k: v for k, v in ITEM.items() if k != "freight_value"}).json()
    assert p["check"] is None


def test_a_high_charge_is_unusual_under_both_rules():
    fair = post("/freight/predict", ITEM).json()["fair_freight"]
    p = post("/freight/predict", dict(ITEM, freight_value=round(fair * 4, 2))).json()
    for rule in ("overall", "band"):
        assert p["check"]["rules"][rule]["verdict"] == "unusual_both"
    assert p["check"]["excess"] > 0


@pytest.mark.parametrize("zip_text", ["01310", "01310-100", "1310", 1310, "01310100", "1310100"])
def test_zip_formats_give_the_same_location(zip_text):
    base = post("/freight/predict", ITEM).json()["inputs"]["distance_km"]
    p = post("/freight/predict", dict(ITEM, seller_zip=zip_text)).json()
    assert p["inputs"]["distance_km"] == pytest.approx(base)


def test_zip_in_another_state_wins_with_a_note():
    p = post("/freight/predict", dict(ITEM, seller_state="RJ")).json()       # zip 01310 is in SP
    assert p["inputs"]["seller_state"] == "SP"
    assert any("not RJ" in n for n in p["notes"])


def test_distance_given_null_or_derived():
    given = post("/freight/predict", dict(ITEM, distance_km=800)).json()
    assert given["inputs"]["distance_km"] == 800
    unknown = post("/freight/predict", dict(ITEM, distance_km=None)).json()
    assert unknown["inputs"]["distance_km"] is None and unknown["tariff"] is None
    assert any("Distance unknown" in n for n in unknown["notes"])
    no_zip = post("/freight/predict", {k: v for k, v in ITEM.items() if not k.endswith("_zip")}).json()
    assert any("centre of SP" in n for n in no_zip["notes"])


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("change, field", [
    (dict(price=0), "price"), (dict(weight_kg=-1), "weight_kg"), (dict(volume_cm3=0), "volume_cm3"),
    (dict(seller_state="XX"), "seller_state"), (dict(customer_zip="12"), "customer_zip"),
    (dict(distance_km=9000), "distance_km"), (dict(freight_value=-5), "freight_value"),
    (dict(n_items=0), "n_items")])
def test_invalid_inputs_are_rejected_clearly(change, field):
    r = post("/freight/predict", dict(ITEM, **change))
    assert r.status_code == 422
    assert any(e.startswith(field) for e in r.json()["errors"])


def test_missing_required_field():
    r = post("/late/predict", {k: v for k, v in ORDER.items() if k != "promised_days"})
    assert r.status_code == 422 and r.json()["errors"] == ["promised_days: Field required"]


def test_more_sellers_than_items_is_rejected():
    r = post("/late/predict", dict(ORDER, n_items=1, n_sellers=2))
    assert r.status_code == 422 and "n_sellers cannot exceed n_items" in r.json()["errors"][0]


def test_batch_size_is_capped():
    r = post("/freight/batch", {"items": [ITEM] * (main.MAX_BATCH + 1)})
    assert r.status_code == 422 and "split the rows" in r.json()["errors"][0]
    assert post("/freight/batch", {"items": []}).status_code == 422


# ------------------------------------------------------------------ drift warnings
def test_drift_warnings_for_unusual_inputs():
    p = post("/freight/predict", dict(ITEM, weight_kg=200, category="space_rockets", n_items=30, n_sellers=1)).json()
    text = " ".join(p["drift_warnings"])
    assert "weight_kg" in text and "space_rockets" in text and "n_items" in text
    assert post("/freight/predict", ITEM).json()["drift_warnings"] == []


def test_late_drift_on_a_very_long_promise():
    p = post("/late/predict", dict(ORDER, promised_days=120)).json()
    assert any(w.startswith("promised_days") for w in p["drift_warnings"])


# ------------------------------------------------------------------ Problem 2: late delivery
def test_late_predict_drivers_explain_the_score():
    a = post("/late/predict", ORDER).json()
    b = post("/late/predict", dict(ORDER, promised_days=30, customer_state="SP")).json()
    assert 0 < a["risk"] < 1 and 0 <= a["percentile"] <= 100
    assert set(a["drivers"]) == set(main.DRIVERS)
    # log-odds = intercept + coef . reference + sum of the contributions, so this difference is the same for any order
    logit = lambda p: np.log(p["risk"] / (1 - p["risk"]))
    assert logit(a) - sum(a["drivers"].values()) == pytest.approx(logit(b) - sum(b["drivers"].values()), abs=1e-9)
    assert a["suggested_follow_up"] in main.ACTIONS.values()


def test_longer_promises_lower_the_risk():
    r = post("/late/batch", {"orders": [dict(ORDER, promised_days=d) for d in range(3, 61)]}).json()
    risk = [p["risk"] for p in r["predictions"]]
    assert all(x > y for x, y in zip(risk, risk[1:]))


def test_review_share():
    assert post("/late/predict", ORDER, share=0.5).json()["share"] == 0.5
    assert post("/late/predict", ORDER, share=0.9).status_code == 422


# ------------------------------------------------------------------ the API reproduces the build
@pytest.mark.skipif(not (HELDOUT / "heldout_items.csv").exists(), reason="held-out records not available")
def test_heldout_records_match_the_local_pipelines():
    items = pd.read_csv(HELDOUT / "heldout_items.csv").sample(400, random_state=0)
    orders = pd.read_csv(HELDOUT / "heldout_orders.csv").sample(400, random_state=0)
    items = pd.concat([items, pd.read_csv(HELDOUT / "heldout_items.csv").query("distance_km != distance_km").head(5)])
    to_rows = lambda d, f: d[f].astype(object).where(d[f].notna(), None).to_dict("records")
    fi = ["category", "price", "weight_kg", "volume_cm3", "distance_km", "n_items", "n_sellers", "seller_state",
          "customer_state", "freight_value"]
    fo = ["promised_days", "price_total", "freight_total", "distance_km", "n_items", "n_sellers", "seller_state",
          "customer_state"]
    api_items = post("/freight/batch", {"items": to_rows(items, fi)}).json()["predictions"]
    api_orders = post("/late/batch", {"orders": to_rows(orders, fo)}).json()["predictions"]
    forest = joblib.load(HERE / "models" / "freight_forest.joblib")
    logit = joblib.load(HERE / "models" / "late_logit.joblib")
    local = np.clip(np.expm1(forest.predict(items[main.FR["columns"]])), 0, None)
    tol = main.FLAG["tol"]
    assert np.abs(np.log1p([p["fair_freight"] for p in api_items]) - np.log1p(local)).max() <= tol
    assert np.abs(np.array([p["risk"] for p in api_orders])
                  - logit.predict_proba(orders[main.LT["columns"]])[:, 1]).max() <= 1e-12


# ------------------------------------------------------------------ logging
def test_every_prediction_is_logged_and_summarised():
    before = client.get("/logs/summary").json()
    post("/late/batch", {"orders": [ORDER, dict(ORDER, promised_days=40)]})
    post("/late/predict", dict(ORDER, promised_days=-1))
    after = client.get("/logs/summary").json()
    assert after["predictions"]["late"] == before["predictions"]["late"] + 2
    assert after["requests"]["rejected_422"] == before["requests"]["rejected_422"] + 1
    lines = [json.loads(x) for x in main.LOG_FILE.read_text(encoding="utf-8").splitlines()]
    last = [e for e in lines if e.get("endpoint") == "/late/batch"][-1]
    assert {"timestamp", "request_id", "input", "outputs", "has_drift", "drift_warnings", "latency_ms"} <= set(last)
