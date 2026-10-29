# -*- coding: utf-8 -*-
"""
Build the model files and data for the Phase 3 deployment: the model API (deployment/api/main.py, FastAPI on Render)
and the model app that calls it (deployment/model_app/app.py, Streamlit).

Both models are trained with the notebooks' own code in src/ on the notebooks' own training splits:

* Problem 1, fair freight (notebooks/Team5_Phase3_IT5006_AY2627Sem1.ipynb): RandomForest regressor (200 trees) on
  log1p(freight), trained on the earliest 80% of items (purchased up to 24 May 2018), with the feature groups,
  settings and model chosen by forward-chaining CV. The test items are the latest 20%. The tuned Ridge model is the
  audit's second opinion: items flagged by both models come first.
* Problem 2, late delivery (notebooks/Team5_Phase2_IT5006_AY2627Sem1.ipynb, Sections 4-8): plain
  LogisticRegression on the time-ordered training split (orders up to 26 Apr 2018).

Unusual freight = residual in the top 5% of the test set, either overall (the notebook's rule) or within each price
band (the report's recommendation, so expensive items do not crowd the review list). Both cut-offs are stored for
both models.

Writes into deployment/api/ (the API service, self-contained so Render can build it from that folder):
  models/freight_forest.joblib   the Problem 1 forest
  models/freight_ridge.joblib    the Problem 1 Ridge model (second opinion for the audit)
  models/late_logit.joblib       the Problem 2 logistic model
  models/model_metadata.json     what the API needs besides the models: features, metrics, cut-offs, bands, tariff
  models/feature_stats.json      training distribution of every input, for the API's drift warnings
  data/zip_centroids.csv         zip-code prefix -> centroid and state, to place new listings and orders
and into deployment/model_app/data/ (the app):
  heldout_items.csv              Problem 1 test items (the latest 20%, purchased after every training item)
  heldout_orders.csv             Problem 2 test window, 26 May - 29 Aug 2018, with outcomes for the after-the-fact check
  facts.json                     every number the app shows

The app's scores for the held-out records (scored_items.csv, scored_orders.csv) come from the API: run
deployment/api/batch_score.py after this script.

The checks at the end stop the script if a split or a test score drifts from the notebooks. Forest scores
differ in the third decimal between Windows and macOS runs (same data, same library versions), so they are checked
with a tolerance; the splits, the tariff and the logistic model must match to rounding.

Usage:
    OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/model_app/build_models.py
"""
import json, os, platform, sys
import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import features as F, models as M, pipeline as P

API = os.path.normpath(os.path.join(HERE, "..", "api"))
MODELS, API_DATA, DATA = os.path.join(API, "models"), os.path.join(API, "data"), os.path.join(HERE, "data")
for folder in (MODELS, API_DATA, DATA):
    os.makedirs(folder, exist_ok=True)

RF_PARAMS = dict(min_samples_leaf=2, max_features=0.33, max_depth=16)   # Phase 3 notebook Section 3, tuning table
RIDGE_ALPHA = 10                                                        # Phase 3 notebook Section 3, tuning table
FINAL_A = "Tree: RandomForest"                                          # Phase 3 notebook Section 4 (paired rule)
FLAG_Q = 0.95                       # unusual freight = top 5% of residuals (Phase 3 notebook Section 9)
# Identical units of one listing in one order share a residual, and the cut-off can land on such a tie. The forest's
# parallel predictions vary in the last bits between calls (about 1e-13), which would flip the tied items, so the
# rule allows a negligible tolerance on the log scale.
FLAG_TOL = 1e-9
SHARE = F.REVIEW_SHARE              # review the riskiest 10% of orders (notebook Section 4)
PRICE_BANDS = ([0, 25, 50, 100, 200, 500, np.inf], ["<25", "25-50", "50-100", "100-200", "200-500", "500+"])
AUDIT_BANDS = ([0, 50, 100, 200, 500, np.inf], ["<50", "50-100", "100-200", "200-500", "500+"])  # pipeline.audit_concentration


def residual(freight, pred_log):
    """Unusual-freight residual exactly as the API computes it: log(1 + charged) - log(1 + expected)."""
    return np.log1p(np.asarray(freight)) - np.log1p(np.clip(np.expm1(pred_log), 0, None))


def audit_band(price):
    return pd.cut(price, AUDIT_BANDS[0], labels=AUDIT_BANDS[1], right=False).astype(str).values


def unusual(res, band, cut, rule):
    """Flag residuals at or above the cut-off: one cut-off overall, or one per price band."""
    thr = cut["overall"] if rule == "overall" else pd.Series(band).map(cut["band"]).values.astype(float)
    return res >= thr - FLAG_TOL

d = F.find_data_dir()
print("Reading raw data from:", d)

# ============================ tables and splits (Phase 3 notebook Section 1; Phase 2 notebook Section 4) ============
items = F.build_item_table(d)
orders = F.build_order_table(d)
tr_a, te_a = M.time_holdout(items, embargo_days=0)      # freight is known at checkout: no label delay
tr_b, te_b = M.time_holdout(orders)
assert not set(tr_a.order_id) & set(te_a.order_id), "Problem 1: an order is in both train and test"
assert tr_a.purchase_ts.max() <= te_a.purchase_ts.min(), "Problem 1: a training item is later than a test item"
assert tr_b.purchase_ts.max() < te_b.purchase_ts.min() - F.EMBARGO_DAYS * M.DAY, "Problem 2: embargo violated"

# Feature sets: Problem 1 = what the forward-chaining feature selection of Phase 3 Section 2 keeps (listing details and
# purchase timing did not help on later periods); Problem 2 = what the forward selection of Phase 2 Section 5 keeps
fs_a = F.FEATURE_SETS["3 + price & basket"]
fs_b = P.merge_sets(P.merge_sets(F.LATE_BASE, F.LATE_GROUPS["route (distance, states)"]),
                    F.LATE_GROUPS["order value & freight"])
cols_a, cols_b = P.cols_for(fs_a), P.cols_for(fs_b)

# ============================ Problem 1: fair freight ============================
model = P.final_reg_models(fs_a, {"ridge_alpha": RIDGE_ALPHA, "rf_reg": RF_PARAMS})[FINAL_A]
model.fit(tr_a[cols_a], tr_a.reg_target)
pred_a = model.predict(te_a[cols_a])
reg_test = M.reg_metrics(te_a.reg_target, pred_a)
print(f"Problem 1 test MAE: R${reg_test['MAE_rs']:.3f}")

ridge = M.reg_estimators(fs_a, ridge_alpha=RIDGE_ALPHA)["Linear: Ridge"].fit(tr_a[cols_a], tr_a.reg_target)
pred_r = ridge.predict(te_a[cols_a])
ridge_test = M.reg_metrics(te_a.reg_target, pred_r)

band_a = audit_band(te_a.price)
cuts, flags = {}, {}
for name, pred in [("model", pred_a), ("ridge", pred_r)]:
    res = residual(te_a.freight_value, pred)
    cuts[name] = {"overall": float(np.quantile(res, FLAG_Q)),
                  "band": {b: float(np.quantile(res[band_a == b], FLAG_Q)) for b in AUDIT_BANDS[1]}}
    flags[name] = {rule: unusual(res, band_a, cuts[name], rule) for rule in ("overall", "band")}
exp_a = np.clip(np.expm1(pred_a), 0, None)
band = pd.cut(te_a.price, PRICE_BANDS[0], labels=PRICE_BANDS[1], right=False)
err = (te_a.freight_value - exp_a).abs()
by_band = (pd.DataFrame({"band": band, "err": err, "freight": te_a.freight_value})
           .groupby("band", observed=True).agg(n=("err", "size"), mae=("err", "mean"), mean_freight=("freight", "mean")))
tariff = P.implied_tariff(tr_a, te_a).Value.tolist()      # base, per kg, per 100 km, same state, test MAE

# ============================ Problem 2: late delivery ============================
logit = M.clf_estimators(fs_b)["Linear: LogisticRegression (plain)"].fit(tr_b[cols_b], tr_b.clf_target)
score_b = M.clf_scores(logit, te_b[cols_b])
flags_b = M.flag_top_share(score_b, SHARE)
clf_test = M.clf_metrics(te_b.clf_target, score_b, pred=flags_b)

# The app's queue works day by day: flag the riskiest 10% of each purchase day's orders
day = te_b.order_purchase_timestamp.dt.normalize()
daily = np.zeros(len(te_b), dtype=int)
for _, idx in te_b.groupby(day).indices.items():
    daily[idx] = M.flag_top_share(score_b[idx], SHARE)
y_b = te_b.clf_target.values
gain = P.gain_table(te_b.clf_target, score_b)
by_month = P.test_by_month(logit, te_b, fs_b)             # skill falls as the promise rules change (Section 10)

# ============================ data files for the app ============================
low = F.load_low_reviews(d).set_index("order_id").low
ITEM_COLS = ["order_id", "order_item_id", "product_id", "seller_id", "order_purchase_timestamp", "freight_value"]
held_items = te_a[ITEM_COLS + [c for c in cols_a if c not in ITEM_COLS]]
ORDER_COLS = ["order_id", "seller_id", "category", "order_purchase_timestamp"]
held_orders = te_b[ORDER_COLS + [c for c in cols_b if c not in ORDER_COLS] + ["clf_target", "days_late"]].assign(
    low_review=low.reindex(te_b.order_id).values)       # outcome only, shown after the fact, never a feature
held_items.to_csv(os.path.join(DATA, "heldout_items.csv"), index=False)      # full precision: the API must
held_orders.to_csv(os.path.join(DATA, "heldout_orders.csv"), index=False)    # reproduce these scores exactly
back = pd.read_csv(os.path.join(DATA, "heldout_items.csv"))                   # re-scored as the API does
app_flags = {(name, rule): int(unusual(residual(back.freight_value, m.predict(back[cols_a])), audit_band(back.price),
                                       cuts[name], rule).sum())
             for name, m in [("model", model), ("ridge", ridge)] for rule in ("overall", "band")}

geo = pd.read_csv(os.path.join(d, "olist_geolocation_dataset.csv"))
geo = geo[geo.geolocation_lat.between(*F.BR_LAT) & geo.geolocation_lng.between(*F.BR_LON)]
zips = (geo.groupby("geolocation_zip_code_prefix")       # same centroids as features._item_frame
        .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"),
             state=("geolocation_state", lambda s: s.value_counts().index[0]))
        .reset_index().rename(columns={"geolocation_zip_code_prefix": "zip_prefix"}))
zips.to_csv(os.path.join(API_DATA, "zip_centroids.csv"), index=False, float_format="%.6f")
state_cen = zips.groupby("state")[["lat", "lng"]].median().round(4)

joblib.dump(model, os.path.join(MODELS, "freight_forest.joblib"), compress=3)
joblib.dump(ridge, os.path.join(MODELS, "freight_ridge.joblib"), compress=3)
joblib.dump(logit, os.path.join(MODELS, "late_logit.joblib"), compress=3)

# ============================ facts.json ============================
r = lambda x, n=3: round(float(x), n)
rng = lambda c: [r(tr_a[c].quantile(0.001), 2), r(tr_a[c].quantile(0.999), 2)]
cats = tr_a.category.value_counts()
facts = {
    "built": {"date": pd.Timestamp.now().strftime("%Y-%m-%d"), "python": platform.python_version(),
              "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
              "seed": F.RANDOM_STATE},
    "p1": {
        "n_train": len(tr_a), "n_test": len(te_a),
        "train_dates": [f"{tr_a.order_purchase_timestamp.min():%Y-%m-%d}", f"{tr_a.order_purchase_timestamp.max():%Y-%m-%d}"],
        "test_dates": [f"{te_a.order_purchase_timestamp.min():%Y-%m-%d}", f"{te_a.order_purchase_timestamp.max():%Y-%m-%d}"],
        "mean_freight_train": r(tr_a.freight_value.mean(), 2), "mean_freight_test": r(te_a.freight_value.mean(), 2),
        "features_num": fs_a["num"], "features_cat": fs_a["cat"],
        "forest_params": dict(RF_PARAMS, n_estimators=200),
        "test": {**{k: r(reg_test[k]) for k in ["MAE_rs", "RMSE_rs", "R2_rs"]},
                 "MAE_pct": r(reg_test["MAE_rs"] / te_a.freight_value.mean() * 100, 1)},
        "ridge_alpha": RIDGE_ALPHA, "ridge_test": {k: r(ridge_test[k]) for k in ["MAE_rs", "RMSE_rs", "R2_rs"]},
        # unusual-freight cut-offs on the log1p residual, unrounded (they are thresholds)
        "flag": {"q": FLAG_Q, "tol": FLAG_TOL, "band_edges": AUDIT_BANDS[0][1:-1], "band_labels": AUDIT_BANDS[1],
                 "cuts": cuts, "ratio_overall": r(np.exp(cuts["model"]["overall"]), 3),
                 "n_unusual": {rule: int(flags["model"][rule].sum()) for rule in ("overall", "band")},
                 "both_share": {rule: r((flags["model"][rule] & flags["ridge"][rule]).sum()
                                        / flags["model"][rule].sum()) for rule in ("overall", "band")}},
        "price_bands": [{"band": b, "n": int(g.n), "mae": r(g.mae, 2), "mae_pct": r(g.mae / g.mean_freight * 100, 1)}
                        for b, g in by_band.iterrows()],
        "tariff": dict(zip(["base", "per_kg", "per_100km", "same_state", "mae"], [r(v) for v in tariff])),
        "categories": cats.index.tolist(), "category_counts": cats.astype(int).tolist(),
        "states": sorted(set(tr_a.seller_state) | set(tr_a.customer_state)),
        "ranges": {c: rng(c) for c in ["weight_kg", "volume_cm3", "price", "distance_km"]},
        "defaults": {c: r(tr_a[c].median(), 0) for c in ["photos_qty", "name_len", "desc_len"]},
    },
    "p2": {
        "n_train": len(tr_b), "n_test": len(te_b),
        "train_dates": [f"{tr_b.order_purchase_timestamp.min():%Y-%m-%d}", f"{tr_b.order_purchase_timestamp.max():%Y-%m-%d}"],
        "test_dates": [f"{te_b.order_purchase_timestamp.min():%Y-%m-%d}", f"{te_b.order_purchase_timestamp.max():%Y-%m-%d}"],
        "late_rate_train": r(tr_b.clf_target.mean(), 4), "late_rate_test": r(te_b.clf_target.mean(), 4),
        "features_num": fs_b["num"], "features_cat": fs_b["cat"], "share": SHARE,
        "test": {k: r(clf_test[k]) for k in ["Lift", "PR_AUC", "ROC_AUC", "Precision", "Recall", "F1", "Brier"]},
        "late_test": int(y_b.sum()), "late_caught": int((flags_b & y_b).sum()),
        "daily": {"flagged": int(daily.sum()), "precision": r((daily & y_b).sum() / daily.sum()),
                  "recall": r((daily & y_b).sum() / y_b.sum()), "late_caught": int((daily & y_b).sum())},
        "gain": gain[["decile", "orders", "late", "cum_capture", "lift"]].round(3).to_dict("records"),
        "by_month": [{"month": m.Month, "orders": int(m.Orders), "late_rate": r(m["Late rate"], 4), "lift": r(m.Lift, 2),
                      "roc_auc": r(m["ROC-AUC"])} for _, m in by_month.iterrows()],
        "score_pct": [r(v, 5) for v in np.quantile(score_b, np.linspace(0, 1, 101))],
        "ranges": {c: [r(tr_b[c].quantile(0.001), 2), r(tr_b[c].quantile(0.999), 2)]
                   for c in ["promised_days", "price_total", "freight_total", "distance_km"]},
    },
    "state_centroids": {s: [row.lat, row.lng] for s, row in state_cen.iterrows()},
    # Notebook results that need more than the deployed models; shown as cited. Problem 1: Phase 3 notebook;
    # Problem 2: Phase 2 notebook (main, 84ba8ab)
    "notebook": {
        "p1_cv_mae": 3.71, "p1_cv_mae_sd": 0.88, "p1_ridge_test_mae": 6.12, "p1_tree_test_mae": 5.69,
        "p1_stack_test_mae": 5.32, "p1_unseen_mae": 5.76, "p1_unseen_pct": 25.6,
        "p1_random_mae": 3.65, "p1_random_pct": 18.3,
        "p2_cv_lift": 2.01, "p2_cv_lift_sd": 0.34, "p2_window_lift": [1.47, 1.71, 2.24, 2.00],
        "p2_window_recall": [0.178, 0.202, 0.289, 0.228], "p2_low_review_ratio": [1.2, 2.0],
    },
}
json.dump(facts, open(os.path.join(DATA, "facts.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# ============================ the API's metadata and drift statistics ============================
p1, p2 = facts["p1"], facts["p2"]
prep_b = logit.named_steps["prep"]
metadata = {
    "service": "Team 5 Olist model API",
    "built": facts["built"],
    "models": {
        "freight_forest": {"file": "freight_forest.joblib", "problem": "Problem 1, fair freight", "role": "fair freight",
                           "estimator": "RandomForestRegressor on log(1 + freight)", "params": p1["forest_params"],
                           "test": p1["test"]},
        "freight_ridge": {"file": "freight_ridge.joblib", "problem": "Problem 1, fair freight",
                          "role": "second opinion on unusual charges", "estimator": "Ridge on log(1 + freight)",
                          "params": {"alpha": RIDGE_ALPHA}, "test": p1["ridge_test"]},
        "late_logit": {"file": "late_logit.joblib", "problem": "Problem 2, late delivery",
                       "role": "late-delivery risk ranking", "estimator": "LogisticRegression (no penalty)",
                       "params": {"penalty": None}, "test": p2["test"], "daily": p2["daily"]},
    },
    "freight": {**{k: p1[k] for k in ["n_train", "n_test", "train_dates", "test_dates", "mean_freight_test",
                                      "features_num", "features_cat", "flag", "tariff", "categories", "states"]},
                "columns": cols_a, "unseen_listing_mae": facts["notebook"]["p1_unseen_mae"],
                # quote range = fair freight +/- the test MAE of the item's price band (upper edges; null = no limit)
                "price_bands": [dict(b, upper=None if np.isinf(hi) else hi)
                                for b, hi in zip(p1["price_bands"], PRICE_BANDS[0][1:])]},
    "late": {**{k: p2[k] for k in ["n_train", "n_test", "train_dates", "test_dates", "late_rate_test", "features_num",
                                   "features_cat", "share", "score_pct"]},
             "columns": cols_b,
             # driver contributions are measured against the average test-window order, as the app's queue shows them
             "reference": {"names": prep_b.get_feature_names_out().tolist(),
                           "values": prep_b.transform(te_b[cols_b]).mean(axis=0).tolist()}},
    "state_centroids": facts["state_centroids"],
}
json.dump(metadata, open(os.path.join(MODELS, "model_metadata.json"), "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)


COUNTS = ["n_items", "n_sellers"]


def num_stats(frame, cols):
    """Training mean and sd of each numeric input; the skewed ones on the log1p scale, as the linear models see them.
    Counts are mostly 1, so any multi-seller order would sit 3 sd out: they are flagged only above the training maximum."""
    out = {}
    for c in cols:
        if c == "same_state":
            continue                                        # a 0/1 flag derived from the states: nothing to drift
        x = frame[c].dropna().astype(float)
        if c in COUNTS:
            out[c] = {"scale": "count", "max": int(x.max()), "n": int(len(x))}
            continue
        log = c in M.SKEWED
        v = np.log1p(x) if log else x
        mu, sd = float(v.mean()), float(v.std())
        back = np.expm1 if log else float
        out[c] = {"scale": "log1p" if log else "raw", "mean": mu, "std": sd, "n": int(len(x)),
                  "low": float(back(mu - 3 * sd)), "high": float(back(mu + 3 * sd)),   # the 3-sd band in R$, kg, km...
                  "train_share_outside": r((np.abs(v - mu) > 3 * sd).mean(), 4)}
    return out


min_freq = model.named_steps["prep"].named_transformers_["cat"].min_frequency
feature_stats = {
    "method": "An input is flagged when it lies more than 3 standard deviations from its training mean. Skewed inputs "
              "(weights, volumes, distances, prices, freight) are compared on the log1p scale, as the linear models "
              "see them. Counts (items and sellers in the order) are flagged above the training maximum, and "
              "categories the model did not see in training are flagged too.",
    "z": 3.0,
    "min_frequency": min_freq,      # rarer training categories are pooled into one 'infrequent' level by the encoder
    "freight": {"numeric": num_stats(tr_a, fs_a["num"]),
                "categorical": {c: tr_a[c].astype(str).value_counts().astype(int).to_dict() for c in fs_a["cat"]}},
    "late": {"numeric": num_stats(tr_b, fs_b["num"]),
             "categorical": {c: tr_b[c].astype(str).value_counts().astype(int).to_dict() for c in fs_b["cat"]}},
}
json.dump(feature_stats, open(os.path.join(MODELS, "feature_stats.json"), "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)

for folder in (DATA, MODELS, API_DATA):
    for f in sorted(os.listdir(folder)):
        print(f"wrote {f:<24} {os.path.getsize(os.path.join(folder, f)) / 1e6:6.2f} MB")

# ============================ CHECKS: rebuilt == notebook ============================
# (computed, notebook, tolerance); transcribed from the notebooks' cell outputs (Problem 1: Phase 3; Problem 2: Phase 2)
NB = {
    "items": (len(items), 112650, 0), "p1 train": (len(tr_a), 90121, 0), "p1 test": (len(te_a), 22529, 0),
    "orders": (len(orders), 96476, 0), "p2 train": (len(tr_b), 70241, 0), "p2 test": (len(te_b), 19295, 0),
    "p2 late in test": (y_b.sum(), 674, 0),
    "forest test MAE": (reg_test["MAE_rs"], 5.398, 0.02), "forest test R2": (reg_test["R2_rs"], 0.591, 0.005),
    "forest MAE, price < R$25": (by_band.mae.iloc[0], 2.16, 0.05), "forest MAE, price R$500+": (by_band.mae.iloc[-1], 16.99, 0.15),
    "tariff base": (tariff[0], 9.614, 0.001), "tariff per kg": (tariff[1], 1.846, 0.001),
    "tariff per 100 km": (tariff[2], 0.811, 0.001), "tariff same state": (tariff[3], -3.985, 0.001),
    "tariff MAE": (tariff[4], 6.221, 0.001),
    "unusual items (forest, overall), notebook": (flags["model"]["overall"].sum(), 1128, 2),
    "logit test lift": (clf_test["Lift"], 1.998, 0.001), "logit test PR-AUC": (clf_test["PR_AUC"], 0.070, 0.001),
    "logit test ROC-AUC": (clf_test["ROC_AUC"], 0.672, 0.001), "logit recall@10%": (clf_test["Recall"], 0.228, 0.001),
    "logit precision@10%": (clf_test["Precision"], 0.080, 0.001), "logit late caught": ((flags_b & y_b).sum(), 154, 0),
    "Ridge test MAE": (ridge_test["MAE_rs"], 6.123, 0.001), "Ridge test R2": (ridge_test["R2_rs"], 0.427, 0.001),
    **{f"logit lift {m}": (by_month.set_index("Month").Lift[m], v, 0.002)
       for m, v in [("2018-05", 10.716), ("2018-06", 6.356), ("2018-07", 2.419), ("2018-08", 1.176)]},
    # the API must reproduce every unusual-freight count when it re-scores the CSV the batch scorer posts
    **{f"unusual items ({name}, {rule}), API path vs build": (app_flags[name, rule], int(flags[name][rule].sum()), 0)
       for name in ("model", "ridge") for rule in ("overall", "band")},
}
fail = [(k, round(float(c), 4), v) for k, (c, v, tol) in NB.items() if abs(c - v) > tol + 1e-9]
if fail:
    print("\n!! MISMATCH vs notebook (check, computed, notebook):")
    for row in fail:
        print("  ", row)
    raise SystemExit("The rebuilt models drifted from the notebooks: fix before shipping.")
print("\nAll checks match the notebooks.")
