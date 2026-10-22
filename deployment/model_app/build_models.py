# -*- coding: utf-8 -*-
"""
Build the model files and data for the Phase 3 model app (deployment/model_app/app.py).

Both models are the Phase 2 final models, trained with the notebook's own code in src/ on the notebook's own
training splits (notebooks/Team5_Phase2_IT5006_AY2627Sem1.ipynb, Sections 4-8):

* Problem 1, fair freight: RandomForest regressor on log1p(freight), order-grouped 80% training split, with the
  tuned settings of Section 6 and 200 trees: the exact final model (141 MB in memory, about 48 MB compressed).
  The tuned Ridge model is the audit's second opinion, as in Section 11: items flagged by both models come first.
* Problem 2, late delivery: plain LogisticRegression on the time-ordered training split (orders up to 26 Apr 2018).

Unusual freight = residual in the top 5% of the test set, either overall (the notebook's rule) or within each price
band (the report's recommendation, so expensive items do not crowd the review list). Both cut-offs are stored for
both models.

Writes into deployment/model_app/:
  models/freight_forest.joblib   the Problem 1 forest
  models/freight_ridge.joblib    the Problem 1 Ridge model (second opinion for the audit)
  models/late_logit.joblib       the Problem 2 logistic model
  data/heldout_items.csv         Problem 1 test items (their orders were never seen in training)
  data/heldout_orders.csv        Problem 2 test window, 26 May - 29 Aug 2018, with outcomes for the after-the-fact check
  data/zip_centroids.csv         zip-code prefix -> centroid and state, to place new listings and orders
  data/facts.json                every number the app shows

The checks at the end stop the script if a split or a test score drifts from the notebook on main. Forest scores
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

MODELS, DATA = os.path.join(HERE, "models"), os.path.join(HERE, "data")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

RF_PARAMS = dict(min_samples_leaf=2, max_features=0.33, max_depth=16,  # notebook Section 6, tuning table
                 n_estimators=200)                                     # as in pipeline.final_reg_models
RIDGE_ALPHA = 10                                                       # notebook Section 6, tuning table
FLAG_Q = 0.95                       # unusual freight = top 5% of residuals (notebook Section 11)
# Identical units of one listing in one order share a residual, and the cut-off can land on such a tie. The forest's
# parallel predictions vary in the last bits between calls (about 1e-13), which would flip the tied items, so the
# rule allows a negligible tolerance on the log scale.
FLAG_TOL = 1e-9
SHARE = F.REVIEW_SHARE              # review the riskiest 10% of orders (notebook Section 4)
PRICE_BANDS = ([0, 25, 50, 100, 200, 500, np.inf], ["<25", "25-50", "50-100", "100-200", "200-500", "500+"])
AUDIT_BANDS = ([0, 50, 100, 200, 500, np.inf], ["<50", "50-100", "100-200", "200-500", "500+"])  # pipeline.audit_concentration


def residual(freight, pred_log):
    """Unusual-freight residual exactly as app.py computes it: log(1 + charged) - log(1 + expected)."""
    return np.log1p(np.asarray(freight)) - np.log1p(np.clip(np.expm1(pred_log), 0, None))


def audit_band(price):
    return pd.cut(price, AUDIT_BANDS[0], labels=AUDIT_BANDS[1], right=False).astype(str).values


def unusual(res, band, cut, rule):
    """Flag residuals at or above the cut-off: one cut-off overall, or one per price band."""
    thr = cut["overall"] if rule == "overall" else pd.Series(band).map(cut["band"]).values.astype(float)
    return res >= thr - FLAG_TOL

d = F.find_data_dir()
print("Reading raw data from:", d)

# ============================ tables and splits (notebook Sections 2 and 4) ============================
items = F.build_item_table(d)
orders = F.build_order_table(d)
strat_a = pd.qcut(items.reg_target.rank(method="first"), 5, labels=False)
tr_a, te_a = M.holdout_split(items, strat_a)
tr_b, te_b = M.time_holdout(orders)
assert not set(tr_a.order_id) & set(te_a.order_id), "Problem 1: an order is in both train and test"
assert tr_b.purchase_ts.max() < te_b.purchase_ts.min() - F.EMBARGO_DAYS * M.DAY, "Problem 2: embargo violated"

# Feature sets: Problem 1 "4 + listing & timing"; Problem 2 = what the forward selection of Section 5 keeps
fs_a = F.FEATURE_SETS[F.FULL_SET]
fs_b = P.merge_sets(P.merge_sets(F.LATE_BASE, F.LATE_GROUPS["route (distance, states)"]),
                    F.LATE_GROUPS["order value & freight"])
cols_a, cols_b = P.cols_for(fs_a), P.cols_for(fs_b)

# ============================ Problem 1: fair freight ============================
forest = M.reg_estimators(fs_a, RF_PARAMS)["Tree: RandomForest"].fit(tr_a[cols_a], tr_a.reg_target)
pred_a = forest.predict(te_a[cols_a])
reg_test = M.reg_metrics(te_a.reg_target, pred_a)
print(f"Problem 1 test MAE: R${reg_test['MAE_rs']:.3f}")

ridge = M.reg_estimators(fs_a, ridge_alpha=RIDGE_ALPHA)["Linear: Ridge"].fit(tr_a[cols_a], tr_a.reg_target)
pred_r = ridge.predict(te_a[cols_a])
ridge_test = M.reg_metrics(te_a.reg_target, pred_r)

band_a = audit_band(te_a.price)
cuts, flags = {}, {}
for name, pred in [("forest", pred_a), ("ridge", pred_r)]:
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
held_items.to_csv(os.path.join(DATA, "heldout_items.csv"), index=False)      # full precision: the app must
held_orders.to_csv(os.path.join(DATA, "heldout_orders.csv"), index=False)    # reproduce these scores exactly
back = pd.read_csv(os.path.join(DATA, "heldout_items.csv"))                   # re-scored as app.py does
app_flags = {(name, rule): int(unusual(residual(back.freight_value, m.predict(back[cols_a])), audit_band(back.price),
                                       cuts[name], rule).sum())
             for name, m in [("forest", forest), ("ridge", ridge)] for rule in ("overall", "band")}

geo = pd.read_csv(os.path.join(d, "olist_geolocation_dataset.csv"))
geo = geo[geo.geolocation_lat.between(*F.BR_LAT) & geo.geolocation_lng.between(*F.BR_LON)]
zips = (geo.groupby("geolocation_zip_code_prefix")       # same centroids as features._item_frame
        .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"),
             state=("geolocation_state", lambda s: s.value_counts().index[0]))
        .reset_index().rename(columns={"geolocation_zip_code_prefix": "zip_prefix"}))
zips.to_csv(os.path.join(DATA, "zip_centroids.csv"), index=False, float_format="%.6f")
state_cen = zips.groupby("state")[["lat", "lng"]].median().round(4)

joblib.dump(forest, os.path.join(MODELS, "freight_forest.joblib"), compress=3)
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
        "n_train": len(tr_a), "n_test": len(te_a), "test_dates": [f"{te_a.order_purchase_timestamp.min():%Y-%m-%d}",
                                                                   f"{te_a.order_purchase_timestamp.max():%Y-%m-%d}"],
        "mean_freight_test": r(te_a.freight_value.mean(), 2),
        "features_num": fs_a["num"], "features_cat": fs_a["cat"],
        "forest_params": RF_PARAMS,
        "test": {**{k: r(reg_test[k]) for k in ["MAE_rs", "RMSE_rs", "R2_rs"]},
                 "MAE_pct": r(reg_test["MAE_rs"] / te_a.freight_value.mean() * 100, 1)},
        "ridge_alpha": RIDGE_ALPHA, "ridge_test": {k: r(ridge_test[k]) for k in ["MAE_rs", "RMSE_rs", "R2_rs"]},
        # unusual-freight cut-offs on the log1p residual, unrounded (they are thresholds)
        "flag": {"q": FLAG_Q, "tol": FLAG_TOL, "band_edges": AUDIT_BANDS[0][1:-1], "band_labels": AUDIT_BANDS[1],
                 "cuts": cuts, "ratio_overall": r(np.exp(cuts["forest"]["overall"]), 3),
                 "n_unusual": {rule: int(flags["forest"][rule].sum()) for rule in ("overall", "band")},
                 "both_share": {rule: r((flags["forest"][rule] & flags["ridge"][rule]).sum()
                                        / flags["forest"][rule].sum()) for rule in ("overall", "band")}},
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
    # Results of the Phase 2 notebook (main, 84ba8ab) that need more than the two deployed models; shown as cited
    "notebook": {
        "p1_cv_mae": 3.59, "p1_cv_mae_sd": 0.07, "p1_ridge_test_mae": 4.74, "p1_tree_test_mae": 4.37,
        "p1_unseen_mae": 4.03, "p1_unseen_pct": 20.2, "p1_recent_mae": 5.46,
        "p2_cv_lift": 2.01, "p2_cv_lift_sd": 0.34, "p2_window_lift": [1.47, 1.71, 2.24, 2.00],
        "p2_window_recall": [0.178, 0.202, 0.289, 0.228], "p2_low_review_ratio": [1.2, 2.0],
        "audit_flags": 1127, "audit_charged": 52.9, "audit_expected": 26.5, "audit_top5_sellers_pct": 58,
        "audit_median_markup": 1.88,
    },
}
json.dump(facts, open(os.path.join(DATA, "facts.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

for folder in (DATA, MODELS):
    for f in sorted(os.listdir(folder)):
        print(f"wrote {f:<24} {os.path.getsize(os.path.join(folder, f)) / 1e6:6.2f} MB")

# ============================ CHECKS: rebuilt == notebook ============================
# (computed, notebook on main, tolerance); transcribed from the notebook's cell outputs
NB = {
    "items": (len(items), 112650, 0), "p1 train": (len(tr_a), 90120, 0), "p1 test": (len(te_a), 22530, 0),
    "orders": (len(orders), 96476, 0), "p2 train": (len(tr_b), 70241, 0), "p2 test": (len(te_b), 19295, 0),
    "p2 late in test": (y_b.sum(), 674, 0),
    "forest test MAE": (reg_test["MAE_rs"], 3.546, 0.02), "forest test R2": (reg_test["R2_rs"], 0.719, 0.005),
    "forest MAE, price < R$25": (by_band.mae.iloc[0], 1.9, 0.05), "forest MAE, price R$500+": (by_band.mae.iloc[-1], 11.5, 0.15),
    "tariff base": (tariff[0], 9.897, 0.001), "tariff per kg": (tariff[1], 1.968, 0.001),
    "tariff per 100 km": (tariff[2], 0.860, 0.001), "tariff same state": (tariff[3], -4.416, 0.001),
    "tariff MAE": (tariff[4], 5.638, 0.001),
    "logit test lift": (clf_test["Lift"], 1.998, 0.001), "logit test PR-AUC": (clf_test["PR_AUC"], 0.070, 0.001),
    "logit test ROC-AUC": (clf_test["ROC_AUC"], 0.672, 0.001), "logit recall@10%": (clf_test["Recall"], 0.228, 0.001),
    "logit precision@10%": (clf_test["Precision"], 0.080, 0.001), "logit late caught": ((flags_b & y_b).sum(), 154, 0),
    "Ridge test MAE": (ridge_test["MAE_rs"], 4.738, 0.001), "Ridge test R2": (ridge_test["R2_rs"], 0.517, 0.001),
    **{f"logit lift {m}": (by_month.set_index("Month").Lift[m], v, 0.002)
       for m, v in [("2018-05", 10.716), ("2018-06", 6.356), ("2018-07", 2.419), ("2018-08", 1.176)]},
    # the app must reproduce every unusual-freight count when it re-scores the CSV it reads
    **{f"unusual items ({name}, {rule}), app vs build": (app_flags[name, rule], int(flags[name][rule].sum()), 0)
       for name in ("forest", "ridge") for rule in ("overall", "band")},
}
fail = [(k, round(float(c), 4), v) for k, (c, v, tol) in NB.items() if abs(c - v) > tol + 1e-9]
if fail:
    print("\n!! MISMATCH vs notebook (check, computed, notebook):")
    for row in fail:
        print("  ", row)
    raise SystemExit("The rebuilt models drifted from the Phase 2 notebook: fix before shipping.")
print("\nAll checks match the Phase 2 notebook.")
