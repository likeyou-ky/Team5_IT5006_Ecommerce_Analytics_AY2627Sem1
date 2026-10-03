"""Headless Phase 2 run: reproduces every table and figure of the notebook, in the same order.

    OLIST_DATA_DIR=/path/to/Olist_CSV python scripts/run_phase2.py        # full run (about 20 min on 12 cores)
    PHASE2_FAST=1 OLIST_DATA_DIR=... python scripts/run_phase2.py         # smoke test on a sample

Problem 1: fair-freight regression (item grain, order-grouped hold-out).
Problem 2: is_late classification (order grain, time-ordered hold-out with forward-chaining CV).
Outputs: docs/phase2/tables/*.csv, docs/phase2/figures/*.png (or $PHASE2_OUT/...)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import features as F
import models as M
import pipeline as P
import plots as PL

FAST = os.environ.get("PHASE2_FAST") == "1"
OUT = os.environ.get("PHASE2_OUT", os.path.join(ROOT, "docs", "phase2"))   # override to keep test runs separate
TAB, FIG = (os.path.join(OUT, d) for d in ("tables", "figures"))
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
np.random.seed(F.RANDOM_STATE)


def save(df, name):
    df.to_csv(os.path.join(TAB, name), index=False)
    print(f"-> {name}  {df.shape}", flush=True)
    return df


# 1. data, documentation tables ---------------------------------------------------------------
items, orders = F.build_item_table(), F.build_order_table()
if FAST:
    ki = items.order_id.drop_duplicates().sample(15000, random_state=F.RANDOM_STATE)
    items = items[items.order_id.isin(ki)].reset_index(drop=True)
    orders = orders.iloc[::5].reset_index(drop=True)          # every 5th order keeps the time structure
save(P.scoping_checklist(orders), "scoping_checklist.csv")
save(P.feature_set_table(F.FEATURE_SETS), "feature_sets_1.csv")
save(P.feature_set_table(F.LATE_FEATURE_SETS), "feature_sets_2.csv")
save(P.leakage_table(), "leakage_audit.csv")
PL.target_overview(items, orders, f"{FIG}/01_targets.png")

# 2. splits -----------------------------------------------------------------------------------
tr_a, te_a = M.holdout_split(items, pd.qcut(items.reg_target.rank(method="first"), 5, labels=False))
tr_b, te_b = M.time_holdout(orders)
assert not set(tr_a.order_id) & set(te_a.order_id) and not set(tr_b.order_id) & set(te_b.order_id)
assert tr_b.purchase_ts.max() < te_b.purchase_ts.min() - F.EMBARGO_DAYS * M.DAY
save(P.split_summary(tr_a, te_a, tr_b, te_b), "split_summary.csv")
save(P.cv_fold_table(tr_b), "cv_folds_2.csv")

# 3. feature-set selection (training data only) -----------------------------------------------
ab_a = save(P.ablation_regression(tr_a, n_rows=8000 if FAST else 40000), "ablation_1_regression.csv")
ab_b = save(P.ablation_classification(tr_b), "ablation_2_classification.csv")
PL.ablation_chart_reg(ab_a, f"{FIG}/02_ablation_1.png")
PL.ablation_chart_clf(ab_b, f"{FIG}/02_ablation_2.png")
fs_a, fs_b = F.FEATURE_SETS[F.FULL_SET], F.LATE_FEATURE_SETS[F.LATE_FULL_SET]
cols_a, cols_b = P.cols_for(fs_a), P.cols_for(fs_b)

# 4. tuning, CV, test -------------------------------------------------------------------------
best_a, tune_a = P.tune_regression(tr_a, fs_a, n_iter=2 if FAST else 8, tune_rows=6000 if FAST else 30000)
best_b, tune_b = P.tune_classification(tr_b, fs_b, n_iter=3 if FAST else 20)
save(pd.concat([tune_a, tune_b], ignore_index=True), "tuning_log.csv")
reg_models, clf_models = P.final_reg_models(fs_a, best_a), P.final_clf_models(fs_b, best_b)
cv_a = save(P.cv_regression(reg_models, tr_a, fs_a), "cv_regression.csv")
cv_b, share_b = P.cv_classification(clf_models, tr_b, fs_b)
save(cv_b, "cv_classification.csv")
fit_a, fit_b = P.fit_all(reg_models, tr_a, "reg_target", fs_a), P.fit_all(clf_models, tr_b, "clf_target", fs_b)
test_a = save(P.test_regression(fit_a, te_a, fs_a), "test_regression.csv")
test_b = save(P.test_classification(fit_b, te_b, fs_b, share_b), "test_classification.csv")
cmp_b, rand_rate = P.split_comparison(clf_models, orders, fs_b, test_b)
cmp_b = save(pd.concat([cmp_b, P.draft_spec_check(orders)], ignore_index=True), "split_comparison_2.csv")
PL.comparison_chart_reg(test_a, f"{FIG}/03_comparison_1.png")
PL.comparison_chart_clf(cmp_b, f"{FIG}/03_comparison_2.png")

FINAL_A = P.pick_final(cv_a, "MAE_rs mean", "MAE_rs sd", False, P.ORDER_A)
FINAL_B = P.pick_final(cv_b, "Lift mean", "Lift sd", True, P.ORDER_B)
print("final models:", FINAL_A, "|", FINAL_B)

# 5. diagnostics, importance, risk ranking ----------------------------------------------------
pred_a = fit_a[FINAL_A].predict(te_a[cols_a])
PL.pred_vs_actual(te_a.reg_target, pred_a, f"{FIG}/04_pred_vs_actual.png")
save(PL.error_by_price_band(te_a, pred_a, f"{FIG}/05_error_by_price_band.png").reset_index(), "error_by_price_band.csv")
PL.clf_curves(fit_b, te_b, cols_b, te_b.clf_target, f"{FIG}/06_roc_pr.png",
              ["Linear: Logistic (L2)", "Tree: RandomForest", "Ensemble: Voting (Logit + RF)"])
score_b = M.clf_scores(fit_b[FINAL_B], te_b[cols_b])
flags_b = M.flag_top_share(score_b, M.purchase_day(te_b), share_b[FINAL_B])
PL.confusion(te_b.clf_target, flags_b, f"{FIG}/07_confusion.png",
             f"{FINAL_B.split(': ')[-1]}, riskiest {share_b[FINAL_B]:.0%} of each day")
imp_a = save(P.perm_importance(fit_a[FINAL_A], te_a, fs_a, "reg_target"), "perm_importance_1.csv")
imp_b = save(P.perm_importance(fit_b[FINAL_B], te_b, fs_b, "clf_target"), "perm_importance_2.csv")
PL.importance_bar(imp_a, "Problem 1: what drives freight", f"{FIG}/08_importance_1.png", unit="(dMAE, log scale)")
PL.importance_bar(imp_b, "Problem 2: what drives late delivery", f"{FIG}/09_importance_2.png", unit="(dPR-AUC)")
save(P.linear_coefs(fit_a["Linear: Ridge"]), "ridge_top_coefs.csv")
save(P.linear_coefs(fit_b["Linear: Logistic (L2)"]), "logit_top_coefs.csv")
gain = save(P.gain_table(te_b.clf_target, score_b), "gain_table_2.csv")
PL.gain_chart(gain, f"{FIG}/10_gain.png")
save(P.test_by_month(fit_b[FINAL_B], te_b, fs_b), "test_by_month_2.csv")
save(P.rolling_windows(clf_models, orders, fs_b), "rolling_windows_2.csv")
save(P.satisfaction_link(orders), "satisfaction_link_2.csv")
wk = save(P.promise_timeline(orders), "promise_timeline_2.csv")
PL.promise_chart(wk, P.window_starts(orders), orders.order_purchase_timestamp.max(), f"{FIG}/12_promise_timeline.png")

# 6. Problem 1 robustness, tariff and unusual-freight audit -----------------------------------
names_a = ["Mean predictor (dummy)", "Linear: Ridge", "Tree: RandomForest"]
rob_a, (ptr, pte, pfit) = P.robustness_regression({n: reg_models[n] for n in names_a}, items, fs_a, test_a)
save(rob_a, "robustness_regression.csv")
save(P.implied_tariff(tr_a, te_a), "implied_tariff_1.csv")
flagged, stab = P.flag_overcharge(pte, pfit["Tree: RandomForest"].predict(pte[cols_a]),
                                  pfit["Linear: Ridge"].predict(pte[cols_a]), q=0.95)
print("freight-audit stability:", {k: round(v, 3) for k, v in stab.items()})
sell = PL.overcharge_charts(flagged, f"{FIG}/11_freight_audit.png")
save(P.audit_summary(flagged), "freight_audit_summary.csv")
save(sell.sort_values("rate", ascending=False).head(10).reset_index(), "freight_audit_top_sellers.csv")
save(P.audit_review_sheet(flagged), "freight_audit_review_sheet.csv")
print("done")
