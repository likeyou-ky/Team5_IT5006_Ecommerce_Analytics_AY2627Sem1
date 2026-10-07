"""Phase 2 experiment steps, shared by the notebook and scripts/run_phase2.py.

Problem 1 (freight regression) works on the ITEM table; Problem 2 (is_late classification)
works on the ORDER table. Each function does one step of the workflow and returns tidy
DataFrames so the notebook can show them and the script can save them.

Validation
* Problem 1: 80/20 hold-out grouped by order_id (stratified on target quintiles), 5-fold grouped CV;
     robustness splits grouped by product_id and ordered in time.
* Problem 2: time-ordered hold-out (latest 20% of orders, 30-day embargo) and 5-fold forward-chaining CV;
     a random split grouped by customer_unique_id is kept only as a comparison.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import features as F
import models as M

SEED = F.RANDOM_STATE
KEYS = ["order_id", "purchase_ts"]          # read by the splitters, dropped by every ColumnTransformer
ORDER_A = ["Mean predictor (dummy)", "Linear: LinearRegression", "Linear: Ridge", "Tree: DecisionTree (baseline)",
           "Tree: RandomForest", "Ensemble: Stacking (Ridge + RF)"]
ORDER_B = ["Prior predictor (dummy)", "Linear: LogisticRegression (plain)", "Linear: Logistic (L2)",
           "Tree: DecisionTree (baseline)", "Tree: RandomForest", "Ensemble: Voting (Logit + RF)"]


def cols_for(fs):
    return list(dict.fromkeys(fs["num"] + fs["cat"] + fs["te"])) + KEYS


def subsample(df, n, seed=SEED):
    """Order-grouped subsample (keeps whole orders together)."""
    if len(df) <= n:
        return df
    orders = df.order_id.drop_duplicates().sample(frac=n / len(df), random_state=seed)
    return df[df.order_id.isin(orders)]


def pick_final(cv, col, sd_col, higher, order):
    """One-standard-deviation rule: the simplest model whose mean CV score is within one fold sd of the best.

    (A fold sd, not a standard error: with 5 folds this is about 2.2 standard errors, so it is the more
    conservative choice.)
    """
    t = cv.set_index("Model")
    best = t[col].max() if higher else t[col].min()
    sd = t.loc[t[col].idxmax() if higher else t[col].idxmin(), sd_col]
    ok = [m for m in order if m in t.index and ((best - t.loc[m, col]) if higher else (t.loc[m, col] - best)) <= sd]
    return ok[0]


# ------------------------------------------------------------------ documentation tables
def scoping_checklist(orders):
    return pd.DataFrame([
        ["1. Target computable from provided tables, no external data",
         "freight_value comes from order_items.", "is_late = delivered date vs estimated date from orders (Phase 1 definition)."],
        ["2. All predictors exist before the outcome (no leakage)",
         "Listing, address and basket fields fixed at checkout.",
         "Checkout-time fields, including the promised date, and history from orders delivered before the purchase; "
         "delivery, carrier, approval and review fields excluded."],
        ["3. EDA shows feature-target relationships",
         "Phase 1: freight correlates 0.61 / 0.59 / 0.39 with weight / volume / distance.",
         "Phase 1: strong state gradient (SP 4.5% late vs AL 21.4%); late rate varies 1-19% by month."],
        ["4. Class imbalance has a mitigation plan", "None needed (continuous target).",
         f"{orders.clf_target.mean():.1%} late: PR-AUC and its lift as primary metrics, class_weight tuned, "
         "time-ordered validation, precision / recall / F1 for the riskiest 10% of orders, no-skill baseline shown."],
        ["5. A real business stakeholder", "Logistics-pricing / finance analyst.", "Operations and customer-experience team."]],
        columns=["Checklist item", "Problem 1 (freight)", "Problem 2 (is_late)"])


def feature_set_table(sets):
    return pd.DataFrame([{"Feature set": k, "Numeric": ", ".join(v["num"]), "Categorical": ", ".join(v["cat"]) or "-",
                          "Target-encoded": ", ".join(v["te"]) or "-"} for k, v in sets.items()])


def leakage_table():
    return pd.DataFrame(F.LEAKAGE_AUDIT, columns=["Problem", "Field / group", "Why it matters", "Treatment"])


def split_summary(tr_a, te_a, tr_b, te_b):
    """Sizes, targets and overlap checks for both hold-outs."""
    cust = te_b.customer_unique_id.isin(set(tr_b.customer_unique_id)).mean()
    prod = te_a.product_id.isin(set(tr_a.product_id)).mean()
    fmt = lambda s: f"{s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}"
    return pd.DataFrame([
        ["1 train", len(tr_a), tr_a.order_id.nunique(), f"mean freight R${tr_a.freight_value.mean():.2f}", fmt(tr_a.order_purchase_timestamp), ""],
        ["1 test", len(te_a), te_a.order_id.nunique(), f"mean freight R${te_a.freight_value.mean():.2f}", fmt(te_a.order_purchase_timestamp),
         f"{prod:.1%} of test items' products also in train"],
        ["2 train", len(tr_b), len(tr_b), f"late rate {tr_b.clf_target.mean():.1%}", fmt(tr_b.order_purchase_timestamp),
         f"{F.EMBARGO_DAYS}-day embargo before the test window"],
        ["2 test", len(te_b), len(te_b), f"late rate {te_b.clf_target.mean():.1%}", fmt(te_b.order_purchase_timestamp),
         f"{cust:.1%} of test orders from a customer also in train"]],
        columns=["Split", "Rows", "Orders", "Target", "Purchase dates", "Check"])


# ------------------------------------------------------------------ Problem 1: tuning
def tune_regression(train, fs, n_iter=8, tune_rows=30000, n_jobs=-1):
    """Ridge alpha: full grid. Forest: randomised search on a grouped sub-sample. 5-fold grouped CV."""
    X = train[cols_for(fs)]
    out, rows = {}, []
    g = GridSearchCV(M.reg_estimators(fs)["Linear: Ridge"], {"m__alpha": [0.01, 0.1, 1, 10, 100, 1000]},
                     cv=M.OrderGroupCV(5), scoring="neg_mean_absolute_error", n_jobs=n_jobs)
    g.fit(X, train.reg_target)
    out["ridge_alpha"] = g.best_params_["m__alpha"]
    rows.append(("1 Regression", "Ridge", "grid (6)", str(g.best_params_), -g.best_score_, "MAE (log)"))

    sub = subsample(train, tune_rows)
    space = {"m__min_samples_leaf": [2, 5, 10, 20], "m__max_features": [0.2, 0.33, 0.5],
             "m__max_depth": [None, 16, 24], "m__n_estimators": [100]}
    r = RandomizedSearchCV(M.reg_estimators(fs, n_jobs=n_jobs)["Tree: RandomForest"], space, n_iter=n_iter,
                           cv=M.OrderGroupCV(5), scoring="neg_mean_absolute_error", random_state=SEED,
                           n_jobs=1, refit=False)
    r.fit(sub[cols_for(fs)], sub.reg_target)
    out["rf_reg"] = {k[3:]: v for k, v in r.best_params_.items()}
    rows.append(("1 Regression", "RandomForest", f"random ({n_iter} of 36)", str(r.best_params_), -r.best_score_, "MAE (log)"))
    return out, pd.DataFrame(rows, columns=["Problem", "Model", "Search", "Best params", "CV score", "Scored by"])


def final_reg_models(fs, best, n_estimators=200, n_jobs=-1):
    rf = dict(best["rf_reg"], n_estimators=n_estimators)
    reg = M.reg_estimators(fs, rf, best["ridge_alpha"], n_jobs)
    reg["Ensemble: Stacking (Ridge + RF)"] = M.stacking_regressor(fs, dict(rf, n_estimators=100),
                                                                  best["ridge_alpha"], n_jobs)
    return reg


# ------------------------------------------------------------------ Problem 2: tuning
def tune_classification(train, fs, n_iter=20, n_jobs=-1):
    """All searches use 5-fold forward-chaining CV, scored by PR-AUC lift over each fold's late rate.

    Logistic C / class_weight: full grid. Forest: randomised search over leaf size, max features, depth and
    class weight, on all training rows. Voting weight: grid, given the tuned base learners.
    """
    X, y, cv = train[cols_for(fs)], train.clf_target, M.TimeForwardCV(5)
    out, rows = {}, []
    g = GridSearchCV(M.clf_estimators(fs)["Linear: Logistic (L2)"],
                     {"m__C": [0.01, 0.1, 1, 10], "m__class_weight": [None, "balanced"]},
                     cv=cv, scoring=M.LIFT_SCORER, n_jobs=n_jobs)
    g.fit(X, y)
    out["logit_C"], out["logit_cw"] = g.best_params_["m__C"], g.best_params_["m__class_weight"]
    rows.append(("2 is_late", "Logistic (L2)", "grid (8)", str(g.best_params_), g.best_score_, "PR-AUC lift"))

    space = {"m__min_samples_leaf": [10, 20, 50, 100, 200], "m__max_features": ["sqrt", 0.2, 0.3, 0.5],
             "m__max_depth": [6, 10, 16, None], "m__class_weight": [None, "balanced", "balanced_subsample"],
             "m__n_estimators": [100]}
    r = RandomizedSearchCV(M.clf_estimators(fs, n_jobs=n_jobs)["Tree: RandomForest"], space, n_iter=n_iter,
                           cv=cv, scoring=M.LIFT_SCORER, random_state=SEED, n_jobs=1, refit=False)
    r.fit(X, y)
    out["rf_clf"] = {k[3:]: v for k, v in r.best_params_.items()}
    rows.append(("2 is_late", "RandomForest", f"random ({n_iter} of 240)", str(r.best_params_), r.best_score_, "PR-AUC lift"))

    vote = M.voting_classifier(fs, out["rf_clf"], out["logit_C"], out["logit_cw"], n_jobs=n_jobs)
    v = GridSearchCV(vote, {"weights": [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]}, cv=cv, scoring=M.LIFT_SCORER, n_jobs=1)
    v.fit(X, y)
    out["vote_w"] = v.best_params_["weights"][0]
    rows.append(("2 is_late", "Voting (Logit + RF)", "grid (3)", f"{{'w_logit': {out['vote_w']}}}", v.best_score_, "PR-AUC lift"))
    return out, pd.DataFrame(rows, columns=["Problem", "Model", "Search", "Best params", "CV score", "Scored by"])


def final_clf_models(fs, best, n_estimators=200, n_jobs=-1):
    rf = dict(best["rf_clf"], n_estimators=n_estimators)
    clf = M.clf_estimators(fs, rf, best["logit_C"], n_jobs)
    clf["Linear: LogisticRegression (plain)"].set_params(m__class_weight=None)
    clf["Linear: Logistic (L2)"].set_params(m__class_weight=best["logit_cw"])
    clf["Ensemble: Voting (Logit + RF)"] = M.voting_classifier(fs, rf, best["logit_C"], best["logit_cw"],
                                                               best["vote_w"], n_jobs)
    return clf


# ------------------------------------------------------------------ cross-validation
def cv_regression(models, train, fs, k=5):
    X, y = train[cols_for(fs)], train.reg_target
    rows = []
    for name, m in models.items():
        folds = []
        for tr, va in M.OrderGroupCV(k).split(X):
            mm = clone(m).fit(X.iloc[tr], y.iloc[tr])
            folds.append(M.reg_metrics(y.iloc[va], mm.predict(X.iloc[va])))
        f = pd.DataFrame(folds)
        rows.append({"Model": name, **{f"{c} mean": f[c].mean() for c in ["MAE_rs", "RMSE_rs", "R2_rs", "R2_log"]},
                     **{f"{c} sd": f[c].std() for c in ["MAE_rs", "R2_rs"]}})
    return pd.DataFrame(rows)


def cv_classification(models, train, fs, k=5, share=F.REVIEW_SHARE):
    """5-fold forward-chaining CV.

    Each round trains on earlier orders and scores the next time block. Precision / recall / F1 use the decision
    rule "flag the riskiest `share` of the orders in the validation block"; ROC-AUC, PR-AUC and lift are
    threshold-free.
    """
    X, y = train[cols_for(fs)], train.clf_target
    folds_idx = list(M.TimeForwardCV(k).split(X))
    rows, shares = [], {}
    for name, m in models.items():
        oof = np.full(len(X), np.nan)
        for tr, va in folds_idx:
            mm = clone(m).fit(X.iloc[tr], y.iloc[tr])
            oof[va] = M.clf_scores(mm, X.iloc[va])
        shares[name] = share
        f = pd.DataFrame([M.clf_metrics(y.iloc[va], oof[va], pred=M.flag_top_share(oof[va], share))
                          for _, va in folds_idx])
        rows.append({"Model": name, "Review share": share,
                     **{f"{c} mean": f[c].mean() for c in ["Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Lift"]},
                     **{f"{c} sd": f[c].std() for c in ["F1", "PR_AUC", "Lift"]}})
    return pd.DataFrame(rows), shares


def cv_fold_table(train, k=5):
    """Dates, sizes and late rate of every forward-chaining fold (training always starts at the first order)."""
    rows = []
    for j, (tr, va) in enumerate(M.TimeForwardCV(k).split(train), 1):
        t, d = train.order_purchase_timestamp.iloc[tr], train.order_purchase_timestamp.iloc[va]
        rows.append({"Fold": j, "Train dates": f"{t.min():%Y-%m-%d} to {t.max():%Y-%m-%d}", "Train orders": len(tr),
                     "Train late rate": train.clf_target.iloc[tr].mean(),
                     "Validation dates": f"{d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}", "Validation orders": len(va),
                     "Validation late rate": train.clf_target.iloc[va].mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ held-out test
def fit_all(models, train, target, fs):
    X, y = train[cols_for(fs)], train[target]
    return {n: clone(m).fit(X, y) for n, m in models.items()}


def test_regression(fitted, test, fs):
    X = test[cols_for(fs)]
    return pd.DataFrame([{"Model": n, **M.reg_metrics(test.reg_target, m.predict(X))} for n, m in fitted.items()])


def test_classification(fitted, test, fs, shares):
    X, rows = test[cols_for(fs)], []
    for n, m in fitted.items():
        s = M.clf_scores(m, X)
        rows.append({"Model": n, "Review share": shares[n],
                     **M.clf_metrics(test.clf_target, s, pred=M.flag_top_share(s, shares[n]))})
    return pd.DataFrame(rows)


def split_comparison(models, orders, fs, time_test_tbl):
    """Same tuned models on a random split (grouped by customer_unique_id) vs the time-ordered test.

    Shows how much a random split overstates skill when the late rate moves in episodes.
    """
    tr, te = M.holdout_split(orders, orders.clf_target, groups=orders.customer_unique_id)
    fitted = fit_all(models, tr, "clf_target", fs)
    X, rows = te[cols_for(fs)], []
    t = time_test_tbl.set_index("Model")
    for n, m in fitted.items():
        s = M.clf_scores(m, X)
        rows.append({"Model": n, "Random PR-AUC": average_precision_score(te.clf_target, s),
                     "Random lift": M.pr_lift(te.clf_target, s), "Random ROC-AUC": roc_auc_score(te.clf_target, s),
                     "Time-ordered PR-AUC": t.loc[n, "PR_AUC"], "Time-ordered lift": t.loc[n, "Lift"],
                     "Time-ordered ROC-AUC": t.loc[n, "ROC_AUC"]})
    return pd.DataFrame(rows), te.clf_target.mean()


def window_starts(orders):
    """Starts of the four consecutive test windows; the last is the main hold-out (latest 20% of orders)."""
    cut = pd.to_datetime(orders.purchase_ts.quantile(1 - F.TEST_FRAC), unit="s")
    return [pd.Timestamp(d) for d in F.ROLLING_STARTS] + [cut]


def rolling_windows(models, orders, fs, share=F.REVIEW_SHARE):
    """Rolling-origin evaluation: for each window, refit on all orders before it (minus the embargo) and test on it.

    The model specification was chosen on data up to April 2018, so windows before the main hold-out are consistency
    checks rather than untouched tests; only the last window is fully held out.
    """
    starts = window_starts(orders)
    ends = starts[1:] + [orders.order_purchase_timestamp.max()]
    low = F.load_low_reviews().set_index("order_id").low
    rows = []
    for w, (a, b) in enumerate(zip(starts, ends), 1):
        # same boundary convention as M.time_holdout, so W4 is exactly the main hold-out
        tr = orders[orders.order_purchase_timestamp <= a - pd.Timedelta(days=F.EMBARGO_DAYS)]
        te = orders[(orders.order_purchase_timestamp > a) & (orders.order_purchase_timestamp <= b)]
        X = te[cols_for(fs)]
        d = te.order_purchase_timestamp
        info = {"Window": f"W{w}", "Test dates": f"{d.min():%d %b %Y} to {d.max():%d %b %Y}",
                "Train orders": len(tr), "Test orders": len(te), "Late rate": te.clf_target.mean(),
                "Median promise (days)": te.promised_days.median()}
        lw = low.reindex(te.order_id).values
        for n, m in models.items():
            s = M.clf_scores(clone(m).fit(tr[cols_for(fs)], tr.clf_target), X)
            flag = M.flag_top_share(s, share)
            met = M.clf_metrics(te.clf_target, s, pred=flag)
            ok = ~np.isnan(lw)
            rows.append({**info, "Model": n, "PR-AUC": met["PR_AUC"], "Lift": met["Lift"], "ROC-AUC": met["ROC_AUC"],
                         "Recall@10%": met["Recall"], "Precision@10%": met["Precision"],
                         "Low-review rate, flagged": np.nanmean(lw[ok & (flag == 1)]),
                         "Low-review rate, not flagged": np.nanmean(lw[ok & (flag == 0)]),
                         "Share of low reviews flagged": lw[ok & (flag == 1)].sum() / lw[ok].sum()})
    return pd.DataFrame(rows)


def satisfaction_link(orders):
    """How lateness relates to 1-2 star reviews (all delivered orders with a review). Descriptive only."""
    d = orders.merge(F.load_low_reviews(), on="order_id", how="inner")
    late, ontime = d[d.clf_target == 1], d[d.clf_target == 0]
    return pd.DataFrame({"metric": ["orders with a review", "low-review rate (1-2 stars), all", "low-review rate, late orders",
                                    "low-review rate, on-time orders", "share of low reviews from late orders"],
                         "value": [len(d), d.low.mean(), late.low.mean(), ontime.low.mean(), late.low.sum() / d.low.sum()]})


def promise_timeline(orders):
    """Weekly median and middle-50% promised days, and the weekly late rate (delivered orders, full weeks only)."""
    ts = orders.set_index("order_purchase_timestamp")
    wk = ts.promised_days.resample("W-SUN").agg(["size", "median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    wk.columns = ["orders", "median", "p25", "p75"]
    wk["late_rate"] = ts.clf_target.resample("W-SUN").mean()
    last_full = orders.order_purchase_timestamp.max().normalize()          # a week counts only if it ended in the data
    return wk[(wk.index >= "2017-01-08") & (wk.index <= last_full) & (wk.orders >= 100)].reset_index(names="week_ending")


def gain_table(y, score, bins=10):
    """Decile gain / lift table: how many late orders sit in the highest-risk deciles."""
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score)}).sort_values("s", ascending=False).reset_index(drop=True)
    d["decile"] = np.minimum(d.index * bins // len(d) + 1, bins)
    g = d.groupby("decile").agg(orders=("y", "size"), late=("y", "sum")).reset_index()
    g["late_rate"] = g.late / g.orders
    g["cum_capture"] = g.late.cumsum() / g.late.sum()
    g["lift"] = g.late_rate / d.y.mean()
    return g


def test_by_month(fitted_model, test, fs):
    """Late rate and ranking skill of the final model in each month of the test window."""
    s = M.clf_scores(fitted_model, test[cols_for(fs)])
    d = test.assign(s=s, ym=test.order_purchase_timestamp.dt.to_period("M").astype(str))
    return (d.groupby("ym").apply(lambda g: pd.Series({"Orders": len(g), "Late rate": g.clf_target.mean(),
                                                       "PR-AUC": average_precision_score(g.clf_target, g.s),
                                                       "Lift": M.pr_lift(g.clf_target, g.s),
                                                       "ROC-AUC": roc_auc_score(g.clf_target, g.s)}), include_groups=False)
            .reset_index().rename(columns={"ym": "Month"}))


# ------------------------------------------------------------------ ablation (feature breadth)
def ablation_regression(train, n_rows=40000, k=3):
    sub = subsample(train, n_rows)
    rows = []
    for name, fs in F.FEATURE_SETS.items():
        X = sub[cols_for(fs)]
        m = M.reg_estimators(fs, dict(n_estimators=100, min_samples_leaf=5, max_features=0.5))["Tree: RandomForest"]
        r = []
        for tr, va in M.OrderGroupCV(k).split(X):
            mm = clone(m).fit(X.iloc[tr], sub.reg_target.iloc[tr])
            r.append(M.reg_metrics(sub.reg_target.iloc[va], mm.predict(X.iloc[va])))
        rows.append({"Feature set": name, "MAE (R$)": np.mean([a["MAE_rs"] for a in r]),
                     "RMSE (R$)": np.mean([a["RMSE_rs"] for a in r]), "R2 (R$)": np.mean([a["R2_rs"] for a in r])})
    return pd.DataFrame(rows)


def ablation_classification(train, k=5):
    """Forward-chaining CV of every nested feature set, with both base families.

    A fixed L2 logistic model (C = 0.1) and a fixed, regularised class-weighted forest are used, so the
    feature choice does not depend on tuning. Score = mean PR-AUC lift over the fold late rate.
    Because the folds differ a lot in difficulty, each added group is judged by its PAIRED gain over the
    previous set, fold by fold ("Check" rows are compared with the set they extend).
    """
    folds = list(M.TimeForwardCV(k).split(train))
    rows, per_fold = [], {}
    for name, fs in F.LATE_FEATURE_SETS.items():
        X, y = train[cols_for(fs)], train.clf_target
        est = M.clf_estimators(fs, dict(n_estimators=100, min_samples_leaf=50, max_features=0.3, max_depth=10,
                                        class_weight="balanced"), logit_c=0.1)
        res = {"Feature set": name}
        for lbl, key in [("Logit", "Linear: Logistic (L2)"), ("RF", "Tree: RandomForest")]:
            lift, roc = [], []
            for tr, va in folds:
                s = M.clf_scores(clone(est[key]).fit(X.iloc[tr], y.iloc[tr]), X.iloc[va])
                lift.append(M.pr_lift(y.iloc[va], s)); roc.append(roc_auc_score(y.iloc[va], s))
            per_fold[(name, lbl)] = np.array(lift)
            res[f"{lbl} lift"], res[f"{lbl} ROC-AUC"] = np.mean(lift), np.mean(roc)
        rows.append(res)
    names = list(F.LATE_FEATURE_SETS)
    for i, r in enumerate(rows):
        base = None if i == 0 else (F.LATE_FULL_SET if r["Feature set"].startswith("Check") else names[i - 1])
        for lbl in ["Logit", "RF"]:
            if base is None:
                r[f"{lbl} gain"], r[f"{lbl} gain sd"] = np.nan, np.nan
                continue
            d = per_fold[(r["Feature set"], lbl)] - per_fold[(base, lbl)]
            r[f"{lbl} gain"], r[f"{lbl} gain sd"] = d.mean(), d.std(ddof=1)
    cols = ["Feature set"] + [f"{l} {c}" for l in ["Logit", "RF"] for c in ["lift", "gain", "gain sd", "ROC-AUC"]]
    return pd.DataFrame(rows)[cols]


# ------------------------------------------------------------------ interpretability
def perm_importance(model, test, fs, target, n=15000, repeats=5):
    """Permutation importance on RAW columns (the pipeline re-encodes after shuffling)."""
    s = test.sample(min(n, len(test)), random_state=SEED)
    X = s[cols_for(fs)]
    scoring = "neg_mean_absolute_error" if target == "reg_target" else "average_precision"
    r = permutation_importance(model, X, s[target], scoring=scoring, n_repeats=repeats,
                               random_state=SEED, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "sd": r.importances_std})
    return out[~out.feature.isin(KEYS)].sort_values("importance", ascending=False).reset_index(drop=True)


def linear_coefs(fitted_linear, top=15):
    prep, m = fitted_linear.named_steps["prep"], fitted_linear.named_steps["m"]
    d = pd.DataFrame({"feature": prep.get_feature_names_out(), "coef": m.coef_.ravel()})
    return d.reindex(d.coef.abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


# ------------------------------------------------------------------ Problem 1: robustness and audit
def product_holdout(items, seed=SEED):
    """Hold-out grouped by product_id: every test item is a listing the model has never seen."""
    strat = pd.qcut(items.reg_target.rank(method="first"), 5, labels=False)
    return M.holdout_split(items, strat, seed=seed, groups=items.product_id)


def robustness_regression(models, items, fs, main_test_tbl):
    """Main (order-grouped) test vs product-grouped and time-ordered hold-outs for the same tuned models.

    Returns the table and the product-grouped split with its fitted models (used by the audit).
    """
    rows = [{"Split": "Order-grouped (main)", **r} for r in main_test_tbl.to_dict("records") if r["Model"] in models]
    ptr, pte = product_holdout(items)
    pfit = fit_all(models, ptr, "reg_target", fs)
    rows += [{"Split": "Product-grouped (unseen listings)", **r} for r in test_regression(pfit, pte, fs).to_dict("records")]
    ttr, tte = M.time_holdout(items, embargo_days=0)        # freight is known at checkout: no label delay
    tfit = fit_all(models, ttr, "reg_target", fs)
    rows += [{"Split": "Time-ordered (latest 20%)", **r} for r in test_regression(tfit, tte, fs).to_dict("records")]
    tbl = pd.DataFrame(rows)[["Split", "Model", "MAE_rs", "RMSE_rs", "R2_rs"]]
    return tbl, (ptr, pte, pfit)


def implied_tariff(train, test):
    """A transparent tariff (Phase 1): freight = base + R$/kg x chargeable kg + R$/100 km x distance + same-state term.

    Plain least squares on the R$ scale with no transformations, so every coefficient reads as a price.
    """
    def design(d):
        return pd.DataFrame({"chargeable_kg": d.chargeable_kg, "distance_100km": d.distance_km / 100,
                             "same_state": d.same_state})
    med = design(train).median()                                    # imputation learned on train only
    lr = LinearRegression().fit(design(train).fillna(med), train.freight_value)
    mae = mean_absolute_error(test.freight_value, np.clip(lr.predict(design(test).fillna(med)), 0, None))
    return pd.DataFrame({"Term": ["Base charge (R$)", "Per chargeable kg (R$)", "Per 100 km (R$)",
                                  "Same-state shipment (R$)", "Test MAE of this tariff (R$)"],
                         "Value": [lr.intercept_, *lr.coef_, mae]})


def flag_overcharge(test, pred_primary, pred_check, q=0.95):
    """Flag the top (1-q) positive residuals; compare two independent models for stability."""
    t = test.copy()
    t["resid_primary"] = t.reg_target - pred_primary
    t["resid_check"] = t.reg_target - pred_check
    t["flag_primary"] = t.resid_primary >= t.resid_primary.quantile(q)
    t["flag_check"] = t.resid_check >= t.resid_check.quantile(q)
    a, b = t.flag_primary, t.flag_check
    stab = dict(flag_rate=float(a.mean()), jaccard=float((a & b).sum() / (a | b).sum()),
                overlap_share=float((a & b).sum() / a.sum()))
    t["expected_rs"] = np.expm1(pred_primary)
    t["excess_rs"] = t.freight_value - t.expected_rs
    return t, stab


def audit_summary(flagged):
    f = flagged[flagged.flag_primary]
    top_sellers = max(1, int(0.05 * flagged.seller_id.nunique()))
    return pd.DataFrame({"metric": ["items flagged", "mean actual freight R$", "mean expected freight R$", "mean excess R$",
                                    "excess as % of all test freight", "flagged items from top 5% of sellers (by flag count)",
                                    "flagged by both forest and Ridge (%)"],
                         "value": [len(f), f.freight_value.mean(), f.expected_rs.mean(), f.excess_rs.mean(),
                                   f.excess_rs.sum() / flagged.freight_value.sum() * 100,
                                   f.groupby("seller_id").size().sort_values(ascending=False).head(top_sellers).sum() / len(f) * 100,
                                   (f.flag_check.mean()) * 100]})


def audit_review_sheet(flagged, n=30):
    """Top-n flagged items with the context a reviewer needs (for manual precision-at-k checks)."""
    f = flagged[flagged.flag_primary & flagged.flag_check].sort_values("resid_primary", ascending=False).head(n)
    cols = ["order_id", "order_item_id", "product_id", "seller_id", "category", "price", "freight_value", "expected_rs",
            "excess_rs", "weight_kg", "volume_cm3", "chargeable_kg", "distance_km", "seller_state", "customer_state"]
    return f[cols].assign(reviewer_verdict="").reset_index(drop=True)
