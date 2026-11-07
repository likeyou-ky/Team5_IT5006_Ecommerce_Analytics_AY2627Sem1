"""Problem 2: train / CV / test precision, recall, PR-AUC and PR curves for the Model card page.

Rebuilds the late-delivery models exactly as the Phase 2 notebook does (same split, feature set, tuned settings and
pipeline functions) and writes data/p2_train_cv_test.csv and data/p2_pr_curves.json. Stops if any model's train, CV or
test lift differs from the notebook's printed values (Phase 2 notebook, Section 7).
Run: OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/model_app/p2_train_cv_test.py"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import features as F, models as M, pipeline as P

OUT = os.path.join(HERE, "data")
orders = F.build_order_table(F.find_data_dir())
tr, te = M.time_holdout(orders)
fs = P.merge_sets(P.merge_sets(F.LATE_BASE, F.LATE_GROUPS["route (distance, states)"]), F.LATE_GROUPS["order value & freight"])
cols = P.cols_for(fs)
best = {"logit_C": 0.1, "logit_cw": None, "vote_w": 0.75,
        "rf_clf": {"n_estimators": 100, "min_samples_leaf": 100, "max_features": "sqrt", "max_depth": None,
                   "class_weight": "balanced"}}
models = P.final_clf_models(fs, best)
SHARE = F.REVIEW_SHARE
NB = {  # notebook cell 48 (train / CV / test lift) and cell 32 (CV precision / recall means)
    "Prior predictor (dummy)": (1.000, 1.000, 1.000), "Linear: LogisticRegression (plain)": (2.444, 2.010, 1.998),
    "Linear: Logistic (L2)": (2.438, 2.028, 1.979), "Tree: DecisionTree (baseline)": (4.826, 1.287, 1.616),
    "Tree: RandomForest": (2.867, 1.881, 1.179), "Ensemble: Voting (Logit + RF)": (2.581, 2.000, 1.762)}

Xtr, ytr, Xte, yte = tr[cols], tr.clf_target, te[cols], te.clf_target
folds = list(M.TimeForwardCV(5).split(Xtr))
rows, curves = [], {}
for name, m in models.items():
    fit = clone(m).fit(Xtr, ytr)
    s_tr, s_te = M.clf_scores(fit, Xtr), M.clf_scores(fit, Xte)
    oof = np.full(len(Xtr), np.nan)
    for a, b in folds:
        oof[b] = M.clf_scores(clone(m).fit(Xtr.iloc[a], ytr.iloc[a]), Xtr.iloc[b])
    cvm = pd.DataFrame([M.clf_metrics(ytr.iloc[b], oof[b], pred=M.flag_top_share(oof[b], SHARE)) for _, b in folds])
    trm = M.clf_metrics(ytr, s_tr, pred=M.flag_top_share(s_tr, SHARE))
    tem = M.clf_metrics(yte, s_te, pred=M.flag_top_share(s_te, SHARE))
    row = {"Model": name}
    for tag, d in [("Train", trm), ("CV", cvm.mean()), ("Test", tem)]:
        for k in ["Precision", "Recall", "PR_AUC", "Lift"]:
            row[f"{tag} {k}"] = float(d[k])
    row["Train late rate"], row["CV late rate"], row["Test late rate"] = float(ytr.mean()), float(np.mean([ytr.iloc[b].mean() for _, b in folds])), float(yte.mean())
    rows.append(row)
    print(name, {k: round(v, 3) for k, v in row.items() if k != "Model"}, flush=True)
    exp = NB[name]
    got = (row["Train Lift"], row["CV Lift"], row["Test Lift"])
    if name != "Prior predictor (dummy)":
        assert all(abs(g - e) < 0.0015 for g, e in zip(got, exp)), f"{name}: lifts {got} differ from notebook {exp}"
    if name == "Linear: LogisticRegression (plain)":
        def pr(y, s, n=400):
            p, r, _ = precision_recall_curve(y, s)
            idx = np.unique(np.linspace(0, len(r) - 1, n).astype(int))
            return {"recall": r[idx].round(4).tolist(), "precision": p[idx].round(4).tolist(),
                    "pr_auc": float(average_precision_score(y, s)), "late_rate": float(np.mean(y))}
        curves = {"train": pr(ytr, s_tr), "test": pr(yte, s_te),
                  "cv_folds": [pr(ytr.iloc[b], oof[b]) for _, b in folds]}
pd.DataFrame(rows).to_csv(os.path.join(OUT, "p2_train_cv_test.csv"), index=False)
json.dump(curves, open(os.path.join(OUT, "p2_pr_curves.json"), "w"))
print("ALL LIFTS MATCH THE NOTEBOOK")
