"""Phase 2 scikit-learn Pipelines, splitters and metrics.

Three model families are used across BOTH tasks (regression + classification):
  1. Linear     : LinearRegression -> Ridge        |  LogisticRegression (plain) -> L2-tuned
  2. Tree-based : DecisionTree (baseline) -> RandomForest (tuned)
  3. Ensemble   : Stacking for regression (Problem 1); soft Voting for classification (Problem 2), because Problem 2 is
                  validated forward in time and stacking's inner out-of-fold fit would need every
                  training row to be predicted by a model fitted on later rows

Every transformation (imputation, scaling, encoding, target encoding) lives inside the
Pipeline, so it is fitted on training folds only.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              StackingRegressor, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, make_scorer, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from features import EMBARGO_DAYS, RANDOM_STATE, TEST_FRAC

DAY = 86400.0

SKEWED = {"weight_kg", "volume_cm3", "chargeable_kg", "distance_km", "price", "price_total", "freight_total",
          "seller_load_7d"}


# --------------------------------------------------------------------------- splitting
class OrderGroupCV:
    """Grouped K-fold that reads the group (order_id) from a column of X.

    Reading the groups from X means the same splitter also works *inside*
    StackingRegressor / RandomizedSearchCV, where `groups=` cannot be passed.
    Items of one order never straddle train and validation folds.
    """

    def __init__(self, n_splits=5, stratify=False, seed=RANDOM_STATE):
        self.n_splits, self.stratify, self.seed = n_splits, stratify, seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        g = X["order_id"].values
        if self.stratify:
            cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed)
            yield from cv.split(X, y, g)
        else:
            order = pd.Series(g).astype("category").cat.codes.values
            rng = np.random.RandomState(self.seed)
            perm = rng.permutation(order.max() + 1)
            yield from GroupKFold(self.n_splits).split(X, y, perm[order])


class TimeForwardCV:
    """Forward-chaining (expanding-window) CV on purchase time, with an embargo.

    Rows are cut into n_splits + 1 consecutive time blocks of equal size; fold j trains on blocks
    1..j and validates on block j+1. Training rows purchased within `embargo_days` of the validation
    start are dropped, because their delivery outcome would not yet be known when the model is fitted.
    Reads `purchase_ts` (seconds) from X, so the same splitter works inside GridSearchCV.
    """

    def __init__(self, n_splits=5, embargo_days=EMBARGO_DAYS):
        self.n_splits, self.embargo_days = n_splits, embargo_days

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        ts = np.asarray(X["purchase_ts"], dtype=float)
        blocks = np.array_split(np.argsort(ts, kind="stable"), self.n_splits + 1)
        for j in range(1, self.n_splits + 1):
            va = blocks[j]
            tr = np.concatenate(blocks[:j])
            tr = tr[ts[tr] < ts[va].min() - self.embargo_days * DAY]
            yield np.sort(tr), np.sort(va)


def holdout_split(df, strat, test_frac=0.2, seed=RANDOM_STATE, groups=None):
    """Grouped train/test split (by order_id unless `groups` is given), stratified on `strat`.

    strat: the class label for classification (clf_target) or quintile bins of the target for
    regression, so train and test see the same target distribution.
    """
    k = int(round(1 / test_frac))
    cv = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    tr, te = next(cv.split(df, strat, df.order_id if groups is None else groups))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def time_holdout(df, test_frac=TEST_FRAC, embargo_days=EMBARGO_DAYS):
    """Train on earlier orders, test on the latest `test_frac` by purchase time.

    Training orders purchased in the `embargo_days` before the test window are dropped: at the cut-off
    many of them would not yet have been delivered, so their label could not have been known.
    """
    cut = df.purchase_ts.quantile(1 - test_frac)
    test = df[df.purchase_ts > cut]
    train = df[df.purchase_ts <= cut - embargo_days * DAY]
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- preprocessing
def make_preprocessor(fs, linear):
    """ColumnTransformer for a feature-set spec {num, cat, te}.

    linear=True : log1p on skewed numerics + standardise (needed by Ridge / Logistic).
    linear=False: raw numerics (trees are scale-invariant).
    """
    num_skew = [c for c in fs["num"] if c in SKEWED]
    num_other = [c for c in fs["num"] if c not in SKEWED]
    parts = []
    if linear:
        if num_skew:
            parts.append(("num_skew", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                                ("sc", StandardScaler())]), num_skew))
        if num_other:
            parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                           ("sc", StandardScaler())]), num_other))
    else:
        parts.append(("num", SimpleImputer(strategy="median", add_indicator=True), fs["num"]))
    if fs["cat"]:
        parts.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                           sparse_output=False), fs["cat"]))
    if fs["te"]:
        te = TargetEncoder(smooth="auto", cv=5, random_state=RANDOM_STATE)
        parts.append(("te", Pipeline([("te", te)] + ([("sc", StandardScaler())] if linear else [])), fs["te"]))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


# --------------------------------------------------------------------------- estimators
def reg_estimators(fs, rf_params=None, ridge_alpha=1.0, n_jobs=-1):
    lin = lambda m: Pipeline([("prep", make_preprocessor(fs, True)), ("m", m)])
    tree = lambda m: Pipeline([("prep", make_preprocessor(fs, False)), ("m", m)])
    rf_params = rf_params or dict(n_estimators=200, min_samples_leaf=5, max_features=0.5)
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs, **rf_params)
    return {
        "Mean predictor (dummy)": Pipeline([("m", DummyRegressor())]),
        "Linear: LinearRegression": lin(LinearRegression()),
        "Linear: Ridge": lin(Ridge(alpha=ridge_alpha)),
        "Tree: DecisionTree (baseline)": tree(DecisionTreeRegressor(max_depth=None, min_samples_leaf=5,
                                                                    random_state=RANDOM_STATE)),
        "Tree: RandomForest": tree(rf),
    }


def clf_estimators(fs, rf_params=None, logit_c=1.0, n_jobs=-1):
    lin = lambda m: Pipeline([("prep", make_preprocessor(fs, True)), ("m", m)])
    tree = lambda m: Pipeline([("prep", make_preprocessor(fs, False)), ("m", m)])
    rf_params = rf_params or dict(n_estimators=200, min_samples_leaf=5, max_features="sqrt")
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs, **rf_params)
    return {
        "Prior predictor (dummy)": Pipeline([("m", DummyClassifier(strategy="prior"))]),
        "Linear: LogisticRegression (plain)": lin(LogisticRegression(penalty=None, max_iter=2000)),
        "Linear: Logistic (L2)": lin(LogisticRegression(C=logit_c, max_iter=2000)),
        "Tree: DecisionTree (baseline)": tree(DecisionTreeClassifier(min_samples_leaf=20,
                                                                     random_state=RANDOM_STATE)),
        "Tree: RandomForest": tree(rf),
    }


def stacking_regressor(fs, rf_params, ridge_alpha, n_jobs=-1):
    est = reg_estimators(fs, rf_params, ridge_alpha, n_jobs)
    return StackingRegressor(
        estimators=[("ridge", est["Linear: Ridge"]), ("rf", est["Tree: RandomForest"])],
        final_estimator=Ridge(alpha=1.0), cv=OrderGroupCV(3), n_jobs=1, passthrough=False)


def voting_classifier(fs, rf_params, logit_c, logit_cw=None, w_logit=0.5, n_jobs=-1):
    """Soft vote of the tuned logistic model and forest (a weighted average of their probabilities).

    Needs no inner out-of-fold fit, so it respects time ordering; the weight is tuned on
    forward-chaining CV. This is the weighted-voting design of Wani et al. (2022), built from our two
    base families.
    """
    est = clf_estimators(fs, rf_params, logit_c, n_jobs)
    est["Linear: Logistic (L2)"].set_params(m__class_weight=logit_cw)
    return VotingClassifier(estimators=[("logit", est["Linear: Logistic (L2)"]), ("rf", est["Tree: RandomForest"])],
                            voting="soft", weights=[w_logit, 1 - w_logit])


# --------------------------------------------------------------------------- metrics
def pr_lift(y, score):
    """PR-AUC divided by the late rate: 1.0 = no skill. Comparable across periods with different late rates."""
    return average_precision_score(y, score) / np.mean(y)


LIFT_SCORER = make_scorer(pr_lift, response_method="predict_proba")


def flag_top_share(score, share):
    """Decision rule: flag the `share` highest-risk orders among those being scored (top-k by risk).

    `share` is an assumed review capacity (10% by default). Ranking needs no calibrated probability or fixed
    cut-off, so it is robust to the late rate drifting over time (1% to 19% by month in these data).
    Ties are broken at random (fixed seed), so a constant-score model flags a random 10%, i.e. random review.
    """
    s = np.asarray(score, dtype=float)
    tie = np.random.RandomState(RANDOM_STATE).rand(len(s))
    top = np.lexsort((tie, s))[::-1][:int(round(share * len(s)))]       # highest score first
    out = np.zeros(len(s), dtype=int)
    out[top] = 1
    return out


def reg_metrics(y_log, pred_log):
    """Metrics on the log scale (model target) and on the original R$ scale."""
    pred_log = np.asarray(pred_log)
    y, p = np.expm1(y_log), np.clip(np.expm1(pred_log), 0, None)
    return dict(MAE_rs=mean_absolute_error(y, p), RMSE_rs=mean_squared_error(y, p) ** 0.5,
                R2_rs=r2_score(y, p), MAE_log=mean_absolute_error(y_log, pred_log),
                RMSE_log=mean_squared_error(y_log, pred_log) ** 0.5, R2_log=r2_score(y_log, pred_log))


def clf_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def clf_metrics(y, score, threshold=0.5, pred=None):
    """Threshold metrics use `pred` if given (e.g. the top-10% review rule), else score >= threshold."""
    y = np.asarray(y)
    pred = (np.asarray(score) >= threshold).astype(int) if pred is None else np.asarray(pred)
    out = dict(Flagged=pred.mean(), Precision=precision_score(y, pred, zero_division=0), Recall=recall_score(y, pred),
               F1=f1_score(y, pred), BalancedAcc=balanced_accuracy_score(y, pred),
               ROC_AUC=roc_auc_score(y, score), PR_AUC=average_precision_score(y, score),
               Lift=average_precision_score(y, score) / y.mean())
    if 0 <= np.min(score) and np.max(score) <= 1:
        out["Brier"] = brier_score_loss(y, score)
    return out
