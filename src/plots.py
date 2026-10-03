"""Phase 2 figures, drawn with the Phase 1 theme (olist_theme)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

import olist_theme as T

T.apply_theme()


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def target_overview(items, orders, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(items.freight_value.clip(upper=items.freight_value.quantile(.995)), bins=60, color=T.CAT[0])
    ax[0].set_xlabel("freight_value (R$, clipped at p99.5)"); ax[0].set_ylabel("order items")
    T.title(ax[0], "Problem 1 target: right-skewed freight", "modelled as log1p(freight)")
    m = (orders.assign(ym=orders.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp())
         .groupby("ym").clf_target.agg(["mean", "size"]).query("size >= 300"))
    ax[1].bar(m.index, m["mean"] * 100, width=22, color=T.CAT[0])
    ax[1].axhline(orders.clf_target.mean() * 100, color=T.CAT[1], lw=2)
    ax[1].set_xlabel("purchase month"); ax[1].set_ylabel("% of delivered orders late")
    T.title(ax[1], f"Problem 2 target: {orders.clf_target.mean():.1%} late, with spikes", "red line = overall late rate")
    fig.tight_layout()
    return _save(fig, path)


def pred_vs_actual(y_log, pred_log, path=None):
    y, p = np.expm1(y_log), np.expm1(pred_log)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    s = np.random.RandomState(42).choice(len(y), min(8000, len(y)), replace=False)
    ax[0].scatter(y.iloc[s] if hasattr(y, "iloc") else y[s], p[s], s=5, alpha=.3, color=T.CAT[0])
    lim = float(np.nanpercentile(y, 99.5))
    ax[0].plot([0, lim], [0, lim], color=T.CAT[1], lw=1.5); ax[0].set(xlim=(0, lim), ylim=(0, lim))
    ax[0].set_xlabel("actual freight (R$)"); ax[0].set_ylabel("predicted (R$)")
    T.title(ax[0], "Predicted vs actual freight", "held-out test set")
    res = np.asarray(y_log) - np.asarray(pred_log)
    ax[1].hist(res, bins=80, range=(-1.5, 1.5), color=T.CAT[0]); ax[1].axvline(0, color=T.CAT[1])
    ax[1].set_xlabel("residual (log scale: actual − predicted)"); ax[1].set_ylabel("order items")
    T.title(ax[1], "Residuals centred, long right tail", "the tail is the overcharge candidates")
    fig.tight_layout()
    return _save(fig, path)


def error_by_price_band(test, pred_log, path=None):
    d = test[["price", "freight_value"]].copy()
    d["pred"] = np.expm1(pred_log)
    d["band"] = pd.cut(d.price, [0, 25, 50, 100, 200, 500, np.inf], labels=["<25", "25–50", "50–100", "100–200", "200–500", "500+"])
    g = d.groupby("band", observed=True).apply(lambda x: pd.Series({"MAE": (x.freight_value - x.pred).abs().mean(),
                                                                    "MAE % of mean freight": (x.freight_value - x.pred).abs().mean() / x.freight_value.mean() * 100,
                                                                    "n": len(x)}), include_groups=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(g.index.astype(str), g.MAE, color=T.CAT[0]); T.label_bars(ax, "{:.1f}")
    ax.set_xlabel("item price band (R$)"); ax.set_ylabel("MAE (R$)")
    T.title(ax, "Error by price band", "errors are not pooled, per the Phase 1 literature review")
    fig.tight_layout(); _save(fig, path)
    return g


def importance_bar(imp, title, path=None, top=12, unit=""):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(d.feature, d.importance, xerr=d.sd, color=T.CAT[0], ecolor=T.MUTED)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel(f"permutation importance {unit}", fontsize=12)
    T.title(ax, title, "drop in test score when the column is shuffled")
    fig.tight_layout()
    return _save(fig, path)


def ablation_chart_reg(ab, path=None):
    """Problem 1: CV MAE as feature groups are added."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(ab["Feature set"][::-1], ab["MAE (R$)"][::-1], color=T.CAT[0]); T.label_bars(ax, "{:.2f}", orient="h")
    ax.set_xlabel("CV MAE (R$), lower is better"); T.title(ax, "Problem 1: MAE by feature set")
    fig.tight_layout()
    return _save(fig, path)


def ablation_chart_clf(ab, path=None):
    """Problem 2: forward-chaining CV lift for both families as feature groups are added."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    b = ab.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(b))
    ax.barh(y + 0.2, b["Logit lift"], height=0.4, color=T.CAT[0], label="Logistic (L2)")
    ax.barh(y - 0.2, b["RF lift"], height=0.4, color=T.CAT[2], label="RandomForest")
    ax.set_yticks(y, b["Feature set"].str.replace(r" \(.*\)", "", regex=True))
    ax.axvline(1, color=T.AXIS, ls="--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9, frameon=False)
    ax.set_xlabel("forward-chaining CV PR-AUC lift (1 = no skill)")
    T.title(ax, "Problem 2: lift by feature set", "time-ordered folds; only route and order value add reliably")
    fig.tight_layout()
    return _save(fig, path)


def gain_chart(gain, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(1, len(gain) + 1)
    ax[0].bar(x, gain.late_rate * 100, color=T.CAT[0]); T.label_bars(ax[0], "{:.1f}", orient="v", size=8)
    ax[0].axhline(gain.late.sum() / gain.orders.sum() * 100, color=T.CAT[1])
    ax[0].set_xlabel("risk decile (1 = highest predicted risk)"); ax[0].set_ylabel("% of orders late")
    T.title(ax[0], "Late rate by risk decile", "red line = overall late rate")
    ax[1].plot(np.r_[0, x] * 10, np.r_[0, gain.cum_capture * 100], color=T.CAT[0], lw=2, marker="o")
    ax[1].plot([0, 100], [0, 100], color=T.AXIS, ls="--")
    ax[1].set_xlabel("% of orders reviewed (highest risk first)"); ax[1].set_ylabel("% of late orders caught")
    T.title(ax[1], "Cumulative capture of late orders", "dashed = random review")
    fig.tight_layout()
    return _save(fig, path)


def clf_curves(fitted, test, fs_cols, y, path=None, names=None):
    names = names or list(fitted)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, n in enumerate(names):
        m = fitted[n]
        sc = m.predict_proba(test[fs_cols])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(test[fs_cols])
        RocCurveDisplay.from_predictions(y, sc, name=n.split(": ")[-1], ax=ax[0], color=T.CAT[i % 8])
        PrecisionRecallDisplay.from_predictions(y, sc, name=n.split(": ")[-1], ax=ax[1], color=T.CAT[i % 8])
    ax[0].plot([0, 1], [0, 1], color=T.AXIS, ls="--"); ax[1].axhline(y.mean(), color=T.AXIS, ls="--")
    ax[0].legend(loc="lower right", fontsize=8); ax[1].legend(loc="upper right", fontsize=8)
    T.title(ax[0], "ROC curves"); T.title(ax[1], "Precision–recall curves", "dashed = prevalence (no skill)")
    fig.tight_layout()
    return _save(fig, path)


def confusion(y, pred, path=None, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, cmap=T.SEQ_CMAP,
                                            colorbar=False, display_labels=["on time", "late"], values_format=",")
    ax.set_title(title); ax.grid(False)
    fig.tight_layout()
    return _save(fig, path)


def comparison_chart_reg(reg_tbl, path=None):
    """Problem 1: test MAE by model."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    r = reg_tbl.set_index("Model").MAE_rs.sort_values(ascending=False)
    ax.barh([i.split(": ")[-1] for i in r.index], r.values, color=T.CAT[0]); T.label_bars(ax, "{:.2f}", orient="h")
    ax.set_xlabel("test MAE (R$), lower is better"); T.title(ax, "Problem 1: model comparison")
    fig.tight_layout()
    return _save(fig, path)


def comparison_chart_clf(split_tbl, path=None):
    """Problem 2: PR-AUC lift on a random split vs the time-ordered test set."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    c = split_tbl.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(c))
    ax.barh(y + 0.2, c["Random lift"], height=0.4, color=T.MUTED, label="random split (optimistic)")
    ax.barh(y - 0.2, c["Time-ordered lift"], height=0.4, color=T.CAT[0], label="time-ordered (used)")
    ax.set_yticks(y, [m.split(": ")[-1] for m in c.Model]); ax.axvline(1, color=T.AXIS, ls="--")
    ax.set_xlabel("test PR-AUC lift over the late rate (1 = no skill)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9, frameon=False)
    T.title(ax, "Problem 2: random vs time-ordered test")
    fig.tight_layout()
    return _save(fig, path)


def overcharge_charts(flagged, path=None, min_items=30, top=10):
    f = flagged
    sell = (f.groupby("seller_id").agg(n=("flag_primary", "size"), rate=("flag_primary", "mean"),
                                       excess=("excess_rs", "sum")).query("n >= @min_items"))
    cat = (f.groupby("category").agg(n=("flag_primary", "size"), rate=("flag_primary", "mean")).query("n >= 100")
           .sort_values("rate", ascending=False).head(top).iloc[::-1])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].barh(cat.index, cat.rate * 100, color=T.CAT[0]); T.label_bars(ax[0], "{:.1f}%", orient="h")
    ax[0].set_xlabel("% of items flagged"); T.title(ax[0], "Flag rate by category", f"top {top}, ≥100 test items")
    ax[1].hist(sell.rate * 100, bins=30, color=T.CAT[0]); ax[1].set_xlabel("% of a seller's items flagged")
    ax[1].set_ylabel("sellers"); T.title(ax[1], "Flags concentrate in few sellers", f"sellers with ≥{min_items} test items")
    fig.tight_layout(); _save(fig, path)
    return sell


def promise_chart(wk, starts, end, path=None):
    """Weekly promised days (median, middle 50%) over the weekly late rate, with the four test windows shaded."""
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.32})
    x = pd.to_datetime(wk.week_ending)
    ax[0].fill_between(x, wk.p25, wk.p75, color=T.CAT[0], alpha=0.18, lw=0)
    ax[0].plot(x, wk["median"], color=T.CAT[0], lw=2)
    ax[0].set_ylabel("promised days at checkout"); ax[0].set_ylim(0, 50)
    T.title(ax[0], "Olist's delivery promise, weekly", "line = median order; band = middle 50% of orders")
    ax[1].plot(x, wk.late_rate * 100, color=T.CAT[1], lw=2)
    ax[1].set_ylabel("% of orders late"); ax[1].set_ylim(0, 30)
    T.title(ax[1], "Late rate, weekly", "shaded = the four test windows (W4 = final hold-out, the stress test)")
    edges = list(starts) + [end]
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]), 1):
        for axis in ax:
            axis.axvspan(a, b, color=T.AXIS, alpha=0.30 if i == len(starts) else 0.12, lw=0)
        ax[1].text(a + (b - a) / 2, 28, f"W{i}", fontsize=9, color=T.MUTED, ha="center", va="top")
    pk = wk.loc[wk.week_ending.between("2018-05-20", "2018-06-17"), "median"].idxmax()
    ax[0].annotate(f"{wk.loc[pk, 'median']:.0f} days, week of {pd.Timestamp(wk.loc[pk, 'week_ending']) - pd.Timedelta(days=6):%d %b}",
                   xy=(x[pk], wk.loc[pk, "median"]), xytext=(x[pk] - pd.Timedelta(days=150), 44), fontsize=9, color=T.MUTED,
                   arrowprops=dict(arrowstyle="-", color=T.AXIS, lw=1))
    ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    return _save(fig, path)
