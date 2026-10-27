# -*- coding: utf-8 -*-
"""
Olist logistics assistant: the Phase 3 model app (Team 5, IT5006).

Deploys the two final models, one tool set per stakeholder: Problem 1 from notebooks/Team5_Phase3_IT5006_AY2627Sem1.ipynb
(re-evaluated on a time-ordered split), Problem 2 from notebooks/Team5_Phase2_IT5006_AY2627Sem1.ipynb.

  Logistics-pricing / finance analyst (Problem 1, fair freight; RandomForest on a time-ordered split, with Ridge as
  the audit's second opinion)
    Freight audit   which sellers and items to audit this month
    Check a charge  is one charge fair; what should a new listing be quoted
    Tariff review   where a simple published tariff over- or under-charges
  Operations and customer-experience team (Problem 2, late delivery; plain LogisticRegression)
    Daily risk queue  which of today's orders to follow up, and how
    Promise check     how risky a new order is, and what promise would keep it off the review list
    Monitoring        when a change in the promise rules means the model needs a review

The model files and data are written by build_models.py, which checks them against the notebooks; every headline
number on these pages is read from data/facts.json. The Olist data end in 2018, so the app scores held-out
historical records as if they were new orders: the freight items come from the Problem 1 test set (the latest 20% of
items, 24 May - 3 Sep 2018) and the orders from the Problem 2 test window (26 May - 29 Aug 2018), none of which the
models saw in training.

Run locally:   streamlit run deployment/model_app/app.py
"""
import json, re, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import features as F, models as M            # the same feature and decision-rule code the notebook uses

st.set_page_config(page_title="Olist logistics assistant", page_icon="📦", layout="wide")

REPO_URL = "https://github.com/likeyou-ky/Team5_IT5006_Ecommerce_Analytics_AY2627Sem1"
EDA_URL = "https://team5-olist-eda-phase1.streamlit.app/"
MAX_UPLOAD_ROWS = 50_000

# Chart colours: the CVD-validated slots and diverging pair of src/olist_theme.py
BLUE, ORANGE, GREY, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#c3c2b7", "#10100f", "#52514e", "#e1e0d9"
DIV_LOW, DIV_MID, DIV_HIGH = "#184f95", "#f0efec", "#e34948"
LABELS = {"promised_days": "Promised delivery time", "distance_km": "Distance seller to customer",
          "same_state": "Same-state shipment", "price_total": "Order value", "freight_total": "Freight charged",
          "freight_ratio": "Freight / order value", "n_items": "Items in the order", "n_sellers": "Sellers in the order",
          "seller_state": "Seller state", "customer_state": "Customer state"}
# Suggested follow-up for a flagged order, by the group of inputs that pushes its risk up most. All are cheap,
# because only about one flagged order in 15 is really late (Phase 2 report, Section 7.2).
DRIVER_GROUP = {"promised_days": "promise", "distance_km": "route", "same_state": "route", "seller_state": "route",
                "customer_state": "route", "price_total": "order", "freight_total": "order", "freight_ratio": "order",
                "n_items": "order", "n_sellers": "order"}
ACTIONS = {"promise": "Update the delivery estimate", "route": "Check carrier tracking",
           "order": "Confirm dispatch with seller", None: "Send a tracking update"}
RULES = {"overall": "Top 5% overall (notebook rule)", "band": "Top 5% within each price band (report)"}


# ------------------------------------------------------------------ loading (cached, with a clear failure message)
@st.cache_resource(show_spinner="Loading the models...")
def load_models():
    load = lambda name: joblib.load(HERE / "models" / name)
    return load("freight_forest.joblib"), load("freight_ridge.joblib"), load("late_logit.joblib")


@st.cache_data(show_spinner=False)
def load_data():
    facts = json.loads((HERE / "data" / "facts.json").read_text(encoding="utf-8"))
    items = pd.read_csv(HERE / "data" / "heldout_items.csv", parse_dates=["order_purchase_timestamp"])
    orders = pd.read_csv(HERE / "data" / "heldout_orders.csv", parse_dates=["order_purchase_timestamp"])
    zips = pd.read_csv(HERE / "data" / "zip_centroids.csv").set_index("zip_prefix")
    return facts, items, orders, zips


try:
    MODEL, RIDGE, LOGIT = load_models()
    FACTS, ITEMS, ORDERS, ZIPS = load_data()
except Exception as exc:                                                    # missing or unreadable files
    st.error("The app could not load its model or data files. Rebuild them with "
             "`python deployment/model_app/build_models.py` and make sure the library versions match "
             "`deployment/model_app/requirements.txt`.")
    st.exception(exc)
    st.stop()

P1, P2, NB = FACTS["p1"], FACTS["p2"], FACTS["notebook"]
FLAG = P1["flag"]
COLS_A, COLS_B = list(MODEL.feature_names_in_), list(LOGIT.feature_names_in_)
USES_LISTING = "photos_qty" in COLS_A           # listing details and purchase time: inputs only if the model kept them
USES_TIMING = "purchase_month" in COLS_A
STATES = P1["states"]
DAILY_YIELD = 1 / P2["daily"]["precision"]          # flagged orders checked per late order found, daily 10% list


# ------------------------------------------------------------------ helpers
def money(x, dp=2):
    """R$ value for st.metric and tables (plain text)."""
    return f"R$ {x:,.{dp}f}"


def md_money(x, dp=2):
    """R$ value inside markdown, where a bare $ pair would render as LaTeX."""
    return f"R\\$ {x:,.{dp}f}"


def layout(fig, h=360, title=None, **kw):
    """Shared chart style: title on its own line, legend between title and plot, hairline grid."""
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top"))
    fig.update_layout(height=h, margin=dict(t=80 if title else 40, b=10, l=10, r=10), template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


def how_to(text):
    with st.expander("How to use this page", expanded=False):
        st.markdown(text)


def parse_zip(text):
    """Zip-code prefix (first five digits of a CEP such as 01310-100), or None if empty; ValueError if malformed."""
    digits = re.sub(r"\D", "", text or "")
    if not digits:
        return None
    if len(digits) < 4:
        raise ValueError(f"'{text}' is not a zip code (enter the first five digits of the CEP)")
    return int(digits[:5])


def locate(zip_text, state, who):
    """(lat, lng, state, note) for a party, from its zip prefix if known, else the centre of its state."""
    try:
        z = parse_zip(zip_text)
    except ValueError as e:
        return None, str(e)
    if z is None:
        lat, lng = FACTS["state_centroids"][state]
        return (lat, lng, state), None
    # Zip codes are allocated by region: the first three digits fix the sub-region (one state in 861 of 862 cases),
    # so an unknown prefix takes the nearest known one that shares them
    same = ZIPS.index.values[ZIPS.index.values // 100 == z // 100]
    if len(same) == 0:
        lat, lng = FACTS["state_centroids"][state]
        return (lat, lng, state), f"{who} zip {z:05d} is not in the Olist zip table: using the centre of {state}."
    near = int(same[np.abs(same - z).argmin()])
    row = ZIPS.loc[near]
    notes = [] if near == z else [f"{who} zip {z:05d} is not in the Olist zip table: using the nearest one, {near:05d}"]
    if row.state != state:
        notes.append(f"{who.lower()} zip {near:05d} is in {row.state}, not {state}: using {row.state}")
    note = "; ".join(notes)
    return (row.lat, row.lng, row.state), (note[0].upper() + note[1:] + ".") if notes else None


def place_inputs(prefix, label):
    c = st.columns(2)
    state = c[0].selectbox(f"{label} state", STATES, index=STATES.index("SP"), key=f"{prefix}_state")
    z = c[1].text_input(f"{label} zip code (optional)", key=f"{prefix}_zip", placeholder="e.g. 01310")
    return z, state


def locate_pair(s_zip, s_state, c_zip, c_state):
    """Seller and customer locations, with notes shown and errors returned."""
    s_loc, s_note = locate(s_zip, s_state, "Seller")
    c_loc, c_note = locate(c_zip, c_state, "Customer")
    errors = [n for loc, n in [(s_loc, s_note), (c_loc, c_note)] if loc is None]
    if not errors:
        for n in (s_note, c_note):
            if n:
                st.info(n)
    return s_loc, c_loc, errors


def range_note(values, ranges):
    """Warn when an input lies outside the middle 99.8% of the training data."""
    out = [f"{name} {v:,.2f} is outside the range seen in training ({lo:,.2f} to {hi:,.2f})"
           for name, (v, (lo, hi)) in ((n, (values[n], ranges[n])) for n in values) if not lo <= v <= hi]
    if out:
        st.warning("The model has seen few cases like this, so treat the result with caution: " + "; ".join(out) + ".")


# ------------------------------------------------------------------ Problem 1 scoring
def error_band(price):
    """The test-set error of items at this price (the bands of the notebook's error-by-price table)."""
    for b, hi in zip(P1["price_bands"], [25, 50, 100, 200, 500, np.inf]):
        if price < hi:
            return b
    return P1["price_bands"][-1]


def audit_band(price):
    """Price band of the unusual-freight rule (the bands of the notebook's flag-rate table)."""
    edges = [0] + FLAG["band_edges"] + [np.inf]
    return pd.cut(pd.Series(np.atleast_1d(price), dtype=float), edges, labels=FLAG["band_labels"],
                  right=False).astype(str).values


def unusual(charged, expected, price, model, rule):
    """Unusual freight: log residual at or above the cut-off of `model` ("model" = the forest, or "ridge")."""
    cut = FLAG["cuts"][model]
    thr = cut["overall"] if rule == "overall" else pd.Series(audit_band(price)).map(cut["band"]).values.astype(float)
    res = np.log1p(np.atleast_1d(charged).astype(float)) - np.log1p(np.atleast_1d(expected).astype(float))
    return res >= thr - FLAG["tol"]              # tolerance: see build_models.py


def predict(model, df):
    """Expected (fair) freight in R$ for rows holding the model's input columns."""
    return np.clip(np.expm1(model.predict(df[COLS_A])), 0, None)


def tariff(chargeable_kg, distance_km, same_state):
    t = P1["tariff"]
    return np.maximum(0.0, t["base"] + t["per_kg"] * np.asarray(chargeable_kg) + t["per_100km"]
                      * np.asarray(distance_km) / 100 + t["same_state"] * np.asarray(same_state))


def add_scores(d, rule):
    """Fair freight from both models, the unusual flags under `rule`, and the audit columns."""
    d = d.copy()
    if "expected" not in d:
        d["expected"], d["expected_ridge"] = predict(MODEL, d), predict(RIDGE, d)
    d["unusual"] = unusual(d.freight_value, d.expected, d.price, "model", rule)
    d["unusual_ridge"] = unusual(d.freight_value, d.expected_ridge, d.price, "ridge", rule)
    d["both"] = d.unusual & d.unusual_ridge
    d["excess"] = d.freight_value - d.expected
    d["markup"] = np.where(d.expected > 0, d.freight_value / d.expected.where(d.expected > 0), np.nan)
    d["band"] = audit_band(d.price)
    return d


@st.cache_data(show_spinner="Scoring the held-out items...")
def heldout_predictions():
    return ITEMS.assign(expected=predict(MODEL, ITEMS), expected_ridge=predict(RIDGE, ITEMS))


@st.cache_data(show_spinner=False)
def heldout_items_scored(rule):
    return add_scores(heldout_predictions(), rule)


def rule_picker():
    """The unusual-freight rule, shared by the analyst's pages."""
    keys = list(RULES)
    choice = st.radio("Unusual-freight rule", keys, index=keys.index(st.session_state.get("rule", "overall")),
                      format_func=RULES.get, horizontal=True,
                      help="Overall: the 5% of items with the largest charge above fair freight, as in the "
                           "notebooks. Within each price band: 5% of every band, as the report recommends, so expensive "
                           "items (harder to predict) do not crowd the review list.")
    st.session_state.rule = choice
    return choice


# ------------------------------------------------------------------ Problem 2 scoring
@st.cache_data(show_spinner="Scoring the held-out orders...")
def heldout_orders_scored():
    d = ORDERS.copy()
    d["risk"] = LOGIT.predict_proba(d[COLS_B])[:, 1]
    d["day"] = d.order_purchase_timestamp.dt.normalize()
    return d


@st.cache_data(show_spinner=False)
def daily_flags(share):
    """The queue rule: flag the riskiest `share` of each purchase day's orders (the same rule build_models.py scores)."""
    d = heldout_orders_scored()
    flags = np.zeros(len(d), dtype=int)
    for _, idx in d.groupby("day").indices.items():
        flags[idx] = M.flag_top_share(d.risk.values[idx], share)
    return flags


def percentile(risk):
    """Where a risk score falls among the orders of the test window (0-100)."""
    return np.interp(risk, P2["score_pct"], np.linspace(0, 100, 101))


@st.cache_data(show_spinner=False)
def _logit_reference():
    return LOGIT.named_steps["prep"].transform(ORDERS[COLS_B]).mean(axis=0)


def logit_drivers(rows):
    """Each input's push on the log-odds of lateness, relative to the average order of the test window."""
    prep, lr = LOGIT.named_steps["prep"], LOGIT.named_steps["m"]
    names = prep.get_feature_names_out()
    contrib = (prep.transform(rows[COLS_B]) - _logit_reference()) * lr.coef_.ravel()
    raw = []
    for n in names:
        n = n.removeprefix("missingindicator_")
        raw.append(next((c for c in ("seller_state", "customer_state") if n.startswith(c + "_")), n))
    return pd.DataFrame(contrib, columns=names, index=rows.index).T.groupby(raw).sum().T


def suggested_actions(rows):
    """The follow-up for each order, from the group of inputs that raises its risk most."""
    groups = logit_drivers(rows).T.groupby(lambda c: DRIVER_GROUP.get(c, "order")).sum().T
    top = groups.idxmax(axis=1).where(groups.max(axis=1) > 0)
    return top.map(lambda g: ACTIONS[g if isinstance(g, str) else None])


def drivers_chart(contrib, title):
    s = contrib.sort_values()
    s = s[s.abs() > 1e-3]
    lab = [LABELS.get(k, k) for k in s.index]
    fig = go.Figure()
    for sign, colour, name in [(-1, BLUE, "lowers the risk"), (1, ORANGE, "raises the risk")]:
        part = s[np.sign(s) == sign]
        fig.add_bar(x=part.values, y=[LABELS.get(k, k) for k in part.index], orientation="h", marker_color=colour,
                    name=name, hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=lab)
    fig.update_xaxes(title="push on risk vs an average order (log-odds)")
    return layout(fig, h=60 + 34 * len(s), title=title, bargap=0.35)


def promise_curve(row, line, measure):
    """Risk of one order for every promise from 3 to 60 days, and the shortest promise that falls below `line`.

    measure="pct": risk as a percentile among test-window orders (a new order at checkout);
    measure="risk": the raw score, for a line set by one day's queue.
    """
    days = np.arange(3, 61)
    rows = pd.concat([row] * len(days), ignore_index=True).assign(promised_days=days.astype(float))
    risk = LOGIT.predict_proba(rows[COLS_B])[:, 1]
    y = percentile(risk) if measure == "pct" else risk
    ok = days[y < line]
    return pd.DataFrame({"days": days, "y": y}), (int(ok.min()) if len(ok) else None)


def promise_section(row, line, measure, line_label, title):
    """Promise what-if: the lever the Phase 2 analysis found strongest."""
    curve, safe = promise_curve(row, line, measure)
    now = float(row.promised_days.iloc[0])
    now_risk = LOGIT.predict_proba(row[COLS_B])[:, 1]
    now_y = float(percentile(now_risk)[0] if measure == "pct" else now_risk[0])
    pct = measure == "pct"
    fig = go.Figure()
    fig.add_scatter(x=curve.days, y=curve.y, mode="lines", line=dict(color=BLUE, width=2), name="risk",
                    hovertemplate=("promise %{x} days → riskier than %{y:.0f}% of orders" if pct else
                                   "promise %{x} days → risk score %{y:.3f}") + "<extra></extra>")
    fig.add_hline(y=line, line=dict(color=INK_2, width=1, dash="dash"), annotation_text=line_label,
                  annotation_position="top right")
    fig.add_scatter(x=[now], y=[now_y], mode="markers", name="current promise",
                    marker=dict(color=ORANGE, size=11, line=dict(color="white", width=2)),
                    hovertemplate="current promise %{x:.0f} days<extra></extra>")
    fig.update_xaxes(title="promised delivery time (days)")
    fig.update_yaxes(title="risk percentile among test-window orders" if pct else "risk score",
                     range=[0, 101] if pct else [0, max(curve.y.max(), line) * 1.1])
    st.plotly_chart(layout(fig, h=330, title=title), width="stretch")
    if safe is None:
        st.markdown("No promise up to 60 days takes this order below the review line: the route and the order "
                    "itself carry the risk, so check it with the carrier or seller.")
    elif now >= safe:
        st.markdown(f"The current promise ({now:.0f} days) already keeps it below the review line, which starts "
                    f"at **{safe} days**.")
    else:
        st.markdown(f"A promise of **{safe} days or more** would take this order below the review line "
                    f"(it was promised {now:.0f} days). A longer promise lowers *recorded* lateness, not the actual "
                    f"delivery time, so use it for routes that genuinely take longer.")


# ------------------------------------------------------------------ Home
def home():
    st.title("Olist logistics assistant")
    st.markdown(
        "Decision tools for the two teams our project serves, each built on the final model of one of our two problems. "
        "The Olist data end in 2018, so the tools score **held-out historical records as if they were new**: freight "
        "items and orders the models never saw in training.")
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Logistics-pricing / finance analyst")
        st.markdown(
            "**Question:** what freight *should* an item cost, and which charges and sellers need an audit?\n\n"
            f"**Model:** a random forest predicts fair freight to within {md_money(P1['test']['MAE_rs'])} on average "
            f"for items bought after its training data ({P1['test']['MAE_pct']}% of mean freight); a Ridge model gives "
            f"a second opinion on unusual charges.")
        st.markdown(
            "| When | Decision | Page |\n|---|---|---|\n"
            "| Monthly | Which sellers and items to audit | Freight audit |\n"
            "| Ad hoc | Is this charge fair? What should a new listing be quoted? | Check a charge |\n"
            "| Quarterly | Where a simple published tariff misprices freight | Tariff review |")
        st.page_link(PAGE_AUDIT, label="Start with the freight audit", icon="🔍")
    with b:
        st.subheader("Operations and customer-experience team")
        st.markdown(
            "**Question:** which of today's orders are most likely to arrive late, and what should we do?\n\n"
            f"**Model:** a logistic model ranks orders at checkout. Checking the riskiest {P2['share']:.0%} of each "
            f"day's orders catches {P2['daily']['recall']:.0%} of late orders, about twice what random checks catch; "
            f"about one checked order in {DAILY_YIELD:.0f} is really late, so follow-ups must be cheap.")
        st.markdown(
            "| When | Decision | Page |\n|---|---|---|\n"
            "| Daily | Which orders to follow up, and how | Daily risk queue |\n"
            "| At checkout | What promise to give a new order | Promise check |\n"
            "| Weekly | Whether a change in the promise rules calls for a model review | Monitoring |")
        st.page_link(PAGE_QUEUE, label="Start with today's risk queue", icon="⏰")

    st.divider()
    st.subheader("How to use the app")
    st.markdown(
        "1. Pick your team's pages in the sidebar. Each page has a **How to use this page** note at the top.\n"
        "2. The data are historical: a *day* in the queue is a day between 26 May and 29 Aug 2018, and *Reveal what "
        "happened* shows the outcome nobody knew at checkout.\n"
        "3. Forms take a listing or an order you describe; zip codes place seller and customer exactly, otherwise the "
        "centre of the state is used.\n"
        "4. Lists can be downloaded as CSV for follow-up. The **Model card** gives the data, validation, test results "
        "and limits of every model.")
    st.caption(f"Team 5 · IT5006 · AY2026/27. Code: [GitHub]({REPO_URL}). "
               f"Exploratory analysis (Phase 1): [EDA dashboard]({EDA_URL}).")


# ------------------------------------------------------------------ Analyst: freight audit
def read_upload(file):
    """Validate an uploaded CSV and derive the model's inputs; returns (frame, messages) or raises ValueError."""
    try:
        u = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"the file could not be read as CSV ({e}).")
    msgs = []
    if len(u) > MAX_UPLOAD_ROWS:
        msgs.append(f"only the first {MAX_UPLOAD_ROWS:,} of {len(u):,} rows are scored.")
        u = u.head(MAX_UPLOAD_ROWS)
    text_cols = ("order_id", "seller_id", "seller_state", "customer_state", "category")
    for c in u.columns:                                         # numbers first, so text in a number column -> NaN
        if c not in text_cols:
            u[c] = pd.to_numeric(u[c], errors="coerce")
    if "distance_km" not in u and {"seller_zip_prefix", "customer_zip_prefix"} <= set(u):
        s, c = ZIPS.reindex(u.seller_zip_prefix.values), ZIPS.reindex(u.customer_zip_prefix.values)
        u["distance_km"] = F.haversine_km(s.lat.values, s.lng.values, c.lat.values, c.lng.values)
    if "chargeable_kg" not in u and {"weight_kg", "volume_cm3"} <= set(u):
        u["chargeable_kg"] = np.maximum(u.weight_kg, u.volume_cm3 / 5000)
    if "same_state" not in u and {"seller_state", "customer_state"} <= set(u):
        u["same_state"] = (u.seller_state == u.customer_state).astype(int)
    need = [c for c in COLS_A if c not in ("order_id", "purchase_ts")] + ["freight_value"]
    missing = [c for c in need if c not in u]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing) + ". Download the template for the layout.")
    if "order_id" not in u:
        u["order_id"] = [f"row {i + 1}" for i in range(len(u))]
    u["purchase_ts"] = 0.0                                                   # a split key only, not a model input
    if "seller_id" not in u:
        u["seller_id"] = "unknown"
    num = [c for c in need if c not in ("seller_state", "customer_state", "category")]
    bad = (u[["price", "freight_value"]].isna().any(axis=1) | (u.price <= 0) | (u.freight_value < 0)
           | (u[[c for c in num if c not in ("price", "freight_value")]] < 0).any(axis=1))
    if bad.any():
        msgs.append(f"{bad.sum():,} row{'s' if bad.sum() != 1 else ''} with a missing or negative price or freight, "
                    f"or a negative input, {'were' if bad.sum() != 1 else 'was'} skipped.")
        u = u[~bad]
    rows = lambda n: f"{n:,} row{'s' if n != 1 else ''}"
    gaps = u[[c for c in num if c not in ("price", "freight_value")]].isna().any(axis=1).sum()
    if gaps:
        msgs.append(f"{rows(gaps)} had a missing or non-numeric input; the model fills it with the training median.")
    unknown = (~u.category.isin(P1["categories"])).sum()
    if unknown:
        msgs.append(f"{rows(unknown)} had a category the model has not seen; it treats it as a rare category.")
    if u.empty:
        raise ValueError("no valid rows left to score.")
    for c in ("seller_state", "customer_state", "category", "order_id", "seller_id"):
        u[c] = u[c].astype(str)
    return u.reset_index(drop=True), msgs


def audit_page():
    st.title("🔍 Freight audit")
    st.markdown(
        "Which sellers and items should the pricing team audit this month? The random forest gives each item's fair "
        "freight; an item is **unusual** when its charge sits in the top 5% above it. The Ridge model gives an "
        "independent second opinion, and items both models flag are reviewed first.")
    how_to(
        "1. Choose the items: all held-out items (the test set), or a CSV of your own with the charged freight "
        "(download the template for the columns; `distance_km` may be replaced by `seller_zip_prefix` and "
        "`customer_zip_prefix`).\n"
        "2. Choose the rule. *Within each price band* keeps expensive items, which are harder to predict, from "
        "crowding the list.\n"
        "3. **Sellers to audit** ranks sellers by the *share* of their items that are unusual (with a minimum number "
        "of items), not by volume: large sellers mostly price freight consistently.\n"
        "4. **Items to review** lists the unusual items, those flagged by both models first. Download either list.\n\n"
        "A flag is a reason to look, not proof of overcharging: fragile goods, data errors or cost recovery can also "
        "explain a high charge.")
    tmpl_cols = ["order_id", "seller_id", "category", "price", "freight_value", "weight_kg", "volume_cm3",
                 "distance_km", "seller_state", "customer_state"]
    tmpl_cols += [c for c in COLS_A if c not in tmpl_cols + ["chargeable_kg", "same_state", "purchase_ts"]]
    c = st.columns([2, 1])
    src = c[0].radio("Items to audit", ["all held-out items", "upload a CSV"], horizontal=True)
    c[1].download_button("Download the CSV template", ITEMS[tmpl_cols].head(5).to_csv(index=False),
                         "freight_audit_template.csv", "text/csv", width="stretch")
    rule = rule_picker()
    if src == "upload a CSV":
        file = st.file_uploader("CSV of order items", type="csv")
        if file is None:
            return
        try:
            raw, msgs = read_upload(file)
        except ValueError as e:
            st.error(f"The upload could not be scored: {e}")
            return
        for m in msgs:
            st.warning(m)
        d = add_scores(raw, rule)
    else:
        d = heldout_items_scored(rule)
    f = d[d.unusual]
    k = st.columns(4)
    k[0].metric("Items scored", f"{len(d):,}")
    k[1].metric("Unusual items", f"{len(f):,}", f"{len(f) / len(d):.1%} of items", delta_color="off")
    k[2].metric("Flagged by both models", f"{int(f.both.sum()):,}", f"{f.both.mean():.0%} of unusual" if len(f) else None,
                delta_color="off")
    k[3].metric("Excess on unusual items", f"{f.excess.sum() / d.freight_value.sum():.1%}" if len(f) else "–",
                help="What the unusual items were charged above their fair freight, as a share of all freight scored")
    if f.empty:
        st.success("No unusual freight in these items.")
        return
    st.markdown(f"Unusual items were charged **{md_money(f.freight_value.mean())}** on average, against a fair "
                f"freight of **{md_money(f.expected.mean())}**.")
    if src == "all held-out items":
        st.caption(f"These are the latest 20% of items (purchased {P1['test_dates'][0]} to {P1['test_dates'][1]}), "
                   f"scored by models trained only on earlier items, as a monthly audit would run. Freight rose over "
                   f"time, so the cut-off is relative to the items audited.")

    st.subheader("1 · Sellers to audit")
    g = (d.assign(excess_u=np.where(d.unusual, d.excess, 0.0), markup_u=d.markup.where(d.unusual))
         .groupby("seller_id").agg(items=("unusual", "size"), n_unusual=("unusual", "sum"), n_both=("both", "sum"),
                                   excess=("excess_u", "sum"), markup=("markup_u", "median"))
         .assign(rate=lambda t: t.n_unusual / t["items"]))
    top_n = int(g["items"].quantile(0.99)) if len(g) > 1 else 1
    min_items = st.slider("Minimum items per seller", 1, max(2, min(50, top_n)),
                          min(30, max(1, top_n)) if src == "all held-out items" else 1,
                          help="Flag rates of sellers with few items are noisy")
    s = g[g["items"] >= min_items].sort_values(["rate", "n_unusual"], ascending=False)
    if s.empty or s.n_unusual.sum() == 0:
        st.info("No seller with that many items has unusual freight: lower the minimum.")
    else:
        overall = len(f) / len(d)
        top = s.head(10).reset_index()
        top["seller"] = top.seller_id.str[:8]
        fig = go.Figure(go.Bar(x=top.rate, y=top.seller, orientation="h", marker_color=BLUE,
                               text=[f"{r:.0%}  ({u} of {n})" for r, u, n in zip(top.rate, top.n_unusual, top["items"])],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="seller %{y}: %{x:.0%} of items unusual<extra></extra>"))
        fig.add_vline(x=overall, line=dict(color=INK_2, width=1, dash="dash"),
                      annotation_text=f"all items {overall:.1%}", annotation_position="top")
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title="share of the seller's items that are unusual", tickformat=".0%",
                         range=[0, top.rate.max() * 1.35])
        st.plotly_chart(layout(fig, h=110 + 32 * len(top), title=f"Highest flag rates among the {len(s):,} sellers "
                               f"with at least {min_items} items", bargap=0.3), width="stretch")
        k_big = max(1, int(0.05 * d.seller_id.nunique()))
        big = d.seller_id.value_counts().head(k_big).index
        st.caption(f"For comparison, the {k_big:,} sellers with the most items ship {d.seller_id.isin(big).mean():.0%} "
                   f"of items but hold {f.seller_id.isin(big).mean():.0%} of the unusual ones.")
        table = s.reset_index().assign(seller=lambda t: t.seller_id.str[:8])
        st.dataframe(table[["seller", "items", "n_unusual", "rate", "n_both", "excess", "markup"]], hide_index=True,
                     width="stretch", height=260, column_config={
                         "n_unusual": st.column_config.NumberColumn("unusual"),
                         "rate": st.column_config.NumberColumn("flag rate", format="percent"),
                         "n_both": st.column_config.NumberColumn("both models"),
                         "excess": st.column_config.NumberColumn("excess (R$)", format="R$ %.2f"),
                         "markup": st.column_config.NumberColumn("median markup", format="%.2f×")})
        st.download_button("Download the seller list (CSV)", table.drop(columns="seller").to_csv(index=False),
                           "sellers_to_audit.csv", "text/csv")

    st.subheader("2 · Items to review")
    items = (f.sort_values(["both", "excess"], ascending=False).reset_index(drop=True)
             .assign(priority=lambda t: np.arange(1, len(t) + 1), order=lambda t: t.order_id.str[:8],
                     seller=lambda t: t.seller_id.str[:8], models=lambda t: np.where(t.both, "both", "forest only")))
    st.dataframe(items[["priority", "models", "freight_value", "expected", "excess", "markup", "order", "seller", "band",
                        "category"]], hide_index=True, width="stretch", height=320, column_config={
                     "band": st.column_config.TextColumn("price (R$)"),
                     "freight_value": st.column_config.NumberColumn("charged", format="R$ %.2f"),
                     "expected": st.column_config.NumberColumn("fair freight", format="R$ %.2f"),
                     "excess": st.column_config.NumberColumn("excess", format="R$ %.2f"),
                     "markup": st.column_config.NumberColumn("markup", format="%.2f×")})
    out = items[["priority", "models", "order_id", "seller_id", "category", "price", "freight_value", "expected",
                 "expected_ridge", "excess", "markup", "weight_kg", "distance_km", "seller_state", "customer_state"]]
    st.download_button("Download the review list (CSV)", out.round(2).to_csv(index=False), "items_to_review.csv",
                       "text/csv")

    st.subheader("3 · Where the unusual charges sit")
    a, b = st.columns([3, 2], gap="large")
    with a:
        ok = d[~d.unusual]
        ok = ok.sample(min(len(ok), 6000), random_state=0)
        fig = go.Figure()
        fig.add_scattergl(x=ok.expected, y=ok.freight_value, mode="markers", name="normal",
                          marker=dict(color=GREY, size=5, opacity=0.5),
                          hovertemplate="fair R$ %{x:.2f}<br>charged R$ %{y:.2f}<extra></extra>")
        for sub, colour, name in [(f[~f.both], BLUE, "unusual, forest only"), (f[f.both], ORANGE, "unusual, both models")]:
            fig.add_scattergl(x=sub.expected, y=sub.freight_value, mode="markers", name=name, text=sub.order_id.str[:8],
                              marker=dict(color=colour, size=7, line=dict(color="white", width=1)),
                              hovertemplate="order %{text}<br>fair R$ %{x:.2f}<br>charged R$ %{y:.2f}<extra></extra>")
        x = np.geomspace(max(d.expected.min(), 1), d.expected.max(), 50)
        fig.add_scatter(x=x, y=x, mode="lines", line=dict(color=INK_2, width=1), name="charged = fair", hoverinfo="skip")
        ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        fig.update_xaxes(type="log", title="fair freight (R$)", tickvals=ticks)
        fig.update_yaxes(type="log", title="charged (R$)", tickvals=ticks)
        st.plotly_chart(layout(fig, h=430, title="Charged against fair freight"), width="stretch")
    with b:
        br = d.groupby("band", sort=False).unusual.agg(["mean", "size"]).reindex(FLAG["band_labels"]).dropna()
        fig = go.Figure(go.Bar(x=br.index, y=br["mean"], marker_color=BLUE, text=[f"{v:.1%}" for v in br["mean"]],
                               textposition="outside", cliponaxis=False, customdata=br["size"],
                               hovertemplate="items R$ %{x}: %{y:.1%} unusual of %{customdata:,}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", title="share unusual", range=[0, max(br["mean"].max() * 1.3, 0.06)])
        fig.update_xaxes(title="item price (R$)")
        st.plotly_chart(layout(fig, h=430, title="Flag rate by item price", bargap=0.35), width="stretch")
        st.caption("Under the overall rule the flag rate rises with price, because expensive items are harder to "
                   "predict; the per-band rule flags 5% in every band.")


# ------------------------------------------------------------------ Analyst: check a charge
def verdict(charged, expected, expected_ridge, price, rule):
    band = error_band(price)
    c = st.columns(4)
    c[0].metric("Charged", money(charged))
    c[1].metric("Fair freight", money(expected))
    c[2].metric("Difference", money(charged - expected),
                f"{(charged / expected - 1) * 100:+.0f}%" if expected > 0 else None, delta_color="inverse")
    c[3].metric(f"Typical error at R$ {band['band']}", f"± {money(band['mae'])}")
    model_flag = bool(unusual(charged, expected, price, "model", rule)[0])
    ridge_flag = bool(unusual(charged, expected_ridge, price, "ridge", rule)[0])
    second = (f"The Ridge model agrees (it expects {md_money(expected_ridge)})." if ridge_flag == model_flag else
              f"The Ridge model disagrees (it expects {md_money(expected_ridge)}).")
    if model_flag:
        st.error(f"🚩 **Unusual freight:** {md_money(charged)} is in the top 5% above the fair freight of "
                 f"{md_money(expected)}. {second} " + ("Review it first." if ridge_flag else "Review it after the "
                                                                                             "items both models flag."))
    else:
        st.success(f"✅ **Within the normal range** for an item like this. {second}")


def check_page():
    st.title("🧾 Check a charge")
    st.markdown("Is one charge fair, and what should a new listing be quoted? Use it when a seller disputes a flag, "
                "a buyer complains about freight, or a new listing needs a sanity check.")
    rule = rule_picker()
    t1, t2 = st.tabs(["A held-out item", "A new listing (quote)"])
    with t1:
        check_item(rule)
    with t2:
        quote_listing(rule)


def check_item(rule):
    d = heldout_items_scored(rule)
    how_to("Draw a random held-out item, or paste an order ID from the audit's review list. The table under the "
           "verdict shows what the model used.")
    c = st.columns([1, 1, 2])
    only = c[0].radio("Draw from", ["all held-out items", "unusual items only"])
    pool = d[d.unusual] if only == "unusual items only" else d
    clicked = c[1].button("🎲 Random item", width="stretch")
    oid = c[2].text_input("…or enter an order ID", placeholder="e.g. " + d.order_id.iloc[0])
    if clicked or "item_id" not in st.session_state:
        st.session_state.item_id = pool.sample(1, random_state=np.random.randint(1 << 30)).index[0]
    elif oid.strip():
        hit = d.index[d.order_id.str.startswith(oid.strip().lower())]
        if len(hit):
            st.session_state.item_id = hit[0]
        else:
            st.warning("No held-out item has that order ID. Only test-set items are held here; use *A new listing* "
                       "for anything else.")
    it = d.loc[st.session_state.item_id]
    st.markdown(f"**Order `{it.order_id}`**, item {it.order_item_id} · {it.category} · seller `{it.seller_id[:8]}` · "
                f"purchased {it.order_purchase_timestamp:%d %b %Y}")
    verdict(it.freight_value, it.expected, it.expected_ridge, it.price, rule)
    st.caption(f"A simple published tariff would charge "
               f"{md_money(float(tariff(it.chargeable_kg, it.distance_km, it.same_state)))}; it errs by "
               f"{md_money(P1['tariff']['mae'])} on average, against {md_money(P1['test']['MAE_rs'])} for the forest.")
    rows = [("Price", money(it.price)), ("Weight", f"{it.weight_kg:,.2f} kg"), ("Volume", f"{it.volume_cm3:,.0f} cm³"),
            ("Chargeable weight", f"{it.chargeable_kg:,.2f} kg"), ("Route", f"{it.seller_state} → {it.customer_state}"),
            ("Distance", f"{it.distance_km:,.0f} km"), ("Category", it.category),
            ("Items / sellers in the order", f"{it.n_items} / {it.n_sellers}")]
    if USES_LISTING:
        rows.append(("Photos, name and description length",
                     f"{it.photos_qty:.0f}, {it.name_len:.0f}, {it.desc_len:.0f} characters"))
    info = pd.DataFrame(rows, columns=["Input", "Value"])
    st.dataframe(info, hide_index=True, width="stretch")


def quote_listing(rule):
    how_to("Describe the item and the route, then press **Estimate** for the fair freight and a quote range "
           "(fair freight ± the typical error at that price). Enter the charged freight to check it as well.")
    with st.form("listing"):
        a, b = st.columns(2, gap="large")
        with a:
            cat = st.selectbox("Product category", P1["categories"], index=0)
            price = st.number_input("Item price (R$)", min_value=0.0, value=120.0, step=10.0)
            weight_g = st.number_input("Weight (g)", min_value=0.0, value=1500.0, step=100.0)
            d1, d2, d3 = st.columns(3)
            length = d1.number_input("Length (cm)", min_value=0.0, value=30.0)
            width = d2.number_input("Width (cm)", min_value=0.0, value=20.0)
            height = d3.number_input("Height (cm)", min_value=0.0, value=15.0)
        with b:
            s_zip, s_state = place_inputs("s", "Seller")
            c_zip, c_state = place_inputs("c", "Customer")
            e1, e2 = st.columns(2)
            n_items = e1.number_input("Items in the order", min_value=1, value=1)
            n_sellers = e2.number_input("Sellers in the order", min_value=1, value=1)
            day, hour = pd.Timestamp("2018-08-01"), 14
            if USES_TIMING:
                w1, w2 = st.columns(2)
                day = w1.date_input("Purchase date", value=day)
                hour = w2.slider("Purchase hour", 0, 23, hour)
        photos, name_len, desc_len = (int(P1["defaults"][c]) for c in ("photos_qty", "name_len", "desc_len"))
        if USES_LISTING:
            with st.expander("Listing details (training medians by default)"):
                l1, l2, l3 = st.columns(3)
                photos = l1.number_input("Photos", min_value=0, value=photos)
                name_len = l2.number_input("Name length (characters)", min_value=0, value=name_len)
                desc_len = l3.number_input("Description length (characters)", min_value=0, value=desc_len)
        charged = st.number_input("Freight charged (R$, optional: leave 0 to skip the check)", min_value=0.0, value=0.0)
        go_ = st.form_submit_button("Estimate", type="primary")
    if not go_:
        return
    errors = [m for m, bad in [("price must be above zero", price <= 0), ("weight must be above zero", weight_g <= 0),
                               ("every dimension must be above zero", min(length, width, height) <= 0)] if bad]
    s_loc, c_loc, loc_errors = locate_pair(s_zip, s_state, c_zip, c_state)
    if errors + loc_errors:
        st.error("Please fix: " + "; ".join(errors + loc_errors) + ".")
        return
    when = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    wkg, vol = weight_g / 1000, length * width * height
    row = pd.DataFrame([dict(weight_kg=wkg, volume_cm3=vol, chargeable_kg=max(wkg, vol / 5000),
                             distance_km=float(F.haversine_km(s_loc[0], s_loc[1], c_loc[0], c_loc[1])),
                             same_state=int(s_loc[2] == c_loc[2]), price=price, n_items=n_items, n_sellers=n_sellers,
                             photos_qty=photos, name_len=name_len, desc_len=desc_len, purchase_month=when.month,
                             purchase_dow=when.dayofweek, purchase_hour=when.hour, seller_state=s_loc[2],
                             customer_state=c_loc[2], category=cat, order_id="new", purchase_ts=when.timestamp())])
    range_note({"weight_kg": wkg, "volume_cm3": vol, "price": price, "distance_km": row.distance_km[0]}, P1["ranges"])
    expected, expected_ridge = float(predict(MODEL, row)[0]), float(predict(RIDGE, row)[0])
    band = error_band(price)
    st.markdown(f"**{cat}**, {wkg:,.2f} kg, {vol:,.0f} cm³, {s_loc[2]} → {c_loc[2]} ({row.distance_km[0]:,.0f} km)")
    c = st.columns(3)
    c[0].metric("Fair freight", money(expected))
    c[1].metric("Quote range (R$)", f"{max(0, expected - band['mae']):,.0f} – {expected + band['mae']:,.0f}",
                help=f"Fair freight ± the average test-set error for items priced R$ {band['band']}")
    c[2].metric("Simple tariff", money(float(tariff(row.chargeable_kg[0], row.distance_km[0], row.same_state[0]))))
    if charged > 0:
        verdict(charged, expected, expected_ridge, price, rule)
    st.caption(f"For a listing the model has never seen, the average error is {md_money(NB['p1_unseen_mae'])} "
               f"({NB['p1_unseen_pct']}% of their mean freight; Phase 3 notebook, Section 6).")


# ------------------------------------------------------------------ Analyst: tariff review
DIST_BANDS = ([0, 200, 500, 1000, 2000, np.inf], ["<200 km", "200–500 km", "500–1,000 km", "1,000–2,000 km", "2,000+ km"])
KG_BANDS = ([0, 0.5, 1, 2, 5, 10, np.inf], ["<0.5 kg", "0.5–1 kg", "1–2 kg", "2–5 kg", "5–10 kg", "10+ kg"])


def tariff_page():
    st.title("📐 Tariff review")
    t = P1["tariff"]
    st.markdown(
        f"A published tariff is transparent but cruder than the forest: **{md_money(t['base'])} base + "
        f"{md_money(t['per_kg'])} per chargeable kg + {md_money(t['per_100km'])} per 100 km "
        f"{'−' if t['same_state'] < 0 else '+'} {md_money(abs(t['same_state']))} within a state** (least squares on the "
        f"training items). Where would it over- or under-charge against the fair freight, and where do sellers "
        f"charge above it?")
    how_to("Pick what to compare with the fair freight. Each cell is a group of held-out items by distance and "
           "chargeable weight; red means more than the fair freight, blue less. Cells with fewer than 30 items are "
           "left blank. Use it to see which lanes a simple tariff would misprice before publishing one.")
    d = heldout_predictions()
    comp = st.radio("Compare fair freight with", ["the published tariff", "what sellers charged"], horizontal=True)
    other = (tariff(d.chargeable_kg.fillna(d.chargeable_kg.median()), d.distance_km.fillna(d.distance_km.median()),
                    d.same_state) if comp == "the published tariff" else d.freight_value.values)
    gap = other - d.expected.values
    k = st.columns(3)
    k[0].metric("Tariff error (MAE)", money(t["mae"]), f"forest {money(P1['test']['MAE_rs'])}", delta_color="off")
    k[1].metric(f"Within ± R$ 2 of fair freight ({'tariff' if comp.endswith('tariff') else 'charged'})",
                f"{(np.abs(gap) <= 2).mean():.0%}")
    k[2].metric("Average gap", money(np.nanmean(gap)), help="Positive: above the fair freight on average")
    g = pd.DataFrame({"dist": pd.cut(d.distance_km, DIST_BANDS[0], labels=DIST_BANDS[1], right=False),
                      "kg": pd.cut(d.chargeable_kg, KG_BANDS[0], labels=KG_BANDS[1], right=False), "gap": gap})
    piv = g.groupby(["dist", "kg"], observed=False).gap.agg(["mean", "size"]).reset_index()
    piv.loc[piv["size"] < 30, "mean"] = np.nan
    z = piv.pivot(index="dist", columns="kg", values="mean").reindex(index=DIST_BANDS[1], columns=KG_BANDS[1])
    n = piv.pivot(index="dist", columns="kg", values="size").reindex(index=DIST_BANDS[1], columns=KG_BANDS[1])
    lim = float(np.nanmax(np.abs(z.values))) if np.isfinite(z.values).any() else 1.0
    fig = go.Figure(go.Heatmap(z=z.values, x=KG_BANDS[1], y=DIST_BANDS[1], zmin=-lim, zmax=lim, zmid=0,
                               colorscale=[[0, DIV_LOW], [0.5, DIV_MID], [1, DIV_HIGH]], customdata=n.values,
                               xgap=2, ygap=2, colorbar=dict(title="R$", thickness=12),
                               hovertemplate="%{y}, %{x}: %{z:+.2f} R$ against fair freight "
                                             "(%{customdata:,} items)<extra></extra>"))
    for i, dist in enumerate(DIST_BANDS[1]):
        for j, kg in enumerate(KG_BANDS[1]):
            v = z.iat[i, j]
            if np.isfinite(v):
                fig.add_annotation(x=kg, y=dist, text=f"{v:+.1f}", showarrow=False,
                                   font=dict(color="white" if abs(v) > 0.6 * lim else INK, size=12))
    fig.update_yaxes(autorange="reversed", title="distance seller to customer")
    fig.update_xaxes(title="chargeable weight", side="bottom")
    who = "tariff" if comp.endswith("tariff") else "charged freight"
    st.plotly_chart(layout(fig, h=420, title=f"Average {who} minus fair freight (R$)"), width="stretch")
    s = z.stack(future_stack=True).dropna()
    if len(s):
        hi, lo = s.idxmax(), s.idxmin()
        st.markdown(f"Largest gap above the fair freight: **{hi[0]}, {hi[1]}** ({md_money(s.max())}). "
                    f"Largest below: **{lo[0]}, {lo[1]}** ({md_money(s.min())}).")
    with st.expander("Show as a table"):
        st.dataframe(z.round(2), width="stretch")


# ------------------------------------------------------------------ Operations: daily risk queue
def queue_page():
    st.title("⏰ Daily risk queue")
    st.markdown(
        f"Which of today's orders should the team follow up? The logistic model scores each order at checkout from "
        f"its promise, route, value and freight, and the riskiest share of the day goes on the list. The score is a "
        f"**ranking**, not a probability: late deliveries were rarer in the test window ({P2['late_rate_test']:.1%}) "
        f"than in training ({P2['late_rate_train']:.1%}).")
    how_to(
        f"1. Pick the day the orders were placed (26 May to 29 Aug 2018) and how many orders the team can check.\n"
        f"2. Work down the list. Each order has a **suggested follow-up** from its biggest risk driver: a short "
        f"promise, a risky lane or a large or multi-seller order. Keep them cheap: only about one checked order in "
        f"{DAILY_YIELD:.0f} is really late.\n"
        f"3. Pick an order to see why it is on the list and what promise would have kept it off.\n"
        f"4. Download the list for the team, or switch on **Reveal what happened** to see the real outcomes.")
    d = heldout_orders_scored()
    days = d.day.drop_duplicates().sort_values()
    c = st.columns([1, 1, 1])
    default = days[d.groupby("day").size().reindex(days).values >= 150].iloc[0]
    day = c[0].date_input("Orders placed on", value=default.date(), min_value=days.min().date(),
                          max_value=days.max().date())
    share = c[1].select_slider("Review capacity (share of the day's orders)", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                               value=P2["share"], format_func=lambda v: f"{v:.0%}")
    reveal = c[2].toggle("Reveal what happened", value=False)
    flags = daily_flags(share)
    y = d.clf_target.values
    precision = (flags & y).sum() / max(flags.sum(), 1)
    mask = (d.day == pd.Timestamp(day)).values
    today = d[mask].assign(flag=flags[mask]).sort_values("risk", ascending=False)
    if today.empty:
        st.info("No delivered orders were placed on that day.")
        return
    flagged = today[today.flag == 1]
    k = st.columns(4)
    k[0].metric("Orders placed", f"{len(today):,}")
    k[1].metric("To check", f"{len(flagged):,}")
    k[2].metric("Expected late on the list", f"≈ {len(flagged) * precision:.1f}",
                help=f"In the test window, about 1 in {1 / precision:.0f} orders on a {share:.0%} daily list was late")
    if reveal:
        k[3].metric("Late orders caught", f"{int((today.flag & today.clf_target).sum())} of {int(today.clf_target.sum())}")
    else:
        k[3].metric("Median promise", f"{today.promised_days.median():.0f} days")
    if flagged.empty:
        st.warning(f"With {len(today)} orders, a {share:.0%} capacity rounds to nobody: raise the capacity or pick a "
                   f"busier day.")
        return
    actions = suggested_actions(flagged)
    show = today.assign(rank=np.arange(1, len(today) + 1), order=today.order_id.str[:8],
                        placed=today.order_purchase_timestamp.dt.strftime("%H:%M"),
                        route=today.seller_state + " → " + today.customer_state,
                        action=actions.reindex(today.index).fillna(""))
    cols = ["rank", "action", "order", "placed", "route", "distance_km", "promised_days", "price_total",
            "freight_total", "risk"]
    if reveal:
        show["outcome"] = np.where(show.clf_target == 1, "late by " + show.days_late.astype(int).astype(str) + " d",
                                   "on time")
        show["review"] = show.low_review.map({1.0: "1–2 stars", 0.0: "3–5 stars"}).fillna("none")
        cols = cols[:2] + ["outcome", "review"] + cols[2:]
    only = st.toggle("Show only the orders to check", value=True)
    st.dataframe((show[show.flag == 1] if only else show)[cols], hide_index=True, width="stretch", height=380,
                 column_config={
                     "action": st.column_config.TextColumn("suggested follow-up", width="medium"),
                     "distance_km": st.column_config.NumberColumn("distance (km)", format="%.0f"),
                     "promised_days": st.column_config.NumberColumn("promised (days)", format="%.0f"),
                     "price_total": st.column_config.NumberColumn("order value", format="R$ %.2f"),
                     "freight_total": st.column_config.NumberColumn("freight", format="R$ %.2f"),
                     "risk": st.column_config.ProgressColumn("risk score", format="%.3f", min_value=0.0,
                                                             max_value=float(d.risk.max()))})
    out = show[show.flag == 1][["rank", "action", "order_id", "seller_id", "order_purchase_timestamp", "seller_state",
                                "customer_state", "distance_km", "promised_days", "price_total", "freight_total", "risk"]]
    st.download_button(f"Download the list for {pd.Timestamp(day):%d %b %Y} (CSV)", out.round(3).to_csv(index=False),
                       f"late_risk_queue_{pd.Timestamp(day):%Y-%m-%d}.csv", "text/csv")

    st.subheader("Why is an order on the list?")
    pick = st.selectbox("Order", show.order[show.flag == 1].tolist(), help="The orders to check, riskiest first")
    row = today[today.order_id.str[:8] == pick].head(1)
    a, b = st.columns(2, gap="large")
    with a:
        st.plotly_chart(drivers_chart(logit_drivers(row).iloc[0], f"What drives order {pick}'s risk"), width="stretch")
        st.markdown(f"Suggested follow-up: **{actions.loc[row.index[0]]}**.")
    with b:
        # off the list = less risky than the riskiest order the day's list leaves out
        line = float(today.loc[today.flag == 0, "risk"].max())
        promise_section(row, line, "risk", f"today's review line ({share:.0%} capacity)",
                        "What promise would have kept it off the list?")

    if reveal:
        st.divider()
        caught = (flags & y).sum()
        st.markdown(
            f"**Across the whole test window**, checking the riskiest {share:.0%} of each day's orders catches "
            f"**{caught / y.sum():.1%}** of late orders ({caught} of {y.sum()}); {share:.0%} random checks would "
            f"catch about {share:.0%}. Ranking the whole window at once, as the Phase 2 evaluation did, catches "
            f"{P2['test']['Recall']:.1%} at 10%: late orders cluster on bad days, which a daily list cannot see.")
        curve = pd.DataFrame({"share": np.arange(0.05, 0.51, 0.05)})
        curve["caught"] = [(daily_flags(round(s, 2)) & y).sum() / y.sum() for s in curve.share]
        fig = go.Figure()
        fig.add_scatter(x=curve.share, y=curve.caught, mode="lines+markers", name="model's daily list",
                        line=dict(color=BLUE, width=2), marker=dict(size=8),
                        hovertemplate="check %{x:.0%} → catch %{y:.1%}<extra></extra>")
        fig.add_scatter(x=[0, 0.5], y=[0, 0.5], mode="lines", name="random checks", line=dict(color=GREY, width=2),
                        hoverinfo="skip")
        sel = curve.loc[(curve.share - share).abs().idxmin()]
        fig.add_annotation(x=sel.share, y=sel.caught, text=f"{sel.caught:.0%} at {sel.share:.0%}", showarrow=True,
                           arrowcolor=INK_2, ax=-50, ay=-30, font=dict(color=INK))
        fig.update_xaxes(tickformat=".0%", title="share of each day's orders checked", range=[0, 0.52])
        fig.update_yaxes(tickformat=".0%", title="share of late orders caught", range=[0, 0.6])
        st.plotly_chart(layout(fig, h=380, title="Checking more orders catches more late ones"), width="stretch")
        with st.expander("Show as a table"):
            st.dataframe(curve.assign(share=curve.share.map("{:.0%}".format), caught=curve.caught.map("{:.1%}".format)),
                         hide_index=True)


# ------------------------------------------------------------------ Operations: promise check
def promise_page():
    st.title("📅 Promise check")
    st.markdown("How risky is a new order, and what delivery promise would keep it off the review list? The promise "
                "is the strongest signal the Phase 2 analysis found: longer promises make lateness much less likely.")
    how_to("Describe the order as it looks at checkout and press **Score**. The result places its risk among the "
           "orders of the test window, shows which inputs push it up or down, and finds the shortest promise that "
           "keeps it below the review line.")
    with st.form("order"):
        a, b = st.columns(2, gap="large")
        with a:
            promised = st.number_input("Promised delivery time (days from purchase)", min_value=1.0, value=20.0)
            value = st.number_input("Order value, all items (R$)", min_value=0.0, value=150.0, step=10.0)
            freight = st.number_input("Freight charged, all items (R$)", min_value=0.0, value=25.0, step=1.0)
            e1, e2 = st.columns(2)
            n_items = e1.number_input("Items", min_value=1, value=1)
            n_sellers = e2.number_input("Sellers", min_value=1, value=1)
        with b:
            s_zip, s_state = place_inputs("os", "Seller")
            c_zip, c_state = place_inputs("oc", "Customer")
            share = st.select_slider("Review capacity", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30], value=P2["share"],
                                     format_func=lambda v: f"{v:.0%}", help="The share of orders the team can check")
        go_ = st.form_submit_button("Score", type="primary")
    if not go_:
        return
    errors = ["order value must be above zero"] if value <= 0 else []
    s_loc, c_loc, loc_errors = locate_pair(s_zip, s_state, c_zip, c_state)
    if errors + loc_errors:
        st.error("Please fix: " + "; ".join(errors + loc_errors) + ".")
        return
    dist = float(F.haversine_km(s_loc[0], s_loc[1], c_loc[0], c_loc[1]))
    row = pd.DataFrame([dict(promised_days=promised, distance_km=dist, same_state=int(s_loc[2] == c_loc[2]),
                             price_total=value, freight_total=freight, freight_ratio=freight / value,
                             n_items=n_items, n_sellers=n_sellers, seller_state=s_loc[2], customer_state=c_loc[2],
                             order_id="new", purchase_ts=0.0)])
    range_note({"promised_days": promised, "price_total": value, "freight_total": freight, "distance_km": dist},
               P2["ranges"])
    risk = float(LOGIT.predict_proba(row[COLS_B])[0, 1])
    pct = float(percentile(risk))
    st.markdown(f"**{s_loc[2]} → {c_loc[2]}**, {dist:,.0f} km, promised {promised:.0f} days")
    c = st.columns(3)
    c[0].metric("Risk score", f"{risk:.3f}")
    c[1].metric("Risk percentile", f"{pct:.0f}th", help="Among the orders of the test window")
    c[2].metric("Review line", f"{100 * (1 - share):.0f}th", help=f"The riskiest {share:.0%} of orders are checked")
    if pct >= 100 * (1 - share):
        st.error(f"🔎 **On the review list:** it would be in the riskiest {share:.0%} of orders.")
    else:
        st.success(f"✅ **Below the review line** for a {share:.0%} capacity.")
    a, b = st.columns(2, gap="large")
    with a:
        promise_section(row, 100 * (1 - share), "pct", f"review line (riskiest {share:.0%})", "Risk for every promise")
    with b:
        st.plotly_chart(drivers_chart(logit_drivers(row).iloc[0], "What drives this order's risk"), width="stretch")


# ------------------------------------------------------------------ Operations: monitoring
def monitor_page():
    st.title("📈 Monitoring")
    st.markdown(
        "When does the model need a review? The Phase 2 analysis found that the ranking weakens when Olist changes "
        "its promise rules, so promise changes should be treated as events. This page follows the test window week "
        "by week: the promise, the late rate and how many late orders the daily list caught.")
    how_to("Read the three panels together. A sharp move in the median promise is the warning sign; the weeks it "
           "flags (⚠) are when the team should re-check the model on recent orders and retrain if the catch rate "
           "falls. Monthly model skill from the Phase 2 notebook is in the table below.")
    d = heldout_orders_scored()
    flags = daily_flags(P2["share"])
    wk = (d.assign(caught=flags & d.clf_target.values, week=d.order_purchase_timestamp.dt.to_period("W").dt.start_time)
          .groupby("week").agg(orders=("clf_target", "size"), late=("clf_target", "sum"),
                               promise=("promised_days", "median"), caught=("caught", "sum")))
    wk = wk[wk.orders >= 300]                                    # drop the part weeks at either end of the window
    wk["late_rate"] = wk.late / wk.orders
    wk["catch_rate"] = np.where(wk.late >= 5, wk.caught / wk.late.where(wk.late > 0), np.nan)
    trailing = wk.promise.shift(1).rolling(4, min_periods=2).median()
    wk["alert"] = (wk.promise / trailing - 1).abs() > 0.2
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=["Median promised delivery time (days)", "Late rate",
                                        f"Late orders caught by the daily {P2['share']:.0%} list"])
    fig.add_scatter(x=wk.index, y=wk.promise, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8),
                    name="promise", hovertemplate="week of %{x|%d %b}: %{y:.0f} days<extra></extra>", row=1, col=1)
    al = wk[wk.alert]
    fig.add_scatter(x=al.index, y=al.promise, mode="markers+text", text=["⚠"] * len(al), textposition="top center",
                    marker=dict(color=ORANGE, size=12, line=dict(color="white", width=2)), name="promise moved > 20%",
                    hovertemplate="week of %{x|%d %b}: promise moved more than 20%<extra></extra>", row=1, col=1)
    fig.add_scatter(x=wk.index, y=wk.late_rate, mode="lines+markers", line=dict(color=BLUE, width=2),
                    marker=dict(size=8), name="late rate", customdata=wk.late,
                    hovertemplate="week of %{x|%d %b}: %{y:.1%} late (%{customdata} orders)<extra></extra>",
                    row=2, col=1, showlegend=False)
    fig.add_scatter(x=wk.index, y=wk.catch_rate, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8),
                    name="caught", customdata=np.c_[wk.caught, wk.late], showlegend=False,
                    hovertemplate="week of %{x|%d %b}: %{customdata[0]} of %{customdata[1]} late orders<extra></extra>",
                    row=3, col=1)
    fig.add_hline(y=P2["share"], line=dict(color=GREY, width=1, dash="dash"), row=3, col=1,
                  annotation_text="random checks", annotation_position="top left")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=3, col=1)
    layout(fig, h=640).update_layout(showlegend=False, margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")
    st.caption("⚠ marks a week whose median promise moved more than 20% from the previous four weeks. Weeks with "
               "fewer than 300 orders (the part weeks at either end) are left out; the catch rate is shown only for "
               "weeks with at least 5 late orders.")
    if len(al):
        st.markdown(f"**⚠ Weeks to review:** {', '.join(f'{w.day} {w:%b}' for w in al.index)}. The median promise fell from "
                    f"{wk.promise.iloc[0]:.0f} to {wk.promise.iloc[-1]:.0f} days across the window while the late rate "
                    f"rose to {wk.late_rate.max():.1%}; the ranking learned on longer promises loses its edge.")
    st.subheader("Model skill by month (Phase 2 notebook, Section 10)")
    bm = pd.DataFrame(P2["by_month"])
    st.dataframe(bm.rename(columns={"month": "Month", "orders": "Orders", "late_rate": "Late rate", "lift": "Lift",
                                    "roc_auc": "ROC-AUC"}), hide_index=True, width="stretch",
                 column_config={"Late rate": st.column_config.NumberColumn(format="percent")})
    st.markdown("**Action:** re-train monthly with forward validation; after any change to the promise rules, check "
                "lift and the catch rate on the newest orders before relying on the ranking.")


# ------------------------------------------------------------------ Model card
def model_card():
    st.title("📋 Model card")
    st.markdown("What each model was trained on, how it was validated and where it should not be trusted. Test "
                "results are recomputed by `build_models.py` from the files this app loads; cross-validation and robustness "
                "results are quoted from the notebooks (Problem 1: Phase 3; Problem 2: Phase 2).")
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Problem 1 · Fair freight")
        fl = FLAG
        st.markdown(
            f"- **User and use:** the logistics-pricing / finance analyst; benchmark freight, audit unusual charges, "
            f"review tariffs.\n"
            f"- **Model:** random forest regressor on log(1 + freight), {P1['forest_params']['n_estimators']} trees, "
            f"leaf size {P1['forest_params']['min_samples_leaf']}, depth {P1['forest_params']['max_depth']}, "
            f"{P1['forest_params']['max_features']:.0%} of features per split; feature groups, settings and model "
            f"chosen by forward-chaining cross-validation. Second opinion: Ridge (alpha {P1['ridge_alpha']}), test MAE "
            f"{md_money(P1['ridge_test']['MAE_rs'])}.\n"
            f"- **Inputs:** weight, volume, chargeable weight, distance, same-state route, seller and customer "
            f"state, category, price and basket size"
            f"{', listing details and purchase timing' if USES_LISTING else ''}. Seller identity is excluded on "
            f"purpose, so a seller's habitual overcharging cannot become the benchmark"
            f"{'' if USES_LISTING else '; listing details and purchase timing did not help on later periods'}.\n"
            f"- **Data:** trained on the {P1['n_train']:,} items bought from {P1['train_dates'][0]} to "
            f"{P1['train_dates'][1]}; tested on the {P1['n_test']:,} items bought from {P1['test_dates'][0]} to "
            f"{P1['test_dates'][1]} (the latest 20%).\n"
            f"- **Test results:** MAE {md_money(P1['test']['MAE_rs'])} ({P1['test']['MAE_pct']}% of mean freight "
            f"{md_money(P1['mean_freight_test'])}), RMSE {md_money(P1['test']['RMSE_rs'])}, R² {P1['test']['R2_rs']:.3f}. "
            f"Cross-validated MAE {md_money(NB['p1_cv_mae'])} ± {NB['p1_cv_mae_sd']:.2f} (the spread is mostly the "
            f"freight level of each period). Alternatives on the same test items: stacking "
            f"with Ridge {md_money(NB['p1_stack_test_mae'])} (no better fold by fold), Ridge {md_money(NB['p1_ridge_test_mae'])}, single tree "
            f"{md_money(NB['p1_tree_test_mae'])}, transparent tariff {md_money(P1['tariff']['mae'])}.\n"
            f"- **Unusual-freight rule:** residual in the top {1 - fl['q']:.0%} of the test set, overall "
            f"({fl['n_unusual']['overall']:,} items; Ridge agrees on {fl['both_share']['overall']:.0%}) or within each "
            f"price band ({fl['n_unusual']['band']:,} items; {fl['both_share']['band']:.0%}).")
        bands = pd.DataFrame(P1["price_bands"]).assign(mae_pct=lambda t: t.mae_pct / 100)
        st.markdown("**Error grows with item price**, so a single tolerance is too tight for expensive items:")
        st.dataframe(bands, hide_index=True, width="stretch", column_config={
            "band": st.column_config.TextColumn("Price (R$)"), "n": st.column_config.NumberColumn("Items"),
            "mae": st.column_config.NumberColumn("MAE", format="R$ %.2f"),
            "mae_pct": st.column_config.NumberColumn("MAE %", format="percent")})
        st.markdown(
            f"**Limits.** Freight per kg rose over time, so the error on later items ({P1['test']['MAE_pct']}% of mean "
            f"freight) misses the 20% target; on a random split, which mixes periods, it would be "
            f"{md_money(NB['p1_random_mae'])} ({NB['p1_random_pct']}%). Retrain monthly on recent items. Listings never "
            f"seen in training: MAE {md_money(NB['p1_unseen_mae'])} ({NB['p1_unseen_pct']}%). A flag is a reason to "
            f"look, not proof of overcharging.")
    with b:
        st.subheader("Problem 2 · Late delivery")
        month_lift = ", ".join(f"{m['lift']:.1f}" for m in P2["by_month"][1:])
        st.markdown(
            f"- **User and use:** the operations and customer-experience team; a daily follow-up list, promises at "
            f"checkout, monitoring.\n"
            f"- **Model:** plain logistic regression (no penalty), the simplest model within one standard deviation "
            f"of the best in forward-chaining cross-validation, compared fold by fold (it trails the L2 model by "
            f"0.02 ± 0.09 of lift).\n"
            f"- **Inputs:** promised delivery time, distance, same-state route, seller and customer state, order "
            f"value, freight, freight share, number of items and sellers. Everything is known at checkout; carrier "
            f"and delivery dates are never used.\n"
            f"- **Data:** trained on {P2['n_train']:,} delivered orders from {P2['train_dates'][0]} to "
            f"{P2['train_dates'][1]}; tested on the {P2['n_test']:,} orders of {P2['test_dates'][0]} to "
            f"{P2['test_dates'][1]}, after a 30-day gap so no training label comes from the future.\n"
            f"- **Test results:** lift {P2['test']['Lift']:.2f} (PR-AUC {P2['test']['PR_AUC']:.3f} against a late "
            f"rate of {P2['late_rate_test']:.1%}), ROC-AUC {P2['test']['ROC_AUC']:.3f}. Checking the riskiest "
            f"{P2['share']:.0%} catches {P2['late_caught']} of {P2['late_test']} late orders "
            f"({P2['test']['Recall']:.1%}; precision {P2['test']['Precision']:.3f}). As a daily list, "
            f"{P2['daily']['recall']:.1%} ({P2['daily']['late_caught']} of {P2['late_test']}). Cross-validated lift "
            f"{NB['p2_cv_lift']:.2f} ± {NB['p2_cv_lift_sd']:.2f}.\n"
            f"- **Over time:** lift {', '.join(f'{v:.2f}' for v in NB['p2_window_lift'])} in four consecutive test "
            f"windows (Sep 2017 to Aug 2018); within the last, {month_lift} in June, July and August. Flagged orders "
            f"received 1–2 star reviews "
            f"{NB['p2_low_review_ratio'][0]}–{NB['p2_low_review_ratio'][1]}× as often as unflagged ones.")
        gain = pd.DataFrame(P2["gain"])
        st.markdown("**Late orders by risk decile** (the whole test window ranked at once):")
        st.dataframe(gain, hide_index=True, width="stretch", column_config={
            "decile": st.column_config.NumberColumn("Decile"), "orders": st.column_config.NumberColumn("Orders"),
            "late": st.column_config.NumberColumn("Late"),
            "cum_capture": st.column_config.NumberColumn("Caught so far", format="percent"),
            "lift": st.column_config.NumberColumn("Lift", format="%.2f")})
        st.markdown(
            "**Limits.** The model ranks, it does not forecast: its scores overstate the late rate of the test window. "
            "Olist cut its promised times sharply from late July 2018, and the short promises to São Paulo customers "
            "account for most missed late orders. The ranking is weakest when promises change: see *Monitoring*.")
    st.divider()
    b_ = FACTS["built"]
    st.caption(f"Models built {b_['date']} with Python {b_['python']}, pandas {b_['pandas']}, numpy {b_['numpy']}, "
               f"scikit-learn {b_['scikit_learn']}, seed {b_['seed']}. Rebuild: "
               f"`OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/model_app/build_models.py`.")


# ------------------------------------------------------------------ navigation, grouped by stakeholder
PAGE_HOME = st.Page(home, title="Home", icon="🏠", default=True)
PAGE_AUDIT = st.Page(audit_page, title="Freight audit", icon="🔍", url_path="audit")
PAGE_CHECK = st.Page(check_page, title="Check a charge", icon="🧾", url_path="check")
PAGE_TARIFF = st.Page(tariff_page, title="Tariff review", icon="📐", url_path="tariff")
PAGE_QUEUE = st.Page(queue_page, title="Daily risk queue", icon="⏰", url_path="queue")
PAGE_PROMISE = st.Page(promise_page, title="Promise check", icon="📅", url_path="promise")
PAGE_MONITOR = st.Page(monitor_page, title="Monitoring", icon="📈", url_path="monitor")
PAGE_CARD = st.Page(model_card, title="Model card", icon="📋", url_path="model-card")
nav = st.navigation({"": [PAGE_HOME],
                     "Pricing and finance analyst": [PAGE_AUDIT, PAGE_CHECK, PAGE_TARIFF],
                     "Operations and customer experience": [PAGE_QUEUE, PAGE_PROMISE, PAGE_MONITOR],
                     "About": [PAGE_CARD]})
st.sidebar.caption("Team 5 · IT5006 · AY2026/27 · Phase 3")
st.sidebar.caption("Scores held-out Olist records (2016–2018) as if they were new orders.")
nav.run()
