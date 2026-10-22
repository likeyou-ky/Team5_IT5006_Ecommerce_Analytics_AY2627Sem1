# -*- coding: utf-8 -*-
"""
Olist logistics assistant: the Phase 3 model app (Team 5, IT5006).

Deploys the two Phase 2 final models (notebooks/Team5_Phase2_IT5006_AY2627Sem1.ipynb):
  Problem 1, fair freight   : RandomForest regressor  -> logistics-pricing / finance analyst
  Problem 2, late delivery  : plain LogisticRegression -> operations and customer-experience team

The model files and data are written by build_models.py, which checks them against the notebook; every headline
number on these pages is read from data/facts.json. The Olist data end in 2018, so the app scores held-out
historical records as if they were new orders: the freight items come from the Problem 1 test set and the orders
from the Problem 2 test window (26 May - 29 Aug 2018), none of which the models saw in training.

Run locally:   streamlit run deployment/model_app/app.py
"""
import json, re, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import features as F, models as M            # the same feature and decision-rule code the notebook uses

st.set_page_config(page_title="Olist logistics assistant", page_icon="📦", layout="wide")

REPO_URL = "https://github.com/likeyou-ky/Team5_IT5006_Ecommerce_Analytics_AY2627Sem1"
EDA_URL = "https://team5-olist-eda-phase1.streamlit.app/"
MAX_UPLOAD_ROWS = 50_000

# Chart colours: the CVD-validated slots of src/olist_theme.py (blue and orange pass every check on this surface)
BLUE, ORANGE, GREY, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#c3c2b7", "#10100f", "#52514e", "#e1e0d9"
LABELS = {"promised_days": "Promised delivery time", "distance_km": "Distance seller to customer",
          "same_state": "Same-state shipment", "price_total": "Order value", "freight_total": "Freight charged",
          "freight_ratio": "Freight / order value", "n_items": "Items in the order", "n_sellers": "Sellers in the order",
          "seller_state": "Seller state", "customer_state": "Customer state"}


# ------------------------------------------------------------------ loading (cached, with a clear failure message)
@st.cache_resource(show_spinner="Loading the models...")
def load_models():
    return joblib.load(HERE / "models" / "freight_forest.joblib"), joblib.load(HERE / "models" / "late_logit.joblib")


@st.cache_data(show_spinner=False)
def load_data():
    facts = json.loads((HERE / "data" / "facts.json").read_text(encoding="utf-8"))
    items = pd.read_csv(HERE / "data" / "heldout_items.csv", parse_dates=["order_purchase_timestamp"])
    orders = pd.read_csv(HERE / "data" / "heldout_orders.csv", parse_dates=["order_purchase_timestamp"])
    zips = pd.read_csv(HERE / "data" / "zip_centroids.csv").set_index("zip_prefix")
    return facts, items, orders, zips


try:
    FOREST, LOGIT = load_models()
    FACTS, ITEMS, ORDERS, ZIPS = load_data()
except Exception as exc:                                                    # missing or unreadable files
    st.error("The app could not load its model or data files. Rebuild them with "
             "`python deployment/model_app/build_models.py` and make sure the library versions match "
             "`deployment/model_app/requirements.txt`.")
    st.exception(exc)
    st.stop()

P1, P2, NB = FACTS["p1"], FACTS["p2"], FACTS["notebook"]
COLS_A, COLS_B = list(FOREST.feature_names_in_), list(LOGIT.feature_names_in_)
STATES = P1["states"]


# ------------------------------------------------------------------ helpers
def money(x, dp=2):
    """R$ value for st.metric and tables (plain text)."""
    return f"R$ {x:,.{dp}f}"


def md_money(x, dp=2):
    """R$ value inside markdown, where a bare $ pair would render as LaTeX."""
    return f"R\\$ {x:,.{dp}f}"


def layout(fig, h=360, title=None, **kw):
    """Shared chart style: title on its own line, legend between title and plot, hairline grid."""
    fig.update_layout(height=h, margin=dict(t=80 if title else 40, b=10, l=10, r=10), template="plotly_white",
                      title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


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


def price_band(price):
    for b, hi in zip(P1["price_bands"], [25, 50, 100, 200, 500, np.inf]):
        if price < hi:
            return b
    return P1["price_bands"][-1]


def is_unusual(charged, expected):
    """Phase 2 audit rule: residual on the log1p scale in the top 5% of the test set (tolerance: see build_models.py)."""
    return np.log1p(charged) - np.log1p(expected) >= P1["flag_log_resid"] - P1["flag_tol"]


def tariff(chargeable_kg, distance_km, same_state):
    t = P1["tariff"]
    return max(0.0, t["base"] + t["per_kg"] * chargeable_kg + t["per_100km"] * distance_km / 100
               + t["same_state"] * same_state)


def predict_freight(df):
    """Expected (fair) freight in R$ for rows holding the model's input columns."""
    return np.clip(np.expm1(FOREST.predict(df[COLS_A])), 0, None)


def range_note(values, ranges):
    """Warn when an input lies outside the middle 99.8% of the training data."""
    out = [f"{name} {v:,.2f} is outside the range seen in training ({lo:,.2f} to {hi:,.2f})"
           for name, (v, (lo, hi)) in ((n, (values[n], ranges[n])) for n in values) if not lo <= v <= hi]
    if out:
        st.warning("The model has seen few cases like this, so treat the result with caution: " + "; ".join(out) + ".")


@st.cache_data(show_spinner="Scoring the held-out items...")
def heldout_items_scored():
    d = ITEMS.copy()
    d["expected"] = predict_freight(d)
    d["unusual"] = is_unusual(d.freight_value, d.expected)
    d["excess"] = d.freight_value - d.expected
    return d


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


def logit_drivers(rows):
    """Each input's push on the log-odds of lateness, relative to the average order of the test window."""
    prep, lr = LOGIT.named_steps["prep"], LOGIT.named_steps["m"]
    names = prep.get_feature_names_out()
    ref = _logit_reference()
    contrib = (prep.transform(rows[COLS_B]) - ref) * lr.coef_.ravel()
    raw = []
    for n in names:
        n = n.removeprefix("missingindicator_")
        raw.append(next((c for c in ("seller_state", "customer_state") if n.startswith(c + "_")), n))
    return pd.DataFrame(contrib, columns=names).T.groupby(raw).sum().T


@st.cache_data(show_spinner=False)
def _logit_reference():
    return LOGIT.named_steps["prep"].transform(ORDERS[COLS_B]).mean(axis=0)


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
    fig.update_xaxes(title="push on the log-odds of a late delivery, against the average order")
    return layout(fig, h=60 + 34 * len(s), title=title, bargap=0.35)


# ------------------------------------------------------------------ pages
def home():
    st.title("Olist logistics assistant")
    st.markdown(
        "Two decision tools built on the Olist Brazilian e-commerce data (2016–2018). Each deploys the final model of "
        "one of our Phase 2 problems. The data end in 2018, so the tools score **held-out historical records as if "
        "they were new**: freight items and orders the models never saw in training.")
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("📦 Fair-freight checker")
        st.markdown(
            f"**For the logistics-pricing / finance analyst.** A random forest predicts the freight an item *should* "
            f"cost from its weight, size, route and category. Compare it with what was charged, price a new listing, "
            f"or audit a batch of items for unusual freight.\n\n"
            f"Test error: **{md_money(P1['test']['MAE_rs'])}** on average ({P1['test']['MAE_pct']}% of mean freight), "
            f"R² {P1['test']['R2_rs']:.2f}.")
        st.page_link(PAGE_FREIGHT, label="Open the fair-freight checker", icon="📦")
    with b:
        st.subheader("⏰ Late-delivery risk queue")
        st.markdown(
            f"**For the operations and customer-experience team.** A logistic model ranks each day's new orders by "
            f"their risk of arriving after the promised date, so the team can check the riskiest "
            f"{P2['share']:.0%} first.\n\n"
            f"In the test window, ranking every order catches **{P2['test']['Recall']:.1%}** of late orders by "
            f"checking {P2['share']:.0%} of them, against {P2['share']:.0%} for random checks "
            f"(lift {P2['test']['Lift']:.2f}).")
        st.page_link(PAGE_LATE, label="Open the risk queue", icon="⏰")

    st.divider()
    st.subheader("How to use the app")
    st.markdown(
        "1. Pick a tool in the sidebar.\n"
        "2. **Fair-freight checker.** *Check a held-out item* draws a real item from the test set and shows its fair "
        "freight. *Price a new listing* takes the product, route and price you enter. *Audit a batch* scores all "
        "held-out items, or a CSV you upload, and lists the unusual ones for review.\n"
        "3. **Late-delivery risk queue.** Choose a purchase day and a review capacity to get that day's review list. "
        "Switch on *Reveal what happened* to see which orders were really late. *Score a new order* rates one "
        "order you describe.\n"
        "4. **Model card** gives the training data, validation, test results and limits of both models.")
    st.caption(f"Team 5 · IT5006 · AY2026/27. Code: [GitHub]({REPO_URL}). "
               f"Exploratory analysis (Phase 1): [EDA dashboard]({EDA_URL}).")


def freight_page():
    st.title("📦 Fair-freight checker")
    st.markdown(
        f"A random forest trained on {P1['n_train']:,} order items predicts the freight an item should be charged. "
        f"An item is **unusual** when its charged freight sits in the top 5% of what the model leaves unexplained, "
        f"which means roughly {P1['flag_ratio']:.2f}× the fair freight (on a log scale with R\\$1 added, so "
        f"small charges need a larger ratio).")
    t1, t2, t3 = st.tabs(["Check a held-out item", "Price a new listing", "Audit a batch"])
    with t1:
        check_item()
    with t2:
        price_listing()
    with t3:
        audit_batch()


def verdict_box(charged, expected, price):
    band = price_band(price)
    c = st.columns(4)
    c[0].metric("Charged", money(charged))
    c[1].metric("Fair freight (model)", money(expected))
    c[2].metric("Difference", money(charged - expected), f"{(charged / expected - 1) * 100:+.0f}%" if expected > 0 else None,
                delta_color="inverse")
    c[3].metric(f"Typical error, items R$ {band['band']}", f"± {money(band['mae'])}")
    if is_unusual(charged, expected):
        st.error(f"🚩 **Unusual freight.** {md_money(charged)} is in the top 5% above the model's fair freight of "
                 f"{md_money(expected)}: send it for review.")
    else:
        st.success(f"✅ **Within the normal range** for an item like this (fair freight {md_money(expected)}).")


def check_item():
    d = heldout_items_scored()
    with st.expander("How to use", expanded=False):
        st.markdown("Draw a random held-out item, or paste an order ID from the audit list. The model never saw "
                    "these orders in training. The table under the verdict shows the inputs it used.")
    c = st.columns([1, 1, 2])
    only = c[0].radio("Draw from", ["all held-out items", "unusual items only"], horizontal=False)
    pool = d[d.unusual] if only == "unusual items only" else d
    clicked = c[1].button("🎲 Random item", width="stretch")
    oid = c[2].text_input("…or enter an order ID", placeholder="e.g. " + d.order_id.iloc[0])
    if clicked or "item_id" not in st.session_state:
        st.session_state.item_id = pool.sample(1, random_state=np.random.randint(1 << 30)).index[0]
    elif oid.strip():
        hit = d.index[d.order_id == oid.strip().lower()]
        if len(hit):
            st.session_state.item_id = hit[0]
        else:
            st.warning("No held-out item has that order ID. The checker only holds test-set items; use *Price a new "
                       "listing* for anything else.")
    it = d.loc[st.session_state.item_id]
    st.markdown(f"**Order `{it.order_id}`**, item {it.order_item_id} · {it.category} · purchased "
                f"{it.order_purchase_timestamp:%d %b %Y}")
    verdict_box(it.freight_value, it.expected, it.price)
    st.caption(f"A simple published tariff (base + per kg + per 100 km + same-state) would charge "
               f"{md_money(tariff(it.chargeable_kg, it.distance_km, it.same_state))}; it errs by "
               f"{md_money(P1['tariff']['mae'])} on average, against {md_money(P1['test']['MAE_rs'])} for the forest.")
    info = pd.DataFrame({"Input": ["Price", "Weight", "Volume", "Chargeable weight", "Route", "Distance",
                                   "Items / sellers in the order", "Photos, name and description length"],
                         "Value": [money(it.price), f"{it.weight_kg:,.2f} kg", f"{it.volume_cm3:,.0f} cm³",
                                   f"{it.chargeable_kg:,.2f} kg", f"{it.seller_state} → {it.customer_state}",
                                   f"{it.distance_km:,.0f} km", f"{it.n_items} / {it.n_sellers}",
                                   f"{it.photos_qty:.0f}, {it.name_len:.0f}, {it.desc_len:.0f} characters"]})
    st.dataframe(info, hide_index=True, width="stretch")


def place_inputs(prefix, label):
    c = st.columns(2)
    state = c[0].selectbox(f"{label} state", STATES, index=STATES.index("SP"), key=f"{prefix}_state")
    z = c[1].text_input(f"{label} zip code (optional)", key=f"{prefix}_zip", placeholder="e.g. 01310")
    return z, state


def price_listing():
    with st.expander("How to use", expanded=False):
        st.markdown("Describe the item and the route, then press **Estimate**. Zip codes place seller and customer "
                    "exactly; without them the app uses the centre of the state. Enter the charged freight to check "
                    "it against the fair freight.")
    cats = P1["categories"]
    with st.form("listing"):
        a, b = st.columns(2, gap="large")
        with a:
            cat = st.selectbox("Product category", cats, index=0)
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
            w1, w2 = st.columns(2)
            day = w1.date_input("Purchase date", value=pd.Timestamp("2018-08-01"))
            hour = w2.slider("Purchase hour", 0, 23, 14)
        with st.expander("Listing details (training medians by default)"):
            l1, l2, l3 = st.columns(3)
            photos = l1.number_input("Photos", min_value=0, value=int(P1["defaults"]["photos_qty"]))
            name_len = l2.number_input("Name length (characters)", min_value=0, value=int(P1["defaults"]["name_len"]))
            desc_len = l3.number_input("Description length (characters)", min_value=0,
                                       value=int(P1["defaults"]["desc_len"]))
        charged = st.number_input("Freight charged (R$, optional: leave 0 to skip the check)", min_value=0.0, value=0.0)
        go_ = st.form_submit_button("Estimate", type="primary")
    if not go_:
        return
    errors = [m for m, bad in [("price must be above zero", price <= 0), ("weight must be above zero", weight_g <= 0),
                               ("every dimension must be above zero", min(length, width, height) <= 0)] if bad]
    s_loc, s_note = locate(s_zip, s_state, "Seller")
    c_loc, c_note = locate(c_zip, c_state, "Customer")
    errors += [n for loc, n in [(s_loc, s_note), (c_loc, c_note)] if loc is None]
    if errors:
        st.error("Please fix: " + "; ".join(errors) + ".")
        return
    for n in (s_note, c_note):
        if n:
            st.info(n)
    when = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    wkg, vol = weight_g / 1000, length * width * height
    row = pd.DataFrame([dict(weight_kg=wkg, volume_cm3=vol, chargeable_kg=max(wkg, vol / 5000),
                             distance_km=float(F.haversine_km(s_loc[0], s_loc[1], c_loc[0], c_loc[1])),
                             same_state=int(s_loc[2] == c_loc[2]), price=price, n_items=n_items, n_sellers=n_sellers,
                             photos_qty=photos, name_len=name_len, desc_len=desc_len, purchase_month=when.month,
                             purchase_dow=when.dayofweek, purchase_hour=when.hour, seller_state=s_loc[2],
                             customer_state=c_loc[2], category=cat, order_id="new", purchase_ts=when.timestamp())])
    range_note({"weight_kg": wkg, "volume_cm3": vol, "price": price, "distance_km": row.distance_km[0]}, P1["ranges"])
    expected = float(predict_freight(row)[0])
    st.markdown(f"**{cat}**, {wkg:,.2f} kg, {vol:,.0f} cm³, {s_loc[2]} → {c_loc[2]} ({row.distance_km[0]:,.0f} km)")
    if charged > 0:
        verdict_box(charged, expected, price)
    else:
        band = price_band(price)
        c = st.columns(3)
        c[0].metric("Fair freight (model)", money(expected))
        c[1].metric(f"Typical error, items R$ {band['band']}", f"± {money(band['mae'])}")
        c[2].metric("Simple tariff", money(tariff(row.chargeable_kg[0], row.distance_km[0], row.same_state[0])))
    st.caption(f"For a listing the model has never seen, the average error is {md_money(NB['p1_unseen_mae'])} "
               f"({NB['p1_unseen_pct']}% of mean freight; Phase 2 notebook, Section 11).")


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
    for c in ("seller_state", "customer_state", "category"):
        u[c] = u[c].astype(str)
    return u.reset_index(drop=True), msgs


def audit_batch():
    with st.expander("How to use", expanded=False):
        st.markdown("Score every held-out item, or upload your own CSV of items with the charged freight. The app "
                    "flags the unusual ones, shows which sellers they come from and lets you download the review "
                    "list. The template shows the columns; `distance_km` can be replaced by `seller_zip_prefix` and "
                    "`customer_zip_prefix`, and `chargeable_kg` and `same_state` are derived when missing.")
    tmpl_cols = ["order_id", "seller_id", "category", "price", "freight_value", "weight_kg", "volume_cm3",
                 "distance_km", "seller_state", "customer_state", "n_items", "n_sellers", "photos_qty", "name_len",
                 "desc_len", "purchase_month", "purchase_dow", "purchase_hour"]
    c = st.columns([2, 1])
    src = c[0].radio("Items to audit", ["all held-out items", "upload a CSV"], horizontal=True)
    c[1].download_button("Download the CSV template", ITEMS[tmpl_cols].head(5).to_csv(index=False),
                         "freight_audit_template.csv", "text/csv", width="stretch")
    if src == "upload a CSV":
        file = st.file_uploader("CSV of order items", type="csv")
        if file is None:
            return
        try:
            d, msgs = read_upload(file)
        except ValueError as e:
            st.error(f"The upload could not be scored: {e}")
            return
        for m in msgs:
            st.warning(m)
        d["expected"] = predict_freight(d)
        d["unusual"] = is_unusual(d.freight_value, d.expected)
        d["excess"] = d.freight_value - d.expected
    else:
        d = heldout_items_scored()
    f = d[d.unusual]
    k = st.columns(3)
    k[0].metric("Items scored", f"{len(d):,}")
    k[1].metric("Unusual items", f"{len(f):,}", f"{len(f) / len(d):.1%} of items", delta_color="off")
    k[2].metric("Excess on unusual items", f"{f.excess.sum() / d.freight_value.sum():.1%}" if len(f) else "–",
                help="What the unusual items were charged above their fair freight, as a share of all freight scored")
    if f.empty:
        st.success("No unusual freight in these items.")
        return
    st.markdown(f"Unusual items were charged **{md_money(f.freight_value.mean())}** on average, against a fair "
                f"freight of **{md_money(f.expected.mean())}**.")
    ok = d[~d.unusual]
    ok = ok.sample(min(len(ok), 6000), random_state=0)
    fig = go.Figure()
    fig.add_scattergl(x=ok.expected, y=ok.freight_value, mode="markers", name="within the normal range",
                      marker=dict(color=GREY, size=5, opacity=0.5),
                      hovertemplate="fair R$ %{x:.2f}<br>charged R$ %{y:.2f}<extra></extra>")
    fig.add_scattergl(x=f.expected, y=f.freight_value, mode="markers", name="unusual", text=f.order_id,
                      marker=dict(color=ORANGE, size=7, line=dict(color="white", width=1)),
                      hovertemplate="order %{text}<br>fair R$ %{x:.2f}<br>charged R$ %{y:.2f}<extra></extra>")
    x = np.geomspace(max(d.expected.min(), 1), d.expected.max(), 50)
    fig.add_scatter(x=x, y=x, mode="lines", line=dict(color=INK_2, width=1), name="charged = fair",
                    hoverinfo="skip")
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    fig.update_xaxes(type="log", title="fair freight from the model (R$)", tickvals=ticks)
    fig.update_yaxes(type="log", title="freight charged (R$)", tickvals=ticks)
    st.plotly_chart(layout(fig, h=420, title="Unusual items sit well above the fair-freight line"),
                    width="stretch")
    if src == "all held-out items":
        st.caption(f"{len(ok):,} of the normal items are drawn, to keep the chart fast; all unusual items are shown.")
    top = (f.groupby("seller_id").agg(n_unusual=("unusual", "size"), excess=("excess", "sum"))
           .sort_values("n_unusual", ascending=False).head(10).reset_index())
    top["seller"] = top.seller_id.astype(str).str[:8]
    share = top.n_unusual.sum() / len(f)
    fig = go.Figure(go.Bar(x=top.n_unusual, y=top.seller, orientation="h", marker_color=BLUE, text=top.n_unusual,
                           textposition="outside", hovertemplate="seller %{y}: %{x} unusual items<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="unusual items", range=[0, top.n_unusual.max() * 1.15])
    st.plotly_chart(layout(fig, h=420, title=f"Ten sellers hold {share:.0%} of the unusual items", bargap=0.3),
                    width="stretch")
    st.subheader("Review list")
    out = (f.sort_values("excess", ascending=False)
           [["order_id", "seller_id", "category", "price", "freight_value", "expected", "excess", "weight_kg",
             "distance_km", "seller_state", "customer_state"]]
           .rename(columns={"freight_value": "charged", "expected": "fair_freight"}))
    st.dataframe(out.round(2), hide_index=True, width="stretch", height=320,
                 column_config={c: st.column_config.NumberColumn(format="R$ %.2f")
                                for c in ["price", "charged", "fair_freight", "excess"]})
    st.download_button("Download the review list (CSV)", out.to_csv(index=False), "unusual_freight_review.csv",
                       "text/csv")


def late_page():
    st.title("⏰ Late-delivery risk queue")
    st.markdown(
        f"A logistic model trained on {P2['n_train']:,} delivered orders (up to {P2['train_dates'][1]}) scores each "
        f"new order at checkout, from the promised delivery time, the route and the order's value and freight. The "
        f"orders here come from the test window ({P2['test_dates'][0]} to {P2['test_dates'][1]}), which the model "
        f"never saw. The score is a **ranking**: late deliveries were rarer in the test window "
        f"({P2['late_rate_test']:.1%}) than in training ({P2['late_rate_train']:.1%}), so read it as an order of "
        f"priority, not a probability.")
    t1, t2 = st.tabs(["Daily review queue", "Score a new order"])
    with t1:
        queue()
    with t2:
        score_order()


def queue():
    d = heldout_orders_scored()
    days = d.day.drop_duplicates().sort_values()
    with st.expander("How to use", expanded=False):
        st.markdown("Choose the day the orders were placed and how many of them the team can check. The list puts the "
                    "riskiest orders first. Switch on **Reveal what happened** to see the real outcomes, which "
                    "nobody knows at checkout.")
    c = st.columns([1, 1, 1])
    default = days[d.groupby("day").size().reindex(days).values >= 150].iloc[0]
    day = c[0].date_input("Orders placed on", value=default.date(), min_value=days.min().date(),
                          max_value=days.max().date())
    share = c[1].select_slider("Review capacity (share of the day's orders)", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                               value=P2["share"], format_func=lambda v: f"{v:.0%}")
    reveal = c[2].toggle("Reveal what happened", value=False)
    flags = daily_flags(share)
    mask = (d.day == pd.Timestamp(day)).values
    today = d[mask].assign(flag=flags[mask]).sort_values("risk", ascending=False)
    if today.empty:
        st.info("No delivered orders were placed on that day.")
        return
    n_flag = int(today.flag.sum())
    k = st.columns(4)
    k[0].metric("Orders placed", f"{len(today):,}")
    k[1].metric("To check first", f"{n_flag:,}")
    k[2].metric("Median promise", f"{today.promised_days.median():.0f} days")
    if reveal:
        late = int(today.clf_target.sum())
        k[3].metric("Late orders caught", f"{int((today.flag & today.clf_target).sum())} of {late}")
    if n_flag == 0:
        st.warning(f"With {len(today)} orders, a {share:.0%} capacity rounds to nobody: raise the capacity or pick a "
                   f"busier day.")
    show = today.assign(rank=np.arange(1, len(today) + 1), order=today.order_id.str[:8],
                        placed=today.order_purchase_timestamp.dt.strftime("%H:%M"),
                        route=today.seller_state + " → " + today.customer_state,
                        check=np.where(today.flag == 1, "🔎 check", ""))
    cols = ["rank", "check", "order", "placed", "route", "distance_km", "promised_days", "price_total",
            "freight_total", "risk"]
    if reveal:
        show["outcome"] = np.where(show.clf_target == 1, "late by " + show.days_late.astype(int).astype(str) + " d",
                                   "on time")
        show["review"] = show.low_review.map({1.0: "1–2 stars", 0.0: "3–5 stars"}).fillna("none")
        cols += ["outcome", "review"]
    st.dataframe(show[cols], hide_index=True, width="stretch", height=380, column_config={
        "distance_km": st.column_config.NumberColumn("distance (km)", format="%.0f"),
        "promised_days": st.column_config.NumberColumn("promised (days)", format="%.0f"),
        "price_total": st.column_config.NumberColumn("order value", format="R$ %.2f"),
        "freight_total": st.column_config.NumberColumn("freight", format="R$ %.2f"),
        "risk": st.column_config.ProgressColumn("risk score", format="%.3f", min_value=0.0,
                                                max_value=float(d.risk.max()))})

    pick = st.selectbox("Why is an order on the list?", show.order.head(max(n_flag, 1)).tolist(),
                        help="The orders to check first, riskiest at the top")
    row = today[today.order_id.str[:8] == pick].head(1)
    st.plotly_chart(drivers_chart(logit_drivers(row).iloc[0], f"What drives order {pick}'s risk"), width="stretch")

    if reveal:
        st.divider()
        y = d.clf_target.values
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


def score_order():
    with st.expander("How to use", expanded=False):
        st.markdown("Describe an order as it looks at checkout and press **Score**. The app places its risk among the "
                    "orders of the test window and shows which inputs push it up or down.")
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
        go_ = st.form_submit_button("Score", type="primary")
    if not go_:
        return
    errors = ["order value must be above zero"] if value <= 0 else []
    s_loc, s_note = locate(s_zip, s_state, "Seller")
    c_loc, c_note = locate(c_zip, c_state, "Customer")
    errors += [n for loc, n in [(s_loc, s_note), (c_loc, c_note)] if loc is None]
    if errors:
        st.error("Please fix: " + "; ".join(errors) + ".")
        return
    for n in (s_note, c_note):
        if n:
            st.info(n)
    dist = float(F.haversine_km(s_loc[0], s_loc[1], c_loc[0], c_loc[1]))
    row = pd.DataFrame([dict(promised_days=promised, distance_km=dist, same_state=int(s_loc[2] == c_loc[2]),
                             price_total=value, freight_total=freight, freight_ratio=freight / value,
                             n_items=n_items, n_sellers=n_sellers, seller_state=s_loc[2], customer_state=c_loc[2],
                             order_id="new", purchase_ts=0.0)])
    range_note({"promised_days": promised, "price_total": value, "freight_total": freight, "distance_km": dist},
               P2["ranges"])
    risk = float(LOGIT.predict_proba(row[COLS_B])[0, 1])
    pct = float(np.interp(risk, P2["score_pct"], np.linspace(0, 100, 101)))
    c = st.columns(3)
    c[0].metric("Risk score", f"{risk:.3f}")
    c[1].metric("Riskier than", f"{pct:.0f}% of test-window orders")
    c[2].metric("Route", f"{s_loc[2]} → {c_loc[2]}, {dist:,.0f} km")
    if pct >= 100 * (1 - P2["share"]):
        st.error(f"🔎 **Check this order.** It would be in the riskiest {P2['share']:.0%} of orders.")
    else:
        st.success(f"✅ Below the review line: the riskiest {P2['share']:.0%} of orders start at the "
                   f"{100 * (1 - P2['share']):.0f}th percentile.")
    st.plotly_chart(drivers_chart(logit_drivers(row).iloc[0], "What drives this order's risk"), width="stretch")


def model_card():
    st.title("Model card")
    st.markdown("What each model was trained on, how it was validated and where it should not be trusted. Test "
                "results are recomputed by `build_models.py` from the files this app loads; robustness results are "
                "quoted from the Phase 2 notebook.")
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Problem 1 · Fair freight")
        st.markdown(
            f"- **Use:** benchmark the freight of an order item; flag unusual charges for review.\n"
            f"- **Model:** random forest regressor on log(1 + freight), {P1['forest_params']['n_estimators']} trees, "
            f"leaf size {P1['forest_params']['min_samples_leaf']}, depth {P1['forest_params']['max_depth']}, "
            f"{P1['forest_params']['max_features']:.0%} of features per split; tuned by grouped cross-validation.\n"
            f"- **Inputs:** weight, volume, chargeable weight, distance, same-state route, seller and customer "
            f"state, category, price, basket size, listing details and purchase timing. Seller identity is "
            f"excluded on purpose, so a seller's habitual overcharging cannot become the benchmark.\n"
            f"- **Data:** {P1['n_train']:,} training items; {P1['n_test']:,} test items whose orders were never seen "
            f"in training (80/20 split grouped by order).\n"
            f"- **Test results:** MAE {md_money(P1['test']['MAE_rs'])} ({P1['test']['MAE_pct']}% of mean freight "
            f"{md_money(P1['mean_freight_test'])}), RMSE {md_money(P1['test']['RMSE_rs'])}, R² {P1['test']['R2_rs']:.3f}. "
            f"Cross-validated MAE {md_money(NB['p1_cv_mae'])} ± {NB['p1_cv_mae_sd']:.2f}. Baselines: Ridge "
            f"{md_money(NB['p1_ridge_test_mae'])}, single tree {md_money(NB['p1_tree_test_mae'])}, transparent "
            f"tariff {md_money(P1['tariff']['mae'])}.\n"
            f"- **Unusual-freight rule:** residual in the top {1 - P1['flag_q']:.0%} of the test set "
            f"({P1['flagged_test']:,} items).")
        bands = pd.DataFrame(P1["price_bands"]).rename(columns={"band": "Item price (R$)", "n": "Test items",
                                                                "mae": "MAE (R$)", "mae_pct": "MAE % of mean freight"})
        st.markdown("**Error grows with item price**, so a single tolerance is too tight for expensive items:")
        st.dataframe(bands, hide_index=True, width="stretch")
        st.markdown(
            f"**Limits.** Listings never seen in training: MAE {md_money(NB['p1_unseen_mae'])} "
            f"({NB['p1_unseen_pct']}%). Freight per kg rose over time, and on the latest 20% of items the MAE is "
            f"{md_money(NB['p1_recent_mae'])}: retrain on recent orders before real use. A flag is a reason to look, "
            f"not proof of overcharging.")
    with b:
        st.subheader("Problem 2 · Late delivery")
        st.markdown(
            f"- **Use:** rank new orders by their risk of arriving after the promised date, so the team checks the "
            f"riskiest first.\n"
            f"- **Model:** plain logistic regression (no penalty), the simplest model within one fold standard "
            f"deviation of the best in forward-chaining cross-validation.\n"
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
            f"{P2['daily']['recall']:.1%}. Cross-validated lift {NB['p2_cv_lift']:.2f} ± {NB['p2_cv_lift_sd']:.2f}.\n"
            f"- **Over time:** lift {', '.join(f'{v:.2f}' for v in NB['p2_window_lift'])} in four consecutive test "
            f"windows (Sep 2017 to Aug 2018). Flagged orders received 1–2 star reviews "
            f"{NB['p2_low_review_ratio'][0]}–{NB['p2_low_review_ratio'][1]}× as often as unflagged ones.")
        gain = pd.DataFrame(P2["gain"]).rename(columns={"decile": "Risk decile", "orders": "Orders", "late": "Late",
                                                        "cum_capture": "Late orders caught (cumulative)", "lift": "Lift"})
        st.markdown("**Late orders by risk decile** (the whole test window ranked at once):")
        st.dataframe(gain, hide_index=True, width="stretch")
        st.markdown(
            "**Limits.** The model ranks, it does not forecast: its scores overstate the late rate of the test window. "
            "Olist cut its promised times sharply from late July 2018, and the short promises to São Paulo customers "
            "account for most missed late orders. The ranking is weakest when promises change, so monitor lift monthly.")
    st.divider()
    b_ = FACTS["built"]
    st.caption(f"Models built {b_['date']} with Python {b_['python']}, pandas {b_['pandas']}, numpy {b_['numpy']}, "
               f"scikit-learn {b_['scikit_learn']}, seed {b_['seed']}. Rebuild: "
               f"`OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/model_app/build_models.py`.")


# ------------------------------------------------------------------ navigation
PAGE_HOME = st.Page(home, title="Home", icon="🏠", default=True)
PAGE_FREIGHT = st.Page(freight_page, title="Fair-freight checker", icon="📦", url_path="freight")
PAGE_LATE = st.Page(late_page, title="Late-delivery risk queue", icon="⏰", url_path="late")
PAGE_CARD = st.Page(model_card, title="Model card", icon="📋", url_path="model-card")
nav = st.navigation([PAGE_HOME, PAGE_FREIGHT, PAGE_LATE, PAGE_CARD])
st.sidebar.caption("Team 5 · IT5006 · AY2026/27 · Phase 3")
st.sidebar.caption("Scores held-out Olist records (2016–2018) as if they were new orders.")
nav.run()
