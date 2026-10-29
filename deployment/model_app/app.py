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
  Both teams
    Map by state      where unusual charges and late orders concentrate; opens the audit or the queue for a state

The lists are clickable (a row opens its detail), the forms update as the inputs change, and an analyst's review
marks (a verdict on a flagged item, "contacted" on an order) are kept for the browser session and can be downloaded.

The app is the frontend of a two-part deployment. The models run in a separate service, the model API
(deployment/api/main.py, FastAPI on Render), and this app holds no model files: the forms and CSV uploads send their
inputs to the API, and the pages that show the whole test period (audit, tariff, queue, monitoring) read scores the API
returned in a batch run (deployment/api/batch_score.py writes data/scored_*.csv and checks them against the build).
Every headline number on these pages is read from data/facts.json, which build_models.py checks against the notebooks.

The Olist data end in 2018, so the app scores held-out historical records as if they were new orders: the freight
items come from the Problem 1 test set (the latest 20% of items, 24 May - 3 Sep 2018) and the orders from the Problem 2
test window (26 May - 29 Aug 2018), none of which the models saw in training.

Run locally:   uvicorn main:app --app-dir deployment/api --port 8000
               API_URL=http://localhost:8000 streamlit run deployment/model_app/app.py
The API address comes from st.secrets["API_URL"], else the API_URL environment variable, else the Render service.
"""
import json, os, re, threading, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Olist logistics assistant", page_icon="📦", layout="wide")

REPO_URL = "https://github.com/likeyou-ky/Team5_IT5006_Ecommerce_Analytics_AY2627Sem1"
EDA_URL = "https://team5-olist-eda-phase1.streamlit.app/"
DEFAULT_API_URL = "https://team5-olist-api.onrender.com"
MAX_UPLOAD_ROWS = 20_000
CHUNK = 2_000                       # rows per API call (the API takes at most 5,000)
API_TIMEOUT = (10, 120)             # connect, read (s)
WAKE_LIMIT = 90                     # s to wait for a sleeping free-tier service
API_CACHE = 200                     # API answers remembered per session, by request

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
RULES = {"overall": "Top 5% overall (evaluated)", "band": "Top 5% within each price band (recommended)"}
MARKS = ["overcharge", "justified", "data error"]   # an analyst's verdict on a flagged item (kept for the session)
# Widget keys kept when the user moves between pages (filters, form inputs), and set by the map and the scenarios
KEEP = ("f_", "q_", "ql_", "pc_", "map_")


# ------------------------------------------------------------------ loading (cached, with a clear failure message)
@st.cache_data(show_spinner=False)
def load_data():
    """facts.json, and the held-out records joined to the scores the API returned for them (batch_score.py)."""
    read = lambda f, **k: pd.read_csv(HERE / "data" / f, **k)
    facts = json.loads((HERE / "data" / "facts.json").read_text(encoding="utf-8"))
    items = read("heldout_items.csv", parse_dates=["order_purchase_timestamp"]).merge(
        read("scored_items.csv"), on=["order_id", "order_item_id"], how="left", validate="one_to_one")
    orders = read("heldout_orders.csv", parse_dates=["order_purchase_timestamp"]).merge(
        read("scored_orders.csv"), on="order_id", how="left", validate="one_to_one")
    if items.expected.isna().any() or orders.risk.isna().any():
        raise ValueError("some held-out records have no API score: re-run deployment/api/batch_score.py")
    return facts, items, orders


try:
    FACTS, ITEMS, ORDERS = load_data()
except Exception as exc:                                                    # missing or unreadable files
    st.error("The app could not load its data files. Rebuild them with "
             "`python deployment/model_app/build_models.py`, then score the held-out records through the API with "
             "`python deployment/api/batch_score.py`.")
    st.exception(exc)
    st.stop()

P1, P2, NB = FACTS["p1"], FACTS["p2"], FACTS["notebook"]
FLAG = P1["flag"]
COLS_A = P1["features_num"] + P1["features_cat"]
USES_LISTING = "photos_qty" in COLS_A           # listing details: shown only if the model kept them
DRIVER_COLS = [c for c in ORDERS.columns if c.startswith("drv_")]
STATES = P1["states"]
DAILY_YIELD = 1 / P2["daily"]["precision"]          # flagged orders checked per late order found, daily 10% list


# ------------------------------------------------------------------ the model API: address, wake-up, calls
def _api_url():
    try:
        url = st.secrets.get("API_URL")
    except Exception:                                           # no secrets file (local runs)
        url = None
    return (url or os.environ.get("API_URL") or DEFAULT_API_URL).rstrip("/")


API_URL = _api_url()


@st.cache_data(ttl=60, show_spinner=False)
def api_health():
    """(ok, health JSON or error) for the status indicator. Any request also wakes a sleeping service."""
    try:
        t0 = time.perf_counter()
        r = requests.get(f"{API_URL}/", timeout=4)
        r.raise_for_status()
        return True, dict(r.json(), ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return False, str(e)


def wake_api():
    """Make sure the API answers. On Render's free tier it sleeps after 15 minutes without traffic, and the first
    request takes 30-60 s while it starts: say so, and keep asking for up to WAKE_LIMIT seconds."""
    if time.time() - st.session_state.get("api_ok_at", 0) < 240:
        return True
    try:
        if requests.get(f"{API_URL}/", timeout=4).ok:
            st.session_state.api_ok_at = time.time()
            return True
    except requests.RequestException:
        pass
    with st.status("Waking the API (30–60 s)…", expanded=True) as box:
        st.write(f"The model service at {API_URL} sleeps after 15 minutes without traffic (Render's free tier); "
                 f"the first request starts it again.")
        t0 = time.time()
        while time.time() - t0 < WAKE_LIMIT:
            try:
                if requests.get(f"{API_URL}/", timeout=20).ok:
                    box.update(label=f"The API is awake ({time.time() - t0:.0f} s)", state="complete", expanded=False)
                    st.session_state.api_ok_at = time.time()
                    api_health.clear()
                    return True
            except requests.RequestException:
                time.sleep(3)
        box.update(label="The API did not answer", state="error")
    st.error(f"The model API at {API_URL} did not answer within {WAKE_LIMIT} s. Try again in a minute.")
    return False


def ping_api():
    """Wake a sleeping API in the background when a session starts, so the first form does not wait for it. Runs in
    its own thread and touches no Streamlit state."""
    try:
        requests.get(f"{API_URL}/", timeout=WAKE_LIMIT)
    except requests.RequestException:
        pass


def show_rejection(body):
    st.error("The API rejected the input:\n\n" + "\n".join(f"- {e}" for e in body.get("errors", [body])))


def api_call(method, path, payload=None, params=None, show_422=True):
    """Call the API with a timeout and up to two retries. Returns (status, JSON); shows what went wrong."""
    if not wake_api():
        return None, None
    for attempt in range(3):
        try:
            r = requests.request(method, f"{API_URL}{path}", json=payload, params=params, timeout=API_TIMEOUT)
        except (requests.ConnectionError, requests.Timeout):
            st.session_state.api_ok_at = 0                       # asleep again, or restarting: wake it and retry
            if attempt < 2 and wake_api():
                continue
            st.error("The model API did not respond. Try again in a minute.")
            return None, None
        if r.status_code in (502, 503, 504) and attempt < 2:      # the service is (re)starting
            time.sleep(5)
            continue
        if r.status_code == 422:
            body = r.json()
            if show_422:
                show_rejection(body)
            return 422, body
        if not r.ok:
            st.error(f"The model API returned an error ({r.status_code}): {r.text[:300]}")
            return r.status_code, None
        st.session_state.api_ok_at = time.time()
        return r.status_code, r.json()
    return None, None


def api_post(path, payload, params=None):
    """POST one request, remembering the answer (or the rejection) for the session: a rerun that leaves the inputs
    unchanged (another widget, a row selection) shows the stored answer instead of calling the API again."""
    key = json.dumps([path, payload, params], sort_keys=True, default=str)
    cache = st.session_state.setdefault("api_cache", {})
    if key not in cache:
        if not wake_api():
            return None
        with st.spinner("Asking the model API…"):
            status, body = api_call("POST", path, payload, params, show_422=False)
        if status not in (200, 422):
            return None
        cache[key] = (status, body)
        while len(cache) > API_CACHE:
            cache.pop(next(iter(cache)))
    status, body = cache[key]
    if status == 422:
        show_rejection(body)
    return body if status == 200 else None


def api_get(path):
    if not wake_api():
        return None
    with st.spinner("Asking the model API…"):
        status, body = api_call("GET", path)
    return body if status == 200 else None


def api_batch(path, key, rows, params=None):
    """Score rows in chunks. Rows the API rejects are left out and reported: ({row: errors}, predictions by row)."""
    preds, rejected = {}, {}
    bar = st.progress(0.0, text="Scoring through the model API…") if len(rows) > CHUNK else None
    for s in range(0, len(rows), CHUNK):
        part = dict(enumerate(rows[s:s + CHUNK], start=s))
        for _ in range(2):                                       # once as sent, once without the rejected rows
            if not part:
                break
            status, body = api_call("POST", path, {key: list(part.values())}, params, show_422=False)
            if status == 422:
                keys = list(part)
                bad = {}
                for d, e in zip(body.get("detail", []), body.get("errors", [])):
                    loc = d.get("loc", [])
                    if len(loc) > 2 and isinstance(loc[2], int):
                        bad.setdefault(keys[loc[2]], []).append(re.sub(r"^\w+\.\d+[.:]\s*", "", e))
                if not bad:
                    st.error("The API rejected the file: " + "; ".join(body.get("errors", [])[:3]))
                    return None, None
                rejected.update(bad)
                part = {k: v for k, v in part.items() if k not in bad}
                continue
            if body is None:
                return None, None
            preds.update(zip(part, body["predictions"]))
            break
        if bar:
            bar.progress(min(1.0, (s + CHUNK) / len(rows)), text="Scoring through the model API…")
    if bar:
        bar.empty()
    return rejected, preds


def show_notes(p):
    """The API's notes on how it placed the inputs, and its drift warnings, for one result."""
    if p["notes"]:
        st.info("  \n".join(p["notes"]))
    if p["drift_warnings"]:
        st.warning("**Drift warning:** this input is unlike the training data, so treat the result with caution.\n\n"
                   + "\n".join(f"- {w}" for w in p["drift_warnings"]))


def flag_top_share(score, share, seed=FACTS["built"]["seed"]):
    """The queue rule of src/models.py (copied, so the app needs no scikit-learn): flag the `share` highest scores,
    breaking ties at random with a fixed seed."""
    s = np.asarray(score, dtype=float)
    tie = np.random.RandomState(seed).rand(len(s))
    top = np.lexsort((tie, s))[::-1][:int(round(share * len(s)))]
    out = np.zeros(len(s), dtype=int)
    out[top] = 1
    return out


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
    with st.popover("How to use this page", icon="💡"):
        st.markdown(text)


def keep(key, value):
    """Key of a widget whose value outlives its page (see KEEP); the widget is then created without a value.
    Streamlit forgets a widget's value when its page is left, and a value changed inside a fragment is lost even
    when the key is stored again, so remember() copies the values to plain session keys and they are restored here."""
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(f"_kept_{key}", value)
    return key


def remember():
    """Copy the kept widgets' values to plain session keys. Runs at the top of every full run and at the end of the
    inputs of each fragment (a fragment rerun does not run the top of the script)."""
    for k in [k for k in st.session_state if str(k).startswith(KEEP)]:
        st.session_state[f"_kept_{k}"] = st.session_state[k]


def place_inputs(prefix, label, default="SP"):
    """State and optional zip code of one party. The API places the party (and says how) from these."""
    c = st.columns(2)
    state = c[0].selectbox(f"{label} state", STATES, key=keep(f"{prefix}_state", default))
    z = c[1].text_input(f"{label} zip code (optional)", key=keep(f"{prefix}_zip", ""), placeholder="e.g. 01310")
    return z.strip(), state


def route_payload(s_zip, s_state, c_zip, c_state):
    """The route fields of an API request; empty zip codes are left out, so the API uses the state centres."""
    out = dict(seller_state=s_state, customer_state=c_state)
    if s_zip:
        out["seller_zip"] = s_zip
    if c_zip:
        out["customer_zip"] = c_zip
    return out


# ------------------------------------------------------------------ Problem 1 scores
def error_band(price):
    """The test-set error of items at this price (the bands of the notebook's error-by-price table)."""
    for b, hi in zip(P1["price_bands"], [25, 50, 100, 200, 500, np.inf]):
        if price < hi:
            return b
    return P1["price_bands"][-1]


def tariff(chargeable_kg, distance_km, same_state):
    """The simple published tariff (coefficients from facts.json), for the tariff review."""
    t = P1["tariff"]
    return np.maximum(0.0, t["base"] + t["per_kg"] * np.asarray(chargeable_kg) + t["per_100km"]
                      * np.asarray(distance_km) / 100 + t["same_state"] * np.asarray(same_state))


def add_scores(d, rule):
    """The API's unusual flags under `rule`, and the audit columns. `d` holds the API's fair freight (expected,
    expected_ridge), its price band and its flags for both rules (unusual_<rule>, unusual_ridge_<rule>)."""
    d = d.copy()
    d["unusual"] = d[f"unusual_{rule}"].astype(bool)
    d["unusual_ridge"] = d[f"unusual_ridge_{rule}"].astype(bool)
    d["both"] = d.unusual & d.unusual_ridge
    d["excess"] = d.freight_value - d.expected
    d["markup"] = np.where(d.expected > 0, d.freight_value / d.expected.where(d.expected > 0), np.nan)
    d["band"] = d.audit_band
    return d


@st.cache_data(show_spinner=False)
def heldout_items_scored(rule):
    return add_scores(ITEMS, rule)


def rule_picker():
    """The unusual-freight rule, shared by the analyst's pages (a kept widget, so every page uses the same rule)."""
    return st.radio("Unusual-freight rule", list(RULES), format_func=RULES.get, horizontal=True,
                    key=keep("f_rule", "overall"),
                    help="Top 5% overall (evaluated): the 5% of items with the largest charge above fair freight; "
                         "the rule the notebooks evaluate, and the one the headline audit numbers use. Top 5% within "
                         "each price band (recommended): 5% of every price band, as the report recommends, so "
                         "expensive items, whose freight is harder to predict, do not crowd the review list.")


def similar_items(cat, s_state, c_state, ckg, km, exclude=None, n=5):
    """Up to n held-out items like this one: the same category going to the same customer state (the same route
    first), with a chargeable weight within ±50% and a distance within ±25% (at least ±150 km)."""
    c = ITEMS[(ITEMS.category == cat) & (ITEMS.customer_state == c_state)]
    if exclude is not None:
        c = c.drop(index=exclude, errors="ignore")
    kg_gap = np.abs(np.log((c.chargeable_kg + 0.1) / (ckg + 0.1))) / np.log(1.5)
    km_gap = (np.abs(c.distance_km - km) / max(0.25 * km, 150) if km is not None and np.isfinite(km)
              else pd.Series(0.0, index=c.index))
    ok = (kg_gap <= 1) & (km_gap <= 1)                                   # an unknown distance is never close
    other_route = (c.seller_state != s_state).astype(float)
    return c[ok].assign(_gap=(other_route * 10 + kg_gap + km_gap)[ok]).nsmallest(n, "_gap")


def why_price(cat, s_state, c_state, ckg, km, same, fair, fair_ridge, charged=None, exclude=None):
    """Why this price: the simple tariff built up from its parts next to the forest and Ridge estimates (and the
    charge), and the most similar held-out items with what they were charged and their fair freight."""
    st.markdown("#### Why this price?")
    t = P1["tariff"]
    a, b = st.columns([5, 4], gap="large")
    with a:
        if km is None or not np.isfinite(km):
            st.info("The distance is unknown, so the simple tariff cannot be priced; the models use the training "
                    "median distance instead.")
        else:
            parts = [("base", t["base"], "base charge"),
                     (f"{ckg:,.2f} kg", t["per_kg"] * ckg, f"{ckg:,.2f} chargeable kg × R$ {t['per_kg']:.3f}"),
                     (f"{km:,.0f} km", t["per_100km"] * km / 100, f"{km:,.0f} km × R$ {t['per_100km']:.3f} per 100 km")]
            if same:
                parts.append(("same state", t["same_state"], "discount within a state"))
            total = max(0.0, sum(v for _, v, _ in parts))
            fig = go.Figure(go.Waterfall(
                x=[p[0] for p in parts] + ["tariff"], y=[p[1] for p in parts] + [0],
                measure=["relative"] * len(parts) + ["total"], customdata=[p[2] for p in parts] + ["simple tariff"],
                text=[f"{v:+.2f}" for _, v, _ in parts] + [f"{total:.2f}"], textposition="outside",
                increasing=dict(marker=dict(color=GREY)), decreasing=dict(marker=dict(color=DIV_MID, line=dict(
                    color=GREY, width=1))), totals=dict(marker=dict(color=INK_2)),
                connector=dict(line=dict(color=GRID, width=1)), cliponaxis=False, showlegend=False,
                hovertemplate="%{customdata}: R$ %{y:+.2f}<extra></extra>"))
            ests = [("forest", fair, BLUE, "fair freight (forest)"), ("Ridge", fair_ridge, DIV_LOW, "Ridge estimate")]
            if charged:
                ests.append(("charged", charged, ORANGE, "charged"))
            for x, v, colour, name in ests:
                fig.add_bar(x=[x], y=[v], marker_color=colour, text=[f"{v:.2f}"], textposition="outside",
                            cliponaxis=False, showlegend=False, hovertemplate=f"{name}: R$ %{{y:.2f}}<extra></extra>")
            top = max([total, *[v for _, v, _, _ in ests]])
            fig.update_yaxes(title="R$", range=[0, top * 1.2])
            st.plotly_chart(layout(fig, h=330, title="The simple tariff, step by step, against the models",
                                   barmode="overlay", waterfallmode="overlay", bargap=0.35), width="stretch")
            st.caption(f"The tariff adds a base, a rate per chargeable kg and a rate per 100 km"
                       f"{', less a discount within a state' if same else ''}: {md_money(total)}. The forest's fair "
                       f"freight ({md_money(fair)}) also weighs the category, the price, both states and how weight "
                       f"and distance interact, so it errs less: {md_money(P1['test']['MAE_rs'])} on average against "
                       f"{md_money(t['mae'])} for the tariff (Ridge {md_money(P1['ridge_test']['MAE_rs'])}).")
    with b:
        sim = similar_items(cat, s_state, c_state, ckg, km, exclude)
        st.markdown(f"**Similar held-out items** ({cat}, to {c_state})")
        if sim.empty:
            st.caption("No held-out item of this category went to this state with a similar weight over a similar "
                       "distance.")
            return
        st.dataframe(sim.assign(order=sim.order_id.str[:8], route=sim.seller_state + " → " + sim.customer_state)[
            ["order", "route", "chargeable_kg", "distance_km", "freight_value", "expected"]], hide_index=True,
            width="stretch", column_config={
                "chargeable_kg": st.column_config.NumberColumn("kg", format="%.2f"),
                "distance_km": st.column_config.NumberColumn("km", format="%.0f"),
                "freight_value": st.column_config.NumberColumn("charged", format="R$ %.2f"),
                "expected": st.column_config.NumberColumn("fair", format="R$ %.2f")})
        st.caption(f"Same category and customer state (same route first), chargeable weight within ±50% and distance "
                   f"within ±25%. They were charged {md_money(sim.freight_value.median())} (median) against a fair "
                   f"freight of {md_money(sim.expected.median())}; fair freight comes from the API's batch run.")


def freight_curves(p, charged=None):
    """Fair freight of a listing as its weight or its distance changes: one /freight/batch call with variants of the
    listing at the route the API placed (explicit distance_km), shaded by the quote range (± the price band's error).
    The grids stay inside the training data's drift limits, so the variants raise no drift warnings."""
    x = p["inputs"]
    w0, km0 = float(x["weight_kg"]), x["distance_km"]
    if km0 is None:
        return
    base = {k: x[k] for k in ("category", "price", "volume_cm3", "n_items", "n_sellers", "seller_state",
                              "customer_state")} | dict(weight_kg=w0, distance_km=float(km0))
    weights = np.unique(np.round(np.r_[np.geomspace(0.05, max(18.0, w0), 30), w0], 3))
    kms = np.unique(np.round(np.r_[np.linspace(10, max(3500.0, km0), 30), km0], 1))
    rows = [dict(base, weight_kg=float(w)) for w in weights] + [dict(base, distance_km=float(k)) for k in kms]
    res = api_post("/freight/batch", {"items": rows})
    if res is None:
        return
    fair = np.array([q["fair_freight"] for q in res["predictions"]])
    mae, vol_kg = p["band_mae"], x["volume_cm3"] / 5000
    st.markdown("#### What if the weight or the distance changed?")
    cols = st.columns(2, gap="large")
    for col, xs, ys, now, what, unit, log in [
            (cols[0], weights, fair[:len(weights)], w0, "weight", "kg", True),
            (cols[1], kms, fair[len(weights):], float(km0), "distance", "km", False)]:
        fig = go.Figure()
        fig.add_scatter(x=xs, y=np.maximum(ys - mae, 0), mode="lines", line=dict(width=0), hoverinfo="skip",
                        showlegend=False)
        fig.add_scatter(x=xs, y=ys + mae, mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(42,120,214,0.15)", name=f"quote range (± R$ {mae:.2f})", hoverinfo="skip")
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=2), name="fair freight",
                        hovertemplate=f"%{{x:,.2f}} {unit} → fair freight R$ %{{y:.2f}}<extra></extra>")
        fig.add_scatter(x=[now], y=[p["fair_freight"]], mode="markers", name="this listing",
                        marker=dict(color=ORANGE, size=11, line=dict(color="white", width=2)),
                        hovertemplate=f"this listing: %{{x:,.2f}} {unit}, R$ %{{y:.2f}}<extra></extra>")
        if charged:
            fig.add_hline(y=charged, line=dict(color=ORANGE, width=1.5, dash="dash"),
                          annotation_text=f"charged R$ {charged:.2f}", annotation_position="top left")
        if log and xs[0] < vol_kg < xs[-1]:
            # a shape and its label by hand: add_vline places its label in linear units on a log axis
            fig.add_shape(type="line", x0=vol_kg, x1=vol_kg, y0=0, y1=1, yref="paper",
                          line=dict(color=INK_2, width=1, dash="dot"))
            fig.add_annotation(x=np.log10(vol_kg), y=1, yref="paper", text="volumetric weight", showarrow=False,
                               xanchor="left", yanchor="top", font=dict(size=11, color=INK_2))
        fig.update_xaxes(title=f"{what} ({unit}){', log scale' if log else ''}", type="log" if log else "linear",
                         range=[np.log10(xs[0]), np.log10(xs[-1])] if log else [0, xs[-1]],
                         tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200] if log else None)
        fig.update_yaxes(title="fair freight (R$)", rangemode="tozero")
        with col:
            st.plotly_chart(layout(fig, h=340, title=f"Fair freight by {what}"), width="stretch")
    st.caption(f"Each curve changes one input and keeps the rest (one `/freight/batch` call with "
               f"{len(rows)} variants of the listing). Left of the volumetric weight (volume / 5,000 = "
               f"{vol_kg:,.2f} kg) carriers charge by volume, so a lighter item costs about the same. The shaded band "
               f"is the quote range: ± {md_money(mae)}, the forest's average error at this price.")


# ------------------------------------------------------------------ review marks (kept for the browser session only)
def item_keys(d):
    """A key per item for the review marks: order and item number (held-out items) or order and file row (uploads)."""
    second = d.order_item_id if "order_item_id" in d else d.file_row
    return d.order_id.astype(str) + "/" + second.astype(str)


def _save_mark(store, key, wkey, info):
    """Widget callback: store (or clear) one mark before the page reruns, so every list shows it at once."""
    v = st.session_state.get(wkey)
    marks = st.session_state.setdefault(store, {})
    if v:
        marks[key] = dict(info, mark="contacted" if v is True else v, marked_at=time.strftime("%H:%M:%S"))
    else:
        marks.pop(key, None)


def mark_item(it, key):
    """The analyst's verdict on one flagged item."""
    marks = st.session_state.setdefault("item_marks", {})
    wkey = f"mk_{key}"
    if wkey not in st.session_state:
        st.session_state[wkey] = marks.get(key, {}).get("mark")
    info = dict(order_id=it.order_id, item=key.split("/", 1)[1], seller_id=it.seller_id, category=it.category,
                price=round(float(it.price), 2), charged=round(float(it.freight_value), 2),
                fair_freight=round(float(it.expected), 2), excess=round(float(it.excess), 2),
                flagged_by="both models" if it.both else "forest only" if it.unusual else "not flagged")
    st.segmented_control("Your verdict on this charge", MARKS, key=wkey, on_change=_save_mark,
                         args=("item_marks", key, wkey, info),
                         help="Recorded for this browser session only; download the marks from the review list.")


def mark_order(row):
    """'Contacted' on one order of the risk queue."""
    oid = row.order_id
    marks = st.session_state.setdefault("order_marks", {})
    wkey = f"ct_{oid}"
    if wkey not in st.session_state:
        st.session_state[wkey] = oid in marks
    info = dict(order_id=oid, seller_id=row.seller_id, placed=f"{row.order_purchase_timestamp:%Y-%m-%d %H:%M}",
                route=f"{row.seller_state} → {row.customer_state}", promised_days=round(float(row.promised_days), 1),
                risk=round(float(row.risk), 4), follow_up=row.action)
    st.toggle("Contacted: the follow-up is done", key=wkey, on_change=_save_mark,
              args=("order_marks", oid, wkey, info))


def marks_summary(store, what, file):
    """The running 'reviewed so far' line, with the marks as CSV. `what` is (singular, plural)."""
    marks = st.session_state.get(store, {})
    with st.container(border=True):
        a, b = st.columns([3, 1], vertical_alignment="center")
        if marks:
            n = pd.Series([m["mark"] for m in marks.values()]).value_counts()
            a.markdown(f"**Reviewed so far:** {len(marks)} {what[len(marks) != 1]}" + (
                " (" + ", ".join(f"{v} {k}" for k, v in n.items()) + ")" if store == "item_marks" else "") + ".")
        else:
            a.markdown(f"**Reviewed so far:** no {what[1]} yet. Open one from the list to record it.")
        a.caption("Kept for this browser session only (no database): download the marks before you close the tab.")
        b.download_button("Download the marks (CSV)", pd.DataFrame(list(marks.values())).to_csv(index=False), file,
                          "text/csv", disabled=not marks, width="stretch")


def item_inputs(it):
    rows = [("Price", money(it.price)), ("Weight", f"{it.weight_kg:,.2f} kg"), ("Volume", f"{it.volume_cm3:,.0f} cm³"),
            ("Chargeable weight", f"{it.chargeable_kg:,.2f} kg"), ("Route", f"{it.seller_state} → {it.customer_state}"),
            ("Distance", f"{it.distance_km:,.0f} km" if pd.notna(it.distance_km) else
             "unknown (zip code not in the Olist table): the model uses the training median"), ("Category", it.category),
            ("Items / sellers in the order", f"{it.n_items} / {it.n_sellers}")]
    if USES_LISTING and "photos_qty" in it:
        rows.append(("Photos, name and description length",
                     f"{it.photos_qty:.0f}, {it.name_len:.0f}, {it.desc_len:.0f} characters"))
    return pd.DataFrame(rows, columns=["Input", "Value"])


def explain_item(it, exclude=None):
    """The 'why this price' panel for a scored item (held-out or uploaded)."""
    why_price(it.category, it.seller_state, it.customer_state, float(it.chargeable_kg),
              float(it.distance_km) if pd.notna(it.distance_km) else None, bool(it.same_state), float(it.expected),
              float(it.expected_ridge), float(it.freight_value), exclude)


# ------------------------------------------------------------------ Problem 2 scores
@st.cache_data(show_spinner=False)
def heldout_orders_scored():
    return ORDERS.assign(day=ORDERS.order_purchase_timestamp.dt.normalize())


@st.cache_data(show_spinner=False)
def daily_flags(share):
    """The queue rule: flag the riskiest `share` of each purchase day's orders (the same rule build_models.py scores)."""
    d = heldout_orders_scored()
    flags = np.zeros(len(d), dtype=int)
    for _, idx in d.groupby("day").indices.items():
        flags[idx] = flag_top_share(d.risk.values[idx], share)
    return flags


def drivers_of(row):
    """One held-out order's driver contributions, as the API returned them (log-odds against an average order)."""
    return row[DRIVER_COLS].iloc[0].rename(lambda c: c.removeprefix("drv_"))


def order_payload(row):
    """API fields of one held-out order. A missing distance is sent as null: the model imputes it, as in training."""
    r = row.iloc[0]
    return dict(promised_days=float(r.promised_days), price_total=float(r.price_total),
                freight_total=float(r.freight_total), n_items=int(r.n_items), n_sellers=int(r.n_sellers),
                seller_state=r.seller_state, customer_state=r.customer_state,
                distance_km=None if pd.isna(r.distance_km) else float(r.distance_km))


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


def promise_curve(base, line, measure):
    """Risk of one order for every promise from 3 to 60 days (one /late/batch call with 58 variants of the order),
    and the shortest promise that falls below `line`.

    measure="pct": risk as a percentile among test-window orders (a new order at checkout);
    measure="risk": the raw score, for a line set by one day's queue.
    """
    days = np.arange(3, 61)
    res = api_post("/late/batch", {"orders": [dict(base, promised_days=float(d)) for d in days]})
    if res is None:
        return None, None
    y = np.array([p["percentile" if measure == "pct" else "risk"] for p in res["predictions"]])
    ok = days[y < line]
    return pd.DataFrame({"days": days, "y": y}), (int(ok.min()) if len(ok) else None)


def promise_section(base, now_y, line, measure, line_label, title):
    """Promise what-if: the lever the Phase 2 analysis found strongest. `base` holds the order's API fields and
    `now_y` its current risk (percentile or score)."""
    curve, safe = promise_curve(base, line, measure)
    if curve is None:
        return
    now = float(base["promised_days"])
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
def start_scenario(name):
    """One-click examples: set the target page's inputs in the session, then open it."""
    ss = st.session_state
    if name == "charge":                                         # the first item on the audit's review list
        d = heldout_items_scored("overall")
        ss.update(f_rule="overall", check_tab=CHECK_TABS[0],
                  item_id=d[d.unusual].sort_values(["both", "excess"], ascending=False).index[0])
        st.switch_page(PAGE_CHECK)
    elif name == "order":
        ss.update(pc_promise=12, pc_s_state="SP", pc_s_zip="01310", pc_c_state="AM", pc_c_zip="69005")
        st.switch_page(PAGE_PROMISE)
    else:
        ss.update(q_day=pd.Timestamp("2018-07-18").date(), q_reveal=True, q_cstate=[], q_action=[])
        st.switch_page(PAGE_QUEUE)


def home():
    st.title("Olist logistics assistant")
    st.markdown(
        "Decision tools for the two teams our project serves, each built on the final model of one of our two problems. "
        "The Olist data end in 2018, so the tools score **held-out historical records as if they were new**: freight "
        "items and orders the models never saw in training.")
    st.caption(f"The models run in a separate service, the [model API]({API_URL}/docs) (FastAPI on Render), which "
               f"this app calls: forms and uploads are scored live, and the population pages show scores the API "
               f"returned in a batch run. See **Model service** for its status and live statistics.")
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
    st.subheader("Try it in one click")
    c = st.columns(3, gap="medium")
    cards = [("charge", "🚩 A suspicious charge", "The held-out item both models flag most strongly: the verdict, why "
              "the fair freight is what it is, and similar items.", "Check the charge"),
             ("order", "📦 A 12-day order to Manaus", "São Paulo (zip 01310) to Manaus (zip 69005), promised in 12 "
              "days: how risky is it, and which promise would keep it off the review list?", "Score the order"),
             ("queue", "📋 The queue of 18 July 2018", "A day's follow-up list with the outcomes revealed: which late "
              "orders the list caught, and why each order is on it.", "Open the queue")]
    for col, (key, title, text, label) in zip(c, cards):
        with col.container(border=True, height="stretch", vertical_alignment="distribute"):
            st.markdown(f"**{title}**")
            st.caption(text)
            if st.button(label, key=f"scenario_{key}", type="primary", width="stretch"):
                start_scenario(key)
    st.page_link(PAGE_MAP, label="Or start from the map: unusual charges and late orders by state", icon="🗺️")

    st.divider()
    st.subheader("How to use the app")
    st.markdown(
        "1. Pick your team's pages in the sidebar. Each page has a **💡 How to use this page** button at the top.\n"
        "2. Lists are clickable: select a row to open its detail (a seller's items, an item's verdict and why its fair "
        "freight is what it is, why an order is on the queue). Filters above the lists narrow them down.\n"
        "3. The data are historical: a *day* in the queue is a day between 26 May and 29 Aug 2018, and *Reveal what "
        "happened* shows the outcome nobody knew at checkout.\n"
        "4. The forms update as you change an input; zip codes place seller and customer exactly, otherwise the "
        "centre of the state is used. The API checks every input and warns when one is unlike the training data.\n"
        "5. Record your verdict on an item or mark an order as contacted: the marks show in the lists and can be "
        "downloaded as CSV, but they are kept for this browser session only. The **Model card** gives the data, "
        "validation, test results and limits of every model.\n"
        "6. The API sleeps after 15 minutes without traffic: the app wakes it as soon as you open it, but the first "
        "result may still take 30–60 seconds. The status in the sidebar shows whether it is awake.")
    st.caption(f"Team 5 · IT5006 · AY2026/27. Code: [GitHub]({REPO_URL}). "
               f"Exploratory analysis (Phase 1): [EDA dashboard]({EDA_URL}).")


# ------------------------------------------------------------------ Analyst: freight audit
UPLOAD_NUM = ["price", "freight_value", "weight_kg", "volume_cm3", "distance_km", "n_items", "n_sellers"]
ZIP_COLS = {"seller_zip_prefix": "seller_zip", "customer_zip_prefix": "customer_zip"}
n_rows = lambda n: f"{n:,} row{'s' if n != 1 else ''}"


def read_upload(file):
    """Read an uploaded CSV of items into API rows; returns (frame, rows, messages) or raises ValueError. Only the
    layout is checked here: the API validates every row, places the parties and flags drift."""
    try:
        u = pd.read_csv(file, dtype={c: str for c in ["seller_zip", "customer_zip", *ZIP_COLS]})
    except Exception as e:
        raise ValueError(f"the file could not be read as CSV ({e}).")
    if u.empty:
        raise ValueError("the file has no rows.")
    u = u.rename(columns=ZIP_COLS).assign(file_row=lambda t: np.arange(len(t)) + 1)
    msgs = []
    if len(u) > MAX_UPLOAD_ROWS:
        msgs.append(f"only the first {MAX_UPLOAD_ROWS:,} of {len(u):,} rows are scored.")
        u = u.head(MAX_UPLOAD_ROWS)
    missing = [c for c in ["price", "freight_value", "weight_kg", "volume_cm3", "seller_state", "customer_state"]
               if c not in u]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing) + ". Download the template for the layout.")
    for c in UPLOAD_NUM:
        if c in u:
            u[c] = pd.to_numeric(u[c], errors="coerce")             # text in a number column -> missing
    zips = {"seller_zip", "customer_zip"} <= set(u)
    if "distance_km" not in u and not zips:
        msgs.append("the file has neither `distance_km` nor both zip-code columns, so distances are measured between "
                    "the centres of the states.")
    no_charge = u.freight_value.isna()
    if no_charge.any():
        msgs.append(f"{n_rows(no_charge.sum())} without a charged freight {'were' if no_charge.sum() != 1 else 'was'} "
                    f"skipped: the audit compares the charge with the fair freight.")
        u = u[~no_charge]
    if u.empty:
        raise ValueError("no row has a charged freight to check.")
    u = u.reset_index(drop=True)
    u["order_id"] = u.order_id.astype(str) if "order_id" in u else [f"row {i}" for i in u.file_row]
    u["seller_id"] = u.seller_id.astype(str) if "seller_id" in u else "unknown"
    fields = [c for c in ["category", *UPLOAD_NUM, "seller_state", "customer_state", "seller_zip", "customer_zip"]
              if c in u]
    rows = []
    for r in u[fields].to_dict("records"):
        # Empty cells are left out, so the API applies its defaults (1 item, 1 seller, category "unknown"). A blank
        # distance with no zip codes stays null: the distance is unknown and the model imputes it, as in training.
        row = {k: v for k, v in r.items() if not (v is None or (isinstance(v, float) and np.isnan(v)))}
        if "distance_km" in r and "distance_km" not in row and not ("seller_zip" in row and "customer_zip" in row):
            row["distance_km"] = None
        rows.append(row)
    return u, rows, msgs


def score_upload(u, rows):
    """Score the uploaded rows through /freight/batch; returns (frame with the audit columns, messages)."""
    rejected, preds = api_batch("/freight/batch", "items", rows)
    if preds is None:
        return None, []
    msgs = []
    if rejected:
        ex = "; ".join(f"row {u.file_row[i]}: {', '.join(e)}" for i, e in list(rejected.items())[:3])
        msgs.append(f"The API rejected {n_rows(len(rejected))}, which {'were' if len(rejected) != 1 else 'was'} "
                    f"skipped ({ex}{'; …' if len(rejected) > 3 else ''}).")
    keep = sorted(preds)
    P = [preds[i] for i in keep]
    d = u.loc[keep].reset_index(drop=True).assign(
        **{c: [p["inputs"][c] for p in P] for c in ["category", "seller_state", "customer_state", "distance_km",
                                                     "chargeable_kg", "same_state", "n_items", "n_sellers"]},
        expected=[p["fair_freight"] for p in P], expected_ridge=[p["fair_freight_ridge"] for p in P],
        audit_band=[p["check"]["audit_band"] for p in P],
        **{f"unusual{'_ridge' if m == 'ridge' else ''}_{rule}": [p["check"]["rules"][rule][m] for p in P]
           for rule in RULES for m in ("forest", "ridge")},
        drift=["; ".join(p["drift_warnings"]) for p in P])
    n_drift = int((d.drift != "").sum())
    if n_drift:
        top = pd.Series([w.split(" ")[0] for p in P for w in p["drift_warnings"]]).value_counts().head(4)
        msgs.append(f"{n_rows(n_drift)} had inputs unlike the training data (drift warnings on "
                    f"{', '.join(f'{k} ({v:,})' for k, v in top.items())}): treat their flags with caution.")
    return d, msgs


AUDIT_FILTERS = {"f_cat": "category", "f_sstate": "seller_state", "f_cstate": "customer_state", "f_band": "band"}


def clear_filters(defaults):
    for k, v in defaults.items():
        st.session_state[k] = v


def audit_filters(d):
    """The audit's filters. With none set every item is kept, so the headline numbers are the notebooks'.
    Returns (filtered items, both-models-only, description of the filters set)."""
    ss = st.session_state
    options = {"f_cat": sorted(set(P1["categories"]) | set(d.category.astype(str))), "f_sstate": STATES,
               "f_cstate": STATES, "f_band": FLAG["band_labels"]}
    for k, opts in options.items():                       # drop values another source had (e.g. an upload's category)
        ss[k] = [v for v in ss[keep(k, [])] if v in opts]
    with st.container(border=True):
        c = st.columns([2.2, 1.2, 1.2, 1.2, 1.9, 0.8], vertical_alignment="bottom")
        c[0].multiselect("Category", options["f_cat"], key="f_cat", placeholder="all categories")
        c[1].multiselect("Seller state", STATES, key="f_sstate", placeholder="all")
        c[2].multiselect("Customer state", STATES, key="f_cstate", placeholder="all")
        c[3].multiselect("Price (R$)", FLAG["band_labels"], key="f_band", placeholder="all")
        c[4].toggle("Both models only", key=keep("f_both", False),
                    help="List only the items both models flag (the review list and a seller's items)")
        c[5].button("Clear", on_click=clear_filters, args=({**{k: [] for k in AUDIT_FILTERS}, "f_both": False},),
                    width="stretch")
    m = np.ones(len(d), dtype=bool)
    set_ = []
    for k, col in AUDIT_FILTERS.items():
        if ss[k]:
            m &= d[col].astype(str).isin(ss[k]).to_numpy()
            set_.append(f"{col.replace('_', ' ')} {', '.join(ss[k])}")
    return d[m], ss.f_both, "; ".join(set_)


def seller_detail(d, sid, overall, both_only):
    """One seller's items, flag rate and excess (the drill-down of the seller table)."""
    s = d[d.seller_id == sid]
    u = s[s.unusual]
    with st.container(border=True):
        st.markdown(f"#### Seller `{sid[:8]}`")
        st.caption(f"Seller ID {sid}")
        k = st.columns(4)
        k[0].metric("Items", f"{len(s):,}")
        k[1].metric("Unusual", f"{len(u):,}", f"{len(u) / len(s):.0%} against {overall:.1%} of all items",
                    delta_color="off")
        k[2].metric("Flagged by both models", f"{int(u.both.sum()):,}")
        k[3].metric("Excess on unusual items", money(u.excess.sum()),
                    f"{u.excess.sum() / s.freight_value.sum():.1%} of their freight" if len(u) else None,
                    delta_color="off")
        marks = st.session_state.get("item_marks", {})
        rows = (s[s.both] if both_only else s).sort_values(["unusual", "both", "excess"], ascending=False)
        rows = rows.assign(flag=np.where(rows.both, "both", np.where(rows.unusual, "forest only", "")),
                           review=item_keys(rows).map(lambda x: marks.get(x, {}).get("mark", "")),
                           order=rows.order_id.str[:8], route=rows.seller_state + " → " + rows.customer_state)
        st.dataframe(rows[["flag", "review", "order", "category", "price", "freight_value", "expected", "excess",
                           "route"]], hide_index=True, width="stretch", height=min(320, 38 + 35 * len(rows)),
                     column_config={
                         "flag": st.column_config.TextColumn("unusual"),
                         "price": st.column_config.NumberColumn("price", format="R$ %.2f"),
                         "freight_value": st.column_config.NumberColumn("charged", format="R$ %.2f"),
                         "expected": st.column_config.NumberColumn("fair freight", format="R$ %.2f"),
                         "excess": st.column_config.NumberColumn("excess", format="R$ %.2f")})
        st.download_button(f"Download seller {sid[:8]}'s items (CSV)",
                           rows.drop(columns=["order", "route"]).round(2).to_csv(index=False),
                           f"seller_{sid[:8]}_items.csv", "text/csv")


def item_detail(it, key, heldout):
    """One item to review (the drill-down of the review list): verdict, inputs, why this price, the analyst's mark."""
    with st.container(border=True):
        when = f" · purchased {it.order_purchase_timestamp:%d %b %Y}" if heldout else ""
        st.markdown(f"#### Order `{it.order_id[:8]}`, item {key.split('/', 1)[1]}")
        st.caption(f"{it.category} · seller `{it.seller_id[:8]}`{when}")
        verdict(it.freight_value, it.expected, it.expected_ridge, error_band(it.price), it.unusual, it.unusual_ridge)
        a, b = st.columns([3, 2], gap="large")
        a.dataframe(item_inputs(it), hide_index=True, width="stretch")
        with b:
            mark_item(it, key)
            if heldout and st.button("Open in Check a charge", icon="🧾", key=f"open_{key}"):
                st.session_state.update(item_id=it.idx, check_tab=CHECK_TABS[0])
                st.switch_page(PAGE_CHECK)
        explain_item(it, exclude=it.idx if heldout else None)


def audit_page():
    st.title("🔍 Freight audit")
    st.markdown(
        "Which sellers and items should the pricing team audit this month? The random forest gives each item's fair "
        "freight; an item is **unusual** when its charge sits in the top 5% above it. The Ridge model gives an "
        "independent second opinion, and items both models flag are reviewed first.")
    how_to(
        "1. Choose the items: all held-out items (the test set, scored by the model API in a batch run), or a CSV of "
        f"your own with the charged freight, which the app sends to the API (up to {MAX_UPLOAD_ROWS:,} rows; download "
        "the template for the columns; `distance_km` may be replaced by `seller_zip_prefix` and "
        "`customer_zip_prefix`). Rows the API rejects are skipped and listed, and rows with inputs unlike the "
        "training data are counted.\n"
        "2. Choose the rule. *Top 5% overall* is the rule the notebooks evaluated; *top 5% within each price band*, "
        "the one the report recommends, keeps expensive items, which are harder to predict, from crowding the list.\n"
        "3. Narrow the items with the filters (category, seller or customer state, price); with none set the numbers "
        "are those of the whole test set. The **Map by state** page opens this page filtered to one state.\n"
        "4. **Sellers to audit** ranks sellers by the *share* of their items that are unusual (with a minimum number "
        "of items), not by volume: large sellers mostly price freight consistently. Click a seller to see their items.\n"
        "5. **Items to review** lists the unusual items, those flagged by both models first. Click an item to see the "
        "verdict, why its fair freight is what it is, similar items, and to record your verdict (overcharge, "
        "justified or data error). Download either list, and your marks.\n\n"
        "A flag is a reason to look, not proof of overcharging: fragile goods, data errors or cost recovery can also "
        "explain a high charge. Your marks are kept for this browser session only.")
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
        key = getattr(file, "file_id", None) or f"{file.name}:{file.size}"
        if st.session_state.get("upload_key") != key:              # score each file once, not on every rerun
            try:
                u, rows, msgs = read_upload(file)
            except ValueError as e:
                st.error(f"The upload could not be scored: {e}")
                return
            scored, more = score_upload(u, rows)
            if scored is None:
                return
            st.session_state.upload_key, st.session_state.upload = key, (scored, msgs + more)
        scored, msgs = st.session_state.upload
        for m in msgs:
            st.warning(m)
        if scored.empty:
            st.error("No row of the file could be scored.")
            return
        d = add_scores(scored, rule)
    else:
        d = heldout_items_scored(rule)
    heldout = src == "all held-out items"
    n_all = len(d)
    d, both_only, active = audit_filters(d)
    if active:
        st.caption(f"Filtered to {len(d):,} of {n_all:,} items: {active}.")
    if d.empty:
        st.warning("No item matches these filters: clear one of them.")
        return
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
        # the selection resets whenever the list changes, so a selected row never points at another seller
        view_key = f"{src}|{rule}|{active}|{min_items}|{st.session_state.get('upload_key')}"
        st.caption("👆 Click a seller to see their items.")
        ev = st.dataframe(table[["seller", "items", "n_unusual", "rate", "n_both", "excess", "markup"]],
                          hide_index=True, width="stretch", height=260, on_select="rerun",
                          selection_mode="single-row", key=f"sel_seller|{view_key}", column_config={
                              "n_unusual": st.column_config.NumberColumn("unusual"),
                              "rate": st.column_config.NumberColumn("flag rate", format="percent"),
                              "n_both": st.column_config.NumberColumn("both models"),
                              "excess": st.column_config.NumberColumn("excess (R$)", format="R$ %.2f"),
                              "markup": st.column_config.NumberColumn("median markup", format="%.2f×")})
        st.download_button("Download the seller list (CSV)", table.drop(columns="seller").to_csv(index=False),
                           "sellers_to_audit.csv", "text/csv")
        sel = [r for r in ev.selection.rows if r < len(table)]
        if sel:
            seller_detail(d, table.seller_id.iloc[sel[0]], overall, both_only)

    st.subheader("2 · Items to review")
    marks = st.session_state.get("item_marks", {})
    items = (f[f.both] if both_only else f).sort_values(["both", "excess"], ascending=False)
    items = (items.assign(key=item_keys(items)).rename_axis("idx").reset_index()
             .assign(priority=lambda t: np.arange(1, len(t) + 1), order=lambda t: t.order_id.str[:8],
                     seller=lambda t: t.seller_id.str[:8], models=lambda t: np.where(t.both, "both", "forest only"),
                     review=lambda t: t.key.map(lambda x: marks.get(x, {}).get("mark", ""))))
    view_key = f"{src}|{rule}|{active}|{both_only}|{st.session_state.get('upload_key')}"
    st.caption("👆 Click an item to see the verdict, why its fair freight is what it is, and to record your review.")
    ev = st.dataframe(items[["priority", "review", "models", "freight_value", "expected", "excess", "markup", "order",
                             "seller", "band", "category"]], hide_index=True, width="stretch", height=320,
                      on_select="rerun", selection_mode="single-row", key=f"sel_item|{view_key}", column_config={
                          "band": st.column_config.TextColumn("price (R$)"),
                          "freight_value": st.column_config.NumberColumn("charged", format="R$ %.2f"),
                          "expected": st.column_config.NumberColumn("fair freight", format="R$ %.2f"),
                          "excess": st.column_config.NumberColumn("excess", format="R$ %.2f"),
                          "markup": st.column_config.NumberColumn("markup", format="%.2f×")})
    out = items[["priority", "review", "models", "order_id", "seller_id", "category", "price", "freight_value",
                 "expected", "expected_ridge", "excess", "markup", "weight_kg", "distance_km", "seller_state",
                 "customer_state"] + (["drift"] if "drift" in items else [])]
    st.download_button("Download the review list (CSV)", out.round(2).to_csv(index=False), "items_to_review.csv",
                       "text/csv")
    marks_summary("item_marks", ("item", "items"), "freight_review_marks.csv")
    sel = [r for r in ev.selection.rows if r < len(items)]
    if sel:
        it = items.iloc[sel[0]]
        item_detail(it, it.key, heldout)

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
def verdict(charged, expected, expected_ridge, band, model_flag, ridge_flag, narrow=False):
    """The charge against the fair freight, and the unusual-freight verdict (the flags are the API's). `narrow`
    puts the four figures in two rows, for a column beside the inputs."""
    c = st.columns(2) + st.columns(2) if narrow else st.columns(4)
    c[0].metric("Charged", money(charged))
    c[1].metric("Fair freight", money(expected))
    c[2].metric("Difference", money(charged - expected),
                f"{(charged / expected - 1) * 100:+.0f}%" if expected > 0 else None, delta_color="inverse")
    c[3].metric(f"Typical error at R$ {band['band']}", f"± {money(band['mae'])}")
    second =(f"The Ridge model agrees (it expects {md_money(expected_ridge)})." if ridge_flag == model_flag else
              f"The Ridge model disagrees (it expects {md_money(expected_ridge)}).")
    if model_flag:
        st.error(f"🚩 **Unusual freight:** {md_money(charged)} is in the top 5% above the fair freight of "
                 f"{md_money(expected)}. {second} " + ("Review it first." if ridge_flag else "Review it after the "
                                                                                             "items both models flag."))
    else:
        st.success(f"✅ **Within the normal range** for an item like this. {second}")


CHECK_TABS = ["A held-out item", "A new listing (quote)"]


def check_page():
    st.title("🧾 Check a charge")
    st.markdown("Is one charge fair, and what should a new listing be quoted? Use it when a seller disputes a flag, "
                "a buyer complains about freight, or a new listing needs a sanity check.")
    rule = rule_picker()
    # lazy tabs: only the open tab runs, so a held-out item never waits for the API
    t1, t2 = st.tabs(CHECK_TABS, key="check_tab", on_change="rerun")
    if t1.open:
        with t1:
            check_item(rule)
    if t2.open:
        with t2:
            quote_listing(rule)


def check_item(rule):
    d = heldout_items_scored(rule)
    how_to("Draw a random held-out item, paste an order ID from the audit's review list, or open one from the audit "
           "(click an item, then **Open in Check a charge**). Under the verdict: what the model used, your verdict "
           "on the charge (kept for this session), and why the fair freight is what it is: the simple tariff step by "
           "step against the two models, and similar held-out items with their charges.")
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
    verdict(it.freight_value, it.expected, it.expected_ridge, error_band(it.price), it.unusual, it.unusual_ridge)
    a, b = st.columns([3, 2], gap="large")
    a.dataframe(item_inputs(it), hide_index=True, width="stretch")
    with b:
        mark_item(it, f"{it.order_id}/{it.order_item_id}")
        n = len(st.session_state.get("item_marks", {}))
        st.caption(f"{n} item{'s' if n != 1 else ''} reviewed so far this session; the list and the CSV are on the "
                   f"*Freight audit* page.")
    explain_item(it, exclude=it.name)


@st.fragment
def quote_listing(rule):
    """A new listing: the inputs on the left, the API's answer on the right, updated as the inputs change. A fragment,
    so a change here reruns only this tab; the answers are remembered per input (api_post)."""
    how_to("Describe the item and the route on the left: the result on the right updates as you change an input (the "
           "app sends the listing to the model API, `POST /freight/predict`, and remembers each answer). It shows the "
           "fair freight and a quote range (fair freight ± the typical error at that price); enter the charged "
           "freight to check it as well. Below: how the fair freight changes with the weight and the distance, and "
           "why the price is what it is. The API rejects impossible inputs with a reason, and warns when an input is "
           "unlike the training data.")
    a, b = st.columns([2, 3], gap="large")
    with a:
        cat = st.selectbox("Product category", P1["categories"], key=keep("ql_cat", P1["categories"][0]))
        price = st.number_input("Item price (R$)", min_value=0.0, step=10.0, key=keep("ql_price", 120.0))
        weight_g = st.number_input("Weight (g)", min_value=0.0, step=100.0, key=keep("ql_weight", 1500.0))
        d1, d2, d3 = st.columns(3)
        length = d1.number_input("Length (cm)", min_value=0.0, key=keep("ql_len", 30.0))
        width = d2.number_input("Width (cm)", min_value=0.0, key=keep("ql_wid", 20.0))
        height = d3.number_input("Height (cm)", min_value=0.0, key=keep("ql_hei", 15.0))
        s_zip, s_state = place_inputs("ql_s", "Seller")
        c_zip, c_state = place_inputs("ql_c", "Customer", "RJ")
        e1, e2 = st.columns(2)
        n_items = e1.number_input("Items in the order", min_value=1, key=keep("ql_items", 1))
        n_sellers = e2.number_input("Sellers in the order", min_value=1, key=keep("ql_sellers", 1))
        charged = st.number_input("Freight charged (R$, optional: leave 0 to skip the check)", min_value=0.0,
                                  key=keep("ql_charged", 0.0))
    remember()
    payload =dict(category=cat, price=price, weight_kg=weight_g / 1000, volume_cm3=length * width * height,
                   n_items=int(n_items), n_sellers=int(n_sellers), **route_payload(s_zip, s_state, c_zip, c_state))
    if charged > 0:
        payload["freight_value"] = charged
    with b:
        p = api_post("/freight/predict", payload)
        if p is None:
            return
        show_notes(p)
        x = p["inputs"]
        st.markdown(f"**{cat}**, {x['weight_kg']:,.2f} kg, {x['volume_cm3']:,.0f} cm³, {x['seller_state']} → "
                    f"{x['customer_state']} ({x['distance_km']:,.0f} km)")
        c = st.columns(3)
        c[0].metric("Fair freight", money(p["fair_freight"]))
        c[1].metric("Quote range (R$)", f"{p['quote_low']:,.0f} – {p['quote_high']:,.0f}",
                    help=f"Fair freight ± the average test-set error for items priced R$ {p['price_band']}")
        c[2].metric("Simple tariff", money(p["tariff"]))
        if p["check"]:
            r = p["check"]["rules"][rule]
            verdict(charged, p["fair_freight"], p["fair_freight_ridge"],
                    dict(band=p["price_band"], mae=p["band_mae"]), r["forest"], r["ridge"], narrow=True)
            other = "band" if rule == "overall" else "overall"
            st.caption(f"Under the other rule ({RULES[other]}): "
                       f"{p['check']['rules'][other]['verdict'].replace('_', ' ')}.")
        else:
            st.info("Enter the freight charged (bottom left) to check it against the fair freight.")
        st.caption(f"API request `{p['request_id']}`, answered in {p['latency_ms']:.0f} ms. "
                   f"For a listing the model has never seen, the average error is {md_money(NB['p1_unseen_mae'])} "
                   f"({NB['p1_unseen_pct']}% of their mean freight; Phase 3 notebook, Section 6).")
    freight_curves(p, charged or None)
    why_price(cat, x["seller_state"], x["customer_state"], x["chargeable_kg"], x["distance_km"], x["same_state"],
              p["fair_freight"], p["fair_freight_ridge"], charged or None)


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
           "left blank. Use it to see which lanes a simple tariff would misprice before publishing one. The fair "
           "freight of every held-out item comes from the model API's batch run.")
    d = ITEMS
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
        f"3. Narrow the list by customer state or follow-up type (the **Map by state** page opens it for one state). "
        f"The list itself is always the riskiest share of the *whole* day.\n"
        f"4. Click an order to see why it is on the list and what promise would have kept it off (the app asks the "
        f"model API for the order's risk under every promise from 3 to 60 days). Mark it **contacted** when the "
        f"follow-up is done: the mark shows in the list and can be downloaded, for this browser session only.\n"
        f"5. Download the list for the team, or switch on **Reveal what happened** to see the real outcomes.\n\n"
        f"The scores, drivers and follow-ups of the test window come from the model API's batch run.")
    d = heldout_orders_scored()
    days = d.day.drop_duplicates().sort_values()
    c = st.columns([1, 1, 1])
    default = days[d.groupby("day").size().reindex(days).values >= 150].iloc[0]
    day = c[0].date_input("Orders placed on", min_value=days.min().date(), max_value=days.max().date(),
                          key=keep("q_day", default.date()))
    share = c[1].select_slider("Review capacity (share of the day's orders)", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                               format_func=lambda v: f"{v:.0%}", key=keep("q_share", P2["share"]))
    reveal = c[2].toggle("Reveal what happened", key=keep("q_reveal", False))
    follow_ups = list(dict.fromkeys(ACTIONS.values()))
    with st.container(border=True):
        c = st.columns([2, 2.6, 1.6, 0.8], vertical_alignment="bottom")
        states = c[0].multiselect("Customer state", STATES, key=keep("q_cstate", []), placeholder="all")
        acts = c[1].multiselect("Suggested follow-up", follow_ups, key=keep("q_action", []), placeholder="all")
        only = c[2].toggle("Show only the orders to check", key=keep("q_only", True))
        c[3].button("Clear", on_click=clear_filters, args=({"q_cstate": [], "q_action": []},), width="stretch")
    flags = daily_flags(share)
    y = d.clf_target.values
    precision = (flags & y).sum() / max(flags.sum(), 1)
    mask = (d.day == pd.Timestamp(day)).values
    whole = d[mask].assign(flag=flags[mask]).sort_values("risk", ascending=False)
    if whole.empty:
        st.info("No delivered orders were placed on that day.")
        return
    whole["rank"] = np.arange(1, len(whole) + 1)                # rank in the whole day's list
    line = float(whole.loc[whole.flag == 0, "risk"].max())       # off the list = below the day's riskiest unflagged
    def keep_rows(t):
        m = np.ones(len(t), dtype=bool)
        if states:
            m &= t.customer_state.isin(states).to_numpy()
        if acts:
            m &= t.action.isin(acts).to_numpy()
        return m

    today = whole[keep_rows(whole)]
    filters = "; ".join(([f"customers in {', '.join(states)}"] if states else []) +
                        ([f"follow-up {', '.join(acts)}"] if acts else []))
    if filters:
        st.caption(f"Filtered to {len(today):,} of the day's {len(whole):,} orders: {filters}.")
    if today.empty:
        st.info("No order of that day matches these filters: clear one of them or pick another day.")
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
        st.warning(f"No order to check among these {len(today)} orders: a {share:.0%} capacity of the day's "
                   f"{len(whole)} orders" + (" flags none of them; clear a filter, " if filters else " rounds to nobody; ")
                   + "raise the capacity or pick a busier day.")
        return
    marks = st.session_state.get("order_marks", {})
    show = today.assign(order=today.order_id.str[:8], placed=today.order_purchase_timestamp.dt.strftime("%H:%M"),
                        route=today.seller_state + " → " + today.customer_state,
                        action=today.action.where(today.flag == 1, ""),
                        contacted=today.order_id.map(lambda o: "✓" if o in marks else ""))
    cols = ["rank", "contacted", "action", "order", "placed", "route", "distance_km", "promised_days", "price_total",
            "freight_total", "risk"]
    if reveal:
        show["outcome"] = np.where(show.clf_target == 1, "late by " + show.days_late.astype(int).astype(str) + " d",
                                   "on time")
        show["review"] = show.low_review.map({1.0: "1–2 stars", 0.0: "3–5 stars"}).fillna("none")
        cols = cols[:3] + ["outcome", "review"] + cols[3:]
    listed = (show[show.flag == 1] if only else show).reset_index(drop=True)
    st.caption("👆 Click an order to see why it is on the list and to mark it contacted.")
    ev = st.dataframe(listed[cols], hide_index=True, width="stretch", height=380, on_select="rerun",
                      selection_mode="single-row", key=f"sel_order|{day}|{share}|{only}|{filters}",
                      column_config={
                          "contacted": st.column_config.TextColumn("contacted", width="small"),
                          "action": st.column_config.TextColumn("suggested follow-up", width="medium"),
                          "distance_km": st.column_config.NumberColumn("distance (km)", format="%.0f"),
                          "promised_days": st.column_config.NumberColumn("promised (days)", format="%.0f"),
                          "price_total": st.column_config.NumberColumn("order value", format="R$ %.2f"),
                          "freight_total": st.column_config.NumberColumn("freight", format="R$ %.2f"),
                          "risk": st.column_config.ProgressColumn("risk score", format="%.3f", min_value=0.0,
                                                                  max_value=float(d.risk.max()))})
    out = show[show.flag == 1][["rank", "contacted", "action", "order_id", "seller_id", "order_purchase_timestamp",
                                "seller_state", "customer_state", "distance_km", "promised_days", "price_total",
                                "freight_total", "risk"]]
    st.download_button(f"Download the list for {pd.Timestamp(day):%d %b %Y} (CSV)", out.round(3).to_csv(index=False),
                       f"late_risk_queue_{pd.Timestamp(day):%Y-%m-%d}.csv", "text/csv")
    marks_summary("order_marks", ("order contacted", "orders contacted"), "contacted_orders.csv")

    st.subheader("Why is an order on the list?")
    sel = [r for r in ev.selection.rows if r < len(listed)]
    row = listed.iloc[[sel[0]]] if sel else flagged.head(1)          # default: the riskiest order to check
    r = row.iloc[0]
    status = "on the list" if r.flag else "not on the list"
    st.markdown(f"**Order `{r.order_id[:8]}`** · rank {r['rank']} of {len(whole):,} that day ({status}) · "
                f"{r.seller_state} → {r.customer_state} · promised {r.promised_days:.0f} days · risk score {r.risk:.3f}"
                + ("" if sel else " · the riskiest order to check: click another in the list"))
    if reveal:
        st.caption("What happened: " + (f"late by {int(r.days_late)} days" if r.clf_target else "on time") + "; "
                   + {1.0: "a 1–2 star review", 0.0: "a 3–5 star review"}.get(r.low_review, "no review") + ".")
    a, b = st.columns(2, gap="large")
    with a:
        st.plotly_chart(drivers_chart(drivers_of(row), f"What drives order {r.order_id[:8]}'s risk"), width="stretch")
        st.markdown(f"Suggested follow-up: **{r.action}**." if r.flag else
                    "Not on today's list, so no follow-up is due.")
        mark_order(r)
    with b:
        promise_section(order_payload(row), float(r.risk), line, "risk",
                        f"today's review line ({share:.0%} capacity)", "What promise would have kept it off the list?")

    if reveal:
        st.divider()
        caught = (flags & y).sum()
        st.markdown(
            f"**Across the whole test window**, checking the riskiest {share:.0%} of each day's orders catches "
            f"**{caught / y.sum():.1%}** of late orders ({caught} of {y.sum()}); {share:.0%} random checks would "
            f"catch about {share:.0%}. Ranking the whole window at once, as the Phase 2 evaluation did, catches "
            f"{P2['test']['Recall']:.1%} at 10%: late orders cluster on bad days, which a daily list cannot see.")
        if filters:
            wm = keep_rows(d)
            c_f, y_f = int((flags & y)[wm].sum()), int(y[wm].sum())
            st.markdown(f"For the filtered orders ({filters}), the daily lists caught **{c_f} of {y_f}** late "
                        f"orders" + (f" ({c_f / y_f:.1%})." if y_f else "."))
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
    how_to("Describe the order as it looks at checkout on the left: the result on the right updates as you change an "
           "input. Drag the **promise** slider to move the order along its risk curve; the curve itself is asked for "
           "once per order (the order under every promise from 3 to 60 days), and each answer is remembered. The "
           "result places the order's risk among the orders of the test window, finds the shortest promise that "
           "keeps it below the review line, and shows which inputs push the risk up or down.")
    promise_tool()


@st.fragment
def promise_tool():
    """The order's inputs on the left and its risk on the right, updated as the inputs change (a fragment, so only
    this part reruns). Moving the promise asks for one /late/predict; the promise curve is fetched once per order."""
    a, b = st.columns([2, 3], gap="large")
    with a:
        promised = st.slider("Promised delivery time (days from purchase)", 3, 60, key=keep("pc_promise", 20),
                             help="Move it to see the order's risk under another promise")
        value = st.number_input("Order value, all items (R$)", min_value=0.0, step=10.0, key=keep("pc_value", 150.0))
        freight = st.number_input("Freight charged, all items (R$)", min_value=0.0, step=1.0,
                                  key=keep("pc_freight", 25.0))
        e1, e2 = st.columns(2)
        n_items = e1.number_input("Items", min_value=1, key=keep("pc_items", 1))
        n_sellers = e2.number_input("Sellers", min_value=1, key=keep("pc_sellers", 1))
        s_zip, s_state = place_inputs("pc_s", "Seller")
        c_zip, c_state = place_inputs("pc_c", "Customer", "RJ")
        share = st.select_slider("Review capacity", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30], format_func=lambda v: f"{v:.0%}",
                                 key=keep("pc_share", P2["share"]), help="The share of orders the team can check")
    remember()
    order =dict(promised_days=float(promised), price_total=value, freight_total=freight, n_items=int(n_items),
                 n_sellers=int(n_sellers), **route_payload(s_zip, s_state, c_zip, c_state))
    with b:
        p = api_post("/late/predict", order, params={"share": share})
        if p is None:
            return
        show_notes(p)
        x = p["inputs"]
        st.markdown(f"**{x['seller_state']} → {x['customer_state']}**, {x['distance_km']:,.0f} km, promised "
                    f"{promised} days")
        c = st.columns(3)
        c[0].metric("Risk score", f"{p['risk']:.3f}")
        c[1].metric("Risk percentile", f"{p['percentile']:.0f}th", help="Among the orders of the test window")
        c[2].metric("Review line", f"{100 * (1 - share):.0f}th", help=f"The riskiest {share:.0%} of orders are checked")
        if p["on_review_list"]:
            st.error(f"🔎 **On the review list:** it would be in the riskiest {share:.0%} of orders. Suggested "
                     f"follow-up: **{p['suggested_follow_up']}**.")
        else:
            st.success(f"✅ **Below the review line** for a {share:.0%} capacity.")
        promise_section(order, p["percentile"], 100 * (1 - share), "pct", f"review line (riskiest {share:.0%})",
                        "Risk for every promise (the dot is this order)")
        st.plotly_chart(drivers_chart(pd.Series(p["drivers"]), "What drives this order's risk"), width="stretch")
        st.caption(f"API request `{p['request_id']}`, answered in {p['latency_ms']:.0f} ms.")


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


# ------------------------------------------------------------------ Both teams: map by state
MAP_VIEWS = {"seller": "Unusual freight, by seller state", "customer": "Unusual freight, by customer state",
             "late": "Late orders, by customer state", "listed": "Orders on the daily list, by customer state"}
MAP_MIN = 30                                    # fewer items or orders than this: the state is greyed out
MAP_SCALE = [[0, "#e8f0fb"], [0.35, "#9cc0ec"], [0.7, BLUE], [1, DIV_LOW]]


@st.cache_data(show_spinner=False)
def brazil_geojson():
    """The 27 states, simplified; each feature's id is its two-letter code."""
    return json.loads((HERE / "data" / "brazil_states.geojson").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def state_table(view, rule):
    """One row per state: the count, the share shown on the map and the summary figures."""
    if view in ("seller", "customer"):
        d = heldout_items_scored(rule)
        g = (d.assign(excess_u=np.where(d.unusual, d.excess, 0.0))
             .groupby(f"{view}_state").agg(n=("unusual", "size"), hits=("unusual", "sum"), both=("both", "sum"),
                                           excess=("excess_u", "sum"), freight=("freight_value", "sum")))
        g["value"] = g.hits / g.n
        national = float(d.unusual.mean())
    else:
        d = heldout_orders_scored()
        flags = daily_flags(P2["share"])
        g = (d.assign(listed=flags, caught=flags & d.clf_target.values)
             .groupby("customer_state").agg(n=("clf_target", "size"), late=("clf_target", "sum"),
                                            listed=("listed", "sum"), caught=("caught", "sum")))
        g["value"] = (g.late if view == "late" else g.listed) / g.n
        national = float(d.clf_target.mean() if view == "late" else flags.mean())
    return g.reindex(STATES), national


def open_filtered(page, **state):
    """Open the audit or the queue filtered to one state (the filters are session state, see KEEP)."""
    st.session_state.update(state)
    st.switch_page(page)


def map_page():
    st.title("🗺️ Map by state")
    st.markdown("Where do unusual freight charges and late orders concentrate? Each state is coloured by the share of "
                "its held-out items or orders; click a state for its figures and to open the freight audit or the "
                "risk queue filtered to it.")
    how_to(f"1. Pick what to map. *Unusual freight* uses the analyst's rule (top of the page) and either the seller's "
           f"or the customer's state; *late orders* shows what happened in the test window; *orders on the daily "
           f"list* shows the share the daily {P2['share']:.0%} list would have checked.\n"
           f"2. Click a state (or pick it from the list beside the map) for its figures.\n"
           f"3. **Open the freight audit** or **Open the risk queue** carries the state over as a filter; clear it "
           f"there with **Clear**.\n\n"
           f"States with fewer than {MAP_MIN} items or orders are grey: their shares are too noisy to compare.")
    keys = list(MAP_VIEWS)
    view = st.radio("Show", keys, format_func=MAP_VIEWS.get, horizontal=True, key=keep("map_view", "seller"))
    freight = view in ("seller", "customer")
    rule = rule_picker() if freight else st.session_state.get("f_rule", "overall")
    g, national = state_table(view, rule)
    ok = g[g.n >= MAP_MIN]
    geo = brazil_geojson()
    noun = "items" if freight else "orders"
    label = {"seller": "unusual", "customer": "unusual", "late": "late", "listed": "on the daily list"}[view]
    # the state shown: a new click on the map picks it (read before the map is drawn, so it can be outlined), and the
    # list beside the map can pick another without the earlier click overriding it
    chart_key = f"sel_map|{view}|{rule}"
    points = (st.session_state.get(chart_key) or {}).get("selection", {}).get("points", [])
    clicked = [p["location"] for p in points if p.get("location")]
    if clicked and clicked[0] != st.session_state.get("map_click"):
        st.session_state.map_state = clicked[0]
    st.session_state.map_click = clicked[0] if clicked else None
    pick = st.session_state[keep("map_state", "SP")]
    plain = dict(marker=dict(opacity=1))                        # no fading of the states not clicked
    fig = go.Figure()
    fig.add_choropleth(geojson=geo, locations=STATES, z=[0] * len(STATES), colorscale=[[0, GRID], [1, GRID]],
                       showscale=False, marker_line=dict(color="white", width=1), customdata=g.n.fillna(0).values,
                       selected=plain, unselected=plain,
                       hovertemplate=f"%{{location}}: %{{customdata:,.0f}} {noun} (too few to compare)<extra></extra>")
    fig.add_choropleth(geojson=geo, locations=ok.index, z=ok.value.values, colorscale=MAP_SCALE, zmin=0,
                       zmax=float(ok.value.max()), marker_line=dict(color="white", width=1), customdata=ok.n.values,
                       colorbar=dict(title="", tickformat=".0%", thickness=12, len=0.7), selected=plain,
                       unselected=plain,
                       hovertemplate=f"%{{location}}: %{{z:.1%}} {label} of %{{customdata:,.0f}} {noun}<extra></extra>")
    fig.add_choropleth(geojson=geo, locations=[pick], z=[0], colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                       showscale=False, marker_line=dict(color=INK, width=2.5), hoverinfo="skip")
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.plotly_chart(layout(fig, h=560, title=f"{MAP_VIEWS[view]} (all states {national:.1%})")
                        .update_layout(dragmode=False), on_select="rerun", selection_mode="points", key=chart_key,
                        config={"scrollZoom": False, "displayModeBar": False})
    with b:
        st.selectbox("State", STATES, key="map_state", help="Or click it on the map")
        r = g.loc[pick]
        n = 0 if pd.isna(r.n) else int(r.n)
        with st.container(border=True):
            st.markdown(f"#### {next(f['properties']['name'] for f in geo['features'] if f['id'] == pick)} ({pick})")
            if n == 0:
                st.markdown(f"No held-out {noun} for this state.")
            elif freight:
                k = st.columns(2)
                k[0].metric("Items", f"{n:,}")
                k[1].metric("Unusual", f"{int(r.hits):,}", f"{r.value:.1%} against {national:.1%}", delta_color="off")
                k = st.columns(2)
                k[0].metric("Flagged by both models", f"{int(r.both):,}")
                k[1].metric("Excess on unusual items", money(r.excess, 0),
                            f"{r.excess / r.freight:.1%} of the freight" if r.freight else None, delta_color="off")
            else:
                k = st.columns(2)
                k[0].metric("Orders", f"{n:,}")
                k[1].metric("Late", f"{int(r.late):,}", f"{r.late / n:.1%} against "
                            f"{heldout_orders_scored().clf_target.mean():.1%}", delta_color="off")
                k = st.columns(2)
                k[0].metric(f"On the daily {P2['share']:.0%} list", f"{int(r.listed):,}",
                            f"{r.listed / n:.1%} of its orders", delta_color="off")
                k[1].metric("Late orders caught", f"{int(r.caught):,} of {int(r.late):,}")
            if 0 < n < MAP_MIN:
                st.caption(f"Fewer than {MAP_MIN} {noun}: the share is too noisy to compare.")
            side = "seller" if view == "seller" else "customer"
            if st.button(f"Open the freight audit for {side}s in {pick}", icon="🔍", width="stretch"):
                open_filtered(PAGE_AUDIT, f_cat=[], f_band=[], f_both=False,
                              f_sstate=[pick] if side == "seller" else [], f_cstate=[pick] if side == "customer" else [])
            if st.button(f"Open the risk queue for customers in {pick}", icon="⏰", width="stretch"):
                open_filtered(PAGE_QUEUE, q_cstate=[pick], q_action=[])
    with st.expander("Show as a table"):
        t = g.assign(state=g.index).dropna(subset=["n"])
        st.dataframe(t[["state", "n", "value"]].sort_values("value", ascending=False), hide_index=True,
                     width="stretch", column_config={
                         "n": st.column_config.NumberColumn("items" if freight else "orders", format="%d"),
                         "value": st.column_config.NumberColumn(f"share {label}", format="percent")})


# ------------------------------------------------------------------ Model card
def model_card():
    st.title("📋 Model card")
    st.markdown("What each model was trained on, how it was validated and where it should not be trusted. Test "
                "results are recomputed by `build_models.py` from the model files the API loads; cross-validation and "
                "robustness results are quoted from the notebooks (Problem 1: Phase 3; Problem 2: Phase 2).")
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
               f"scikit-learn {b_['scikit_learn']}, seed {b_['seed']}, and served by the model API (see *Model "
               f"service*). Rebuild: `OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/model_app/build_models.py`, "
               f"then `python deployment/api/batch_score.py`.")


# ------------------------------------------------------------------ About: the model service
def service_page():
    st.title("🛰️ Model service")
    st.markdown(
        "The models do not run inside this app. They run in a separate **model API** (FastAPI, hosted on Render), "
        "which loads them once when it starts; this app is its frontend and sends every form and upload to it as "
        "JSON. Other clients use the same API: the batch job that scores the test period for the audit, tariff, queue "
        "and monitoring pages, and anyone with a script or a notebook.")
    st.markdown(
        "| Part | Where | What it does |\n|---|---|---|\n"
        "| Model API | Render, `deployment/api` | loads the three pipelines once; validates every input; returns "
        "predictions with drift warnings; logs every prediction |\n"
        "| This app | Streamlit Community Cloud, `deployment/model_app` | forms, uploads, lists and charts; no model "
        "files |\n"
        "| Batch job | `deployment/api/batch_score.py` | scores every held-out record through the API for the "
        "population pages, and checks the scores against the build |")
    api_health.clear()
    ok, info = api_health()
    st.subheader("Status")
    c = st.columns(4)
    c[0].metric("Model API", "online" if ok else "asleep")
    if ok:
        up = info["uptime_s"]
        c[1].metric("Response time", f"{info['ms']:.0f} ms")
        c[2].metric("Version", info["version"], f"models built {info['models_built']}", delta_color="off")
        c[3].metric("Up for", f"{up // 3600} h {up % 3600 // 60} min" if up >= 3600 else f"{up // 60} min {up % 60} s")
    st.markdown(f"Address: `{API_URL}` · [interactive docs]({API_URL}/docs) · [model info]({API_URL}/model-info) · "
                f"[log summary]({API_URL}/logs/summary)")
    st.markdown(
        "| Method | Path | Used by |\n|---|---|---|\n"
        "| GET | `/` | the status in the sidebar |\n"
        "| GET | `/model-info` | this page |\n"
        "| POST | `/freight/predict` | Check a charge, *A new listing* |\n"
        "| POST | `/freight/batch` | Freight audit uploads; the freight curves of *A new listing*; the batch job |\n"
        "| POST | `/late/predict` | Promise check |\n"
        "| POST | `/late/batch` | the promise what-if curves; the batch job |\n"
        "| GET | `/logs/summary` | this page |")
    if not ok:
        st.info("The API sleeps after 15 minutes without traffic (Render's free tier). Wake it to see its statistics.")
        if not st.button("Wake the API", type="primary"):
            return
    info, logs = api_get("/model-info"), api_get("/logs/summary")
    if info is None or logs is None:
        return

    st.subheader("Model statistics (`/model-info`)")
    m = info["models"]
    st.dataframe(pd.DataFrame([
        ("Forest (fair freight)", m["freight_forest"]["estimator"], f"MAE R$ {m['freight_forest']['test']['MAE_rs']:.2f}"
         f" ({m['freight_forest']['test']['MAE_pct']}% of mean freight), R² {m['freight_forest']['test']['R2_rs']:.3f}"),
        ("Ridge (second opinion)", m["freight_ridge"]["estimator"], f"MAE R$ {m['freight_ridge']['test']['MAE_rs']:.2f}, "
         f"R² {m['freight_ridge']['test']['R2_rs']:.3f}"),
        ("Logistic (late delivery)", m["late_logit"]["estimator"], f"lift {m['late_logit']['test']['Lift']:.2f}, "
         f"ROC-AUC {m['late_logit']['test']['ROC_AUC']:.3f}, recall at 10% {m['late_logit']['test']['Recall']:.1%}")],
        columns=["Model", "Estimator", "Test results"]), hide_index=True, width="stretch")
    rt = info["runtime"]
    st.caption(f"Served with Python {rt['python']}, scikit-learn {rt['scikit_learn']}, numpy {rt['numpy']}, pandas "
               f"{rt['pandas']}; models built {info['built']['date']}. Batches of up to "
               f"{info['validation']['max_batch']:,} rows. Drift: {info['drift']['method']}")

    st.subheader("Live monitoring (`/logs/summary`)")
    st.caption(logs["storage"])
    n, req = logs["predictions"], logs["requests"]
    k = st.columns(4)
    k[0].metric("Predictions logged", f"{n['total']:,}", f"{n['freight']:,} freight · {n['late']:,} late",
                delta_color="off")
    k[1].metric("Requests", f"{req['total']:,}", f"{req['rejected_422']:,} rejected inputs", delta_color="off")
    k[2].metric("With drift warnings", f"{logs['drift']['rate']:.1%}" if logs["drift"]["rate"] is not None else "–")
    lat = logs["latency_ms"]
    k[3].metric("Latency, median", f"{lat['p50']:.0f} ms" if lat else "–",
                f"95th percentile {lat['p95']:.0f} ms" if lat else None, delta_color="off")
    fr, la = logs["freight"], logs["late"]
    k = st.columns(4)
    for i, r in enumerate(RULES):
        v = fr["unusual_rate"][r]
        k[i].metric(f"Unusual, {'overall' if r == 'overall' else 'per-band'} rule",
                    f"{v:.1%}" if v is not None else "–", f"of {fr['with_charge']:,} with a charge", delta_color="off",
                    help=RULES[r])
    k[2].metric("Orders on the review list", f"{la['on_review_list_rate']:.1%}" if la["on_review_list_rate"] is not None
                else "–")
    k[3].metric("Average risk score", f"{la['mean_risk']:.3f}" if la["mean_risk"] is not None else "–")
    a, b = st.columns(2, gap="large")
    with a:
        by = pd.Series(logs["drift"]["by_input"], dtype=float)
        if len(by):
            fig = go.Figure(go.Bar(x=by.values, y=by.index, orientation="h", marker_color=ORANGE,
                                   hovertemplate="%{y}: %{x:,} predictions<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="predictions with a drift warning on this input")
            st.plotly_chart(layout(fig, h=90 + 28 * len(by), title="Drift warnings by input", bargap=0.3),
                            width="stretch")
        else:
            st.markdown("No drift warnings logged yet.")
    with b:
        fu = pd.Series(la["follow_ups"], dtype=float)
        if len(fu):
            fig = go.Figure(go.Bar(x=fu.values, y=fu.index, orientation="h", marker_color=BLUE,
                                   hovertemplate="%{y}: %{x:,} orders<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="orders scored")
            st.plotly_chart(layout(fig, h=90 + 28 * len(fu), title="Suggested follow-ups", bargap=0.3),
                            width="stretch")
    if logs["since"]:
        st.caption(f"Logged since {logs['since'][:19].replace('T', ' ')} UTC; last prediction "
                   f"{logs['last'][:19].replace('T', ' ')} UTC. By endpoint: "
                   + ", ".join(f"`{e}` {v:,}" for e, v in req["by_endpoint"].items()) + ".")


def api_status():
    """API status indicator for the sidebar (checked at most once a minute)."""
    ok, info = api_health()
    if ok:
        st.sidebar.success(f"Model API online · {info['ms']:.0f} ms", icon="🟢")
    else:
        st.sidebar.warning("Model API asleep or unreachable: the forms wake it (30–60 s).", icon="🟠")


# ------------------------------------------------------------------ navigation, grouped by stakeholder
PAGE_HOME = st.Page(home, title="Home", icon="🏠", default=True)
PAGE_MAP = st.Page(map_page, title="Map by state", icon="🗺️", url_path="map")
PAGE_AUDIT = st.Page(audit_page, title="Freight audit", icon="🔍", url_path="audit")
PAGE_CHECK = st.Page(check_page, title="Check a charge", icon="🧾", url_path="check")
PAGE_TARIFF = st.Page(tariff_page, title="Tariff review", icon="📐", url_path="tariff")
PAGE_QUEUE = st.Page(queue_page, title="Daily risk queue", icon="⏰", url_path="queue")
PAGE_PROMISE = st.Page(promise_page, title="Promise check", icon="📅", url_path="promise")
PAGE_MONITOR = st.Page(monitor_page, title="Monitoring", icon="📈", url_path="monitor")
PAGE_CARD = st.Page(model_card, title="Model card", icon="📋", url_path="model-card")
PAGE_SERVICE = st.Page(service_page, title="Model service", icon="🛰️", url_path="service")
nav = st.navigation({"": [PAGE_HOME, PAGE_MAP],
                     "Pricing and finance analyst": [PAGE_AUDIT, PAGE_CHECK, PAGE_TARIFF],
                     "Operations and customer experience": [PAGE_QUEUE, PAGE_PROMISE, PAGE_MONITOR],
                     "About": [PAGE_CARD, PAGE_SERVICE]})
if not st.session_state.get("api_pinged"):                  # once per session: wake the API while the user reads
    st.session_state.api_pinged = True
    threading.Thread(target=ping_api, daemon=True).start()
remember()                                                  # filters and form inputs outlive their page (keep())
api_status()
st.sidebar.caption("Team 5 · IT5006 · AY2026/27 · Phase 3")
st.sidebar.caption("Scores held-out Olist records (2016–2018) as if they were new orders.")
nav.run()
