# -*- coding: utf-8 -*-
"""
Olist Brazilian E-Commerce — Consolidated EDA dashboard (Team 5, IT5006).

An interactive retelling of ``notebooks/00_Consolidated_EDA_final.ipynb``. The eight pages follow the
notebook's flow in order — dataset & data quality, temporal, delivery, geography, the customer /
seller / product distributions, freight economics (the spine), reviews, and the problem selection —
and every headline number is read from ``data/facts.json``, which is generated straight from the
notebook's own logic (see ``gen_dashboard_data.py``). It reads only the small precomputed aggregates in
``data/``, so it deploys on Streamlit Cloud without the 121 MB raw dataset. EDA only — no model.

Run locally:   streamlit run deployment/app.py
"""
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

st.set_page_config(page_title="Olist Consolidated EDA", page_icon="📦", layout="wide")
DATA = os.path.join(os.path.dirname(__file__), "data")
ASSETS = os.path.join(os.path.dirname(__file__), "assets")


@st.cache_data
def csv(name):
    return pd.read_csv(os.path.join(DATA, name))


@st.cache_data
def js(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


@st.cache_data
def geojson():
    p = os.path.join(DATA, "brazil_states.geojson")
    return json.load(open(p)) if os.path.exists(p) else None


F = js("facts.json")

# ---- Olist design-system palette (designsystem.olist.io tokens) ----
PRIMARY = "#0a4ee4"; ACCENT = "#2766ec"; WARM = "#ed6e5a"; SLATE = "#3e3e3d"; GREEN = "#779e3d"
AMBER = "#f0a028"; PURPLE = "#cf77ad"; TEAL = "#54b6b6"
SEQ = ["#e7edf8", "#a1b9ed", "#2766ec", "#0a4ee4"]
SEQ_WARM = ["#f2f0e8", "#f7ada1", "#ed6e5a", "#e64e36"]
OLIST_CATS = [ACCENT, WARM, GREEN, AMBER, SLATE, PURPLE, TEAL]
pio.templates["olist"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#10100f"), colorway=OLIST_CATS))
pio.templates.default = "plotly_white+olist"


def H(fig, h=380, **kw):
    """Standard layout: generous height and inner padding so charts breathe."""
    fig.update_layout(height=h, margin=dict(t=42, b=10, l=10, r=10), **kw)
    return fig


def money(x, dp=0):
    """R$ value for plain (non-markdown) contexts such as st.metric."""
    return f"R$ {x:,.{dp}f}"


def spacer(px=14):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar / nav
st.sidebar.title("📦 Olist Consolidated EDA")
st.sidebar.caption("Team 5 · IT5006 · AY2026/27 — Phase 1")
PAGES = [
    "1 · Overview & data quality",
    "2 · Temporal patterns",
    "3 · Delivery performance",
    "4 · Geography",
    "5 · Customers, sellers & products",
    "6 · Freight economics",
    "7 · Reviews & satisfaction",
    "8 · Problem selection",
]
PAGE = st.sidebar.radio("Follow the EDA story", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"{F['date_min']} → {F['date_max']} · {F['n_orders']:,} orders · {F['n_items']:,} item lines · "
    "9 tables. Numbers mirror the consolidated notebook; charts read precomputed aggregates.")
st.sidebar.caption("Flow: data → temporal → delivery → geography → actors → **freight** → reviews → problem.")


# ================================================================== 1. OVERVIEW & DATA QUALITY
if PAGE == PAGES[0]:
    st.title("Olist Brazilian E-Commerce — Overview and Data Quality")
    st.markdown(
        "Nine relational tables centred on `orders`, spanning **%s → %s**. Before choosing a problem we "
        "read every chart for two things: **what it implies for our choice of problem**, and **whether the "
        "underlying data is usable**. This page covers the dataset and its quality; the four distributions "
        "below are the ones that frame the whole story." % (F["date_min"], F["date_max"]))

    m = st.columns(6)
    m[0].metric("Orders", f"{F['n_orders']:,}")
    m[1].metric("Item lines", f"{F['n_items']:,}")
    m[2].metric("Unique customers", f"{F['n_customer_unique']:,}")
    m[3].metric("Sellers", f"{F['n_sellers']:,}")
    m[4].metric("Delivered", f"{F['pct_delivered']}%")
    m[5].metric("Freight of GMV", f"{F['freight_pct_gmv']}%")

    st.divider()
    # ---- four framing distributions ----
    st.subheader("Four distributions that frame the problem")
    fd = csv("freight_dist.csv"); dh = csv("delivery_days_hist.csv"); rv = csv("review_score.csv")
    a, b = st.columns(2, gap="large")
    with a:
        d = fd[fd.freight_value <= fd.freight_value.quantile(0.99)]
        fig = px.histogram(d, x="freight_value", nbins=60, color_discrete_sequence=[WARM],
                           labels={"freight_value": "freight (R$)"}, title="Freight value (R$)")
        fig.add_vline(x=F["freight_p50"], line_dash="dash", line_color=SLATE)
        st.plotly_chart(H(fig), width="stretch")
        st.info(f"Large, right-skewed cost — median **R\\${F['freight_p50']:.0f}**, tail out to "
                f"R\\${F['freight_max']:.0f}. **A quantity worth predicting.**")
    with b:
        d = fd[fd.freight_ratio <= 2]
        fig = px.histogram(d, x="freight_ratio", nbins=70, color_discrete_sequence=[ACCENT],
                           labels={"freight_ratio": "freight / price"}, title="Freight as a share of item price")
        fig.add_vline(x=0.25, line_dash="dash", line_color="#e64e36")
        st.plotly_chart(H(fig), width="stretch")
        st.info(f"Median **{F['median_freight_ratio']}%**; **{F['pct_high_freight']}%** of items exceed 25% "
                "(dashed line). **A ready-made, nearly balanced classification target.**")
    spacer()
    a, b = st.columns(2, gap="large")
    with a:
        fig = px.bar(dh, x="bin_mid", y="count", color_discrete_sequence=[GREEN],
                     labels={"bin_mid": "delivery days", "count": "orders"}, title="Delivery time (days)")
        fig.add_vline(x=F["median_delivery_days"], line_dash="dash", line_color=SLATE)
        st.plotly_chart(H(fig), width="stretch")
        st.info(f"Median **{F['median_delivery_days']:.0f} days**, **{F['late_rate']}%** late vs estimate. "
                "**A second candidate target (delivery / lateness).**")
    with b:
        fig = px.bar(rv, x="review_score", y="n", color_discrete_sequence=[SLATE],
                     labels={"review_score": "review score", "n": "reviews"}, title="Review score")
        st.plotly_chart(H(fig), width="stretch")
        st.warning(f"Bimodal (mostly 5★, spike at 1★). But review **text is {F['review_text_empty']}% empty** — "
                   "so no reliable text mining; only the numeric score is usable.")
    st.success("**Reading across the four:** two targets have derivable labels and real spread — **freight** "
               "(cost) and **delivery** (time). Freight is the larger, more systematic, less-studied signal, so "
               "the rest of the dashboard tests whether it holds up as the lead problem — and it does.")

    st.divider()
    # ---- the nine tables ----
    st.subheader("Dataset Overview and Structure")
    ov = csv("overview.csv").rename(columns={
        "total_missing_cells": "missing cells", "exact_dup_rows": "dup rows"})
    st.dataframe(ov, width="stretch", hide_index=True)
    st.image(os.path.join(ASSETS, "ERD-final.png"),
             caption="Database schema — the nine Olist tables and their relationships", width="stretch")
    st.markdown(
        f"**Grain matters.** `orders`, `customers`, `reviews` are one row per order; `order_items` is one "
        f"row per **item line** (up to **{F['max_items_in_order']}** per order); `payments` is one row per "
        f"instalment record. `customer_id` is issued **per order** — the real person is `customer_unique_id` "
        f"({F['n_customer_rows']:,} ids collapse to **{F['n_customer_unique']:,}** people). `geolocation` has "
        f"{F['n_geo_rows']:,} rows across {F['n_geo_zip']:,} zip prefixes, so it must be reduced to one "
        "coordinate per zip before use (median lat/long per zip, on the Geography page).")

    st.divider()
    # ---- data quality ----
    st.subheader("Data quality — missing values and the traps we avoid")
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.markdown("**Columns with missing data**")
        ms = csv("missing.csv").sort_values("missing_pct", ascending=False)
        fig = px.bar(ms.head(13), x="missing_pct", y="column", orientation="h", color="table",
                     color_discrete_sequence=OLIST_CATS, labels={"missing_pct": "% missing", "column": ""})
        st.plotly_chart(H(fig, 360, legend=dict(orientation="h", y=-0.2, title="")), width="stretch")
    with c2:
        st.markdown("**Three traps, all real:**")
        st.markdown(
            f"- **Review text is mostly empty** — comment message **{F['review_text_empty']}%** empty, title "
            f"**{F['review_title_empty']}%**. Any review analysis must use the **1–5 numeric score**, not NLP.\n"
            f"- **Delivery timestamps missing on ~3%** — only for orders that were never delivered (cancelled "
            "before dispatch). Restrict delivery analysis to delivered orders; never use delivery dates as "
            "predictors.\n"
            f"- **City-name columns split one city across spellings** — `sao paulo` appears **{F['city_plain']:,}** "
            f"times plainly and **{F['city_accented']:,}** more under accented/encoded spellings. **Do not join or "
            "group on city names** — key on `zip_code_prefix` and the lat/long centroids instead.")
    st.divider()
    st.subheader("Missing-data & outlier handling decisions")
    st.caption("Documented so Phase-2 preprocessing is reproducible.")
    st.markdown(
            "| Issue | Extent | Decision |\n|---|---|---|\n"
            "| Review comment text mostly empty | ~59–88% | Drop text; use numeric `review_score`. |\n"
            "| Delivery timestamps missing | ~3% | Restrict delivery analysis to delivered orders; never use delivery dates as predictors. |\n"
            "| Product category / dimensions missing | ~2% | Impute category as `unknown`; drop the few rows missing weight/volume for physical-attribute modelling. |\n"
            "| Geolocation out-of-Brazil + duplicates | 42 rows + ~262k dup | Bounding-box filter, then median lat/long per zip prefix. |\n"
            "| City-name columns unreliable | accented vs plain spellings | **Do not use city-name columns** — key on `zip_code_prefix` and centroids. |\n"
            "| `freight_ratio` when price = 0 | tiny | Exclude (division undefined). |\n"
            "| Right-skewed money (price, freight) | long tail | Keep rows; **log-transform** for modelling, clip only for chart legibility. |\n"
            "| Extreme freight residuals | by design | The anomaly *signal* in Candidate A, not noise to remove. |\n"
            "| Distance is great-circle | all rows | Approximation (road distance unavailable); stated limitation. |")


# ================================================================== 2. TEMPORAL
elif PAGE == PAGES[1]:
    st.title("Temporal — enough history to model, with caveats")
    st.markdown("Order volume and revenue over the ~two-year window, plus the intraweek rhythm of when "
                "customers buy. The window is only **late 2016 → Oct 2018** with a partial final month, so any "
                "time model uses time-ordered splits and cautious seasonality.")
    m = csv("monthly.csv")
    metric = st.radio("Series", ["orders", "revenue", "late_rate"], horizontal=True,
                      format_func=lambda s: {"orders": "Orders", "revenue": "Revenue (R$)",
                                             "late_rate": "Late rate (%)"}[s])
    col = {"orders": ACCENT, "revenue": WARM, "late_rate": "#e64e36"}[metric]
    fig = px.line(m, x="purchase_month", y=metric, markers=True, color_discrete_sequence=[col])
    lab = {"orders": "orders", "revenue": "revenue (R$)", "late_rate": "late rate (%)"}[metric]
    if metric != "late_rate":
        fig.add_annotation(x="2017-11", y=m.set_index("purchase_month").loc["2017-11", metric],
                           text="Black Friday<br>Nov 2017", showarrow=True, arrowhead=2, ay=-40)
    st.plotly_chart(H(fig, 400, yaxis_title=lab, xaxis_title=""), width="stretch")

    st.divider()
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown("**Order volume by year — is Black Friday seasonal?**")
        yoy = csv("yoy.csv"); piv = yoy.pivot(index="mon", columns="year", values="orders")
        mn = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = go.Figure()
        for i, yr in enumerate(piv.columns):
            fig.add_scatter(x=[mn[k-1] for k in piv.index], y=piv[yr], mode="lines+markers",
                            name=str(int(yr)), line=dict(color=OLIST_CATS[i % len(OLIST_CATS)]))
        st.plotly_chart(H(fig, 360, yaxis_title="orders", legend=dict(title="year")), width="stretch")
    with b:
        st.markdown("**When customers buy (day × hour)**")
        dh = csv("dow_hour.csv"); dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        piv = dh.pivot(index="dow", columns="hour", values="orders").reindex(dow)
        st.plotly_chart(H(px.imshow(piv, color_continuous_scale=SEQ_WARM, aspect="auto",
                        labels=dict(x="hour of day", y="", color="orders")), 360), width="stretch")
    st.info("The marketplace **grows through 2017–2018** with a pronounced **November 2017 Black Friday** spike; "
            "overlaying the years shows 2018 above 2017 for most months and marks Black Friday as a **one-off, "
            "not a recurring season**. Buying concentrates on **weekday afternoons**. For selection: there is "
            "enough history for month/holiday features and time-ordered splits, but only ~2 years — this sets the "
            "modelling ground rules without favouring either candidate.")


# ================================================================== 3. DELIVERY
elif PAGE == PAGES[2]:
    st.title("Delivery performance — logistics, not seller handling")
    m = st.columns(5)
    m[0].metric("Delivered", f"{F['pct_delivered']}%")
    m[1].metric("Late vs estimate", f"{F['late_rate']}%")
    m[2].metric("Early", f"{F['early_rate']}%")
    m[3].metric("Median delivery", f"{F['median_delivery_days']:.0f} days")
    m[4].metric("Median vs estimate", f"{F['median_days_late']} days")
    st.markdown(
        f"**{F['pct_delivered']}%** of orders were delivered, median **{F['median_delivery_days']:.0f} days**. "
        f"Only **{F['late_rate']}%** arrived late while **{F['early_rate']}%** beat the estimate — a median "
        f"**{abs(F['median_days_late'])} days early**, so Olist's estimated dates are **padded / conservative**.")

    st.divider()
    st.markdown("#### How long customers wait, and how that compares to the promise")
    a, b = st.columns(2, gap="large")
    with a:
        dh = csv("delivery_days_hist.csv")
        fig = px.bar(dh, x="bin_mid", y="count", color_discrete_sequence=[ACCENT],
                     labels={"bin_mid": "days from purchase to delivery", "count": "orders"},
                     title="Purchase → delivery lead time")
        fig.add_vline(x=F["median_delivery_days"], line_dash="dash", line_color="#e64e36")
        st.plotly_chart(H(fig), width="stretch")
    with b:
        dl = csv("days_late_hist.csv")
        fig = px.bar(dl, x="bin_mid", y="count", color_discrete_sequence=[WARM],
                     labels={"bin_mid": "days vs estimate (negative = early)", "count": "orders"},
                     title="Delivery vs estimate (±30-day window)")
        fig.add_vline(x=0, line_dash="dash", line_color=SLATE)
        st.plotly_chart(H(fig), width="stretch")

    st.divider()
    st.markdown("#### Where the lead time goes")
    a, b = st.columns([2, 3], gap="large")
    with a:
        lt = csv("leadtime.csv")
        fig = px.bar(lt, x="stage", y="avg_days", color="stage", color_discrete_sequence=[ACCENT, AMBER, GREEN],
                     labels={"avg_days": "average days", "stage": ""})
        st.plotly_chart(H(fig, 340, showlegend=False), width="stretch")
    with b:
        st.markdown("")
        st.markdown(f"The wait splits into three operational legs. Most of it is the **carrier→customer leg "
                    f"(~{F['lead_carrier_to_cust']:.0f} of ~{F['mean_delivery_days']:.0f} days)** — the delay is "
                    f"**logistics in transit**, not seller/Olist handling: purchase→approval is only "
                    f"~{F['lead_to_approval']:.1f} day and approval→carrier ~{F['lead_to_carrier']:.1f} days. So a "
                    "delivery-time model is really predicting how long the courier takes to reach the customer.")

    st.divider()
    st.markdown("#### Delivery Performance Across Customer States")
    st.caption("Compare customer states by median delivery time and late-delivery rate to identify "
               "differences in delivery performance.")
    s = csv("state_summary.csv")
    sort_by = st.radio("Rank customer states by", ["median_delivery_days", "late_rate"], horizontal=True,
                       format_func=lambda x: {"median_delivery_days": "Median delivery time (days)",
                                              "late_rate": "Late-delivery rate (%)"}[x])
    top = s.sort_values(sort_by, ascending=False)
    barcol = ACCENT if sort_by == "median_delivery_days" else "#e64e36"
    title = "Median delivery time by customer state (days)" if sort_by == "median_delivery_days" \
        else "Late-delivery rate by customer state (%)"
    fig = px.bar(top, x=sort_by, y="customer_state", orientation="h", color_discrete_sequence=[barcol],
                 labels={sort_by: title.split(" by ")[0], "customer_state": "customer state"}, title=title)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(H(fig, 560), width="stretch")
    st.info(
        f"A strong gradient: **SP** (the seller hub) gets orders in **~{F['state_fast_days']} days with "
        f"<6% late**; remote northern states (AM, RR, AP) wait **~24–25 days**. But **longest transit ≠ "
        f"highest late**: **{F['state_late_top']}** has only mid-range transit yet the **highest late rate "
        f"({F['state_late_top_rate']}%)**, while the slowest-by-transit states sit among the *lowest* late rates "
        "(~3%) — Olist evidently sets a more generous estimate for the known-remote states. Distance drives "
        "delivery time, but late-rate is shaped by more than distance alone.")


# ================================================================== 4. GEOGRAPHY
elif PAGE == PAGES[3]:
    st.title("Geography — distance is the hidden driver")
    st.markdown("Sellers cluster in the south-east (SP), so **where the seller sits relative to the customer** "
                "drives both freight and delivery. All three maps darken toward the north/north-east together.")
    s = csv("state_summary.csv"); lanes = csv("lanes.csv"); gj = geojson()

    maps = [("median_delivery_days", "Median delivery days", SEQ),
            ("median_freight", "Median freight (R$)", SEQ_WARM),
            ("median_distance", "Median distance (km)", SEQ)]
    cols = st.columns(3, gap="medium")
    for (colname, title, scale), cc in zip(maps, cols):
        with cc:
            st.markdown(f"**{title}**")
            if gj is not None:
                fig = px.choropleth(s, geojson=gj, locations="customer_state",
                                    featureidkey="properties.sigla", color=colname, color_continuous_scale=scale)
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(height=320, margin=dict(t=0, b=0, l=0, r=0),
                                  coloraxis_colorbar=dict(title="", thickness=8))
                st.plotly_chart(fig, width="stretch")
    st.info("States far from the SP seller hub wait **longer**, pay **more freight**, and sit at greater "
            "**distance** — the three move together. This is the core evidence that **freight and delivery are "
            "both distance-driven, and therefore predictable from the parcel and route.**")

    st.divider()
    st.subheader("Which seller states serve a given customer state best?")
    states = sorted(s.customer_state.unique())
    cust = st.selectbox("Customer state", states, index=states.index("SP") if "SP" in states else 0)
    lc = lanes[(lanes.customer_state == cust) & (lanes.n >= 15)].copy()
    instate = lc[lc.seller_state == cust]
    k = st.columns(3)
    if len(instate):
        r = instate.iloc[0]
        k[0].metric("In-state sellers — delivery", f"{r.median_delivery_days:.0f} days")
        k[1].metric("In-state sellers — freight", money(r.median_freight))
        k[2].metric("In-state sellers — distance", f"{r.median_distance:.0f} km")
    spacer()
    metrics = [("median_delivery_days", "Delivery days (lower = better)", ACCENT),
               ("median_freight", "Freight R$ (lower = better)", WARM),
               ("median_distance", "Distance km", GREEN)]
    for (colname, title, colr), cc in zip(metrics, st.columns(3, gap="medium")):
        with cc:
            top = lc.sort_values(colname).head(10)
            fig = px.bar(top, x=colname, y="seller_state", orientation="h", color_discrete_sequence=[colr],
                         labels={colname: "", "seller_state": "seller state"}, title=title)
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(H(fig, 340), width="stretch")
    st.dataframe(
        lc.sort_values("median_delivery_days")[["seller_state", "n", "median_delivery_days",
                                                "median_freight", "median_distance"]]
        .rename(columns={"n": "item lines", "median_delivery_days": "delivery days",
                         "median_freight": "freight R$", "median_distance": "distance km"}),
        width="stretch", hide_index=True)
    st.info(f"For **{cust}** customers, in-state (short-distance) sellers are fastest and cheapest; distant "
            "seller states cost more freight and more days. The pattern holds for every state — **freight is a "
            "function of distance and the parcel, which we can compute and therefore predict.**")


# ================================================================== 5. CUSTOMERS / SELLERS / PRODUCTS
elif PAGE == PAGES[4]:
    st.title("Customers, sellers & products — three skews that rule targets in and out")
    st.markdown("All three actor tables are highly skewed. The customer and seller skews point **away** from "
                "customer- and seller-centred targets; the product/freight skew points **toward** freight.")
    sub = st.radio("Sub-section", ["Customers", "Sellers", "Products", "Payments"], horizontal=True)
    st.divider()

    if sub == "Customers":
        m = st.columns(3)
        m[0].metric("Ever place a 2nd order", f"{F['repeat_rate']}%")
        m[1].metric("Revenue from repeaters", f"{F['repeat_rev_share']}%")
        m[2].metric("Days to 2nd order (median)", f"{F['repeat_gap_median']}")
        spacer()
        a, b = st.columns(2, gap="large")
        with a:
            cr = csv("cust_repeat.csv")
            fig = px.bar(cr, x="orders_placed", y="customers", color_discrete_sequence=[ACCENT],
                         labels={"orders_placed": "orders placed by a customer", "customers": "customers"},
                         title=f"Orders per customer — only {F['repeat_rate']}% ever return")
            fig.update_yaxes(type="log", title_text="customers (log scale)")
            st.plotly_chart(H(fig), width="stretch")
        with b:
            ds = csv("demand_state.csv")
            fig = px.bar(ds, x="pct_orders", y="customer_state", orientation="h", color_discrete_sequence=[AMBER],
                         labels={"pct_orders": "% of all orders", "customer_state": "customer state"},
                         title="Top 10 states by order volume")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(H(fig), width="stretch")
        st.info(f"Repeat purchasing is essentially absent — only **{F['repeat_rate']}%** of customers place a "
                f"second order, and they carry just **{F['repeat_rev_share']}%** of revenue. A repeat / CLV target "
                f"would be a rare-event problem on a thin signal, so we drop it. Demand is also lopsided "
                f"(**SP {F['demand_sp']}%**), the same geography behind the delivery gradient.")

    elif sub == "Sellers":
        m = st.columns(4)
        m[0].metric("Revenue from top 10% of sellers", f"{F['seller_top10']}%")
        m[1].metric("Median lines / seller", f"{F['seller_items_median']}")
        m[2].metric("Sellers shipping ≤10 lines", f"{F['seller_le10_pct']}%")
        m[3].metric("Item lines shipped interstate", f"{F['interstate_pct']}%")
        spacer()
        a, b = st.columns(2, gap="large")
        with a:
            sp = csv("seller_pareto.csv")
            fig = go.Figure()
            fig.add_scatter(x=sp.pct_sellers, y=sp.cum_revenue, mode="lines", line=dict(color=WARM, width=3),
                            fill="tozeroy", fillcolor="rgba(237,110,90,0.10)", name="cumulative revenue")
            fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="#9AA0A6", dash="dot"),
                            name="perfectly even")
            for p, v in [(1, F["seller_top1"]), (5, F["seller_top5"]), (10, F["seller_top10"]), (20, F["seller_top20"])]:
                fig.add_annotation(x=p, y=v, text=f"top {p}% → {v}%", showarrow=False, yshift=11, font=dict(size=10))
            st.plotly_chart(H(fig, 380, title="Seller revenue is Pareto-concentrated",
                              xaxis_title="% of sellers (ranked by revenue)", yaxis_title="cumulative % of revenue"),
                            width="stretch")
        with b:
            si = csv("seller_items.csv")["items"]
            # Bin in log10 space -> equal-width bars on a linear axis, ticks relabelled to real
            # counts. (A log-axis bar chart gives uneven widths and messy minor ticks.)
            fig = px.histogram(x=np.log10(si), nbins=40, title="Most sellers are tiny (log scale)")
            fig.update_traces(marker_color=ACCENT, marker_line_width=0)   # one solid blue
            ticks = [1, 10, 100, 1000]                                    # decade ticks (10^n), matching the notebook
            fig.update_xaxes(tickvals=[np.log10(v) for v in ticks],
                             ticktext=["10⁰", "10¹", "10²", "10³"],
                             title_text="item lines sold")
            fig.update_yaxes(title_text="sellers")
            fig.add_vline(x=np.log10(F["seller_items_median"]), line_dash="dash", line_color=SLATE,
                          annotation_text=f"median {F['seller_items_median']}")
            fig.add_vline(x=np.log10(F["seller_items_p90"]), line_dash="dash", line_color=WARM,
                          annotation_text=f"p90 {F['seller_items_p90']}")
            st.plotly_chart(H(fig, 380), width="stretch")
        st.info(f"Revenue is sharply concentrated — top 1% of sellers take **{F['seller_top1']}%**, top 10% "
                f"**{F['seller_top10']}%**, and {F['seller_top80_pct']}% cover 80%. But the median seller ships "
                f"only **{F['seller_items_median']}** lines and **{F['seller_le10_pct']}%** ship ≤10 — far too "
                f"thin to model individually. The real hand-off to freight: supply sits in SP, demand does not, so "
                f"**{F['interstate_pct']}% of lines ship interstate** (median **{F['interstate_dist']} km**, "
                f"R\\${F['interstate_freight']:.2f}) vs within-state ({F['intrastate_dist']} km, "
                f"R\\${F['intrastate_freight']:.2f}). Distance is the default, not the edge case.")

    elif sub == "Products":
        m = st.columns(3)
        m[0].metric("Distinct categories", f"{F['n_categories']}")
        m[1].metric("Top 10 categories' share of lines", f"{F['top10_vol_pct']}%")
        m[2].metric("Corr(price, freight burden)", f"{F['corr_price_burden']}")
        spacer()
        cat = csv("category_summary.csv")
        a, b = st.columns(2, gap="large")
        with a:
            cc = cat.sort_values("lines", ascending=False).head(15)
            fig = px.bar(cc, x="lines", y="category", orientation="h", color_discrete_sequence=[GREEN],
                         labels={"lines": "item lines", "category": ""}, title="Top 15 categories by item lines")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(H(fig, 420), width="stretch")
        with b:
            cr = cat.sort_values("revenue", ascending=False).head(15).copy()
            cr["rev_m"] = cr.revenue / 1e6
            fig = px.bar(cr, x="rev_m", y="category", orientation="h", color_discrete_sequence=["#3A7D44"],
                         labels={"rev_m": "revenue (R$ millions)", "category": ""}, title="Top 15 categories by revenue")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(H(fig, 420), width="stretch")
        spacer()
        st.markdown("**Weight explains only part of price** — freight follows the parcel, revenue follows price")
        wp = csv("weight_price_sample.csv").copy()
        wp = wp[(wp.weight_kg > 0) & (wp.price > 0)]        # guard log10 against rounded-to-0 weights
        wp["log_w"] = np.log10(wp.weight_kg); wp["log_p"] = np.log10(wp.price)
        fig = px.density_heatmap(wp, x="log_w", y="log_p", nbinsx=45, nbinsy=45,
                                 color_continuous_scale="Greens", labels={"count": "item lines"})
        ticks = [0.1, 1, 10, 100]
        fig.update_xaxes(tickvals=[np.log10(v) for v in ticks], ticktext=[str(v) for v in ticks],
                         title_text="product weight (kg, log scale)")
        fig.update_yaxes(tickvals=[np.log10(v) for v in [10, 100, 1000]], ticktext=["10", "100", "1000"],
                         title_text="item price (R$, log scale)")
        st.plotly_chart(H(fig, 420), width="stretch")
        st.info(f"Volume is concentrated — top 10 of {F['n_categories']} categories carry **{F['top10_vol_pct']}%** "
                f"of lines and **{F['top10_rev_pct']}%** of revenue. Freight is near-flat in reais while prices "
                f"span three orders of magnitude, so **cheap categories carry the heaviest burden** "
                f"(price-vs-burden corr **{F['corr_price_burden']}**). And weight is only moderately tied to price "
                f"(Spearman **{F['spearman_weight_price']}**) — the gap between what sets freight (weight, size, "
                "distance) and what sets revenue (price) is exactly the freight-burden problem.")

    else:  # Payments
        m = st.columns(3)
        m[0].metric("Credit card", f"{F['pay_credit']}%")
        m[1].metric("Card orders in instalments", f"{F['inst_pct']}%")
        m[2].metric("Median instalments", f"{F['inst_median']}")
        spacer()
        a, b, c = st.columns(3, gap="medium")
        with a:
            pt = csv("payment_type.csv")
            fig = px.bar(pt, x="pct", y="payment_type", orientation="h", color_discrete_sequence=[PURPLE],
                         labels={"pct": "% of payment records", "payment_type": ""}, title="Payment method mix")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(H(fig, 340), width="stretch")
        with b:
            ins = csv("installments.csv")
            fig = px.bar(ins, x="installments", y="pct", color_discrete_sequence=["#CCB974"],
                         labels={"pct": "% of card payments", "installments": "instalments on the order"},
                         title="Card instalment plans")
            st.plotly_chart(H(fig, 340), width="stretch")
        with c:
            ov = csv("order_value_by_installment.csv")
            fig = px.line(ov, x="max_installments", y="median_order_value", markers=True, color_discrete_sequence=[WARM],
                          labels={"max_installments": "instalments on the order",
                                  "median_order_value": "median order value (R$)"},
                          title="Order value climbs with instalments")
            st.plotly_chart(H(fig, 340), width="stretch")
        st.info(f"Payments are **credit-card dominated** ({F['pay_credit']}%) then boleto ({F['pay_boleto']}%). "
                f"**{F['inst_pct']}%** of card orders are split into instalments (median {F['inst_median']}), and "
                f"instalment count **tracks basket size, not distress** — median order value rises from "
                f"R\\${F['ordval_1inst']} at one instalment to R\\${F['ordval_maxinst']} at "
                f"{F['ordval_maxinst_n']}. Usable as a proxy for order value, not a risk signal.")

    st.divider()
    st.success("**This page hands forward the freight case:** supply concentrated in SP against dispersed demand "
               f"forces **{F['interstate_pct']}%** of lines interstate, freight is near-flat in reais while prices "
               "span three orders of magnitude, and the burden falls hardest on the cheapest categories. That is "
               "the densest, most actionable signal in the dataset — pursued on the Freight economics page.")


# ================================================================== 6. FREIGHT ECONOMICS
elif PAGE == PAGES[5]:
    st.title("Freight economics — the central theme")
    st.markdown("Freight is the largest systematic cost in the data after the goods themselves. Two questions: "
                "**what drives it**, and **how evenly does it fall**? The answers make it the lead problem.")
    m = st.columns(4)
    m[0].metric("Freight as % of GMV", f"{F['freight_pct_gmv']}%")
    m[1].metric("Median freight / price", f"{F['median_freight_ratio']}%")
    m[2].metric("Items >25% of price", f"{F['pct_high_freight']}%")
    m[3].metric("Median freight", money(F["freight_p50"]))

    st.divider()
    st.markdown("#### Question 1 — What drives freight?  *(interactive: pick the x-axis and a category)*")
    sample = csv("freight_sample.csv")
    c1, c2 = st.columns(2)
    xaxis = c1.selectbox("Freight vs", ["distance_km", "weight_kg", "volume_cm3", "price"],
                         format_func=lambda s: {"distance_km": "Distance (km)", "weight_kg": "Weight (kg)",
                                                "volume_cm3": "Volume (cm³)", "price": "Item price (R$)"}[s])
    cats = ["All categories"] + sorted(sample.category.dropna().unique().tolist())
    pick = c2.selectbox("Category", cats)
    d = sample if pick == "All categories" else sample[sample.category == pick]
    d = d[d.freight_value < d.freight_value.quantile(0.99)]
    plot_d = d.sample(min(3000, len(d)), random_state=1)
    xlab = {"distance_km": "distance (km)", "weight_kg": "weight (kg)",
            "volume_cm3": "volume (cm³)", "price": "item price (R$)"}[xaxis]
    fig = px.scatter(plot_d, x=xaxis, y="freight_value", opacity=0.3, color_discrete_sequence=[ACCENT],
                     labels={"freight_value": "freight (R$)", xaxis: xlab},
                     title=f"Freight vs {xlab} · {pick} ({len(d):,} items)")
    tmp = d[[xaxis, "freight_value"]].dropna()
    if len(tmp) > 50:                                            # fast binned-median trend (no statsmodels)
        tmp = tmp.assign(bin=pd.qcut(tmp[xaxis], 12, duplicates="drop"))
        tr = tmp.groupby("bin", observed=True).agg(x=(xaxis, "median"), y=("freight_value", "median")).dropna()
        fig.add_scatter(x=tr.x, y=tr.y, mode="lines+markers", name="median trend", line=dict(color=WARM, width=3))
    st.plotly_chart(H(fig, 440), width="stretch")
    st.caption(f"Scope: **{pick}** ({len(d):,} items). Correlations across **all** items — freight vs weight "
               f"**{F['corr_freight_weight']}**, volume **{F['corr_freight_volume']}**, distance "
               f"**{F['corr_freight_distance']}**, but only **{F['corr_freight_price']}** vs price. Freight is "
               "**priced off the parcel, not the price** — so it can be modelled.")

    spacer()
    st.markdown("**Chargeable weight = max(actual, volumetric)**  ·  "
                "carriers bill on whichever is larger")
    st.markdown(
        f"Volumetric (dimensional) weight is `L×W×H / 5000` — the DHL parcel convention. The single "
        f"chargeable-weight measure correlates **{F['corr_freight_chargeable']}** with freight — essentially "
        f"tied with actual weight (**{F['corr_freight_weight']}**), just above volumetric "
        f"(**{F['corr_freight_volume']}**). For Olist's small, dense parcels, **actual weight is usually the "
        "binding constraint**, so bulk rarely overrides it.")
    st.markdown("**Correlation — freight economics drivers**")
    cm = csv("corr_freight.csv").set_index("var")
    fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(H(fig, 460), width="stretch")

    st.divider()
    st.markdown("#### Question 2 — How deterministic is freight?  *(do near-identical parcels cost the same?)*")
    a, b = st.columns([3, 2], gap="large")
    with a:
        dc = csv("determinism_cells.csv")
        fig = px.histogram(dc, x="p90_over_p10", nbins=40, color_discrete_sequence=[PURPLE],
                           labels={"p90_over_p10": "within-band freight spread (p90 / p10)", "count": "bands"},
                           title="Freight spread inside near-identical weight×volume×distance bands")
        fig.add_vline(x=F["det_ratio"], line_dash="dash", line_color="#e64e36",
                      annotation_text=f"median {F['det_ratio']}×")
        st.plotly_chart(H(fig, 380), width="stretch")
    with b:
        st.markdown("**Same band, freight actually charged**")
        st.dataframe(csv("determinism_sample.csv").rename(columns={
            "weight_kg": "kg", "volume_cm3": "cm³", "distance_km": "km", "freight_value": "freight R$"}),
            width="stretch", hide_index=True)
    st.warning(f"Across **{F['det_cells']}** near-identical bands (≥30 items each), the top-decile freight is "
               f"**~{F['det_ratio']}×** the bottom, and only **{F['det_explained']}%** of freight variation lines "
               f"up with the weight/volume/distance band — the other **{F['det_residual']}%** varies between "
               "otherwise-identical parcels. That residual is **seller-set pricing, free-shipping deals and "
               "handling** — the overcharge signal a classifier would target. Freight is systematic enough to "
               "study, but not fully determined by the parcel.")

    st.divider()
    st.markdown("#### Question 3 — How evenly does freight fall?  *(regressive, and category-dependent)*")
    a, b = st.columns(2, gap="large")
    with a:
        pb = csv("price_band.csv")
        fig = px.bar(pb, x="price_band", y="median_burden", color_discrete_sequence=[WARM], text="median_burden",
                     labels={"median_burden": "freight as % of price", "price_band": "item price band (R$)"},
                     title="Freight burden is regressive · all items")
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        st.plotly_chart(H(fig), width="stretch")
    with b:
        cf = csv("category_freight.csv").sort_values("median_burden", ascending=False).head(15)
        fig = px.bar(cf.sort_values("median_burden"), x="median_burden", y="category", orientation="h",
                     color_discrete_sequence=[ACCENT],
                     labels={"median_burden": "median freight as % of price", "category": ""},
                     title="Freight burden by category · all items")
        st.plotly_chart(H(fig), width="stretch")
    st.info("Median freight is roughly flat across price bands (~R\\$14–31), so as a **share of price** it falls "
            "from **~60% on the cheapest items to <4% on the dearest** — regressive by construction. Electronics "
            "and watches pay almost the **same absolute freight (~R\\$15)**, yet that is **~62% burden for "
            "electronics vs ~12% for watches** — a **price** story (cheap vs dear), not a cost one. "
            f"**Freight is material ({F['freight_pct_gmv']}% of GMV), structured (a real cost), and unevenly "
            "applied — the richest thread in the dataset.**")


# ================================================================== 7. REVIEWS
elif PAGE == PAGES[6]:
    st.title("Reviews — why satisfaction is a delivery story, not a freight one")
    m = st.columns(4)
    m[0].metric("Review text empty", f"{F['review_text_empty']}%")
    m[1].metric("Review title empty", f"{F['review_title_empty']}%")
    m[2].metric("Corr(review, delivery)", f"{F['corr_review_delivery']:.2f}")
    m[3].metric("Corr(review, freight)", f"{F['corr_review_freight']:.2f}")

    st.divider()
    a, b = st.columns(2, gap="large")
    with a:
        rv = csv("review_score.csv")
        fig = px.bar(rv, x="review_score", y="n", color_discrete_sequence=[GREEN],
                     labels={"review_score": "review score (1–5)", "n": "reviews"}, title="Review score distribution")
        st.plotly_chart(H(fig), width="stretch")
        st.caption(f"Bimodal — many more 5★ than 1★ (~5×). Text is unusable ({F['review_text_empty']}% of "
                   f"messages, {F['review_title_empty']}% of titles empty), so only the **numeric score** is.")
    with b:
        rm = csv("review_monthly.csv")
        fig = go.Figure()
        fig.add_scatter(x=rm.month, y=rm.num_reviews, mode="lines+markers", name="reviews answered", line=dict(color=ACCENT))
        fig.add_scatter(x=rm.month, y=rm.num_orders, mode="lines+markers", name="orders delivered", line=dict(color=WARM))
        st.plotly_chart(H(fig, 380, title="Reviews track orders over time",
                          legend=dict(orientation="h", y=-0.2)), width="stretch")
        st.caption("Reviews closely track delivered-order volume — customers review within a month of receiving, "
                   "so reviews reflect **delivery and appearance**, not long-term quality.")

    st.divider()
    st.subheader("What moves the score — delivery, not freight")
    st.markdown("**Order-level correlations** — review score lights up against delivery timing, stays pale "
                "against every freight column.")
    cm = csv("corr_reviews.csv").set_index("var")
    fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(H(fig, 460), width="stretch")

    spacer()
    st.markdown("**Review-score distribution by bin (box plots)** — pick a driver; each box shows how the "
                "*spread* of scores shifts across bins, as in the notebook.")
    which = st.radio("Review score by", ["delivery days", "days late", "freight R$", "freight ratio"],
                     horizontal=True)
    fmap = {"delivery days": ("review_box_delivery.csv", "delivery_bin", ACCENT),
            "days late": ("review_box_days_late.csv", "days_late_bin", "#e64e36"),
            "freight R$": ("review_box_freight.csv", "freight_bin", WARM),
            "freight ratio": ("review_box_freight_ratio.csv", "freight_ratio_bin", GREEN)}
    fn, xcol, colr = fmap[which]
    rb = csv(fn)
    fig = go.Figure(go.Box(x=rb[xcol], q1=rb["q1"], median=rb["median"], q3=rb["q3"],
                           lowerfence=rb["lowerfence"], upperfence=rb["upperfence"],
                           fillcolor=colr, line=dict(color=SLATE, width=1), name=""))
    fig.update_yaxes(range=[0.5, 5.5], dtick=1, title_text="review score")
    fig.update_xaxes(title_text=which)
    st.plotly_chart(H(fig, 400, title=f"Review score vs {which}", showlegend=False), width="stretch")
    st.success(f"Review score is far more correlated with **delivery** ({F['corr_review_delivery']:.2f}) than "
               f"**freight** ({F['corr_review_freight']:.2f}). It drops sharply as delivery slips late — roughly "
               f"**{F['review_ontime_star']}★ on-time vs {F['review_late_star']}★ late** — but is essentially "
               "**flat across freight burden**. Satisfaction belongs to a **delivery** problem; that keeps our "
               "scope honest and frames freight as a **cost / pricing** problem, not a customer-happiness one.")


# ================================================================== 8. PROBLEM SELECTION
elif PAGE == PAGES[7]:
    st.title("Problem selection — where the EDA lands")
    st.markdown("Each page narrowed the choice (page numbers match the sidebar):")
    st.markdown(
        "| Page | What it showed | Effect on selection |\n|---|---|---|\n"
        "| 1 · Overview | freight & delivery both have derivable, well-spread targets | two candidates on the table |\n"
        "| 2 · Temporal | ~2 years, Black-Friday spike | modelling ground rules, no winner yet |\n"
        "| 3 · Delivery | strong state gradient; longest-transit ≠ highest-late | delivery is a real, distance-driven target |\n"
        "| 4 · Geography | distance drives freight & delivery | freight is computable from parcel + route |\n"
        "| 5 · Actors | repeat 3%, sellers too thin; freight burden regressive | rules out customer/seller targets |\n"
        "| 6 · Freight | tracks weight/volume/distance; regressive; residual spread | freight is **predictable and matters** |\n"
        "| 7 · Reviews | text mostly empty; score tracks delivery, not freight | satisfaction ≠ freight; keep freight a cost problem |")

    st.divider()
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("### A · Fair-freight pricing & overcharge detection")
        st.caption("Primary — signal from the Freight economics page (6)")
        st.markdown("**Regression:** predict `freight_value` per item from distance, weight, volume, category.  \n"
                    "**Classification:** flag items whose actual freight deviates abnormally from the predicted "
                    "fair value (a residual-based overcharge / undercharge flag).")
        st.markdown(
            "| Check | Verdict |\n|---|---|\n"
            "| Derivable target | `freight_value` is in order_items; anomaly label derived from model residual |\n"
            "| Realistic features | distance, weight, size, volume, category all known at checkout — no leakage |\n"
            "| Manageable imbalance | anomaly flag = chosen residual percentile, so its rate is controllable |\n"
            "| Clear stakeholder | Finance / Logistics cost analyst — audit sellers, routes, product types over/under-charging |")
        st.markdown("**Sufficient signal?** Yes, on both halves of the problem:")
        st.markdown(
            f"- **For the regression:** freight is tightly tied to the parcel — it correlates "
            f"**{F['corr_freight_weight']}** with weight, **{F['corr_freight_volume']}** with volume and "
            f"**{F['corr_freight_distance']}** with distance, all well above **{F['corr_freight_price']}** with "
            "price. So a model can learn a *fair* freight from the parcel and route.\n"
            f"- **For the classification:** among parcels of near-identical weight, volume and distance, freight "
            f"still ranges **~{F['det_ratio']}×** from the 10th to the 90th percentile, and only "
            f"**~{F['det_explained']}%** of the variation is explained by the band. That leftover "
            f"**~{F['det_residual']}%** is the seller-driven over/under-charging the classifier flags.")
    with b:
        st.markdown("### B · Delivery-time / late-delivery prediction")
        st.caption("Companion — signal from the Delivery page (3)")
        st.markdown("**Regression:** predict `delivery_days`.  \n"
                    "**Classification:** predict `is_late` (vs Olist's estimate).")
        st.markdown(
            "| Check | Verdict |\n|---|---|\n"
            "| Derivable target | `delivery_days` and `is_late` come from the orders dataset |\n"
            "| Realistic features | must use only pre-dispatch info (distance, product, seller, category) |\n"
            f"| Manageable imbalance | ~{F['late_rate']}% late — use precision/recall/F1 and stratified splits |\n"
            "| Clear stakeholder | Operations / CX — set delivery promises, flag at-risk orders |")
        st.markdown("**Sufficient signal?** Yes — delivery varies strongly and learnably:")
        st.markdown(
            f"- **Geography splits it cleanly:** SP (the seller hub) delivers in **~{F['state_fast_days']} days "
            "with under 6% late**, while remote states (AM, RR, AP) take **~24–25 days**; state late-rates run "
            f"from under 3% up to **{F['state_late_top_rate']}% ({F['state_late_top']})**.\n"
            f"- **The wait is concentrated:** the carrier→customer leg is **~{F['lead_carrier_to_cust']:.0f} of "
            f"~{F['mean_delivery_days']:.0f} days**, and the slowest states are *not* the latest ones — so both "
            "`delivery_days` and `is_late` have clear, separable signal.")

    st.divider()
    st.success("**Recommended Phase-2 direction — Candidate A (fair-freight pricing + overcharge detection) as "
               "the primary problem** (one regression + one derived classification, satisfying the dual-framing "
               "requirement), with **Candidate B (delivery / lateness)** as the companion logistics problem.")
    st.info("**Limitations (apply to all).** `freight_value` is **revenue, not carrier cost** — no true margin "
            "claims. Distances are **great-circle, not road**. Everything conditions on **completed orders** "
            "(pre-checkout demand loss is unobservable). The window is **~2 years ending Oct 2018** with a "
            "partial final month.")

st.sidebar.markdown("---")
st.sidebar.caption("Source: notebooks/00_Consolidated_EDA_final.ipynb · Team 5")
