# -*- coding: utf-8 -*-
"""
Precompute small aggregate files for the Streamlit dashboard.

This regenerates the lightweight files in ``deployment/data/`` that the deployed app reads, so the
app runs on Streamlit Cloud without shipping the 121 MB raw dataset.

Everything here mirrors ``notebooks/Team5_Phase1_IT5006_AY2627Sem1.ipynb`` cell-for-cell: the ``item`` and
``order`` analytical tables are built with the notebook's exact logic (§3), and every aggregate below
is the same groupby the notebook charts from. The headline numbers are also written verbatim to
``facts.json`` and a block of ``assert``s at the end fails loudly if a recomputed number ever drifts
from the notebook. No prediction model is produced — the dashboard is EDA-only.

Usage:
    OLIST_DATA_DIR=/path/to/Olist_CSV python deployment/gen_dashboard_data.py
"""
import os, json, unicodedata
import numpy as np
import pandas as pd


def find_data_dir():
    for c in [os.environ.get("OLIST_DATA_DIR"),
              "/content/drive/MyDrive/IT5006_Project-Data/Olist_CSV",
              "data", "../data", "Olist_CSV", "../Olist_CSV",
              "../../IT5006_Project-Data/Olist_CSV"]:
        if c and os.path.exists(os.path.join(c, "olist_orders_dataset.csv")):
            return c
    raise FileNotFoundError("Set OLIST_DATA_DIR to the Olist_CSV folder.")


DATA = find_data_dir()
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)
print("Reading raw data from:", DATA)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    print("wrote", name, df.shape)


# ============================ LOAD (notebook §0) ============================
DATE_COLS = {
    "orders": ["order_purchase_timestamp", "order_approved_at",
               "order_delivered_carrier_date", "order_delivered_customer_date",
               "order_estimated_delivery_date"],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}
FILES = {
    "orders": "olist_orders_dataset.csv", "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv", "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv", "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv", "geolocation": "olist_geolocation_dataset.csv",
    "cat_trans": "product_category_name_translation.csv",
}
t = {name: pd.read_csv(os.path.join(DATA, fn), parse_dates=DATE_COLS.get(name))
     for name, fn in FILES.items()}

# ============================ §1 dataset overview ============================
overview = pd.DataFrame([{
    "table": name, "rows": len(df), "cols": df.shape[1],
    "total_missing_cells": int(df.isna().sum().sum()),
    "exact_dup_rows": int(df.duplicated().sum()),
} for name, df in t.items()])
save(overview, "overview.csv")

# ============================ §2 missing values ============================
col_missing = []
for name, df in t.items():
    for col, mc in df.isna().sum().items():
        col_missing.append({"table": name, "column": col, "missing_rows": int(mc),
                            "missing_pct": round(mc / len(df) * 100, 2)})
missing = pd.DataFrame(col_missing)
save(missing[missing.missing_rows > 0].reset_index(drop=True), "missing.csv")

# item-level price/freight extremes + the city-name split (notebook cell 13)
it0 = t["items"].copy()
it0["fr"] = it0.freight_value / it0.price.replace(0, np.nan)
gc = t["geolocation"]["geolocation_city"].astype(str)
folded = gc.map(lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode())
city_plain = int((gc == "sao paulo").sum())
city_accented = int(((folded == "sao paulo") & (gc != "sao paulo")).sum())

# ============================ §3 analytical tables ============================
BR_LAT, BR_LON = (-34.0, 5.5), (-74.0, -34.0)
geo = t["geolocation"]
geo = geo[geo.geolocation_lat.between(*BR_LAT) & geo.geolocation_lng.between(*BR_LON)]
geo_centroid = (geo.groupby("geolocation_zip_code_prefix")
                .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))
                .reset_index().rename(columns={"geolocation_zip_code_prefix": "zip"}))

prod = t["products"].merge(t["cat_trans"], on="product_category_name", how="left")
prod["category"] = (prod["product_category_name_english"]
                    .fillna(prod["product_category_name"]).fillna("unknown"))
prod["volume_cm3"] = prod.product_length_cm * prod.product_height_cm * prod.product_width_cm
prod["volumetric_weight_kg"] = prod.volume_cm3 / 5000.0
prod["weight_kg"] = prod.product_weight_g / 1000.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


sellers_geo = (t["sellers"].merge(geo_centroid, left_on="seller_zip_code_prefix",
                                  right_on="zip", how="left")
               .rename(columns={"lat": "sell_lat", "lng": "sell_lng"}))
cust_geo = (t["customers"].merge(geo_centroid, left_on="customer_zip_code_prefix",
                                 right_on="zip", how="left")
            .rename(columns={"lat": "cust_lat", "lng": "cust_lng"}))

item = (t["items"]
        .merge(t["orders"][["order_id", "customer_id", "order_status",
                            "order_purchase_timestamp"]], on="order_id", how="left")
        .merge(prod[["product_id", "category", "weight_kg", "volumetric_weight_kg", "volume_cm3"]],
               on="product_id", how="left")
        .merge(sellers_geo[["seller_id", "sell_lat", "sell_lng", "seller_state"]],
               on="seller_id", how="left")
        .merge(cust_geo[["customer_id", "cust_lat", "cust_lng", "customer_state"]],
               on="customer_id", how="left"))
item["distance_km"] = haversine_km(item.sell_lat, item.sell_lng, item.cust_lat, item.cust_lng)
item["freight_ratio"] = item.freight_value / item.price.replace(0, np.nan)

items_agg = (t["items"].groupby("order_id")
             .agg(n_items=("order_item_id", "count"), item_total=("price", "sum"),
                  freight_total=("freight_value", "sum"),
                  n_sellers=("seller_id", "nunique")).reset_index())
pay_agg = (t["payments"].groupby("order_id")
           .agg(payment_total=("payment_value", "sum"),
                max_installments=("payment_installments", "max"),
                n_payment_rows=("payment_sequential", "count")).reset_index())
pay_type = (t["payments"].sort_values("payment_value", ascending=False)
            .drop_duplicates("order_id")[["order_id", "payment_type"]]
            .rename(columns={"payment_type": "primary_payment_type"}))
rev = (t["reviews"].sort_values("review_creation_date")
       .drop_duplicates("order_id", keep="last")[["order_id", "review_score"]])
item_cat = (t["items"].merge(prod[["product_id", "category"]], on="product_id", how="left")
            .sort_values("price", ascending=False).drop_duplicates("order_id")
            [["order_id", "category"]])

order = (t["orders"]
         .merge(t["customers"][["customer_id", "customer_unique_id", "customer_state"]],
                on="customer_id", how="left")
         .merge(items_agg, on="order_id", how="left").merge(pay_agg, on="order_id", how="left")
         .merge(pay_type, on="order_id", how="left").merge(rev, on="order_id", how="left")
         .merge(item_cat, on="order_id", how="left"))
order["order_total"] = order.item_total.fillna(0) + order.freight_total.fillna(0)
order["freight_ratio"] = order.freight_total / order.item_total.replace(0, np.nan)
order["delivery_days"] = (order.order_delivered_customer_date - order.order_purchase_timestamp).dt.days
order["est_days"] = (order.order_estimated_delivery_date - order.order_purchase_timestamp).dt.days
order["days_late"] = (order.order_delivered_customer_date - order.order_estimated_delivery_date).dt.days
order["is_late"] = order.days_late > 0
order["purchase_month"] = order.order_purchase_timestamp.dt.to_period("M").astype(str)
order["purchase_dow"] = order.order_purchase_timestamp.dt.day_name()
order["purchase_hour"] = order.order_purchase_timestamp.dt.hour

delivered = order[order.order_status == "delivered"].copy()
d = delivered.dropna(subset=["delivery_days"]).copy()

# ============================ §4 temporal ============================
monthly = (delivered.groupby("purchase_month")
           .agg(orders=("order_id", "nunique"), revenue=("order_total", "sum"),
                late_rate=("is_late", "mean")).reset_index())
monthly = monthly.iloc[:-1]                       # drop partial final month (notebook)
monthly["late_rate"] = (monthly.late_rate * 100).round(1)
save(monthly, "monthly.csv")

oy = t["orders"].copy()
oy["year"] = oy.order_purchase_timestamp.dt.year
oy["mon"] = oy.order_purchase_timestamp.dt.month
yoy = oy.groupby(["mon", "year"]).order_id.count().reset_index(name="orders")
save(yoy, "yoy.csv")

dowh = order.groupby(["purchase_dow", "purchase_hour"]).size().reset_index(name="orders")
dowh.columns = ["dow", "hour", "orders"]
save(dowh, "dow_hour.csv")

# ============================ §5 delivery ============================
dd = delivered.copy()
dd["to_approval"] = (dd.order_approved_at - dd.order_purchase_timestamp).dt.total_seconds()/86400
dd["to_carrier"] = (dd.order_delivered_carrier_date - dd.order_approved_at).dt.total_seconds()/86400
dd["carrier_to_cust"] = (dd.order_delivered_customer_date - dd.order_delivered_carrier_date).dt.total_seconds()/86400
stages = dd[["to_approval", "to_carrier", "carrier_to_cust"]].mean()
leadtime = pd.DataFrame({"stage": ["purchase->approval", "approval->carrier", "carrier->customer"],
                         "avg_days": stages.values.round(2)})
save(leadtime, "leadtime.csv")

# delivery-days & days-late histograms, pre-binned (notebook cell 28's two charts)
def hist_df(series, lo, hi, bins):
    counts, edges = np.histogram(series.clip(lo, hi), bins=bins, range=(lo, hi))
    mids = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"bin_mid": mids.round(2), "count": counts.astype(int)})

save(hist_df(d.delivery_days, 0, 60, 40), "delivery_days_hist.csv")
mid = d.days_late[d.days_late.between(-30, 30)]        # notebook shows the +/-30d window only
save(hist_df(mid, -30, 30, 40), "days_late_hist.csv")

state = (d.groupby("customer_state")
         .agg(orders=("order_id", "count"), median_delivery_days=("delivery_days", "median"),
              late_rate=("is_late", "mean")).reset_index())
state["late_rate"] = (state.late_rate * 100).round(1)
stf = (item.dropna(subset=["freight_value"]).groupby("customer_state")
       .agg(median_freight=("freight_value", "median"), median_distance=("distance_km", "median"),
            median_burden=("freight_ratio", "median")).reset_index())
state = state.merge(stf, on="customer_state", how="left")
state["median_burden"] = (state.median_burden * 100).round(1)
state["median_freight"] = state.median_freight.round(2)
state["median_distance"] = state.median_distance.round(1)
save(state, "state_summary.csv")

# customer_state x seller_state lanes (geography drill-down)
il = item.merge(order[["order_id", "delivery_days"]], on="order_id", how="left")
lanes = (il.dropna(subset=["seller_state", "customer_state"])
         .groupby(["customer_state", "seller_state"])
         .agg(n=("freight_value", "size"), median_freight=("freight_value", "median"),
              median_distance=("distance_km", "median"),
              median_delivery_days=("delivery_days", "median")).reset_index())
lanes[["median_freight", "median_distance", "median_delivery_days"]] = \
    lanes[["median_freight", "median_distance", "median_delivery_days"]].round(1)
save(lanes, "lanes.csv")

# ============================ §6 customers / sellers / products / payments ============================
cust_orders = (order.groupby("customer_unique_id")
               .agg(orders=("order_id", "nunique"), spend=("order_total", "sum"),
                    first_month=("purchase_month", "min")))
repeat_rate = (cust_orders.orders > 1).mean()*100
repeat_rev_share = cust_orders.loc[cust_orders.orders > 1, "spend"].sum()/cust_orders.spend.sum()*100
rep = order[order.customer_unique_id.isin(cust_orders.index[cust_orders.orders > 1])]
gap = (rep.sort_values("order_purchase_timestamp").groupby("customer_unique_id")
       ["order_purchase_timestamp"].apply(lambda s: (s.iloc[1] - s.iloc[0]).days))
# orders per customer, bucketed 1..4,5+ (notebook cell 37 left panel)
od = cust_orders.orders.value_counts().sort_index()
od = pd.concat([od[od.index <= 4], pd.Series({5: od[od.index >= 5].sum()})])
cust_hist = pd.DataFrame({"orders_placed": [str(i) if i < 5 else "5+" for i in od.index],
                          "customers": od.values})
save(cust_hist, "cust_repeat.csv")
cs = order.customer_state.value_counts()
demand_top = (cs.head(10)/len(order)*100).round(1).rename_axis("customer_state").reset_index(name="pct_orders")
save(demand_top, "demand_state.csv")

seller_rev = item.groupby("seller_id")["price"].sum().sort_values(ascending=False)
cum = seller_rev.cumsum()/seller_rev.sum()
top_share = int((cum <= 0.8).sum())
qs = {p: float(cum.iloc[int(len(cum)*p/100)-1]*100) for p in (1, 5, 10, 20)}
pct = np.arange(1, len(cum)+1)/len(cum)*100
idx = np.unique(np.linspace(0, len(cum)-1, 250).astype(int))   # thin the curve for the CSV
save(pd.DataFrame({"pct_sellers": pct[idx].round(3), "cum_revenue": (cum.values[idx]*100).round(3)}),
     "seller_pareto.csv")
seller_items = item.groupby("seller_id")["order_item_id"].count()
save(seller_items.rename("items").reset_index(drop=True).to_frame(), "seller_items.csv")
x = item.dropna(subset=["seller_state", "customer_state"]).copy()
x["interstate"] = x.seller_state != x.customer_state
inter = x.groupby("interstate").agg(share=("order_id", "size"), med_freight=("freight_value", "median"),
                                    med_dist=("distance_km", "median"))
inter["share"] = inter.share/len(x)*100

cat = (item.groupby("category")
       .agg(lines=("order_item_id", "count"), revenue=("price", "sum"),
            med_price=("price", "median"), med_freight=("freight_value", "median"),
            med_ratio=("freight_ratio", "median"), med_weight=("weight_kg", "median")).reset_index())
save(cat.round(3), "category_summary.csv")
big = cat[cat.lines >= 300]
corr_price_burden = float(np.corrcoef(np.log(big.med_price), big.med_ratio)[0, 1])
pw = item.dropna(subset=["weight_kg", "price"])
pw = pw[(pw.weight_kg > 0) & (pw.price > 0)]
spearman_wp = float(pw[["weight_kg", "price"]].corr(method="spearman").iloc[0, 1])
# small sample for the weight-vs-price hexbin/scatter
save(pw.sample(min(9000, len(pw)), random_state=42)[["weight_kg", "price"]].round(2),
     "weight_price_sample.csv")

pay = t["payments"]
pv = pay.payment_type.value_counts(normalize=True)*100
pv = pv[pv >= 0.1]
save(pv.round(1).rename_axis("payment_type").reset_index(name="pct"), "payment_type.csv")
card = pay[(pay.payment_type == "credit_card") & (pay.payment_installments > 0)]
ic = card.payment_installments.clip(upper=11).value_counts().sort_index()
inst = pd.DataFrame({"installments": [str(int(i)) if i < 11 else "11+" for i in ic.index],
                     "pct": (ic/ic.sum()*100).round(1).values})
save(inst, "installments.csv")
inst_val = order.dropna(subset=["max_installments", "order_total"])
inst_val = inst_val[inst_val.max_installments.between(1, 12)]
by_inst = inst_val.groupby("max_installments").agg(median_order_value=("order_total", "median"),
                                                   n=("order_id", "size")).reset_index()
by_inst = by_inst[by_inst.n >= 500]
save(by_inst.round(1), "order_value_by_installment.csv")

# ============================ §7 freight economics ============================
f = item.dropna(subset=["freight_ratio"]).copy()
f = f[(f.price > 0) & (f.freight_value >= 0)]
median_freight_ratio = f.freight_ratio.median()*100
mean_freight_ratio = f.freight_ratio.mean()*100
pct_high_freight = (f.freight_ratio > 0.25).mean()*100
freight_pct_gmv = item.freight_value.sum()/(item.freight_value.sum()+item.price.sum())*100

# freight value + ratio distribution sample
save(f.sample(min(12000, len(f)), random_state=42)[["freight_value", "freight_ratio"]].round(3),
     "freight_dist.csv")

core = f.dropna(subset=["distance_km", "weight_kg", "volumetric_weight_kg"])
core = core[core.weight_kg > 0].copy()
core["chargeable_kg"] = np.maximum(core.weight_kg, core.volumetric_weight_kg)
corr = {
    "weight": float(core[["weight_kg", "freight_value"]].corr().iloc[0, 1]),
    "volume": float(core[["volumetric_weight_kg", "freight_value"]].corr().iloc[0, 1]),
    "distance": float(core[["distance_km", "freight_value"]].corr().iloc[0, 1]),
    "price": float(core[["price", "freight_value"]].corr().iloc[0, 1]),
    "chargeable": float(core[["chargeable_kg", "freight_value"]].corr().iloc[0, 1]),
}
# scatter sample for the "what drives freight" chart (freight vs distance/weight/volume/price)
sc = core[["freight_value", "price", "distance_km", "weight_kg", "volume_cm3",
           "freight_ratio", "category", "customer_state"]].copy()
sc = sc.sample(min(9000, len(sc)), random_state=42)
sc["freight_ratio"] = sc.freight_ratio.round(3)
save(sc.round(2), "freight_sample.csv")

# §7 driver correlation matrix
cols = ["price", "freight_value", "freight_ratio", "weight_kg", "volumetric_weight_kg", "distance_km"]
cm7 = core[cols].corr().round(2)
cm7.insert(0, "var", cm7.index)
save(cm7, "corr_freight.csv")

# determinism: spread within near-identical weight x volume x distance bands (notebook cell 56)
db = core[(core.volume_cm3 > 0) & (core.freight_value > 0)].copy()
db["wbin"] = pd.cut(db.weight_kg*1000, [0, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, np.inf])
db["vbin"] = pd.cut(db.volume_cm3, [0, 1000, 2000, 4000, 8000, 16000, 32000, np.inf])
db["dbin"] = pd.cut(db.distance_km, [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, np.inf])
grp = db.groupby(["wbin", "vbin", "dbin"], observed=True)["freight_value"]
cell = grp.agg(n="size", p10=lambda s: s.quantile(.1), p50="median", p90=lambda s: s.quantile(.9))
cell = cell[cell.n >= 30].copy()
cell["p90_over_p10"] = cell.p90/cell.p10
bigcell = db.groupby(["wbin", "vbin", "dbin"], observed=True)["freight_value"].transform("size") >= 30
sub = db[bigcell]
cmean = sub.groupby(["wbin", "vbin", "dbin"], observed=True).freight_value.transform("mean")
explained = 1 - ((sub.freight_value - cmean)**2).sum()/((sub.freight_value - sub.freight_value.mean())**2).sum()
det = {"n_cells": int(len(cell)), "p10": round(float(cell.p10.median()), 1),
       "p90": round(float(cell.p90.median()), 1), "ratio": round(float(cell.p90_over_p10.median()), 2),
       "explained_pct": round(float(explained)*100), "residual_pct": round(float(1-explained)*100)}
key = cell.sort_values("n", ascending=False).index[0]
ex = db[(db.wbin == key[0]) & (db.vbin == key[1]) & (db.dbin == key[2])]
ex = (ex.sample(min(8, len(ex)), random_state=3)
      [["weight_kg", "volume_cm3", "distance_km", "freight_value", "category"]]
      .sort_values("freight_value").round(1))
save(ex, "determinism_sample.csv")
# p90/p10 spread across the well-populated cells, for a histogram of the spread
save(cell.reset_index(drop=True)[["n", "p10", "p50", "p90", "p90_over_p10"]].round(2),
     "determinism_cells.csv")

# regressive price bands (notebook cell 58)
band = pd.cut(f.price, [0, 30, 60, 100, 200, 500, np.inf],
              labels=["<30", "30-60", "60-100", "100-200", "200-500", "500+"])
pb = f.groupby(band, observed=True).agg(median_freight=("freight_value", "median"),
                                        median_burden=("freight_ratio", "median"),
                                        n=("freight_ratio", "size")).reset_index()
pb.columns = ["price_band", "median_freight", "median_burden", "n"]
pb["median_burden"] = (pb.median_burden*100).round(1)
save(pb, "price_band.csv")

# category burden (notebook cell 60, n>=200)
catburden = (f.groupby("category").agg(median_burden=("freight_ratio", "median"),
                                       median_freight=("freight_value", "median"),
                                       median_price=("price", "median"),
                                       n=("freight_ratio", "size"))
             .query("n >= 200").sort_values("median_burden", ascending=False).reset_index())
catburden["median_burden"] = (catburden.median_burden*100).round(1)
save(catburden, "category_freight.csv")

# ============================ §8 reviews ============================
save(t["reviews"].review_score.value_counts().sort_index().rename_axis("review_score")
     .reset_index(name="n"), "review_score.csv")
review_title_null = t["reviews"].review_comment_title.isna().mean()*100
review_msg_null = t["reviews"].review_comment_message.isna().mean()*100

reviews_monthly = (t["reviews"].dropna(subset=["review_answer_timestamp"])
                   .assign(m=lambda x: x.review_answer_timestamp.dt.to_period("M").astype(str))
                   .groupby("m").review_id.count().reset_index(name="num_reviews"))
orders_monthly = (t["orders"].assign(m=lambda x: x.order_delivered_customer_date.dt.to_period("M").astype(str))
                  .groupby("m").order_id.nunique().reset_index(name="num_orders"))
rm = reviews_monthly.merge(orders_monthly, on="m", how="outer").sort_values("m")
rm = rm[rm.m != "NaT"]
save(rm.rename(columns={"m": "month"}), "review_monthly.csv")

# order-level correlation matrix (notebook cell 71)
rel = order[(order.order_status == "delivered") & order.review_score.notna()].copy()
num = ["order_total", "freight_total", "freight_ratio", "n_items",
       "delivery_days", "days_late", "review_score", "max_installments"]
cm8 = rel[num].corr().round(2)
cm8.insert(0, "var", cm8.index)
save(cm8, "corr_reviews.csv")

# mean review score across delivery / days-late / freight bins (notebook cell 72 bins)
def binned_mean(series_col, bins, labels, name):
    b = pd.cut(rel[series_col], bins=bins, labels=labels, right=False)
    g = rel.groupby(b, observed=True).agg(mean_review=("review_score", "mean"),
                                          n=("review_score", "size")).reset_index()
    g.columns = [name, "mean_review", "n"]
    g["mean_review"] = g.mean_review.round(2)
    return g

def binned_box(series_col, bins, labels, name):
    """Per-bin box-plot statistics (Tukey whiskers, no outliers) — matches the notebook's boxplots."""
    b = pd.cut(rel[series_col], bins=bins, labels=labels, right=False)
    rows = []
    for lab, grp in rel.groupby(b, observed=True):
        s = grp.review_score.dropna()
        if len(s) == 0:
            continue
        q1, med, q3 = s.quantile([.25, .5, .75])
        iqr = q3 - q1
        lo = float(s[s >= q1 - 1.5*iqr].min())
        hi = float(s[s <= q3 + 1.5*iqr].max())
        rows.append({name: lab, "q1": q1, "median": med, "q3": q3,
                     "lowerfence": lo, "upperfence": hi, "n": int(len(s))})
    return pd.DataFrame(rows)


DRIVER_BINS = {
    "delivery": ("delivery_days", [0, 5, 10, 15, 20, 30, rel.delivery_days.max()+1],
                 ["0-5", "6-10", "11-15", "16-20", "21-30", ">30"], "delivery_bin"),
    "days_late": ("days_late", [rel.days_late.min()-1, -10, -5, 0, 5, 10, 20, rel.days_late.max()+1],
                  ["< -10", "-10 to -5", "-5 to 0", "0 to 5", "5 to 10", "10 to 20", "> 20"], "days_late_bin"),
    "freight": ("freight_total", [0, 15, 25, 40, 60, rel.freight_total.max()+1],
                ["0-15", "15-25", "25-40", "40-60", ">60"], "freight_bin"),
    "freight_ratio": ("freight_ratio", [0, 0.1, 0.2, 0.3, 0.5, 1.0, rel.freight_ratio.max()+0.1],
                      ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-1.0", ">1.0"], "freight_ratio_bin"),
}
for key, (col, bins, labels, name) in DRIVER_BINS.items():
    save(binned_box(col, bins, labels, name), f"review_box_{key}.csv")

save(binned_mean("delivery_days", [0, 5, 10, 15, 20, 30, rel.delivery_days.max()+1],
                 ["0-5", "6-10", "11-15", "16-20", "21-30", ">30"], "delivery_bin"),
     "review_by_delivery.csv")
save(binned_mean("days_late", [rel.days_late.min()-1, -10, -5, 0, 5, 10, 20, rel.days_late.max()+1],
                 ["< -10", "-10 to -5", "-5 to 0", "0 to 5", "5 to 10", "10 to 20", "> 20"], "days_late_bin"),
     "review_by_days_late.csv")
save(binned_mean("freight_total", [0, 15, 25, 40, 60, rel.freight_total.max()+1],
                 ["0-15", "15-25", "25-40", "40-60", ">60"], "freight_bin"),
     "review_by_freight.csv")
save(binned_mean("freight_ratio", [0, 0.1, 0.2, 0.3, 0.5, 1.0, rel.freight_ratio.max()+0.1],
                 ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-1.0", ">1.0"], "freight_ratio_bin"),
     "review_by_freight_ratio.csv")

# review-score means used in the §10 narrative (on-time vs late)
ontime_star = float(rel.loc[~rel.is_late, "review_score"].mean())
late_star = float(rel.loc[rel.is_late, "review_score"].mean())
corr_review_delivery = float(rel[["review_score", "delivery_days"]].corr().iloc[0, 1])
corr_review_freight = float(rel[["review_score", "freight_ratio"]].corr().iloc[0, 1])

# ============================ FACTS (headline numbers, verbatim to the app) ============================
facts = {
    # §1
    "n_orders": int(t["orders"].order_id.nunique()), "n_items": int(len(t["items"])),
    "n_payments": int(len(t["payments"])), "n_reviews": int(len(t["reviews"])),
    "n_customer_rows": int(t["customers"].customer_id.nunique()),
    "n_customer_unique": int(t["customers"].customer_unique_id.nunique()),
    "n_sellers": int(t["sellers"].seller_id.nunique()), "n_products": int(len(t["products"])),
    "n_geo_rows": int(len(t["geolocation"])), "n_geo_zip": int(t["geolocation"].geolocation_zip_code_prefix.nunique()),
    "n_categories_trans": int(len(t["cat_trans"])),
    "orders_in_items": int(t["items"].order_id.nunique()), "max_items_in_order": int(t["items"].order_item_id.max()),
    "date_min": str(t["orders"].order_purchase_timestamp.min().date()),
    "date_max": str(t["orders"].order_purchase_timestamp.max().date()),
    # §2
    "city_plain": city_plain, "city_accented": city_accented,
    "item_price_p50": round(float(it0.price.median()), 2), "item_price_p99": round(float(it0.price.quantile(.99)), 2),
    "item_price_max": round(float(it0.price.max()), 2),
    "freight_p50": round(float(it0.freight_value.median()), 2), "freight_p99": round(float(it0.freight_value.quantile(.99)), 2),
    "freight_max": round(float(it0.freight_value.max()), 2),
    "freight_gt_price_pct": round(float((it0.fr > 1).mean()*100), 1),
    "review_title_null": round(review_title_null, 2), "review_msg_null": round(review_msg_null, 2),
    # §5
    "pct_delivered": round(len(delivered)/len(order)*100, 1),
    "late_rate": round(float(d.is_late.mean()*100), 1), "early_rate": round(float((d.days_late < 0).mean()*100), 1),
    "median_days_late": int(d.days_late.median()),
    "median_delivery_days": float(d.delivery_days.median()), "mean_delivery_days": round(float(d.delivery_days.mean()), 1),
    "max_delivery_days": int(d.delivery_days.max()),
    "lead_to_approval": round(float(stages.to_approval), 2), "lead_to_carrier": round(float(stages.to_carrier), 2),
    "lead_carrier_to_cust": round(float(stages.carrier_to_cust), 2),
    "state_fast": "SP", "state_fast_days": int(state.set_index("customer_state").loc["SP", "median_delivery_days"]),
    "state_fast_late": float(state.set_index("customer_state").loc["SP", "late_rate"]),
    "state_late_top": "AL", "state_late_top_rate": float(state.set_index("customer_state").loc["AL", "late_rate"]),
    # §6
    "repeat_rate": round(float(repeat_rate), 2), "repeat_rev_share": round(float(repeat_rev_share), 1),
    "repeat_gap_median": int(gap.median()),
    "demand_sp": int(round(cs.head(1).iloc[0]/len(order)*100)),
    "seller_top80_n": top_share, "seller_top80_pct": round(top_share/len(seller_rev)*100),
    "seller_top1": round(qs[1]), "seller_top5": round(qs[5]), "seller_top10": round(qs[10]), "seller_top20": round(qs[20]),
    "seller_items_median": int(seller_items.median()), "seller_items_p90": int(seller_items.quantile(.9)),
    "seller_items_max": int(seller_items.max()), "seller_le10_pct": round(float((seller_items <= 10).mean()*100)),
    "interstate_pct": round(float(inter.loc[True, "share"])), "interstate_dist": round(float(inter.loc[True, "med_dist"])),
    "interstate_freight": round(float(inter.loc[True, "med_freight"]), 2),
    "intrastate_dist": round(float(inter.loc[False, "med_dist"])), "intrastate_freight": round(float(inter.loc[False, "med_freight"]), 2),
    "n_categories": int(len(cat)),
    "top10_vol_pct": round(cat.sort_values("lines", ascending=False).head(10).lines.sum()/cat.lines.sum()*100),
    "top10_rev_pct": round(cat.sort_values("revenue", ascending=False).head(10).revenue.sum()/cat.revenue.sum()*100),
    "corr_price_burden": round(corr_price_burden, 2), "spearman_weight_price": round(spearman_wp, 2),
    "pay_credit": round(float(pv.get("credit_card", np.nan)), 1), "pay_boleto": round(float(pv.get("boleto", np.nan)), 1),
    "pay_voucher": round(float(pv.get("voucher", np.nan)), 1), "pay_debit": round(float(pv.get("debit_card", np.nan)), 1),
    "inst_pct": round(float((card.payment_installments > 1).mean()*100)),
    "inst_median": int(card.payment_installments.median()), "inst_p90": int(card.payment_installments.quantile(.9)),
    "ordval_1inst": round(by_inst.iloc[0].median_order_value), "ordval_maxinst": round(by_inst.iloc[-1].median_order_value),
    "ordval_maxinst_n": int(by_inst.iloc[-1].max_installments),
    # §7
    "freight_pct_gmv": round(float(freight_pct_gmv), 1), "median_freight_ratio": round(float(median_freight_ratio), 1),
    "mean_freight_ratio": round(float(mean_freight_ratio), 1), "pct_high_freight": round(float(pct_high_freight), 1),
    "corr_freight_weight": round(corr["weight"], 3), "corr_freight_volume": round(corr["volume"], 3),
    "corr_freight_distance": round(corr["distance"], 3), "corr_freight_price": round(corr["price"], 3),
    "corr_freight_chargeable": round(corr["chargeable"], 3),
    "det_cells": det["n_cells"], "det_p10": det["p10"], "det_p90": det["p90"], "det_ratio": det["ratio"],
    "det_explained": det["explained_pct"], "det_residual": det["residual_pct"],
    # §8
    "review_text_empty": round(review_msg_null, 1), "review_title_empty": round(review_title_null, 2),
    "corr_review_delivery": round(corr_review_delivery, 3), "corr_review_freight": round(corr_review_freight, 3),
    "review_ontime_star": round(ontime_star, 1), "review_late_star": round(late_star, 1),
}
json.dump(facts, open(os.path.join(OUT, "facts.json"), "w"), indent=2)
print("wrote facts.json")

# ============================ ASSERTIONS: recomputed == notebook ============================
# If any of these fail, an aggregate has drifted from Team5_Phase1_IT5006_AY2627Sem1.ipynb.
NB = {  # transcribed from the final notebook's cell outputs
    "n_orders": 99441, "n_items": 112650, "n_customer_unique": 96096, "n_sellers": 3095,
    "orders_in_items": 98666, "max_items_in_order": 21, "date_max": "2018-10-17",
    "city_plain": 135800, "city_accented": 24919, "freight_max": 409.68,
    "pct_delivered": 97.0, "late_rate": 6.8, "early_rate": 91.9, "median_days_late": -12,
    "median_delivery_days": 10.0, "mean_delivery_days": 12.1, "max_delivery_days": 209,
    "lead_carrier_to_cust": 9.33, "state_fast_days": 7, "state_late_top_rate": 21.4,
    "repeat_rate": 3.12, "repeat_rev_share": 5.8, "repeat_gap_median": 27,
    "seller_top80_pct": 18, "seller_top10": 67, "seller_items_median": 8, "seller_le10_pct": 58,
    "interstate_pct": 64, "n_categories": 74, "top10_vol_pct": 64, "corr_price_burden": -0.86,
    "spearman_weight_price": 0.51, "pay_credit": 73.9, "pay_boleto": 19.0,
    "freight_pct_gmv": 14.2, "median_freight_ratio": 23.1, "mean_freight_ratio": 32.1, "pct_high_freight": 46.0,
    "corr_freight_weight": 0.612, "corr_freight_distance": 0.390, "corr_freight_chargeable": 0.610,
    "det_cells": 430, "det_p10": 14.1, "det_p90": 23.3, "det_ratio": 1.72, "det_explained": 53,
    "review_text_empty": 58.7, "review_title_empty": 88.34,
}
fail = [(k, facts[k], v) for k, v in NB.items() if facts.get(k) != v]
if fail:
    print("\n!! MISMATCH vs notebook (key, computed, notebook):")
    for row in fail:
        print("  ", row)
    raise SystemExit("Aggregates drifted from the notebook — fix before shipping.")
print("\nAll headline numbers match the notebook.")
print("Total data size:", round(sum(os.path.getsize(os.path.join(OUT, x))
      for x in os.listdir(OUT) if os.path.isfile(os.path.join(OUT, x)))/1024, 1), "KB")
