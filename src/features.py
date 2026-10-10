"""Phase 2 data loading and feature engineering.

Two problems, two tables.

Problem 1 (regression, ITEM grain)     : build_item_table()
    reg_target = log1p(freight_value)   -> "fair freight" at checkout
Problem 2 (classification, ORDER grain): build_order_table()
    clf_target = is_late                -> delivered on a calendar day after the promised date
    (same definition as Phase 1: (delivered - estimated).dt.days > 0, delivered orders only;
    the estimated date is a midnight timestamp, so this means "after the promised day")

Leakage rules (the full audit is a table in Section 3 of the notebook)
---------------------------------------------------------------------
* Problem 1: freight_value is the target and is never a predictor.
* Problem 2: every predictor is known when the order is placed, including the PROMISED delivery date
  (order_estimated_delivery_date is set at checkout). Carrier hand-over, approval, delivery and
  review fields are never used as predictors of the order itself.
* Seller identity (Problem 1 only) is tested through a TargetEncoder inside the sklearn Pipeline
  (cross-fitted, fitted on training folds only) and excluded from the model by design.
"""
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42

BR_LAT, BR_LON = (-34.0, 5.5), (-74.0, -34.0)

NUM_CORE = ["weight_kg", "volume_cm3", "chargeable_kg", "distance_km"]
NUM_PRICE = ["price", "n_items", "n_sellers"]
NUM_LISTING = ["photos_qty", "name_len", "desc_len"]
NUM_TIME = ["purchase_month", "purchase_dow", "purchase_hour"]
CAT_GEO = ["seller_state", "customer_state"]
CAT_PROD = ["category"]
BIN_GEO = ["same_state"]

# ---- Problem 1 (freight regression, item grain) -------------------------------------------
FEATURE_SETS = {
    "0 Price only (reference)": dict(num=["price"], cat=[], te=[]),
    "1 Core (weight, volume, distance)": dict(num=NUM_CORE, cat=[], te=[]),
    "2 + category & states": dict(num=NUM_CORE + BIN_GEO, cat=CAT_GEO + CAT_PROD, te=[]),
    "3 + price & basket": dict(num=NUM_CORE + BIN_GEO + NUM_PRICE, cat=CAT_GEO + CAT_PROD, te=[]),
    "4 + listing & timing": dict(num=NUM_CORE + BIN_GEO + NUM_PRICE + NUM_LISTING + NUM_TIME,
                                 cat=CAT_GEO + CAT_PROD, te=[]),
    "5 + seller (target-encoded)": dict(num=NUM_CORE + BIN_GEO + NUM_PRICE + NUM_LISTING + NUM_TIME,
                                        cat=CAT_GEO + CAT_PROD, te=["seller_id"]),
}
FULL_SET = "4 + listing & timing"

# ---- Problem 2 (is_late classification, order grain) --------------------------------------
# Forward selection (pipeline.ablation_classification): start from the promise and add the groups below in this
# order; a group is kept only if its paired fold-by-fold gain in PR-AUC lift exceeds its fold sd for at least one
# family. purchase_month is tested last although it is not expected to help: only one November has delivered orders
# (2017), and February-March was late in 2018 (14-19%) but not in 2017 (3-5%), so month marks one-off episodes.
L_ROUTE = ["distance_km", "same_state"]
L_PARCEL = ["weight_kg", "volume_cm3", "chargeable_kg"]
L_ORDER = ["price_total", "freight_total", "freight_ratio", "n_items", "n_sellers"]
L_TIME = ["purchase_dow", "purchase_hour"]
L_LOAD = ["platform_surge_7d", "seller_load_7d"]
LATE_BASE = dict(num=["promised_days"], cat=[], te=[])
LATE_GROUPS = {
    "route (distance, states)": dict(num=L_ROUTE, cat=CAT_GEO, te=[]),
    "parcel & category": dict(num=L_PARCEL, cat=CAT_PROD, te=[]),
    "order value & freight": dict(num=L_ORDER, cat=[], te=[]),
    "purchase weekday & hour": dict(num=L_TIME, cat=[], te=[]),
    "congestion (7-day volume surge)": dict(num=L_LOAD, cat=[], te=[]),
    "purchase month": dict(num=["purchase_month"], cat=[], te=[]),
}

# Problem 2 validation: time-ordered, with an embargo so that no training label is "from the future"
EMBARGO_DAYS = 30        # 95% of orders are delivered within 30 days of purchase
TEST_FRAC = 0.2
REVIEW_SHARE = 0.10      # decision rule: flag the riskiest 10% of orders (an assumed review capacity)
# Rolling-origin test windows (about three months each); the fourth window starts at the main hold-out cut-off
ROLLING_STARTS = ["2017-09-01", "2017-12-01", "2018-03-01"]

def find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.environ.get("OLIST_DATA_DIR"),
              "/content/drive/MyDrive/IT5006_Project-Data/Olist_CSV",
              os.path.join(here, "..", "data"), "data", "../data",
              os.path.join(here, "..", "..", "IT5006_Project-Data", "Olist_CSV")]:
        if c and os.path.exists(os.path.join(c, "olist_orders_dataset.csv")):
            return c
    raise FileNotFoundError("Set OLIST_DATA_DIR to the folder holding the nine Olist CSVs.")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _tables(d):
    rd = lambda f, **k: pd.read_csv(os.path.join(d, f), **k)
    return dict(
        items=rd("olist_order_items_dataset.csv"),
        orders=rd("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                                           "order_estimated_delivery_date"]),
        cust=rd("olist_customers_dataset.csv", usecols=["customer_id", "customer_zip_code_prefix", "customer_state"]),
        sell=rd("olist_sellers_dataset.csv"), prod=rd("olist_products_dataset.csv"),
        trans=rd("product_category_name_translation.csv"), geo=rd("olist_geolocation_dataset.csv"))


def _item_frame(d):
    """Items joined to orders, customers, sellers, products and zip-centroid distance."""
    t = _tables(d)
    geo = t["geo"]
    geo = geo[geo.geolocation_lat.between(*BR_LAT) & geo.geolocation_lng.between(*BR_LON)]
    cen = (geo.groupby("geolocation_zip_code_prefix")
           .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median")).reset_index())

    prod = t["prod"].merge(t["trans"], on="product_category_name", how="left")
    prod["category"] = prod.product_category_name_english.fillna(prod.product_category_name).fillna("unknown")
    prod["volume_cm3"] = prod.product_length_cm * prod.product_height_cm * prod.product_width_cm
    prod["weight_kg"] = prod.product_weight_g / 1000.0
    prod["chargeable_kg"] = np.maximum(prod.weight_kg, prod.volume_cm3 / 5000.0)   # volumetric rule
    prod = prod.rename(columns={"product_photos_qty": "photos_qty", "product_name_lenght": "name_len",
                                "product_description_lenght": "desc_len"})

    sell = t["sell"].merge(cen, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")
    sell = sell.rename(columns={"lat": "s_lat", "lng": "s_lng"})[["seller_id", "seller_state", "s_lat", "s_lng"]]
    cust = t["cust"].merge(cen, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")
    cust = cust.rename(columns={"lat": "c_lat", "lng": "c_lng"})[["customer_id", "customer_state", "c_lat", "c_lng"]]

    items = t["items"]
    basket = items.groupby("order_id").agg(n_items=("order_item_id", "count"),
                                           n_sellers=("seller_id", "nunique")).reset_index()
    df = (items.merge(t["orders"][["order_id", "customer_id", "order_purchase_timestamp",
                                   "order_estimated_delivery_date", "order_delivered_customer_date"]], on="order_id")
          .merge(cust, on="customer_id").merge(sell, on="seller_id")
          .merge(prod[["product_id", "category", "weight_kg", "volume_cm3", "chargeable_kg",
                       "photos_qty", "name_len", "desc_len"]], on="product_id", how="left")
          .merge(basket, on="order_id"))
    df["distance_km"] = haversine_km(df.s_lat, df.s_lng, df.c_lat, df.c_lng)
    df["same_state"] = (df.seller_state == df.customer_state).astype(int)
    ts = df.order_purchase_timestamp
    df["purchase_month"], df["purchase_dow"], df["purchase_hour"] = ts.dt.month, ts.dt.dayofweek, ts.dt.hour
    return df[df.price > 0].copy()                                 # freight/price undefined at price 0


def _seconds(s):
    return s.values.astype("datetime64[ns]").astype("int64") / 1e9


def build_item_table(data_dir=None):
    """Problem 1: order-item table with reg_target = log1p(freight_value)."""
    df = _item_frame(data_dir or find_data_dir())
    df["reg_target"] = np.log1p(df.freight_value)
    df["freight_ratio"] = df.freight_value / df.price
    df["purchase_ts"] = _seconds(df.order_purchase_timestamp)
    keep = (["order_id", "order_item_id", "product_id", "seller_id", "order_purchase_timestamp", "purchase_ts",
             "freight_value", "freight_ratio", "reg_target"]
            + sorted(set(NUM_CORE + NUM_PRICE + NUM_LISTING + NUM_TIME + BIN_GEO + CAT_GEO + CAT_PROD)))
    return df[keep].reset_index(drop=True)


def load_low_reviews(data_dir=None):
    """order_id -> low (review score 1-2), latest review per order.

    Used ONLY to measure the business impact of the Problem 2 ranking, never as a feature: the review is written
    after the order's outcome is known.
    """
    d = data_dir or find_data_dir()
    rv = pd.read_csv(os.path.join(d, "olist_order_reviews_dataset.csv"), usecols=["order_id", "review_score", "review_creation_date"],
                     parse_dates=["review_creation_date"])
    rv = rv.sort_values("review_creation_date").drop_duplicates("order_id", keep="last")
    return rv.assign(low=(rv.review_score <= 2).astype(float))[["order_id", "low"]]


def build_order_table(data_dir=None):
    """Problem 2: one row per DELIVERED order with clf_target = is_late (Phase 1 definition)."""
    d = data_dir or find_data_dir()
    df = _item_frame(d)
    df = df[df.order_delivered_customer_date.notna() & df.order_estimated_delivery_date.notna()]
    first = (df.sort_values(["order_id", "price"], ascending=[True, False])
             .drop_duplicates("order_id")[["order_id", "customer_id", "seller_id", "seller_state", "category",
                                           "customer_state", "same_state", "order_purchase_timestamp",
                                           "purchase_month", "purchase_dow", "purchase_hour",
                                           "order_estimated_delivery_date", "order_delivered_customer_date"]])
    agg = df.groupby("order_id").agg(
        price_total=("price", "sum"), freight_total=("freight_value", "sum"), weight_kg=("weight_kg", "sum"),
        volume_cm3=("volume_cm3", "sum"), chargeable_kg=("chargeable_kg", "sum"), distance_km=("distance_km", "max"),
        n_items=("n_items", "first"), n_sellers=("n_sellers", "first")).reset_index()
    o = first.merge(agg, on="order_id")

    # congestion: orders placed in the 7 days BEFORE this purchase (all orders, whatever their later fate)
    allo = pd.read_csv(os.path.join(d, "olist_orders_dataset.csv"), usecols=["order_id", "order_purchase_timestamp"],
                       parse_dates=["order_purchase_timestamp"])
    t7 = np.sort(allo.order_purchase_timestamp.values.astype("datetime64[ns]").astype("int64"))
    q = o.order_purchase_timestamp.values.astype("datetime64[ns]").astype("int64")
    w = int(7 * 86400 * 1e9)
    o["platform_load_7d"] = np.searchsorted(t7, q, side="left") - np.searchsorted(t7, q - w, side="left")
    # the platform grows over time, so raw volume is not stationary; the surge ratio compares the last week
    # with the average week of the previous 28 days (1.0 = normal; about 2.5 just after Black Friday 2017).
    # Clipped at 3 because the tiny 2016 volumes give meaningless ratios in the hundreds.
    load_28d = np.searchsorted(t7, q - w, side="left") - np.searchsorted(t7, q - 5 * w, side="left")
    o["platform_surge_7d"] = np.clip(o.platform_load_7d / np.maximum(load_28d / 4, 1), 0, 3)
    so = (pd.read_csv(os.path.join(d, "olist_order_items_dataset.csv"), usecols=["order_id", "seller_id"])
          .drop_duplicates().merge(allo, on="order_id"))
    seller_times = {k: np.sort(v.values.astype("datetime64[ns]").astype("int64"))
                    for k, v in so.groupby("seller_id").order_purchase_timestamp}
    o["seller_load_7d"] = [int(np.searchsorted(seller_times[sid], qi, side="left")
                               - np.searchsorted(seller_times[sid], qi - w, side="left"))
                           for sid, qi in zip(o.seller_id, q)]
    o["freight_ratio"] = o.freight_total / o.price_total
    o["promised_days"] = (o.order_estimated_delivery_date - o.order_purchase_timestamp).dt.total_seconds() / 86400
    days_late = (o.order_delivered_customer_date - o.order_estimated_delivery_date).dt.days
    o["clf_target"] = (days_late > 0).astype(int)
    o["days_late"] = days_late                                     # reporting only, never a predictor
    o["purchase_ts"] = _seconds(o.order_purchase_timestamp)

    cu = pd.read_csv(os.path.join(d, "olist_customers_dataset.csv"), usecols=["customer_id", "customer_unique_id"])
    o = o.merge(cu, on="customer_id", how="left")                  # the real person, for split checks only
    drop = ["order_estimated_delivery_date", "order_delivered_customer_date", "customer_id"]
    return o.drop(columns=drop).sort_values("purchase_ts", kind="stable").reset_index(drop=True)
