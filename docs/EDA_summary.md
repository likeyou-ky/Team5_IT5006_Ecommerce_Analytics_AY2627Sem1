# Olist E-Commerce — EDA Summary

Theme: **Freight Economics & Pricing** (classification + regression)


## Part A — Data Quality Profile

| Table | Rows | Cols | Total missing | Dup rows |
|---|---:|---:|---:|---:|
| customers | 99,441 | 5 | 0 | 0 |
| geolocation | 1,000,163 | 5 | 0 | 261,831 |
| orders | 99,441 | 8 | 4,908 | 0 |
| order_items | 112,650 | 7 | 0 | 0 |
| payments | 103,886 | 5 | 0 | 0 |
| reviews | 99,224 | 7 | 145,903 | 0 |
| products | 32,951 | 9 | 2,448 | 0 |
| sellers | 3,095 | 4 | 0 | 0 |
| cat_translation | 71 | 2 | 0 | 0 |

**Columns with missing values:**

- `orders.order_approved_at`: 160 (0.2%)
- `orders.order_delivered_carrier_date`: 1,783 (1.8%)
- `orders.order_delivered_customer_date`: 2,965 (3.0%)
- `reviews.review_comment_title`: 87,656 (88.3%)
- `reviews.review_comment_message`: 58,247 (58.7%)
- `products.product_category_name`: 610 (1.9%)
- `products.product_name_lenght`: 610 (1.9%)
- `products.product_description_lenght`: 610 (1.9%)
- `products.product_photos_qty`: 610 (1.9%)
- `products.product_weight_g`: 2 (0.0%)
- `products.product_length_cm`: 2 (0.0%)
- `products.product_height_cm`: 2 (0.0%)
- `products.product_width_cm`: 2 (0.0%)

**Key integrity / notable facts:**

- customers: `customer_id` unique per order = True; `customer_unique_id` distinct people = 96,096
- orders: order_status distribution -> {'delivered': 96478, 'shipped': 1107, 'canceled': 625, 'unavailable': 609, 'invoiced': 314, 'processing': 301, 'created': 5, 'approved': 2}
- order_items: 98,666 distinct orders, max items in one order = 21
- geolocation: 19,015 distinct zip prefixes (from 1,000,163 rows -> 261,831 exact dups)
- products: 73 categories; translation table covers 71

## Part B — Analytical Base Table (order-item grain)

- Base table shape: **112,650 rows × 33 cols** (one row per order-item line)
- Coverage of key engineered fields (non-null %):
    - `freight_value`: 100.0%
    - `price`: 100.0%
    - `freight_ratio`: 100.0%
    - `volumetric_weight_kg`: 100.0%
    - `weight_kg`: 100.0%
    - `distance_km`: 99.5%
    - `product_category_english`: 100.0%
- Saved merged base table -> `EDA_output/order_item_base_table.csv`

## Part C — Freight Economics Findings

- Price:  median R$74.99, mean R$120.65, p95 R$349.90
- Freight: median R$16.26, mean R$19.99
- Freight-to-price ratio: median 0.231, mean 0.321
- **46.0%** of order-items have freight > 25% of price (class-imbalance for the classification target)
- Correlation distance_km vs freight_value: 0.389
- Correlation weight_kg vs freight_value: 0.610
- Correlation volumetric_weight_kg vs freight_value: 0.587

## Figures (EDA_output/figures/)

- 01_orders_missingness.png
- 02_order_status.png
- 03_price_freight_dist.png
- 04_freight_ratio.png
- 05_freight_vs_weight.png
- 06_freight_vs_distance.png
- 07_category_freight_burden.png
- 08_correlation_heatmap.png
- 09_orders_timeseries.png
- 10_payments_reviews.png

## Key takeaways for modelling

- Classification target `high_freight_flag` (freight > 25% of price) is ~46% positive — moderate imbalance, watch for it.
- Freight is driven more by volumetric weight & distance than raw price — good regression signal.
- Missing delivery timestamps only affect non-delivered orders; filter to `delivered` for lead-time work.
- Prices/freight are heavily right-skewed → use log transforms for regression.