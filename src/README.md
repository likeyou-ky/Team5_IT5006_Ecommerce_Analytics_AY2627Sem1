# src/

Reusable Python modules (importable from notebooks and the deployment app), so cleaning and
feature logic lives in one place instead of being copy-pasted across notebooks.

Planned modules:
- `data_loading.py` — load the 9 tables, parse dates.
- `features.py` — build the order-item base table (joins, volumetric weight, Haversine distance, freight ratio).
- `models.py` — scikit-learn Pipelines for the classification/regression tasks.

Keep functions pure and documented; set/record random seeds for reproducibility.
