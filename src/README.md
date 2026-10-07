# src/

Reusable Python modules (importable from notebooks and the deployment app), so cleaning and
feature logic lives in one place instead of being copy-pasted across notebooks.

Modules:
- `features.py` — builds the item table (Problem 1) and the delivered-order table (Problem 2) with leakage-safe features known at checkout; the nested feature sets of Problem 1 and the candidate feature groups of Problem 2.
- `models.py` — order-grouped and forward-chaining (time-ordered, embargoed) splitters, preprocessing, regression/classification pipelines, stacking and voting ensembles, metrics and the top-10% review rule.
- `pipeline.py` — every experiment step called by the Phase 2 notebook: feature-set selection (paired fold-by-fold gains; forward selection for Problem 2), tuning, cross-validation, test evaluation, random vs time-ordered split comparison, baseline-vs-advanced and train/validation/test tables, missed-late-order analysis, permutation importance, robustness checks, implied tariff and the unusual-freight audit.
- `plots.py` — Phase 2 figures.
- `olist_theme.py` — shared chart theme.

Keep functions pure and documented; set/record random seeds for reproducibility.
