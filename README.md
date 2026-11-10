# Feature selection for coal / iron-ore price models (PySpark)

Picks 70-100 of ~800+ weekly market series as features for coal and iron-ore price prediction.
Full plan, scenarios and synthetic-data results: see the project doc (Claude Docs link shared in chat).

Requires Java 8/11/17 and `pip install -r requirements.txt`.

```bash
# 1. check column names against the catalog (fast)
python scripts/reconcile_catalog.py --set input.path=/secure/weekly.parquet --set run.output_dir=outputs/real
#    -> fix outputs/real/01_catalog/catalog_report.json issues via config/aliases.yaml or a hand-edited catalog

# 2. full run
python scripts/run_pipeline.py --set input.path=/secure/weekly.parquet --set run.output_dir=outputs/real
#    -> outputs/real/selected_features.json, feature_table.parquet, selection_report.csv, run_summary.json

# self-test on synthetic data with ground truth
python scripts/make_synthetic_data.py
python scripts/run_scenarios.py
python scripts/run_pipeline.py && python scripts/evaluate_synthetic.py
```

All settings live in `config/pipeline.yaml`; any key can be overridden with `--set dotted.key=value`.
