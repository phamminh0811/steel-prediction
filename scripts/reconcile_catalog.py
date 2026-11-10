"""Only reconcile the catalog against the data columns (fast; run this first on new data).

Writes <output_dir>/01_catalog/{catalog_reconciled.json, column_mapping.json, catalog_report.json}.
Fix what the report lists (aliases.yaml or a hand-edited grouped catalog), then run the full pipeline.

    python scripts/reconcile_catalog.py --config config/pipeline.yaml [--set catalog.path=my_groups.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fe_pipeline import catalog, config, io  # noqa: E402
from fe_pipeline.spark_utils import get_spark  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = config.load(a.config, a.set)
    spark = get_spark(cfg)
    raw, _ = io.load_raw_long(spark, cfg)
    # same rule as the pipeline: columns without any value are ignored
    cols = sorted(r[0] for r in raw.where("value IS NOT NULL").select("series").distinct().collect())
    out = os.path.join(cfg["run"]["output_dir"], "01_catalog")
    res = catalog.reconcile(cols, cfg["catalog"], out)
    rep = res["report"]
    # report missing targets instead of failing: this script is for diagnosis
    tg = catalog.resolve_targets({k: v for k, v in cfg["targets"].items() if k != "required"} | {"required": []},
                                 res["mapping"])
    print(json.dumps({k: rep[k] for k in ["n_data_columns", "n_catalog_entries", "n_matched", "match_methods",
                                          "n_extra_removed"]}, indent=2))
    print(f"missing in catalog (thiếu): {len(rep['missing_in_catalog'])} -> {list(rep['missing_in_catalog'])[:15]}")
    print(f"ambiguous: {len(rep['ambiguous'])} -> {list(rep['ambiguous'])[:10]}")
    print(f"targets found: {tg['targets']}  missing: {tg['missing']}")
    print(f"written to {out}")
