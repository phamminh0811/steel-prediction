"""Run the full feature-selection pipeline.

    python scripts/run_pipeline.py --config config/pipeline.yaml
    python scripts/run_pipeline.py --config config/pipeline.yaml --set input.path=/secure/data.parquet --set run.output_dir=outputs/real
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fe_pipeline import config, pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value override, repeatable")
    a = ap.parse_args()
    pipeline.run(config.load(a.config, a.set))
