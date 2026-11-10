"""Run the pipeline over data variants that mimic what the real data may look like, and check the outcome.

    python scripts/run_scenarios.py                 # all scenarios
    python scripts/run_scenarios.py --only long_format short_history

Results: outputs/scenarios/scenario_results.json (+ one output folder per scenario).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SYN = os.path.join(ROOT, "data", "synthetic")
VAR = os.path.join(SYN, "variants")
OUT = os.path.join(ROOT, "outputs", "scenarios")
FAST = ["model.num_trees=60", "model.n_folds=3"]  # scenarios test robustness, not model quality


def make_variants():
    os.makedirs(VAR, exist_ok=True)
    wide = pd.read_parquet(os.path.join(SYN, "wide_mixed.parquet"))
    wide.drop(columns=["Coking Coal", "Iron Ore CNY"]).to_parquet(os.path.join(VAR, "no_optional_targets.parquet"), index=False)
    wide.drop(columns=["Iron Ore"]).to_parquet(os.path.join(VAR, "no_required_target.parquet"), index=False)
    renamed = wide.rename(columns={"Iron Ore": "iron_ore_62fe_cfr_qingdao", "Coal": "newcastle_6000_fob"})
    renamed.to_parquet(os.path.join(VAR, "renamed_targets.parquet"), index=False)


def hand_edited_catalog(src_dir: str) -> str:
    """Simulate a person moving every _UNASSIGNED column into a real group in the reconciled catalog."""
    cat = json.load(open(os.path.join(src_dir, "01_catalog", "catalog_reconciled.json")))
    un = [g for g in cat if g["group"] == "_UNASSIGNED"]
    cat = [g for g in cat if g["group"] != "_UNASSIGNED"]
    if un:
        cat.append({"asset_class": "macro", "group": "Macro (manual)", "members": un[0]["members"]})
    path = os.path.join(VAR, "catalog_hand_edited.json")
    json.dump(cat, open(path, "w"), indent=2)
    return path


def aliases_file() -> str:
    path = os.path.join(VAR, "aliases_test.yaml")
    open(path, "w").write('aliases:\n  "iron_ore_62fe_cfr_qingdao": "SCO:COM"\n  "newcastle_6000_fob": "XAL1:COM"\n'
                          '  "Natural Gas Henry Hub": "NG1:COM"\n')
    return path


def run(name, sets, script="run_pipeline.py"):
    out = os.path.join(OUT, name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", script), "--config", os.path.join(ROOT, "config", "pipeline.yaml"),
           "--set", f"run.output_dir={out}"] + [x for s in sets for x in ("--set", s)]
    t = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    res = {"exit_code": p.returncode, "seconds": round(time.time() - t), "out": out,
           "stderr_tail": "\n".join(l for l in p.stderr.splitlines() if "Error" in l or "error" in l)[-600:]}
    summ = os.path.join(out, "run_summary.json")
    if os.path.exists(summ) and p.returncode == 0:
        s = json.load(open(summ))
        res.update({k: s.get(k) for k in ["n_selected", "relax_step_used", "funnel", "warnings", "targets", "catalog"]})
    rep = os.path.join(out, "01_catalog", "catalog_report.json")
    if os.path.exists(rep):
        r = json.load(open(rep))
        res["catalog_report"] = {"n_matched": r["n_matched"], "missing": len(r["missing_in_catalog"]),
                                 "ambiguous": len(r["ambiguous"]), "methods": r["match_methods"]}
    return res


def scenarios():
    base = os.path.join(OUT, "wide_parquet")
    return {
        # name: (overrides, script, check(result) -> (ok, note))
        "wide_parquet": ([f"input.path={SYN}/wide_mixed.parquet"] + FAST, "run_pipeline.py",
                         lambda r: (r["exit_code"] == 0 and 70 <= r["n_selected"] <= 100, "70-100 selected")),
        "long_format": ([f"input.path={SYN}/long.parquet"] + FAST, "run_pipeline.py",
                        lambda r: (r["exit_code"] == 0 and 70 <= r["n_selected"] <= 100, "long + mixed weekdays + daily rows")),
        "csv_input": ([f"input.path={SYN}/wide_mixed.csv", "model.enabled=false"], "run_pipeline.py",
                      lambda r: (r["exit_code"] == 0 and 70 <= r["n_selected"] <= 100, "csv reader")),
        "short_history": ([f"input.path={SYN}/wide_mixed.parquet", "input.start_date=2022-01-01"] + FAST, "run_pipeline.py",
                          lambda r: (r["exit_code"] == 0 and (r["relax_step_used"] or 0) >= 1, "must relax (< 260 weeks)")),
        "no_optional_targets": ([f"input.path={VAR}/no_optional_targets.parquet"] + FAST, "run_pipeline.py",
                                lambda r: (r["exit_code"] == 0 and any("optional targets" in w for w in r["warnings"]),
                                           "continues with 2 targets + warning")),
        "no_required_target": ([f"input.path={VAR}/no_required_target.parquet"], "run_pipeline.py",
                               lambda r: (r["exit_code"] != 0 and "required targets" in r["stderr_tail"], "clean error")),
        "renamed_targets_with_aliases": ([f"input.path={VAR}/renamed_targets.parquet", f"catalog.aliases_path={aliases_file()}"],
                                         "reconcile_catalog.py",
                                         lambda r: (r["exit_code"] == 0 and r["catalog_report"]["methods"].get("alias") == 3,
                                                    "3 alias matches incl. both targets")),
        "hand_edited_catalog": (lambda: [f"input.path={SYN}/wide_mixed.parquet", f"catalog.path={hand_edited_catalog(base)}"],
                                "reconcile_catalog.py",
                                lambda r: (r["exit_code"] == 0 and r["catalog_report"]["missing"] == 0
                                           and r["catalog_report"]["methods"].get("column", 0) > 800, "all columns matched by column")),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    make_variants()
    results = {}
    for name, (sets, script, check) in scenarios().items():
        if a.only and name not in a.only:
            continue
        sets = sets() if callable(sets) else sets
        print(f"--- {name}", flush=True)
        r = run(name, sets, script)
        ok, note = check(r)
        r.update({"pass": bool(ok), "expectation": note})
        results[name] = r
        print(f"    {'PASS' if ok else 'FAIL'} ({note}) exit={r['exit_code']} {r['seconds']}s selected={r.get('n_selected')}", flush=True)
        json.dump(results, open(os.path.join(OUT, "scenario_results.json"), "w"), indent=2, default=str)
