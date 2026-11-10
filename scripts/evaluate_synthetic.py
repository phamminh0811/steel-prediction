"""Score a pipeline run on synthetic data against its ground truth.

    python scripts/evaluate_synthetic.py --run outputs/run_latest --truth data/synthetic/ground_truth.json
"""
import argparse
import json
import os

import pandas as pd


def evaluate(run_dir: str, truth_path: str, naming: str = "column_mixed") -> dict:
    truth = json.load(open(truth_path))
    rep = pd.read_csv(os.path.join(run_dir, "selection_report.csv")).set_index("series")
    sel = set(rep.index[rep["selected"]])

    def col(d):  # ground-truth driver -> data column name for this naming
        return d[naming] if naming != "symbol" else (d["symbol"] or d["column_mixed"])

    out = {"n_selected": len(sel), "drivers": {}}
    for fam, drivers in truth["true_drivers"].items():
        rows = []
        for d in drivers:
            c = col(d)
            r = rep.loc[c] if c in rep.index else None
            rows.append({"driver": c, "lag": d["lag"], "selected": c in sel,
                         "drop_stage": None if r is None else r["drop_stage"],
                         "same_cluster_as": None if r is None or pd.isna(r.get("rep_of_cluster")) or r["rep_of_cluster"] == c
                         else r["rep_of_cluster"]})
        found = sum(x["selected"] for x in rows)
        # a driver dropped only because a near-duplicate represents it still counts as covered
        covered = sum(x["selected"] or (x["same_cluster_as"] in sel) for x in rows)
        out["drivers"][fam] = {"recall": round(found / len(rows), 3), "recall_incl_cluster_rep": round(covered / len(rows), 3),
                               "n_true": len(rows), "detail": rows}
    names = {v: k for k, v in truth.get("typos", {}).items()}
    issues = truth["issues"]

    def stage(series_list):
        cols = [s for s in series_list if s in rep.index]
        return rep.loc[cols, "drop_stage"].value_counts().to_dict() if cols else {}

    inv = {}
    if naming == "column_mixed":
        # issue lists are in symbols; map them to mixed column names through the column mapping
        mp = json.load(open(os.path.join(run_dir, "01_catalog", "column_mapping.json")))
        inv = {v["symbol"]: c for c, v in mp.items() if v["symbol"]}
    m = lambda xs: [inv.get(x, names.get(x, x)) for x in xs]  # noqa: E731
    out["issues_caught"] = {
        "pegged_fx -> quality": stage(m(issues["pegged_fx"])),
        "discontinued -> quality": stage(m(issues["discontinued"])),
        "late_start -> quality/other": stage(m(issues["late_start"])),
        "unit_break flagged": bool(rep.loc[m(issues["unit_break_x10"]), "possible_unit_break"].all()),
        "negative_price transform": rep.loc[m(issues["negative_price"]), "transform"].tolist(),
        "near_duplicate": rep.loc[[k for k in issues["near_duplicate"]], ["drop_stage", "rep_of_cluster"]].to_dict("records"),
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/run_latest")
    ap.add_argument("--truth", default="data/synthetic/ground_truth.json")
    ap.add_argument("--naming", default="column_mixed", choices=["column_mixed", "symbol"])
    a = ap.parse_args()
    res = evaluate(a.run, a.truth, a.naming)
    json.dump(res, open(os.path.join(a.run, "synthetic_evaluation.json"), "w"), indent=2, default=str)
    for fam, v in res["drivers"].items():
        print(f"{fam}: recall {v['recall']:.0%} ({v['recall_incl_cluster_rep']:.0%} incl. cluster representative) of {v['n_true']} true drivers")
        for d in v["detail"]:
            if not d["selected"]:
                print(f"   missed {d['driver']} (lag {d['lag']}): {d['drop_stage']}" + (f", represented by {d['same_cluster_as']}" if d["same_cluster_as"] else ""))
    print(json.dumps(res["issues_caught"], indent=1, default=str))
