"""End-to-end feature selection: catalog -> quality -> transform -> relevance -> redundancy -> model -> pick -> table."""
import json
import os
import time

import pandas as pd
from pyspark.sql import functions as F

from . import catalog, features, io, model_select, quality, redundancy, relevance, selection, transforms
from .spark_utils import get_spark


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cfg: dict) -> dict:
    t0 = time.time()
    out = cfg["run"]["output_dir"]
    os.makedirs(out, exist_ok=True)
    spark = get_spark(cfg)
    summary = {"warnings": []}

    # 1. load + weekly alignment -------------------------------------------------------------
    raw, declared = io.load_raw_long(spark, cfg)
    weekly = io.align_weekly(raw, cfg["input"]["frequency"]).cache()
    columns = sorted(r[0] for r in weekly.select("series").distinct().collect())
    empty = sorted(set(declared or []) - set(columns))
    if empty:
        summary["warnings"].append(f"{len(empty)} columns have no values at all and were ignored: {empty[:20]}")
    summary["empty_series"] = empty
    _log(f"loaded {len(columns)} series ({len(empty)} empty columns ignored)")

    # 2. catalog reconciliation ----------------------------------------------------------------
    rec = catalog.reconcile(columns, cfg["catalog"], os.path.join(out, "01_catalog"))
    mapping, crep = rec["mapping"], rec["report"]
    _log(f"catalog: {crep['n_matched']} matched, {len(crep['missing_in_catalog'])} missing in catalog, "
         f"{len(crep['ambiguous'])} ambiguous, {crep['n_extra_removed']} catalog entries removed")
    tg = catalog.resolve_targets(cfg["targets"], mapping)
    targets = tg["targets"]
    if tg["missing"]:
        summary["warnings"].append(f"optional targets not found, skipped: {tg['missing']}")
    _log(f"targets: { {k: v['column'] for k, v in targets.items()} }")

    # 3. grid + quality stats ------------------------------------------------------------------
    qcfg = cfg["quality"]
    grid = io.to_grid(weekly, qcfg["max_ffill_weeks"]).cache()
    stats = quality.series_stats(grid, qcfg["jump_flag_logret"]).toPandas()

    # 4. transforms ----------------------------------------------------------------------------
    tcfg = cfg["transform"]
    methods = transforms.choose_methods(mapping, stats, tcfg)
    scannable = stats.loc[stats["n_obs"] >= 104, "series"].tolist()   # >= 2 years: anything less cannot be scored
    d1 = transforms.weekly_change(grid.where(F.col("series").isin(scannable)), {s: methods[s] for s in scannable},
                                  qcfg["winsor_mad"]).cache()
    xwin = transforms.window_changes(d1, tcfg["feature_windows"]).cache()
    y_wide = transforms.forward_returns(grid, targets, cfg["targets"]["horizons"]).cache()
    y_pdf = y_wide.toPandas()
    tasks = [c for c in y_pdf.columns if c.startswith("y__")]
    task_family = {t: targets[t.split("__")[1]]["family"] for t in tasks}
    horizons_by_task = {t: int(t.split("__h")[-1]) for t in tasks}

    # 5. relevance scan (Spark, once) ------------------------------------------------------------
    rcfg = cfg["relevance"]
    overall, by_period, tasks = relevance.scan(xwin, y_pdf, tcfg["max_lag"], tcfg.get("min_lag", 0),
                                               rcfg["n_periods"], cfg["run"]["seed"])
    _log(f"relevance scan done: {overall['series'].nunique()} series x {len(tasks)} tasks")
    os.makedirs(os.path.join(out, "02_scan"), exist_ok=True)
    overall.to_parquet(os.path.join(out, "02_scan", "lead_lag_overall.parquet"), index=False)
    by_period.to_parquet(os.path.join(out, "02_scan", "lead_lag_by_period.parquet"), index=False)
    stats.to_csv(os.path.join(out, "02_scan", "series_stats.csv"), index=False)
    corr_all = redundancy.candidate_corr(d1, scannable)

    # 6. relax loop: quality -> relevance -> redundancy (pandas, cheap) ---------------------------
    scfg = cfg["selection"]
    by_sym = {catalog.norm_symbol(v["symbol"]): k for k, v in mapping.items() if v["symbol"]}
    must = [by_sym[catalog.norm_symbol(s)] for s in scfg.get("must_include_symbols", []) if catalog.norm_symbol(s) in by_sym]
    must += [v["column"] for v in targets.values() if v["column"] not in must]
    for step in range(scfg["relax_steps"] + 1):
        relax = 1 + 0.5 * step
        q = quality.apply_filters(stats, qcfg, relax)
        qpass = set(q.loc[q["quality_pass"], "series"]) & set(scannable)
        rel = relevance.score(overall[overall["series"].isin(qpass)], by_period[by_period["series"].isin(qpass)],
                              tasks, task_family, rcfg, relax)
        must_ok = [s for s in must if s in qpass]
        cands = sorted(set(rel.loc[rel["relevance_pass"], "series"]) | set(must_ok))
        thr = min(cfg["redundancy"]["corr_threshold"] + 0.03 * step, 0.95)
        cl = redundancy.cluster(corr_all.loc[cands, cands], rel.set_index("series")["relevance_score"], thr, set(must_ok))
        reps = cl.loc[cl["representative"], "series"].tolist()
        _log(f"relax step {step}: quality {len(qpass)}, relevance candidates {len(cands)}, representatives {len(reps)}")
        if len(reps) >= scfg["k_min"]:
            break
    summary["relax_step_used"] = step
    summary["thresholds"] = {"relax_factor": relax, "corr_threshold": thr,
                             "null_thresholds": {t[3:]: float(rel[f"thr__{t[3:]}"].iloc[0]) for t in tasks}}

    filled = []
    if len(reps) < scfg["k_min"]:
        # last resort: add best-scored quality-passing series that are not near-duplicates of what we have
        need = scfg["k_min"] - len(reps)
        for s in rel.sort_values("relevance_score", ascending=False)["series"]:
            if need == 0:
                break
            if s in reps or corr_all.loc[s, reps].abs().max() > thr:
                continue
            reps.append(s)
            filled.append(s)
            need -= 1
        summary["warnings"].append(f"only {len(reps) - len(filled)} series passed all filters; "
                                   f"{len(filled)} added by relevance score alone (low evidence)")

    # 7. model-based importance --------------------------------------------------------------------
    mcfg = cfg["model"]
    rel_idx = rel.set_index("series")
    pool = pd.DataFrame({"series": reps})
    pool["relevance_score"] = pool["series"].map(rel_idx["relevance_score"])
    if mcfg.get("enabled", True):
        _log(f"model stage: {len(reps)} features x {len(tasks)} tasks x {mcfg['n_folds']} folds")
        mscore, imp, metrics = model_select.run(xwin, y_wide, rel, reps, tasks, mcfg, horizons_by_task, cfg["run"]["seed"])
        imp.to_csv(os.path.join(out, "model_importance.csv"), index=False)
        summary["model_metrics"] = metrics
        pool = pool.merge(mscore, on="series", how="left").fillna({"model_score": 0})
        w = mcfg["weight"]
        pool["combined_score"] = (1 - w) * pool["relevance_score"].rank(pct=True) + w * pool["model_score"]
    else:
        pool["model_score"] = None
        pool["combined_score"] = pool["relevance_score"].rank(pct=True)
    target_names = sorted(targets)
    for t in target_names:
        pool[f"target_score__{t}"] = pool["series"].map(rel_idx[f"target_score__{t}"])
    pool["group_key"] = pool["series"].map(lambda s: f"{mapping[s]['asset_class']}/{mapping[s]['group']}")

    # 8. final pick ---------------------------------------------------------------------------------
    picked = selection.pick(pool, scfg, target_names, must, corr_all.loc[reps, reps])
    selected = picked.loc[picked["selected"]].sort_values("selection_order")
    _log(f"selected {len(selected)} features")

    # 9. reports --------------------------------------------------------------------------------------
    report = q[["series", "n_obs", "first_date", "last_date", "missing_ratio", "zero_change_ratio", "stale_weeks",
                "possible_unit_break", "quality_pass", "quality_reason"]].copy()
    report["name"] = report["series"].map(lambda s: mapping[s]["name"])
    report["symbol"] = report["series"].map(lambda s: mapping[s]["symbol"])
    report["asset_class"] = report["series"].map(lambda s: mapping[s]["asset_class"])
    report["group"] = report["series"].map(lambda s: mapping[s]["group"])
    report["transform"] = report["series"].map(methods)
    keep_rel = ["series", "relevance_score", "relevance_pass", "n_tasks_passed"] + \
        [c for c in rel.columns if c.startswith(("rho__", "lag__", "win__", "cons__"))]
    report = report.merge(rel[keep_rel], on="series", how="left")
    report = report.merge(cl[["series", "cluster", "representative", "rep_of_cluster"]], on="series", how="left")
    report = report.merge(picked[["series", "model_score", "combined_score", "selected", "selected_by", "selection_order"]],
                          on="series", how="left")
    report["selected"] = report["selected"].astype("boolean").fillna(False).astype(bool)
    report["drop_stage"] = "selected"
    report.loc[~report["selected"], "drop_stage"] = "selection(rank/group cap)"
    report.loc[report["representative"].isna() & ~report["selected"], "drop_stage"] = "relevance"
    report.loc[(report["representative"] == False) & ~report["selected"], "drop_stage"] = "redundancy"  # noqa: E712
    report.loc[~report["quality_pass"].astype(bool), "drop_stage"] = "quality"
    report.loc[report["series"].isin(filled), "selected_by"] = report["selected_by"].fillna("") + "|low_evidence_fill"
    report.sort_values(["selected", "selection_order", "relevance_score"], ascending=[False, True, False]) \
        .to_csv(os.path.join(out, "selection_report.csv"), index=False)

    sel_rep = report[report["selected"]].sort_values("selection_order")
    detail = []
    for _, r in sel_rep.iterrows():
        best = {}
        for t in tasks:
            tag = t[3:]
            best[tag] = {"rho": round(float(r[f"rho__{tag}"]), 4) if pd.notna(r[f"rho__{tag}"]) else None,
                         "lag_weeks": int(r[f"lag__{tag}"]) if pd.notna(r[f"lag__{tag}"]) else None,
                         "window_weeks": int(r[f"win__{tag}"]) if pd.notna(r[f"win__{tag}"]) else None}
        detail.append({"series": r["series"], "name": r["name"], "symbol": r["symbol"], "asset_class": r["asset_class"],
                       "group": r["group"], "transform": r["transform"], "selected_by": r["selected_by"],
                       "combined_score": round(float(r["combined_score"]), 4), "best_by_task": best})
    grouped = {}
    for d in detail:
        g = grouped.setdefault((d["asset_class"], d["group"]), {"asset_class": d["asset_class"], "group": d["group"], "members": []})
        g["members"].append(d["name"])
    json.dump({"n_selected": len(detail), "groups": list(grouped.values()), "features": detail},
              open(os.path.join(out, "selected_features.json"), "w"), ensure_ascii=False, indent=2)

    # 10. model-ready feature table ---------------------------------------------------------------------
    table = features.build(d1, y_wide, selected["series"].tolist(), cfg["features"])
    table.write.mode("overwrite").parquet(os.path.join(out, "feature_table.parquet"))

    stage_counts = report["drop_stage"].value_counts().to_dict()
    summary.update({
        "n_series_in_data": len(columns), "targets": {k: v["column"] for k, v in targets.items()}, "tasks": tasks,
        "catalog": {k: crep[k] for k in ["n_matched", "n_extra_removed", "match_methods"]} |
                   {"n_missing_in_catalog": len(crep["missing_in_catalog"]), "n_ambiguous": len(crep["ambiguous"])},
        "funnel": {"in_data": len(columns), "quality_pass": int(report["quality_pass"].sum()),
                   "relevance_candidates": len(cands), "after_redundancy": len(reps), "selected": len(detail)},
        "drop_stage_counts": stage_counts, "n_selected": len(detail),
        "selected_per_group": sel_rep.assign(g=sel_rep["asset_class"] + "/" + sel_rep["group"])["g"].value_counts().to_dict(),
        "runtime_sec": round(time.time() - t0, 1),
    })
    json.dump(summary, open(os.path.join(out, "run_summary.json"), "w"), ensure_ascii=False, indent=2, default=str)
    for df in (weekly, grid, d1, xwin, y_wide):
        df.unpersist()
    _log(f"done in {summary['runtime_sec']}s -> {out}")
    return summary
