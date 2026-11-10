"""Multivariate importance with Spark ML on expanding-window time-series folds.

For each task, every representative enters with its best (window, lag) variant from the relevance stage.
Importance = tree feature importance averaged over folds; we also report the out-of-fold rank IC so a
reader can see whether the selected set carries signal at all.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def _design(xwin: DataFrame, rel: pd.DataFrame, reps: list, tag: str) -> DataFrame:
    """Wide frame date x feature, each feature shifted by its best lag for this task."""
    r = rel.set_index("series").loc[reps]
    spec = [(s, int(r.at[s, f"win__{tag}"]), int(r.at[s, f"lag__{tag}"])) for s in reps if pd.notna(r.at[s, f"win__{tag}"])]
    spark = xwin.sparkSession
    sdf = spark.createDataFrame(spec, "series string, win int, lag int")
    shifted = (xwin.join(F.broadcast(sdf), ["series", "win"])
               .withColumn("date", F.date_add("date", F.col("lag") * 7)))
    cols = [s for s, _, _ in spec]
    return shifted.groupBy("date").pivot("series", cols).agg(F.first("x")), cols


def run(xwin: DataFrame, y_wide: DataFrame, rel: pd.DataFrame, reps: list, tasks: list, mcfg: dict,
        horizons_by_task: dict, seed: int):
    # tasks are independent: fit them from a thread pool so Spark schedules their jobs side by side
    with ThreadPoolExecutor(max_workers=max(1, mcfg.get("parallel_tasks", 1))) as ex:
        results = list(ex.map(lambda t: _fit_task(xwin, y_wide, rel, reps, t, mcfg, horizons_by_task, seed), tasks))
    imp_rows = [row for rows, _ in results for row in rows]
    metrics = [m for _, m in results]
    imp = pd.DataFrame(imp_rows)
    imp["imp_rank"] = imp.groupby("task")["importance"].rank(pct=True)
    agg = imp.pivot(index="series", columns="task", values="imp_rank")
    model_score = 0.5 * agg.max(axis=1) + 0.5 * agg.mean(axis=1)
    return model_score.rename("model_score").reset_index(), imp, metrics


def _fit_task(xwin, y_wide, rel, reps, t, mcfg, horizons_by_task, seed):
    tag = t[3:]
    X, cols = _design(xwin, rel, reps, tag)
    # safe names first: series names may contain dots/colons that Spark would parse
    safe = {c: f"f{i}" for i, c in enumerate(cols)}
    X = X.select("date", *[F.col(f"`{c}`").alias(s) for c, s in safe.items()])
    df = (X.join(y_wide.select("date", F.col(f"`{t}`").alias("label")), "date")
          .where(F.col("label").isNotNull()).fillna(0.0, subset=list(safe.values())))
    df = VectorAssembler(inputCols=list(safe.values()), outputCol="features").transform(df).select("date", "features", "label").cache()
    dates = [r[0] for r in df.select("date").distinct().orderBy("date").collect()]
    k = mcfg["n_folds"]
    cuts = [dates[int(len(dates) * (i + 1) / (k + 1))] for i in range(k)] + [dates[-1]]
    embargo = horizons_by_task[t]
    imp = np.zeros(len(cols))
    ics = []
    for i in range(k):
        train = df.where(F.col("date") < F.date_sub(F.lit(cuts[i]), 7 * embargo))
        test = df.where((F.col("date") >= F.lit(cuts[i])) & (F.col("date") < F.lit(cuts[i + 1])))
        if mcfg.get("algorithm") == "gbt":
            est = GBTRegressor(maxIter=min(mcfg["num_trees"], 100), maxDepth=mcfg["max_depth"], seed=seed)
        else:
            est = RandomForestRegressor(numTrees=mcfg["num_trees"], maxDepth=mcfg["max_depth"],
                                        featureSubsetStrategy="onethird", seed=seed)
        model = est.fit(train)
        imp += model.featureImportances.toArray()
        pred = model.transform(test).select("label", "prediction").toPandas()
        if len(pred) > 10:
            ics.append(pred["label"].rank().corr(pred["prediction"].rank()))
    df.unpersist()
    imp /= k
    rows = [{"series": c, "task": tag, "importance": v} for c, v in zip(cols, imp)]
    metric = {"task": tag, "oof_rank_ic_mean": float(np.nanmean(ics)) if ics else None,
              "oof_rank_ic_by_fold": [round(float(x), 4) for x in ics], "n_features": len(cols)}
    return rows, metric
