"""Univariate lead-lag relevance of every feature to every target task (Spark scan, pandas scoring).

For each feature series, change window `win` and lag `lag`, and each task y = forward return of a target
over h weeks, we compute the rank correlation corr(rank x[t - lag], rank y[t]).

Picking the best of (windows x lags) variants inflates correlations by chance, so each task gets a
data-driven threshold: the same best-of statistic against a circularly shifted target (keeps the
autocorrelation, destroys the timing), taken at the `null_quantile` across series.
"""
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def _null_shift(y: pd.Series, offset: int) -> pd.Series:
    vals = y.to_numpy()
    ok = ~np.isnan(vals)
    shifted = np.full_like(vals, np.nan)
    shifted[ok] = np.roll(vals[ok], offset)
    return pd.Series(shifted, index=y.index)


def scan(xwin: DataFrame, y_wide_pdf: pd.DataFrame, max_lag: int, min_lag: int, n_periods: int, seed: int):
    """Returns (overall, by_period) pandas frames of correlations."""
    spark = xwin.sparkSession
    y = y_wide_pdf.sort_values("date").reset_index(drop=True)
    tasks = [c for c in y.columns if c.startswith("y__")]
    rng = np.random.default_rng(seed)
    n = len(y)
    for t in tasks:
        y[t] = y[t].rank(pct=True)
        offset = int(rng.integers(max(52, n // 4), max(53, n - max(52, n // 4))))
        y[f"null{t[1:]}"] = _null_shift(y[t], offset)
    null_tasks = [f"null{t[1:]}" for t in tasks]
    y["period"] = np.minimum((np.arange(n) * n_periods) // n, n_periods - 1)
    ydf = spark.createDataFrame(y[["date", "period"] + tasks + null_tasks]).withColumn("date", F.to_date("date"))
    # pandas NaN arrives as NaN (not null) and would turn every corr() into NaN
    ydf = ydf.select("date", "period", *[F.when(~F.isnan(c), F.col(c)).alias(c) for c in tasks + null_tasks])

    wr = Window.partitionBy("series", "win")
    x = xwin.withColumn("xr", F.percent_rank().over(wr.orderBy("x")))
    lags = F.explode(F.array(*[F.lit(L) for L in range(min_lag, max_lag + 1)])).alias("lag")
    x = x.select("series", "win", "date", "xr", lags).withColumn("date", F.date_add("date", F.col("lag") * 7))
    j = x.join(F.broadcast(ydf), "date").cache()

    keys = ["series", "win", "lag"]
    overall = j.groupBy(*keys).agg(F.count("xr").alias("n"),
                                   *[F.corr("xr", t).alias(t) for t in tasks + null_tasks]).toPandas()
    by_period = j.groupBy(*keys, "period").agg(*[F.corr("xr", t).alias(t) for t in tasks]).toPandas()
    j.unpersist()
    return overall, by_period, tasks


def score(overall: pd.DataFrame, by_period: pd.DataFrame, tasks: list, task_family: dict, rcfg: dict,
          relax: float = 1.0) -> pd.DataFrame:
    """One row per series: best variant per task, null-calibrated pass flags, relevance scores."""
    null_q = max(0.95 - 0.05 * (relax - 1) * 2, 0.75)
    min_abs = rcfg["min_abs_spearman"] / relax
    min_cons = max(rcfg["min_sign_consistency"] - 0.1 * (relax - 1) * 2, 0.5)
    res = pd.DataFrame({"series": sorted(overall["series"].unique())}).set_index("series")
    for t in tasks:
        o = overall[["series", "win", "lag", t, f"null{t[1:]}"]].dropna(subset=[t]).copy()
        o["abs"] = o[t].abs()
        best = o.sort_values("abs", ascending=False).drop_duplicates("series").set_index("series")
        null_best = o.assign(nabs=o[f"null{t[1:]}"].abs()).groupby("series")["nabs"].max()
        thr = max(float(null_best.quantile(null_q)), min_abs)
        bp = by_period.merge(best[["win", "lag", t]].rename(columns={t: "rho_all"}).reset_index(),
                             on=["series", "win", "lag"])
        bp["agree"] = np.sign(bp[t]) == np.sign(bp["rho_all"])
        cons = bp.groupby("series")["agree"].mean()
        tag = t[3:]
        res[f"rho__{tag}"] = best[t]
        res[f"win__{tag}"] = best["win"]
        res[f"lag__{tag}"] = best["lag"]
        res[f"cons__{tag}"] = cons
        res[f"pass__{tag}"] = (best["abs"] >= thr) & (cons >= min_cons)
        res[f"thr__{tag}"] = thr
        strength = best["abs"] * cons.reindex(best.index).fillna(0)
        res[f"score__{tag}"] = strength.rank(pct=True)
    for c in [c for c in res.columns if c.startswith("pass__")]:
        res[c] = res[c].astype("boolean").fillna(False).astype(bool)
    score_cols = [c for c in res.columns if c.startswith("score__")]
    res["relevance_score"] = 0.5 * res[score_cols].max(axis=1) + 0.5 * res[score_cols].mean(axis=1)
    for target in sorted({t.split("__")[1] for t in tasks}):
        cols = [f"score__{t[3:]}" for t in tasks if t.split("__")[1] == target]
        res[f"target_score__{target}"] = 0.5 * res[cols].max(axis=1) + 0.5 * res[cols].mean(axis=1)
    pass_cols = [c for c in res.columns if c.startswith("pass__")]
    res["n_tasks_passed"] = res[pass_cols].sum(axis=1)
    keep = set()
    for t in tasks:
        tag = t[3:]
        passing = res[res[f"pass__{tag}"]]
        keep |= set(passing.sort_values(f"score__{tag}", ascending=False).head(rcfg["keep_top_per_target"]).index)
    res["relevance_pass"] = res.index.isin(keep)
    return res.reset_index()
