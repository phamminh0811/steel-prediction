"""Stationary transforms of features and forward returns of targets (Spark)."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def choose_methods(mapping: dict, stats_pdf, tcfg: dict) -> dict:
    """series -> logret | diff | pct. logret falls back to diff when a series has values <= 0."""
    rules = tcfg["by_asset_class"]
    min_val = dict(zip(stats_pdf["series"], stats_pdf["min_value"]))
    out = {}
    for s, info in mapping.items():
        m = rules.get(info["asset_class"], rules.get("_default", "logret"))
        if m in ("logret", "pct") and (min_val.get(s) is None or min_val[s] <= 0):
            m = "diff"
        out[s] = m
    return out


def weekly_change(grid: DataFrame, methods: dict, winsor_mad: float) -> DataFrame:
    """(date, series, value) -> (date, series, d1) with d1 the 1-week change, winsorized per series."""
    spark = grid.sparkSession
    meth = spark.createDataFrame([(k, v) for k, v in methods.items()], "series string, method string")
    w = Window.partitionBy("series").orderBy("date")
    g = grid.join(F.broadcast(meth), "series").withColumn("prev", F.lag("value").over(w))
    d1 = (F.when(F.col("method") == "logret", F.log(F.col("value") / F.col("prev")))
          .when(F.col("method") == "pct", F.col("value") / F.col("prev") - 1)
          .otherwise(F.col("value") - F.col("prev")))
    g = g.withColumn("d1", d1)
    # robust clip: median +- k * MAD (per series)
    st = g.groupBy("series").agg(F.percentile_approx("d1", 0.5).alias("med"))
    g = g.join(st, "series").withColumn("absdev", F.abs(F.col("d1") - F.col("med")))
    mad = g.groupBy("series").agg(F.percentile_approx("absdev", 0.5).alias("mad"))
    g = g.join(mad, "series")
    lo = F.col("med") - winsor_mad * 1.4826 * F.col("mad")
    hi = F.col("med") + winsor_mad * 1.4826 * F.col("mad")
    # greatest/least skip nulls, so a missing d1 must stay null explicitly
    clipped = (F.when(F.col("d1").isNull(), None)
               .when(F.col("mad") > 0, F.least(F.greatest(F.col("d1"), lo), hi)).otherwise(F.col("d1")))
    return g.select("date", "series", clipped.alias("d1"))


def window_changes(d1: DataFrame, windows: list) -> DataFrame:
    """(date, series, d1) -> (date, series, win, x): change over `win` weeks (sum of weekly changes)."""
    out = None
    for win in windows:
        w = Window.partitionBy("series").orderBy("date").rowsBetween(-(win - 1), 0)
        part = (d1.withColumn("x", F.sum("d1").over(w)).withColumn("n", F.count("d1").over(w))
                .where(F.col("n") == win).select("date", "series", F.lit(win).alias("win"), "x"))
        out = part if out is None else out.unionByName(part)
    return out


def forward_returns(grid: DataFrame, target_cols: dict, horizons: list) -> DataFrame:
    """Wide frame: date + y__<target>__h<h> = log(P[t+h] / P[t]) (diff if the target has values <= 0)."""
    cols = [c["column"] for c in target_cols.values()]
    sub = grid.where(F.col("series").isin(cols))
    mins = dict(sub.groupBy("series").agg(F.min("value")).collect())
    w = Window.partitionBy("series").orderBy("date")
    parts = []
    for name, spec in target_cols.items():
        c = spec["column"]
        t = sub.where(F.col("series") == c)
        for h in horizons:
            fut = F.lead("value", h).over(w)
            y = F.log(fut / F.col("value")) if (mins.get(c) or 0) > 0 else fut - F.col("value")
            parts.append(t.select("date", F.lit(f"y__{name}__h{h}").alias("task"), y.alias("y")))
    long = parts[0]
    for p in parts[1:]:
        long = long.unionByName(p)
    return long.groupBy("date").pivot("task").agg(F.first("y")).orderBy("date")
