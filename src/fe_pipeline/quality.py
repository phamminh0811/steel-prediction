"""Per-series data-quality statistics and pass/fail filter (all in Spark)."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def series_stats(grid: DataFrame, jump_flag_logret: float) -> DataFrame:
    """grid: (date, series, value) on the weekly grid, starting at each series' first value."""
    w = Window.partitionBy("series").orderBy("date")
    g = (grid.withColumn("prev", F.lag("value").over(w))
         .withColumn("chg", F.col("value") - F.col("prev"))
         .withColumn("logret", F.when((F.col("value") > 0) & (F.col("prev") > 0),
                                      F.log(F.col("value") / F.col("prev")))))
    data_end = grid.agg(F.max("date")).first()[0]
    return (g.groupBy("series").agg(
        F.count("*").alias("n_rows"),
        F.count("value").alias("n_obs"),
        F.min(F.when(F.col("value").isNotNull(), F.col("date"))).alias("first_date"),
        F.max(F.when(F.col("value").isNotNull(), F.col("date"))).alias("last_date"),
        F.min("value").alias("min_value"),
        F.count("chg").alias("n_chg"),
        F.sum(F.when(F.col("chg") == 0, 1).otherwise(0)).alias("n_zero_chg"),
        F.sum(F.when(F.abs("logret") > jump_flag_logret, 1).otherwise(0)).alias("n_jumps"),
    ).withColumn("missing_ratio", 1 - F.col("n_obs") / F.col("n_rows"))
     .withColumn("zero_change_ratio", F.col("n_zero_chg") / F.greatest(F.col("n_chg"), F.lit(1)))
     .withColumn("stale_weeks", F.datediff(F.lit(data_end), F.col("last_date")) / 7)
     .withColumn("history_weeks", F.datediff(F.col("last_date"), F.col("first_date")) / 7 + 1))


def apply_filters(stats_pdf, qcfg: dict, relax: float = 1.0):
    """Adds `quality_pass` and `quality_reason` columns to a pandas copy of the stats.

    relax > 1 loosens the thresholds (used when too few series survive)."""
    df = stats_pdf.copy()
    rules = [
        ("history<min", df["n_obs"] < qcfg["min_history_weeks"] / relax),
        ("missing>max", df["missing_ratio"] > min(qcfg["max_missing_ratio"] * relax, 0.6)),
        ("stale(discontinued)", df["stale_weeks"] > qcfg["max_stale_weeks"] * relax),
        ("constant/pegged", df["zero_change_ratio"] > min(qcfg["max_zero_change_ratio"] * relax, 0.9)),
    ]
    reason = [""] * len(df)
    for name, mask in rules:
        for i in mask[mask].index:
            pos = df.index.get_loc(i)
            reason[pos] = (reason[pos] + ";" if reason[pos] else "") + name
    df["quality_reason"] = reason
    df["quality_pass"] = df["quality_reason"] == ""
    df["possible_unit_break"] = df["n_jumps"] > 0
    return df
