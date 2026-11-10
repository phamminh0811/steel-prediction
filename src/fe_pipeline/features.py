"""Model-ready feature table for the selected series (Spark): lags and rolling momentum of weekly changes."""
import re

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()


def build(d1: DataFrame, y_wide: DataFrame, selected: list, fcfg: dict) -> DataFrame:
    sel = d1.where(F.col("series").isin(selected))
    w = Window.partitionBy("series").orderBy("date")
    exprs = []
    for L in fcfg["lags"]:
        sel = sel.withColumn(f"d1_lag{L}", F.lag("d1", L).over(w) if L else F.col("d1"))
        exprs.append(f"d1_lag{L}")
    for win in fcfg["rolling_windows"]:
        sel = sel.withColumn(f"mom{win}", F.avg("d1").over(w.rowsBetween(-(win - 1), 0)))
        exprs.append(f"mom{win}")
    # long (date, series, feature, value) -> wide date x "<series>__<feature>"
    stacked = sel.select("date", "series", F.explode(F.array(*[
        F.struct(F.lit(e).alias("feat"), F.col(e).alias("v")) for e in exprs])).alias("kv"))
    stacked = stacked.select("date", F.concat_ws("__", F.col("series"), F.col("kv.feat")).alias("name"), F.col("kv.v").alias("v"))
    names = sorted(f"{s}__{e}" for s in selected for e in exprs)
    wide = stacked.groupBy("date").pivot("name", names).agg(F.first("v"))
    wide = wide.select("date", *[F.col(f"`{n}`").alias(safe_name(n)) for n in names])
    return wide.join(y_wide, "date", "left").orderBy("date")
