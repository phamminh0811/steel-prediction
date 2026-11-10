"""Load raw data (wide or long, csv or parquet) into one long Spark frame aligned to a weekly grid.

Internal representation everywhere downstream: long DataFrame (date: date, series: string, value: double),
where `series` is the original data column name (wide) or series label (long).
"""
import os

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

WEEKDAY = {"W-MON": "Mon", "W-TUE": "Tue", "W-WED": "Wed", "W-THU": "Thu", "W-FRI": "Fri", "W-SAT": "Sat", "W-SUN": "Sun"}


def _read_any(spark: SparkSession, path: str) -> DataFrame:
    p = path.lower().rstrip("/")
    if p.endswith(".csv") or (os.path.isdir(path) and any(f.endswith(".csv") for f in os.listdir(path))):
        return spark.read.option("header", True).option("inferSchema", True).csv(path)
    return spark.read.parquet(path)


def _to_date(df: DataFrame, col: str):
    """Date/timestamp/string as is; integer epochs (s, ms, us or ns, e.g. pandas nanos read as long) by magnitude."""
    dtype = dict(df.dtypes)[col]
    if dtype in ("bigint", "int", "double"):
        c = F.col(col).cast("double")
        secs = (F.when(F.abs(c) > 1e17, c / 1e9).when(F.abs(c) > 1e14, c / 1e6)
                .when(F.abs(c) > 1e11, c / 1e3).otherwise(c))
        return F.to_date(F.timestamp_seconds(secs))
    return F.to_date(F.col(col))


def _detect_format(df: DataFrame, icfg: dict) -> str:
    fmt = icfg.get("format", "auto")
    if fmt != "auto":
        return fmt
    cols = set(df.columns)
    return "long" if {icfg["series_col"], icfg["value_col"]} <= cols and len(cols) <= 5 else "wide"


def wide_to_long(df: DataFrame, date_col: str) -> DataFrame:
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise ValueError("wide input has no value columns")
    pairs = ", ".join(f"'{c.replace(chr(39), chr(39) * 2)}', CAST(`{c}` AS DOUBLE)" for c in value_cols)
    return df.select(F.col(date_col).alias("date"), F.expr(f"stack({len(value_cols)}, {pairs}) AS (series, value)"))


def load_raw_long(spark: SparkSession, cfg: dict):
    """Returns (long frame, declared series names or None). Declared = wide value columns, so series that
    are entirely empty can be reported instead of silently disappearing."""
    icfg = cfg["input"]
    df = _read_any(spark, icfg["path"])
    if icfg["date_col"] not in df.columns:
        raise ValueError(f"date column '{icfg['date_col']}' not found; columns start with {df.columns[:5]}")
    declared = None
    if _detect_format(df, icfg) == "long":
        long = df.select(F.col(icfg["date_col"]).alias("date"), F.col(icfg["series_col"]).cast("string").alias("series"),
                         F.col(icfg["value_col"]).cast("double").alias("value"))
    else:
        declared = [c for c in df.columns if c != icfg["date_col"]]
        long = wide_to_long(df, icfg["date_col"])
    long = long.withColumn("date", _to_date(long, "date")).where(F.col("date").isNotNull() & F.col("series").isNotNull())
    if icfg.get("start_date"):
        long = long.where(F.col("date") >= F.lit(icfg["start_date"]).cast("date"))
    if icfg.get("end_date"):
        long = long.where(F.col("date") <= F.lit(icfg["end_date"]).cast("date"))
    # NaN (from pandas/parquet) and +-inf behave like missing values
    bad = F.isnan("value") | (F.abs("value") == float("inf"))
    return long.withColumn("value", F.when(bad, None).otherwise(F.col("value"))), declared


def align_weekly(long: DataFrame, frequency: str = "W-FRI") -> DataFrame:
    """Snap each observation to its week-end anchor and keep the last observation of that week.

    Handles daily input, weekly input stamped on different weekdays, and duplicated rows.
    """
    day = WEEKDAY[frequency]
    # next_day(d - 1, 'Fri') = the Friday on or after d
    snapped = long.where(F.col("value").isNotNull()).withColumn("week", F.next_day(F.date_sub("date", 1), day))
    w = Window.partitionBy("series", "week").orderBy(F.col("date").desc())
    return (snapped.withColumn("_rn", F.row_number().over(w)).where("_rn = 1")
            .select(F.col("week").alias("date"), "series", "value"))


def to_grid(weekly: DataFrame, max_ffill_weeks: int) -> DataFrame:
    """Full date x series grid; gaps up to `max_ffill_weeks` are forward-filled, longer gaps stay null.

    Rows before a series' first value are dropped so that late starters keep a clean history.
    """
    bounds = weekly.agg(F.min("date").alias("lo"), F.max("date").alias("hi")).first()
    spark = weekly.sparkSession
    dates = spark.sql(f"SELECT explode(sequence(DATE'{bounds.lo}', DATE'{bounds.hi}', INTERVAL 7 DAYS)) AS date")
    series_first = weekly.groupBy("series").agg(F.min("date").alias("first_date"))
    grid = (dates.crossJoin(series_first).where(F.col("date") >= F.col("first_date")).drop("first_date")
            .join(weekly, ["date", "series"], "left"))
    w = Window.partitionBy("series").orderBy("date").rowsBetween(-max_ffill_weeks, 0)
    return grid.withColumn("value", F.last("value", ignorenulls=True).over(w))


def long_to_wide(long: DataFrame, value_col: str = "value") -> DataFrame:
    return long.groupBy("date").pivot("series").agg(F.first(value_col)).orderBy("date")
