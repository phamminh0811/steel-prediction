"""SparkSession helper: reuses an existing session (cluster / Databricks) or builds a local one."""
import os
import shutil

from pyspark.sql import SparkSession


def get_spark(cfg: dict) -> SparkSession:
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    if not os.environ.get("JAVA_HOME") and not shutil.which("java"):
        raise RuntimeError("Spark needs Java 8/11/17: set JAVA_HOME or put `java` on PATH.")
    sc = cfg.get("spark", {})
    spark = (
        SparkSession.builder.master(sc.get("master", "local[*]"))
        .appName(sc.get("app_name", "fe-pipeline"))
        .config("spark.sql.shuffle.partitions", sc.get("shuffle_partitions", 16))
        .config("spark.driver.memory", sc.get("driver_memory", "4g"))
        .config("spark.sql.session.timeZone", "UTC")
        # pandas writes TIMESTAMP(NANOS) parquet, which Spark 3.x refuses; read it as a long, io converts it
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
