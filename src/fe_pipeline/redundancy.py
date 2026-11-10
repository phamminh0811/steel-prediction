"""Correlation clustering of candidate series; one representative per cluster.

The correlation matrix is computed from a Spark pivot of weekly changes; only the candidate x candidate
matrix (a few hundred squared) is brought to the driver for clustering.
"""
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def candidate_corr(d1: DataFrame, candidates: list, min_periods: int = 104) -> pd.DataFrame:
    wide = (d1.where(F.col("series").isin(candidates)).groupBy("date").pivot("series", candidates)
            .agg(F.first("d1")).orderBy("date").toPandas().set_index("date"))
    return wide.corr(min_periods=min_periods)


def cluster(corr: pd.DataFrame, score: pd.Series, threshold: float, must_keep: set) -> pd.DataFrame:
    names = list(corr.columns)
    if not names:  # nothing passed (e.g. first relax step on short data)
        return pd.DataFrame(columns=["series", "cluster", "score", "representative", "rep_of_cluster"])
    if len(names) == 1:
        return pd.DataFrame({"series": names, "cluster": [1], "representative": [True], "rep_of_cluster": names})
    dist = 1 - corr.abs().fillna(0).to_numpy()  # pairs without enough overlap are treated as unrelated
    np.fill_diagonal(dist, 0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=1 - threshold, criterion="distance")
    df = pd.DataFrame({"series": names, "cluster": labels})
    df["score"] = df["series"].map(score).fillna(0)
    df["must"] = df["series"].isin(must_keep)
    # representative: a must-keep member if any, else the best score; other must-keep members are kept too
    df = df.sort_values(["cluster", "must", "score"], ascending=[True, False, False])
    df["representative"] = ~df.duplicated("cluster") | df["must"]
    rep = df[~df.duplicated("cluster")].set_index("cluster")["series"]
    df["rep_of_cluster"] = df["cluster"].map(rep)
    return df.drop(columns=["must"]).reset_index(drop=True)
