"""Final pick of k features: must-include priors, per-target minimum, then best combined score under a group cap.

Picks are greedy and redundancy-aware (mRMR style): each next feature maximises
    score - redundancy_penalty * max |corr| with the features already chosen,
so a cluster of close substitutes (corr 0.6-0.85, below the clustering threshold) does not fill the slots.
"""
import math

import pandas as pd


def pick(pool: pd.DataFrame, scfg: dict, targets: list, must: list, corr: pd.DataFrame = None) -> pd.DataFrame:
    """pool: one row per eligible series with columns series, group_key, combined_score, target_score__<target>.

    Returns pool with `selected` and `selected_by` columns."""
    pool = pool.sort_values("combined_score", ascending=False).copy()
    k = min(scfg["k_target"], len(pool))
    cap = max(1, math.ceil(scfg["max_share_per_group"] * scfg["k_target"]))
    chosen, why, per_group = [], {}, {}
    lam = scfg.get("redundancy_penalty", 0.0) if corr is not None else 0.0
    absc = corr.abs().fillna(0) if lam else None

    def take(s, reason, respect_cap=True):
        if s in why or len(chosen) >= k:
            return False
        g = pool.at[s, "group_key"]
        if respect_cap and per_group.get(g, 0) >= cap:
            return False
        chosen.append(s)
        why[s] = reason
        per_group[g] = per_group.get(g, 0) + 1
        return True

    def greedy(score: pd.Series, n: int, reason: str):
        """Take up to n new features by score minus redundancy with everything chosen so far."""
        taken = 0
        left = [s for s in score.dropna().index if s not in why]
        while taken < n and left and len(chosen) < k:
            adj = score[left].copy()
            if lam and chosen:
                adj -= lam * absc.loc[left, chosen].max(axis=1)
            for s in adj.sort_values(ascending=False).index:
                left.remove(s)
                if take(s, reason):
                    taken += 1
                    break
            else:
                break

    pool = pool.set_index("series")
    for s in must:
        if s in pool.index:
            take(s, "must_include", respect_cap=False)
    # each target keeps its own best features, so a target whose drivers rank lower overall is not crowded out
    for t in targets:
        greedy(pool[f"target_score__{t}"], scfg["min_per_target"], f"top_{t}")
    greedy(pool["combined_score"], k, "combined_score")
    pool["selected"] = pool.index.isin(chosen)
    pool["selected_by"] = pool.index.map(why)
    pool["selection_order"] = pool.index.map({s: i + 1 for i, s in enumerate(chosen)})
    return pool.reset_index()
