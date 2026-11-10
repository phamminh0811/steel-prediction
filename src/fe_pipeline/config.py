"""Config loading with dotted-key overrides: --set input.path=data/x.parquet --set selection.k_target=90"""
import yaml


def load(path: str, overrides: list = ()) -> dict:
    cfg = yaml.safe_load(open(path))
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg
