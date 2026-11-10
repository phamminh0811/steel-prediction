"""Catalog <-> data reconciliation.

Accepts either catalog format:
  * detailed: [{"symbol", "name", "asset_class", "group", "subgroup"?, "url"?}, ...]
  * grouped:  [{"asset_class", "group", "members": ["name or symbol" | {"name", "symbol"}], "subgroups"?}, ...]
    (the format of tradingeconomics_groups_by_category.json, and of any hand-edited copy of it)

Every data column is matched to at most one catalog entry, in this order:
  alias (config/aliases.yaml) -> symbol -> exact name -> name without asset hints ("10Y", "bond", ...)
  -> url slug -> fuzzy (only suggested unless allow_fuzzy).
Outputs:
  * catalog entries without data are dropped ("extra" / thừa),
  * data columns without a catalog entry go to the unassigned group and are listed ("missing" / thiếu).
"""
import difflib
import json
import os
import re
import unicodedata
from collections import defaultdict

import yaml

# words in a column name that hint at the asset class; also stripped before name matching
ASSET_HINTS = {
    "bond": ["10y", "10 y", "bond", "bonds", "yield", "govt", "gov", "government"],
    "currency": ["fx", "currency", "exchange rate"],
    "stock_index": ["index", "stock market", "idx", "stocks"],
    "share": ["share", "equity"],
    "commodity": ["commodity", "futures", "spot"],
}
ELECTRICITY_HINTS = ["electricity", "power price"]


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_symbol(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def strip_hints(s: str) -> str:
    out = " " + norm_text(s) + " "
    for words in list(ASSET_HINTS.values()) + [ELECTRICITY_HINTS]:
        for w in words:
            out = out.replace(" " + norm_text(w) + " ", " ")
    return re.sub(r"\s+", " ", out).strip()


def hinted_classes(col: str) -> set:
    t = " " + norm_text(col) + " "
    hits = {ac for ac, words in ASSET_HINTS.items() if any(" " + norm_text(w) + " " in t for w in words)}
    if any(" " + norm_text(w) + " " in t for w in ELECTRICITY_HINTS):
        hits.add("electricity")
    return hits


# ---------------------------------------------------------------- loading

def load_catalog(path: str) -> list:
    raw = json.load(open(path, encoding="utf-8"))
    if not raw:
        return []
    entries = []
    if "members" in raw[0]:  # grouped format
        for g in raw:
            sub_of = {}
            for sub, names in (g.get("subgroups") or {}).items():
                for n in names:
                    sub_of[n if isinstance(n, str) else n.get("name")] = sub
            for m in g["members"]:
                name = m if isinstance(m, str) else m.get("name")
                symbol = None if isinstance(m, str) else m.get("symbol")
                column = None if isinstance(m, str) else m.get("column")  # present in a reconciled catalog
                entries.append({"symbol": symbol, "name": name, "asset_class": g["asset_class"], "group": g["group"],
                                "subgroup": sub_of.get(name), "url": None, "column": column})
    else:
        for r in raw:
            entries.append({k: r.get(k) for k in ["symbol", "name", "asset_class", "group", "subgroup", "url"]})
    for i, e in enumerate(entries):
        e["entry_id"] = e["symbol"] or f"{e['asset_class']}/{e['group']}/{e['name']}#{i}"
    return entries


def load_aliases(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    return (yaml.safe_load(open(path)) or {}).get("aliases") or {}


# ---------------------------------------------------------------- matching

class Matcher:
    def __init__(self, entries: list):
        self.entries = entries
        self.by_symbol = defaultdict(list)
        self.by_name = defaultdict(list)
        self.by_slug = defaultdict(list)
        self.by_column = defaultdict(list)
        for e in entries:
            if e.get("column"):
                self.by_column[e["column"]].append(e)
            if e["symbol"]:
                self.by_symbol[norm_symbol(e["symbol"])].append(e)
            if e["name"]:
                # a grouped catalog may list symbols as members: index them as symbols too
                self.by_name[norm_text(e["name"])].append(e)
                if re.fullmatch(r"[A-Za-z0-9.\- ]+:[A-Za-z]+", e["name"].strip()):
                    self.by_symbol[norm_symbol(e["name"])].append(e)
            if e.get("url"):
                path = re.sub(r"^https?://[^/]+/", "", e["url"]).strip("/")
                self.by_slug[norm_text(path)].append(e)
                self.by_slug[norm_text(path.split("/")[-1])].append(e)
        self.name_keys = list(self.by_name)

    @staticmethod
    def _disambiguate(col: str, cands: list) -> list:
        cands = list({c["entry_id"]: c for c in cands}.values())
        if len(cands) <= 1:
            return cands
        hints = hinted_classes(col)
        if "electricity" in hints:
            narrowed = [c for c in cands if c["group"] == "Electricity"]
        elif hints:
            narrowed = [c for c in cands if c["asset_class"] in hints and c["group"] != "Electricity"]
        else:
            narrowed = cands
        return narrowed or cands

    def match(self, col: str, aliases: dict) -> dict:
        if col in aliases:
            target = norm_symbol(aliases[col])
            hits = self.by_symbol.get(target) or self.by_name.get(norm_text(aliases[col]), [])
            if hits:
                return {"status": "matched", "method": "alias", "entry": hits[0]}
            return {"status": "unmatched", "method": "alias_not_in_catalog", "alias": aliases[col]}
        steps = [
            ("column", self.by_column.get(col, [])),
            ("symbol", self.by_symbol.get(norm_symbol(col), [])),
            ("name", self.by_name.get(norm_text(col), [])),
            ("name_without_hints", self.by_name.get(strip_hints(col), [])),
            ("url_slug", self.by_slug.get(norm_text(col), [])),
        ]
        for method, cands in steps:
            if not cands:
                continue
            cands = self._disambiguate(col, cands)
            if len(cands) == 1:
                return {"status": "matched", "method": method, "entry": cands[0]}
            return {"status": "ambiguous", "method": method,
                    "candidates": [f"{c['asset_class']}/{c['group']}/{c['name']} ({c['symbol']})" for c in cands]}
        return {"status": "unmatched", "method": None}

    def fuzzy(self, col: str, cutoff: float) -> list:
        key = strip_hints(col) or norm_text(col)
        keys = difflib.get_close_matches(key, self.name_keys, n=3, cutoff=cutoff)
        # containment: a multi-word catalog name inside the column name ("natural gas" in "natural gas henry hub")
        padded = f" {key} "
        keys += [k for k in self.name_keys if k not in keys and " " in k and f" {k} " in padded][:3]
        return [e for k in keys for e in self.by_name[k]]


# ---------------------------------------------------------------- reconcile

def reconcile(columns: list, ccfg: dict, out_dir: str) -> dict:
    """Match data columns to the catalog, write the reconciled catalog + report, return column mapping."""
    entries = load_catalog(ccfg["path"])
    aliases = load_aliases(ccfg.get("aliases_path"))
    m = Matcher(entries)
    unassigned = ccfg.get("unassigned_group", "_UNASSIGNED")

    mapping, ambiguous, unmatched, fuzzy_applied = {}, {}, {}, {}
    used = defaultdict(list)
    for col in columns:
        res = m.match(col, aliases)
        if res["status"] != "matched" and ccfg.get("allow_fuzzy"):
            fz = m.fuzzy(col, ccfg.get("fuzzy_cutoff", 0.92))
            if len(fz) == 1:
                res = {"status": "matched", "method": "fuzzy", "entry": fz[0]}
                fuzzy_applied[col] = fz[0]["name"]
        if res["status"] == "matched":
            e = res["entry"]
            used[e["entry_id"]].append(col)
            mapping[col] = {"symbol": e["symbol"], "name": e["name"], "asset_class": e["asset_class"],
                            "group": e["group"], "subgroup": e.get("subgroup"), "match": res["method"]}
            continue
        if res["status"] == "ambiguous":
            ambiguous[col] = res["candidates"]
        else:
            unmatched[col] = {"reason": res["method"] or "no catalog entry",
                              "fuzzy_suggestions": [f"{e['asset_class']}/{e['group']}/{e['name']} ({e['symbol']})"
                                                    for e in m.fuzzy(col, ccfg.get("fuzzy_cutoff", 0.92) - 0.1)]}
        mapping[col] = {"symbol": None, "name": col, "asset_class": "_unassigned", "group": unassigned,
                        "subgroup": None, "match": res["status"]}

    duplicates = {eid: cols for eid, cols in used.items() if len(cols) > 1}
    extra = [e for e in entries if e["entry_id"] not in used]

    # reconciled catalog in the grouped format (same shape as the input grouped json)
    groups = {}
    for col, info in mapping.items():
        g = groups.setdefault((info["asset_class"], info["group"]),
                              {"asset_class": info["asset_class"], "group": info["group"], "members": []})
        g["members"].append({"name": info["name"], "symbol": info["symbol"], "column": col})
        if info.get("subgroup"):
            g.setdefault("subgroups", {}).setdefault(info["subgroup"], []).append(info["name"])
    extra_by_group = defaultdict(list)
    for e in extra:
        extra_by_group[f"{e['asset_class']}/{e['group']}"].append(e["name"])

    report = {
        "n_data_columns": len(columns),
        "n_catalog_entries": len(entries),
        "n_matched": sum(1 for v in mapping.values() if v["asset_class"] != "_unassigned"),
        "match_methods": dict(sorted({k: sum(1 for v in mapping.values() if v["match"] == k)
                                      for k in {v["match"] for v in mapping.values()}}.items())),
        "missing_in_catalog": unmatched,          # thiếu: in data, not in catalog -> group by hand / add alias
        "ambiguous": ambiguous,                   # several catalog entries fit -> add alias
        "fuzzy_applied": fuzzy_applied,
        "duplicate_columns_same_entry": duplicates,
        "n_extra_removed": len(extra),            # thừa: in catalog, not in data -> removed from reconciled json
        "extra_removed_by_group": {k: v for k, v in sorted(extra_by_group.items())},
    }
    os.makedirs(out_dir, exist_ok=True)
    json.dump(list(groups.values()), open(os.path.join(out_dir, "catalog_reconciled.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(mapping, open(os.path.join(out_dir, "column_mapping.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(report, open(os.path.join(out_dir, "catalog_report.json"), "w"), ensure_ascii=False, indent=2)
    return {"mapping": mapping, "report": report}


def resolve_targets(tcfg: dict, mapping: dict) -> dict:
    """target name -> data column, using `column` if given else the catalog symbol."""
    by_symbol = {norm_symbol(v["symbol"]): col for col, v in mapping.items() if v["symbol"]}
    out, missing = {}, []
    for name, spec in tcfg.items():
        if not isinstance(spec, dict):
            continue
        col = spec.get("column") if spec.get("column") in mapping else by_symbol.get(norm_symbol(spec.get("symbol", "")))
        if col:
            out[name] = {"column": col, "family": spec.get("family", name)}
        else:
            missing.append(name)
    req_missing = [t for t in tcfg.get("required", []) if t in missing]
    if req_missing:
        raise ValueError(f"required targets not found in data: {req_missing}. "
                         f"Set targets.<name>.column or add an alias in config/aliases.yaml.")
    return {"targets": out, "missing": missing}
