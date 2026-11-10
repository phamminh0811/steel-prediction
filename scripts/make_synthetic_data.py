"""Synthetic weekly market data shaped like the real dataset, with known ground truth.

* ~860 columns drawn from the Trading Economics catalog (commodities, indexes, FX, 10Y bonds, shares)
  plus uncatalogued macro columns, generated from a factor model (global, China, USD, energy, metals, ...).
* Coal and iron-ore returns are driven by LAGGED weekly changes of a known list of series (`true_drivers`),
  so a feature-selection run can be scored on how many true drivers it recovers.
* Real-data problems are injected on purpose: late starts, discontinued series, gaps, pegged currencies,
  a unit break, a negative price, a near-duplicate column, mixed column naming, typos, uncatalogued columns.

    python scripts/make_synthetic_data.py --out data/synthetic
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
REGIONS = ["Europe", "Asia", "America", "Africa", "Oceania"]
CHINA_LINKED = {"china", "hong-kong", "australia", "brazil", "south-africa", "chile", "indonesia", "new-zealand", "peru"}
COMMODITY_FX = {"AUD", "BRL", "ZAR", "CLP", "CAD", "NZD", "NOK", "IDR", "PEN", "CNY", "MNT"}
PEGGED = {"USDHKD", "USDPAB", "USDSAR", "USDAED", "USDQAR", "USDOMR", "USDBHD", "USDJOD", "USDDJF"}

# true lead drivers: (symbol or extra column, lag in weeks, sign)
TRUE_DRIVERS = {
    "iron_ore": [("HRC:COM", 1, 1), ("JBP:COM", 1, 1), ("SHCOMP:IND", 2, 1), ("AUDUSD:CUR", 1, 1),
                 ("USDCNY:CUR", 2, -1), ("SPSCFI:COM", 3, 1), ("SSC:COM", 2, 1), ("HG1:COM", 1, 1),
                 ("NUE:US", 1, 1), ("BHP:AU", 1, 1), ("GCNY10YR:GOV", 4, 1), ("China Caixin Manufacturing PMI", 4, 1),
                 ("Baltic Dry Index", 2, 1)],
    "coal": [("NG1:COM", 1, 1), ("NGEU:COM", 1, 1), ("LNGJKM:COM", 2, 1), ("CL1:COM", 2, 1), ("USDIDR:CUR", 1, -1),
             ("JCI:IND", 2, 1), ("AS51:IND", 3, 1), ("DEUELEPRI:COM", 1, 1), ("SENSEX:IND", 3, 1),
             ("USDINR:CUR", 2, -1), ("China Thermal Power Generation", 4, 1), ("Baltic Dry Index", 2, 1)],
}
EXTRA_COLUMNS = ["China Caixin Manufacturing PMI", "China Thermal Power Generation", "Baltic Dry Index",
                 "China Steel Production", "Australia Iron Ore Exports", "US Initial Jobless Claims",
                 "Euro Area Sentiment", "Global Shipping Cost Index", "China Credit Impulse", "India Coal Imports",
                 "Japan Machine Orders", "Korea Exports YoY", "US ISM Manufacturing", "Germany IFO",
                 "China Property Sales", "Brazil Iron Ore Exports", "Indonesia Coal Exports", "VIX Proxy",
                 "Copper Inventory LME", "Aluminium Inventory LME"]
MUST_SHARES = ["NUE:US", "STLD:US", "FCX:US", "BHP:AU", "RIO:AU", "FMG:AU", "VALE3:BZ", "600019:CH"]


def ar1(rng, n, phi=0.1, df=5):
    e = rng.standard_t(df, n) / np.sqrt(df / (df - 2))
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def regime_vol(dates):
    v = np.ones(len(dates))
    v[(dates >= "2020-03-01") & (dates <= "2020-07-01")] = 2.5
    v[(dates >= "2022-02-20") & (dates <= "2022-12-31")] = 1.6
    v[(dates >= "2015-08-01") & (dates <= "2016-02-28")] = 1.4
    return v


def loadings(e):
    ac, g, name, sym = e["asset_class"], e["group"], e["name"], e["symbol"] or ""
    country = e.get("country") or ""
    L = {}
    if ac == "commodity":
        L = {"Energy": {"energy": 1, "global": .3, "china": .2}, "Metals": {"metals": 1, "china": .6, "usd": -.4},
             "Industrial": {"metals": .7, "china": .5}, "Agricultural": {"agri": 1, "usd": -.2},
             "Livestock": {"agri": .5}, "Index": {"energy": .5, "metals": .5, "agri": .3, "global": .3},
             "Electricity": {"energy": .8, "Europe": .3}}.get(g, {"global": .3})
    elif ac == "stock_index":
        L = {"global": 1, g: .6}
        if country in CHINA_LINKED:
            L["china"] = .5
    elif ac == "currency":
        base, quote = name[:3], name[3:6]
        if name == "DXY":
            L = {"usd": 1}
        elif base == "USD":
            L = {"usd": 1, "china": -.5 if quote in COMMODITY_FX else 0, g: .2}
        else:
            L = {"usd": -1, "china": .5 if base in COMMODITY_FX else 0, g: .2}
    elif ac == "bond":
        L = {"rates": 1, g: .5, "global": .3}
    elif ac == "share":
        L = {"global": .8, f"sector:{g}": 1}
        sub = e.get("subgroup") or ""
        if g == "Materials" or re.search(r"metal|steel|mining", sub, re.I):
            L.update({"metals": .6, "china": .5})
        if g == "Energy":
            L["energy"] = .7
        if country in CHINA_LINKED:
            L["china"] = L.get("china", 0) + .4
    return L


SCALE = {"commodity": .035, "stock_index": .02, "currency": .007, "bond": .06, "share": .035}
IDIO = {"commodity": .03, "stock_index": .012, "currency": .006, "bond": .05, "share": .035}
BASE = {"commodity": 100., "stock_index": 5000., "currency": 1., "bond": 3., "share": 50.}


def main(out):
    rng = np.random.default_rng(7)
    cat = json.load(open(os.path.join(ROOT, "catalog", "tradingeconomics_feature_groups.json")))
    dates = pd.date_range("2011-01-07", "2026-10-30", freq="W-FRI")
    n = len(dates)
    vol = regime_vol(dates)

    # --- choose ~860 catalog series
    pick = [e for e in cat if e["asset_class"] in ("commodity", "stock_index", "currency", "bond")]
    shares = [e for e in cat if e["asset_class"] == "share"]
    must = [e for e in shares if e["symbol"] in MUST_SHARES]
    rest = [e for e in shares if e["symbol"] not in MUST_SHARES]
    pick += must + list(rng.choice(rest, 470 - len(must), replace=False))
    # remove a few catalog series so the data is not a superset of the catalog
    drop_idx = set(rng.choice(len(pick), 25, replace=False)) - {i for i, e in enumerate(pick) if e["symbol"] in
                                                                   {s for v in TRUE_DRIVERS.values() for s, _, _ in v} |
                                                                   {"SCO:COM", "IOE:COM", "XAL1:COM", "DJM:COM"}}
    pick = [e for i, e in enumerate(pick) if i not in drop_idx]

    factors = {k: ar1(rng, n) * vol for k in ["global", "china", "usd", "energy", "metals", "rates", "agri"] + REGIONS}
    for g in {e["group"] for e in pick if e["asset_class"] == "share"}:
        factors[f"sector:{g}"] = ar1(rng, n) * vol

    rets, meta = {}, {}
    for e in pick:
        ac = e["asset_class"]
        r = sum(w * factors[f] for f, w in loadings(e).items() if w) * SCALE[ac] / 1.5
        r = r + rng.standard_t(5, n) / np.sqrt(5 / 3) * IDIO[ac] * vol
        if e["name"] in PEGGED:
            r = np.where(rng.random(n) < .85, 0, rng.normal(0, 1e-4, n))
        key = e["symbol"]
        rets[key], meta[key] = r, e
    for c in EXTRA_COLUMNS:
        f = "china" if "China" in c or "Iron" in c else "global"
        rets[c] = factors[f] * .02 + rng.normal(0, .02, n) * vol
        meta[c] = {"symbol": None, "name": c, "asset_class": "extra", "group": "extra"}

    # --- targets: contemporaneous factor part + lagged true drivers + noise
    def z(x):
        return (x - x.mean()) / x.std()

    def driven(fam, contemp, beta=.17, noise=1.0):
        pred = np.zeros(n)
        for s, lag, sign in TRUE_DRIVERS[fam]:
            pred[lag:] += sign * beta * z(rets[s])[:-lag]
        return (contemp + pred + rng.normal(0, noise, n)) * vol

    io_r = driven("iron_ore", .6 * z(factors["china"]) + .5 * z(factors["metals"]) - .3 * z(factors["usd"])) * .02
    coal_r = driven("coal", .6 * z(factors["energy"]) + .3 * z(factors["china"])) * .025
    rets["SCO:COM"] = io_r
    rets["XAL1:COM"] = coal_r
    rets["DJM:COM"] = .6 * io_r + .3 * coal_r + rng.normal(0, .015, n) * vol
    rets["IOE:COM"] = io_r + rets["USDCNY:CUR"] + rng.normal(0, .004, n)  # CNY price = USD price x USDCNY

    # --- levels
    levels = {}
    for k, r in rets.items():
        ac = meta[k]["asset_class"]
        if ac == "bond":
            # mean-reverting yield in % (a clipped random walk would get stuck at the floor and look pegged)
            mu, y = BASE[ac] + rng.normal(0, 1.5), np.empty(n)
            y[0] = mu
            for i in range(1, n):
                y[i] = y[i - 1] + 0.01 * (mu - y[i - 1]) + r[i]
            levels[k] = y
        elif ac == "extra":
            levels[k] = 50 + 10 * np.cumsum(r)  # index-like, may go negative -> diff transform
        else:
            levels[k] = BASE[ac] * np.exp(rng.normal(0, .5) + np.cumsum(r))
    wide = pd.DataFrame(levels, index=dates)

    # --- inject data problems
    issues = {}
    late = [k for k in wide.columns if meta[k]["asset_class"] == "share"]
    late = list(rng.choice([k for k in late if k not in MUST_SHARES], 40, replace=False))
    for k in late:
        start = dates[rng.integers(260, 760)]
        wide.loc[wide.index < start, k] = np.nan
    wide.loc[wide.index < "2018-01-01", "LC:COM"] = np.nan
    issues["late_start"] = late + ["LC:COM"]
    disc = list(rng.choice([k for k in wide.columns if meta[k]["asset_class"] in ("share", "stock_index")
                            and k not in MUST_SHARES and k not in {s for v in TRUE_DRIVERS.values() for s, _, _ in v}], 10, replace=False))
    for k in disc:
        wide.loc[wide.index > "2022-06-30", k] = np.nan
    issues["discontinued"] = disc
    gappy = list(rng.choice(wide.columns, 100, replace=False))
    for k in gappy:
        wide.loc[rng.random(n) < .03, k] = np.nan
    long_gap = list(rng.choice([c for c in wide.columns if meta[c]["asset_class"] == "share"], 5, replace=False))
    for k in long_gap:
        i = rng.integers(100, n - 20)
        wide.iloc[i:i + 10, wide.columns.get_loc(k)] = np.nan
    issues["random_gaps_3pct"] = gappy
    issues["gap_10_weeks"] = long_gap
    issues["pegged_fx"] = [k for k in wide.columns if meta[k]["name"] in PEGGED]
    wide.loc[wide.index >= "2019-03-01", "PLO:COM"] *= 10
    issues["unit_break_x10"] = ["PLO:COM"]
    wide.loc["2020-04-24", "CL1:COM"] = -12.0
    issues["negative_price"] = ["CL1:COM"]
    wide["Brent Crude (copy)"] = wide["CO1:COM"] * np.exp(rng.normal(0, .002, n))
    meta["Brent Crude (copy)"] = {"symbol": None, "name": "Brent Crude (copy)", "asset_class": "extra", "group": "extra"}
    # second vendor for a true driver: should land in the same cluster as HRC:COM
    wide["HRC Steel (vendor B)"] = wide["HRC:COM"] * np.exp(rng.normal(0, .003, n))
    meta["HRC Steel (vendor B)"] = {"symbol": None, "name": "HRC Steel (vendor B)", "asset_class": "extra", "group": "extra"}
    issues["near_duplicate"] = {"Brent Crude (copy)": "CO1:COM", "HRC Steel (vendor B)": "HRC:COM"}

    # --- column naming like a hand-built dataset: mixed conventions + a few typos
    typos = {"NG1:COM": "Natural Gas Henry Hub", "HG1:COM": "Copper Spot", "XAUUSD:CUR": "Gold Spot", "W 1:COM": "Wheat Futures"}

    def mixed_name(k):
        e = meta[k]
        if k in typos:
            return typos[k]
        ac = e["asset_class"]
        if ac == "commodity":
            return f"{e['name']} Electricity" if e["group"] == "Electricity" else e["name"]
        if ac == "currency":
            return e["name"]
        if ac == "bond":
            return f"{e['name']} 10Y"
        if ac == "share" and rng.random() < .1:
            return e["name"]
        return k  # symbol (indexes, most shares) or extra column label

    names = {k: mixed_name(k) for k in wide.columns}
    assert len(set(names.values())) == len(names), "column name collision"
    os.makedirs(out, exist_ok=True)
    w_sym = wide.reset_index().rename(columns={"index": "date"})
    w_mix = wide.rename(columns=names).reset_index().rename(columns={"index": "date"})
    w_sym.to_parquet(os.path.join(out, "wide_symbols.parquet"), index=False)
    w_mix.to_parquet(os.path.join(out, "wide_mixed.parquet"), index=False)
    w_mix.to_csv(os.path.join(out, "wide_mixed.csv"), index=False)

    # long format with heterogeneous stamping: bonds on Thursday, 20 shares daily
    lg = wide.rename(columns=names).stack().rename("value").reset_index()
    lg.columns = ["date", "series", "value"]
    bond_cols = [names[k] for k in wide.columns if meta[k]["asset_class"] == "bond"]
    lg.loc[lg["series"].isin(bond_cols), "date"] -= pd.Timedelta(days=1)
    daily_cols = [names[k] for k in wide.columns if meta[k]["asset_class"] == "share"][:20]
    extra_rows = []
    for c in daily_cols:
        s = lg[lg["series"] == c]
        for off in range(1, 5):  # Mon..Thu noise around the Friday value
            extra_rows.append(s.assign(date=s["date"] - pd.Timedelta(days=off),
                                       value=s["value"] * np.exp(rng.normal(0, .01, len(s)))))
    lg = pd.concat([lg] + extra_rows, ignore_index=True)
    lg.to_parquet(os.path.join(out, "long.parquet"), index=False)
    issues["long_format"] = {"bonds_stamped_thursday": len(bond_cols), "daily_series": daily_cols}

    truth = {"true_drivers": {fam: [{"symbol": s if s in meta and meta[s]["symbol"] else None,
                                     "column_mixed": names[s], "lag": lag, "sign": sign} for s, lag, sign in v]
                              for fam, v in TRUE_DRIVERS.items()},
             "targets": {"iron_ore": names["SCO:COM"], "iron_ore_dce": names["IOE:COM"],
                         "coal_newcastle": names["XAL1:COM"], "coal_coking": names["DJM:COM"]},
             "issues": issues, "typos": {names[k]: k for k in typos if k in names},
             "catalog_series_dropped_from_data": len(drop_idx), "extra_columns": EXTRA_COLUMNS + ["Brent Crude (copy)", "HRC Steel (vendor B)"],
             "n_columns": wide.shape[1], "n_weeks": n}
    json.dump(truth, open(os.path.join(out, "ground_truth.json"), "w"), indent=2, default=str)
    print(f"wrote {wide.shape[1]} columns x {n} weeks to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "synthetic"))
    main(ap.parse_args().out)
