"""Scrape Trading Economics market pages and emit feature groups.

Writes two files into out_dir (with --country-stocks also one stock group per country from /<country>/stocks;
that crawl is slow on purpose because Trading Economics rate-limits after ~15 fast page loads):
  tradingeconomics_feature_groups.json     one entry per series (symbol, name, group, ...)
  tradingeconomics_groups_by_category.json one entry per group with its member names
"""
import json, os, re, sys
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE = "https://tradingeconomics.com"
PAGES = {  # path -> asset_class
    "/commodities": "commodity",
    "/stocks": "stock_index",
    "/currencies": "currency",
    "/bonds": "bond",
    "/shares": "share",
}
# /crypto excluded (not used)
CROSS_CUTTING = {"Major", "Major10Y"}


def clean(t):
    return re.sub(r"\s+", " ", t or "").strip()


def parse_tables(html, path, asset_class):
    s = BeautifulSoup(html, "html.parser")
    out = []
    for t in s.find_all("table"):
        # first column may be an empty flag column; the group label is the first non-empty header
        group = next((clean(th.get_text(" ")) for th in t.select("thead th") if clean(th.get_text(" "))), "")
        for r in t.select("tbody tr[data-symbol]"):
            td = r.select_one("td.datatable-item-first") or r.find("td")
            a = td.find("a", href=True) if td else None
            unit_div = td.find("div") if td else None
            href = a["href"] if a else ""
            # /united-kingdom/stock-market -> "united-kingdom"; /commodity/... has no country
            parts = href.strip("/").split("/")
            country = parts[0] if len(parts) == 2 and parts[0] != "commodity" else None
            out.append({
                "symbol": r["data-symbol"].strip(),
                "name": clean(a.get_text(" ")) if a else clean(td.get_text(" ")),
                "asset_class": asset_class,
                "group": group,
                "subgroup": None,
                "country": country,
                "unit": clean(unit_div.get_text(" ")) if unit_div else None,
                "url": BASE + href if href else None,
                "source_page": BASE + path,
            })
    return out


def parse_shares_tree(html, path):
    """/shares embeds a sector -> industry -> company tree as `const data = {...};` for its treemap."""
    s = BeautifulSoup(html, "html.parser")
    script = next(sc.string for sc in s.find_all("script") if sc.string and '"d3_id"' in sc.string)
    start = script.find('{"name"')
    tree, _ = json.JSONDecoder().raw_decode(script[start:])
    out = []
    for sector in tree["children"] or []:
        for industry in sector["children"] or []:
            for co in industry["children"] or []:
                out.append({
                    "symbol": co["url"].strip("/").upper(),  # /aapl:us -> AAPL:US
                    "name": co["name"],
                    "asset_class": "share",
                    "group": sector["name"],
                    "subgroup": industry["name"],
                    "country": "united-states",
                    "unit": None,
                    "url": BASE + co["url"],
                    "source_page": BASE + path,
                })
    return out


def parse_country_stocks(html, country):
    """/<country>/stocks lists the country's shares as links inside div.card-body#<country>."""
    s = BeautifulSoup(html, "html.parser")
    card = s.select_one(f"div.card-body#{country}")
    out = []
    for a in (card.find_all("a", href=True) if card else []):
        href = a["href"].strip()  # some hrefs carry trailing whitespace
        if re.match(r"^/[^/]+:[a-z]+$", href):
            group = " ".join(w.capitalize() for w in country.split("-"))
            out.append({"symbol": href.strip("/").upper(), "name": clean(a.get_text(" ")), "asset_class": "share",
                        "group": group, "subgroup": None, "country": country, "unit": None,
                        "url": BASE + href, "source_page": f"{BASE}/{country}/stocks"})
    return out


def main(out_dir, country_stocks=False, delay_sec=8.0):
    rows = []
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        ctx = b.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
            locale="en-US", viewport={"width": 1400, "height": 900})
        pg = ctx.new_page()
        for path, ac in PAGES.items():
            resp = pg.goto(BASE + path, wait_until="domcontentloaded", timeout=60000)
            pg.wait_for_timeout(2500)
            html = pg.content()
            got = parse_shares_tree(html, path) if ac == "share" else parse_tables(html, path, ac)
            print(f"{path}: HTTP {resp.status if resp else '?'} -> {len(got)} rows", file=sys.stderr)
            rows += got
        if country_stocks:
            countries = sorted({r["country"] for r in rows if r["asset_class"] == "stock_index" and r["country"]})
            for c in countries:
                pg.goto(f"{BASE}/{c}/stocks", wait_until="domcontentloaded", timeout=60000)
                pg.wait_for_timeout(int(delay_sec * 1000))
                got = parse_country_stocks(pg.content(), c)
                print(f"/{c}/stocks -> {len(got)} shares", file=sys.stderr)
                rows += got
        b.close()

    # Merge duplicates (same symbol listed under "Major" and a region, etc.)
    merged = {}
    for r in rows:
        key = (r["asset_class"], r["symbol"])
        m = merged.setdefault(key, {**r, "group": None, "all_groups": []})
        if r["group"] not in m["all_groups"]:
            m["all_groups"].append(r["group"])
    result = []
    for m in merged.values():
        specific = [g for g in m["all_groups"] if g not in CROSS_CUTTING]
        m["group"] = specific[0] if specific else m["all_groups"][0]
        m["is_major"] = any(g in CROSS_CUTTING for g in m["all_groups"])
        result.append({k: m[k] for k in ["symbol", "name", "asset_class", "group", "subgroup", "all_groups",
                                         "is_major", "country", "unit", "url", "source_page"]})
    path = os.path.join(out_dir, "tradingeconomics_feature_groups.json")
    json.dump(result, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {len(result)} unique series to {path}", file=sys.stderr)

    groups = {}
    for r in result:
        g = groups.setdefault((r["asset_class"], r["group"]), {"asset_class": r["asset_class"], "group": r["group"], "members": []})
        g["members"].append(r["name"])
        if r["subgroup"]:
            g.setdefault("subgroups", {}).setdefault(r["subgroup"], []).append(r["name"])
    path = os.path.join(out_dir, "tradingeconomics_groups_by_category.json")
    json.dump(list(groups.values()), open(path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {len(groups)} groups to {path}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--country-stocks", action="store_true")
    ap.add_argument("--delay", type=float, default=8.0, help="seconds between country pages")
    a = ap.parse_args()
    main(a.out_dir, a.country_stocks, a.delay)
