"""Publish the committee's blind votes on the current holdings.

NOTE: the site section this fed (the blind-vote grid) was archived on
2026-08-22 (it lives in git history at 9e4ecd9); the script is kept because
its screen-reading plumbing is the starting point for publishing committee
statements if that ships. It no longer needs to run weekly.

Reads the newest screens/screen_YYYY-MM-DD.json plus the published
holdings.json and writes data/committee.json for the site (both the
furton_website/ staging copy and the deployed docs/ copy): for each name the
account holds, each member's BLIND vote (buy / pass) and its conviction,
taken before the deliberation round. Nothing else leaves the archive: no
briefs, no statements, no deliberation text, no token accounting.

Weekly use, after the rebalance screen and holdings snapshot:

    py -3 furton_publish_committee.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

INVESTORS = [
    ("buffett", "Warren Buffett", "Buffett", "W"),
    ("marks", "Howard Marks", "Marks", "H"),
    ("greenblatt", "Joel Greenblatt", "Greenblatt", "J"),
    ("wood", "Cathie Wood", "Wood", "C"),
    ("aschenbrenner", "Leopold Aschenbrenner", "Aschenbrenner", "L"),
]
KEY_BY_NAME = {name: key for key, name, _, _ in INVESTORS}


def main():
    screens = sorted(glob.glob(os.path.join(ROOT, "screens", "screen_*.json")))
    if not screens:
        sys.exit("no screens/screen_*.json found")
    with open(screens[-1], encoding="utf-8") as f:
        screen = json.load(f)
    with open(os.path.join(ROOT, "furton_website", "data", "holdings.json"), encoding="utf-8") as f:
        holdings = json.load(f)

    rows = []
    for h in holdings.get("holdings", []):
        ticker = h["ticker"].upper()
        stock = screen.get("stocks", {}).get(ticker)
        votes = {}
        if stock:
            for b in stock.get("blind", []):
                key = KEY_BY_NAME.get(b.get("investor"))
                if not key or b.get("parse_error"):
                    continue
                position = (b.get("position") or "").upper()
                votes[key] = {
                    "vote": "buy" if position == "BUY" else "pass",
                    "strength": b.get("conviction"),
                }
        rows.append({
            "ticker": ticker,
            "name": h.get("name") or (stock or {}).get("name") or ticker,
            "votes": votes,
        })

    out = {
        "date": screen.get("date"),
        "holdings_date": holdings.get("date"),
        "investors": [{"key": k, "name": n, "short": s, "initial": i} for k, n, s, i in INVESTORS],
        "rows": rows,
    }

    dests = [
        os.path.join(ROOT, "furton_website", "data", "committee.json"),
        os.path.join(ROOT, "docs", "data", "committee.json"),
    ]
    for dest in dests:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    buys = sum(1 for r in rows for v in r["votes"].values() if v["vote"] == "buy")
    total = sum(len(r["votes"]) for r in rows)
    print(f"wrote committee.json ({len(rows)} holdings, {buys} blind buys of {total} votes, "
          f"screen {out['date']}) to:")
    for dest in dests:
        print("  " + dest)


if __name__ == "__main__":
    main()
