"""Publish each holding's most recent official committee statement.

Reads screens/screen_YYYY-MM-DD.json (newest first) plus the published
holdings.json and writes data/statements.json for the site (both the
furton_website/ staging copy and the deployed docs/ copy): for each name the
account holds, the last screen where the committee actually deliberated it,
that screen's date, and the statement body verbatim with only the bold
letterhead lines removed. Nothing else leaves the archive: no briefs, no
blind votes, no deliberation transcripts, no token accounting.

Supersedes the archived blind-vote publisher (furton_publish_committee.py,
git history at 9e4ecd9).

Weekly use, after the rebalance screen and holdings snapshot:

    py -3 furton_publish_statements.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NL = chr(10)


def strip_letterhead(text):
    """Drop up to three leading whole-line-bold letterhead lines."""
    lines = text.split(NL)
    i, dropped = 0, 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if dropped < 3 and s.startswith("**") and s.endswith("**") and len(s) > 4 and "**" not in s[2:-2]:
            dropped += 1
            i += 1
            continue
        break
    return NL.join(lines[i:]).strip()


def main():
    paths = sorted(glob.glob(os.path.join(ROOT, "screens", "screen_*.json")), reverse=True)
    if not paths:
        sys.exit("no screens/screen_*.json found")
    screens = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            screens.append(json.load(f))
    with open(os.path.join(ROOT, "furton_website", "data", "holdings.json"), encoding="utf-8") as f:
        holdings = json.load(f)

    rows = []
    for h in holdings.get("holdings", []):
        ticker = h["ticker"].upper()
        row = {"ticker": ticker, "name": h.get("name") or ticker}
        for screen in screens:  # newest first
            stock = (screen.get("stocks") or {}).get(ticker)
            if stock and stock.get("advanced") and (stock.get("statement") or "").strip():
                row["screen_date"] = screen.get("date")
                row["statement"] = strip_letterhead(stock["statement"])
                break
        rows.append(row)

    out = {"holdings_date": holdings.get("date"), "rows": rows}
    dests = [
        os.path.join(ROOT, "furton_website", "data", "statements.json"),
        os.path.join(ROOT, "docs", "data", "statements.json"),
    ]
    for dest in dests:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    have = [r for r in rows if r.get("statement")]
    dates = sorted(set(r["screen_date"] for r in have))
    print(f"wrote statements.json: {len(have)} of {len(rows)} holdings have a statement (dates {dates})")
    for dest in dests:
        print("  " + dest)


if __name__ == "__main__":
    main()
