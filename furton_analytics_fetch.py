"""Fetch and cache the daily price series the analytics generator needs.

One Tiingo request per symbol (full date range in a single call), cached to
market_cache/<SYMBOL>.json so furton_analytics.py can be iterated without
touching the rate limit (free tier ~50 requests/hour; the weekly screen needs
30 of that budget on Mondays). ^DJI comes from Yahoo — the documented
exception (Tiingo's free tier has no index series).

Each cache file: [{"date": "YYYY-MM-DD", "close": raw, "adjClose": adj,
"splitFactor": f, "divCash": d}, ...] ascending. Yahoo's ^DJI rows carry
close only (raw == adj for an index).

Usage:
    py furton_analytics_fetch.py                  # default range
    py furton_analytics_fetch.py --start 2025-09-02 --end 2026-09-04
    py furton_analytics_fetch.py --force SYM ...  # refetch named symbols
"""
import argparse
import datetime
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import furton_server as fs

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "market_cache"

# Every symbol the account has ever held (union of tradelog positions), plus
# SPY for the modeled-risk block. HONA (the June spin-off) has no reliable
# series; the generator carries it at snapshot price per the published note.
SYMBOLS = ["AMGN", "AMZN", "AXP", "CRM", "CVX", "DIS", "GOOGL", "HD", "HON",
           "JNJ", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "SHW", "TRV",
           "UNH", "V", "VZ", "SPY"]
YAHOO_SYMBOLS = ["^DJI"]


def fetch_tiingo(sym, start, end, token):
    url = (f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
           f"?startDate={start}&endDate={end}&token={token}")
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = json.loads(r.read())
    return [{"date": row["date"][:10], "close": row["close"],
             "adjClose": row["adjClose"], "splitFactor": row.get("splitFactor", 1.0),
             "divCash": row.get("divCash", 0.0)} for row in rows]


def fetch_yahoo(sym, start, end):
    t0 = int(datetime.datetime.fromisoformat(start).replace(
        tzinfo=datetime.timezone.utc).timestamp())
    t1 = int(datetime.datetime.fromisoformat(end).replace(
        tzinfo=datetime.timezone.utc).timestamp()) + 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}"
           f"?period1={t0}&period2={t1}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    res = data["chart"]["result"][0]
    stamps = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    out = []
    for ts, c in zip(stamps, closes):
        if c is None:
            continue
        d = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        # Yahoo stamps the session open; the date component is the trading day
        # (US sessions never cross midnight UTC at the open).
        out.append({"date": d.strftime("%Y-%m-%d"), "close": round(c, 4),
                    "adjClose": round(c, 4), "splitFactor": 1.0, "divCash": 0.0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-09-02")
    ap.add_argument("--end", default="2026-09-04")
    ap.add_argument("--force", nargs="*", default=None,
                    help="refetch these symbols even if cached")
    args = ap.parse_args()

    CACHE.mkdir(exist_ok=True)
    token = fs.load_tiingo_key()
    force = {s.upper() for s in (args.force or [])}

    fetched = skipped = 0
    for sym in SYMBOLS:
        dest = CACHE / f"{sym}.json"
        if dest.exists() and sym not in force:
            skipped += 1
            continue
        rows = fetch_tiingo(sym, args.start, args.end, token)
        dest.write_text(json.dumps(rows), encoding="utf-8")
        print(f"  {sym:6} {len(rows)} rows  {rows[0]['date']} .. {rows[-1]['date']}")
        fetched += 1
        time.sleep(0.25)

    for sym in YAHOO_SYMBOLS:
        dest = CACHE / f"{sym.replace('^', '_')}.json"
        if dest.exists() and sym not in force and sym.replace("^", "_") not in force:
            skipped += 1
            continue
        rows = fetch_yahoo(sym, args.start, args.end)
        dest.write_text(json.dumps(rows), encoding="utf-8")
        print(f"  {sym:6} {len(rows)} rows  {rows[0]['date']} .. {rows[-1]['date']} (yahoo)")
        fetched += 1

    print(f"fetched {fetched}, cached-skip {skipped} -> {CACHE}")


if __name__ == "__main__":
    main()
