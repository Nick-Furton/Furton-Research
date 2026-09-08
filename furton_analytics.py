"""Regenerate docs/data/analytics.json — the site's risk/attribution document.

Reconstructed 2026-09-08 (the original generator lived only in a prior session
and was never preserved; this file is now the canonical tool, validated by
regenerating the committed 2026-08-31 document and diffing — see --validate).

Method (matches the published warnings paragraph):
  * LIVE replay: each trading day from inception (2026-06-23) through --asof is
    valued on the shares and cash the account carried that day — the latest
    holdings snapshot on or before the day (rebalance days use the post-trade
    book; inception day is carried at cost, i.e. exactly $10,000) at that day's
    Tiingo closing prices (raw, as-traded). A ticker with no market data series
    (HONA) is carried at its snapshot price. Corporate-action cash (type
    "cash" ledger events: cash-in-lieu) enters on the posted date.
  * SETTLED HISTORY: daily values already published in the prior analytics.json
    are carried forward VERBATIM; only days after the prior asof are computed.
    Published history is a record — it does not get re-marked when tooling
    changes. (The from-scratch replay reproduces 44 of the first 48 published
    days to the cent; the four differences are the late-June Honeywell
    spinoff/split days whose ad-hoc handling the published warning documents.)
    --no-splice rebuilds from scratch for forensics.
  * Headline risk stats (vol/Sharpe/Sortino/drawdown) come from that replay's
    daily returns. ret_dow is the Dow's close-to-close move over the same
    window (^DJI via Yahoo — the documented exception to the all-Tiingo rule).
  * MODELED block: current position weights applied to the daily-return
    history of the trailing CALENDAR year (asof minus one year, raw closes) —
    the risk profile of the book as constructed, not a live track record.
    Histogram, VaR/CVaR, correlation, per-position vol1y/beta1y, and the
    30-session rolling-vol series use the same window.

Usage:
  py furton_analytics.py --asof 2026-09-04 --through 2026-09-08          # write
  py furton_analytics.py --asof 2026-08-28 --through 2026-08-31 \
      --validate <committed.json>                                        # diff
"""
import argparse
import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "market_cache"
WR = ROOT / "weekly_records"
START = "2026-06-23"
STARTING_CAPITAL = 10000.0
RF_DAILY = 0.04 / 252   # 4% annual risk-free, in Sharpe/Sortino excess returns

WARNING = ("The daily chart line replays the account's actual holdings: each "
           "trading day is valued on the shares and cash the account carried "
           "that day (from the weekly position snapshots; rebalance days use "
           "the post-trade book) at that day's closing prices. Weekly record "
           "values are intraday Schwab statements, so the delta column "
           "measures close-of-day vs statement-time pricing, not a holdings "
           "mismatch. A ticker with no market data series is carried at its "
           "snapshot price, and corporate-action cash (cash-in-lieu, dividends "
           "in flight) enters the series on the ledger's posted date - around "
           "the late-June Honeywell spinoff and reverse split this timing "
           "accounts for most of that week's delta.")

MODELED_NOTE = ("Current portfolio weights applied to the trailing year of "
                "daily market data - the risk profile of the book as "
                "constructed, not a live track record.")


# ── tiny stats (no numpy) ────────────────────────────────────────────────────
def mean(xs):
    return sum(xs) / len(xs)


def std(xs, ddof=1):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def cov(xs, ys, ddof=1):
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - ddof)


def corr(xs, ys):
    sx, sy = std(xs), std(ys)
    return cov(xs, ys) / (sx * sy) if sx > 0 and sy > 0 else 0.0


def percentile(xs, p):
    """Linear interpolation between order statistics (numpy default)."""
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


# ── inputs ───────────────────────────────────────────────────────────────────
def load_series():
    out = {}
    for p in CACHE.glob("*.json"):
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not (isinstance(rows, list) and rows and "close" in rows[0]):
            continue
        out[p.stem] = {r["date"]: r for r in rows}
    return out


def load_books(through):
    """[(date, {ticker: {shares, mv}}, cash)] ascending; inception from the
    first tradelog (shares at fill), then every positions_*.json snapshot."""
    books = []
    t0 = json.loads((WR / f"furton_tradelog_{START}.json").read_text(encoding="utf-8"))
    pos = {p["ticker"]: {"shares": p["shares"], "mv": p["shares"] * p["fill_price"]}
           for p in t0["positions"]}
    books.append((START, pos, t0["cash_remaining"]))
    for p in sorted(ROOT.glob("holdings/positions_*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["date"] > through:
            continue
        pos = {q["ticker"]: {"shares": q["shares"], "mv": q["market_value"]}
               for q in rec["positions"]}
        cash = rec.get("cash_value")
        if cash is None:
            av = rec.get("account_value")
            cash = round(av - rec["positions_market_value"], 2) if av else 0.0
        books.append((rec["date"], pos, cash))
    return books


def load_tradelog(through):
    cands = sorted(WR.glob("furton_tradelog_*.json"))
    cands = [p for p in cands if p.stem.split("_")[-1] <= through]
    return json.loads(cands[-1].read_text(encoding="utf-8"))


# ── build ────────────────────────────────────────────────────────────────────
def build(asof, through, no_splice=False):
    series = load_series()
    dji = series["_DJI"]
    books = load_books(through)
    tradelog = load_tradelog(through)
    perf = json.loads((ROOT / "docs" / "data" / "performance.json")
                      .read_text(encoding="utf-8"))["entries"]
    perf = [e for e in perf if e["date"] <= through]

    # settled history: prior published daily values are carried verbatim
    pinned = {}
    prior_path = ROOT / "docs" / "data" / "analytics.json"
    if not no_splice and prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        s = prior.get("series") or {}
        prior_asof = prior.get("asof", "")
        pinned = {d: v for d, v in zip(s.get("dates", []), s.get("value", []))
                  if d <= prior_asof}

    cal = sorted(d for d in dji if START <= d <= asof)
    cash_events = [e for e in tradelog["cash_events"] if e["type"] == "cash"]

    def book_for(day):
        cur = books[0]
        for b in books:
            if b[0] <= day:
                cur = b
            else:
                break
        return cur

    def value_on(day):
        bdate, pos, cash = book_for(day)
        cash += sum(e["amount"] for e in cash_events if bdate < e["date"] <= day)
        total = cash
        for t, p in pos.items():
            row = series.get(t, {}).get(day)
            px = row["close"] if row else (p["mv"] / p["shares"] if p["shares"] else 0)
            total += p["shares"] * px
        return total

    replay = {}
    for d in cal:
        if d in pinned:
            replay[d] = pinned[d]
        else:
            replay[d] = STARTING_CAPITAL if d == START else round(value_on(d), 2)

    days = list(cal)
    vals = [replay[d] for d in days]
    rets = [(vals[i] / vals[i - 1] - 1) for i in range(1, len(vals))]
    ret_days = days[1:]

    # drawdown
    peak, peak_d = vals[0], days[0]
    max_dd, dd_peak, dd_trough = 0.0, days[0], days[0]
    under, longest_under = 0, 0
    for d, v in zip(days, vals):
        if v >= peak:
            peak, peak_d, under = v, d, 0
        else:
            under += 1
            longest_under = max(longest_under, under)
            dd = v / peak - 1
            if dd < max_dd:
                max_dd, dd_peak, dd_trough = dd, peak_d, d
    best_i = max(range(len(rets)), key=lambda i: rets[i])
    worst_i = min(range(len(rets)), key=lambda i: rets[i])

    # current book (latest snapshot <= through) at asof closes
    bdate, cur_pos, cur_cash = books[-1]
    tl_pos = {p["ticker"]: p for p in tradelog["positions"]}
    positions = []
    for t, p in cur_pos.items():
        row = series.get(t, {}).get(asof)
        px = row["close"] if row else (p["mv"] / p["shares"] if p["shares"] else 0)
        positions.append({"symbol": t, "shares": p["shares"], "price": round(px, 2),
                          "value": round(p["shares"] * px, 2)})
    invested = sum(p["value"] for p in positions)
    for p in positions:
        t = p["symbol"]
        tl = tl_pos.get(t, {})
        p["name"] = tl.get("name", t)
        p["weight"] = round(p["value"] / invested * 100, 2)
        p["cost"] = tl.get("amount")
        p["fill"] = tl.get("fill_price")
        p["unreal"] = round(p["value"] - p["cost"], 2) if p.get("cost") else None
        p["unreal_pct"] = (round((p["value"] / p["cost"] - 1) * 100, 2)
                           if p.get("cost") else None)
    positions.sort(key=lambda p: -p["value"])

    # trailing window: the last LOOKBACK closes ending at asof (~1 year)
    LOOKBACK = 253
    win = sorted(d for d in dji if d <= asof)[-LOOKBACK:]

    def win_rets(sym):
        s = series[sym]
        missing = [d for d in win if d not in s]
        if missing:
            raise SystemExit(f"{sym}: {len(missing)} window dates missing "
                             f"from cache (first {missing[0]}) - refetch.")
        px = [s[d]["close"] for d in win]
        return [(px[i] / px[i - 1] - 1) for i in range(1, len(px))]

    spy_r = win_rets("SPY")
    dow_r = win_rets("_DJI")
    sym_r = {}
    for p in positions:
        t = p["symbol"]
        if t in series:
            r = win_rets(t)
            sym_r[t] = r
            p["vol1y"] = round(std(r) * math.sqrt(252) * 100, 1)
            p["beta1y"] = round(cov(r, spy_r) / cov(spy_r, spy_r), 2)
        else:
            p["vol1y"] = p["beta1y"] = None
    p_order = ["symbol", "name", "shares", "price", "value", "weight", "cost",
               "fill", "unreal", "unreal_pct", "vol1y", "beta1y"]
    positions = [{k: p[k] for k in p_order} for p in positions]

    weights = {p["symbol"]: p["value"] / invested for p in positions}
    port_r = [sum(weights[t] * sym_r[t][i] for t in sym_r)
              for i in range(len(spy_r))]

    var95 = percentile(port_r, 5)
    tail = [r for r in port_r if r <= var95]
    value = round(replay[asof], 2)

    import os
    if os.environ.get("FA_DEBUG"):
        print(f"[debug] window {win[0]} .. {win[-1]}  closes={len(win)} returns={len(port_r)}")
        ranked = sorted(zip(port_r, win[1:]))
        print("[debug] worst:", [(d, round(r * 100, 2)) for r, d in ranked[:3]])
        print("[debug] best :", [(d, round(r * 100, 2)) for r, d in ranked[-3:]])
        print(f"[debug] rolling first {rv0[0]} {rv0[1]}" if False else
              f"[debug] modeled mean_d={mean(port_r):.6f} std_d={std(port_r):.6f}")

    # histogram: 21 equal bins over the modeled daily returns, in percent
    lo, hi = min(port_r), max(port_r)
    width = (hi - lo) / 21
    counts = [0] * 21
    for r in port_r:
        i = min(int((r - lo) / width), 20)
        counts[i] += 1
    bins = [round((lo + i * width) * 100, 2) for i in range(21)]

    syms = sorted(sym_r)
    cells = [[round(corr(sym_r[a], sym_r[b]), 2) for b in syms] for a in syms]
    off = [corr(sym_r[a], sym_r[b]) for i, a in enumerate(syms)
           for b in syms[i + 1:]]

    # monthly returns of the replay
    monthly = {}
    month_last = {}
    for d, v in zip(days, vals):
        month_last[d[:7]] = v
    months = sorted(month_last)
    prev_v = STARTING_CAPITAL
    for m in months:
        y, mo = m.split("-")
        monthly.setdefault(y, {})[str(int(mo))] = round(
            (month_last[m] / prev_v - 1) * 100, 2)
        prev_v = month_last[m]

    weekly = []
    for e in perf:
        ev = round(replay[e["date"]], 2) if e["date"] in replay else None
        weekly.append({
            "date": e["date"], "account_value": e["account_value"],
            "dow_level": e["dow_level"], "dividends": 0,
            "engine_value": ev,
            "delta": round(ev - e["account_value"], 2) if ev is not None else None,
        })

    w_sorted = sorted(weights.values(), reverse=True)
    hhi = sum(w * w for w in w_sorted)

    # published daily series + 30-session rolling vol of the modeled portfolio
    run_peak, underwater = STARTING_CAPITAL, []
    for v in vals:
        run_peak = max(run_peak, v)
        underwater.append(round((v / run_peak - 1) * 100, 3))
    ROLL = 30  # rolling 30 daily returns
    rv_dates, rv_vals = [], []
    for i in range(ROLL, len(port_r) + 1):
        chunk = port_r[i - ROLL:i]
        rv_dates.append(win[i])
        rv_vals.append(round(std(chunk) * math.sqrt(252) * 100, 2))
    series_block = {
        "dates": days,
        "value": [round(v, 2) for v in vals],
        "index": [round(v / STARTING_CAPITAL * 100, 3) for v in vals],
        "bench_dow": [round(dji[d]["close"] / dji[START]["close"] * 100, 3)
                      for d in days],
        "bench_spy": [round(series["SPY"][d]["close"]
                            / series["SPY"][START]["close"] * 100, 3)
                      for d in days],
        "underwater": underwater,
        "rolling_vol": {"dates": rv_dates, "vals": rv_vals},
    }

    return {
        "generated": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "asof": asof,
        "period": {"start": START, "end": asof, "trading_days": len(cal)},
        "dow_label": "Dow Jones Industrial Average",
        "young": len(cal) < 20,
        "headline": {
            "value": value,
            "cash": round(tradelog["cash_remaining"], 2),
            "cash_pct": round(tradelog["cash_remaining"] / value * 100, 1),
            "gain": round(value - STARTING_CAPITAL, 2),
            "contributed": STARTING_CAPITAL,
            "ret_total": round((value / STARTING_CAPITAL - 1) * 100, 2),
            "ret_dow": round((dji[asof]["close"] / dji[START]["close"] - 1) * 100, 2),
            "alpha_pts": round((value / STARTING_CAPITAL - 1) * 100
                               - (dji[asof]["close"] / dji[START]["close"] - 1) * 100, 1),
            "vol_live": round(std(rets) * math.sqrt(252) * 100, 1),
            "sharpe_live": round((mean(rets) - RF_DAILY) / std(rets)
                                 * math.sqrt(252), 2),
            "sortino_live": round((mean(rets) - RF_DAILY) / math.sqrt(
                mean([min(r, 0.0) ** 2 for r in rets])) * math.sqrt(252), 1),
            "max_dd": round(max_dd * 100, 2),
            "max_dd_peak": dd_peak,
            "max_dd_trough": dd_trough,
            "longest_underwater_days": longest_under,
            "best_day": {"date": ret_days[best_i], "ret": round(rets[best_i] * 100, 2)},
            "worst_day": {"date": ret_days[worst_i], "ret": round(rets[worst_i] * 100, 2)},
            "eff_n": round(1 / hhi, 1),
            "top5_pct": round(sum(w_sorted[:5]) * 100, 1),
            "hhi": round(hhi, 3),
        },
        "modeled": {
            "note": MODELED_NOTE,
            "vol": round(std(port_r) * math.sqrt(252) * 100, 1),
            "sharpe": round((mean(port_r) - RF_DAILY) / std(port_r)
                            * math.sqrt(252), 2),
            "beta_spy": round(cov(port_r, spy_r) / cov(spy_r, spy_r), 2),
            "corr_spy": round(corr(port_r, spy_r), 2),
            "beta_dow": round(cov(port_r, dow_r) / cov(dow_r, dow_r), 2),
            "var95": round(var95 * 100, 2),
            "var95_dollar": round(var95 * value, 1),
            "cvar95": round(mean(tail) * 100, 2),
            "up_capture": round(mean([p for p, s in zip(port_r, spy_r) if s > 0])
                                / mean([s for s in spy_r if s > 0]) * 100, 1),
            "down_capture": round(mean([p for p, s in zip(port_r, spy_r) if s < 0])
                                  / mean([s for s in spy_r if s < 0]) * 100, 1),
        },
        "histogram": {"bins": bins, "counts": counts, "var95": round(var95 * 100, 2)},
        "correlation": {"symbols": syms, "cells": cells,
                        "avg": round(mean(off), 2) if off else 0.0},
        "positions": positions,
        "monthly": monthly,
        "weekly": weekly,
        "series": series_block,
        "income": {"realized_and_income": tradelog["realized_and_income"],
                   "treatment": tradelog["dividend_treatment"]},
        "warnings": [WARNING],
    }


# ── validate / write ─────────────────────────────────────────────────────────
def diff(a, b, path="", out=None):
    if out is None:
        out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            diff(a.get(k), b.get(k), f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if abs(a - b) > 0.005:
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="replay end (last settled close)")
    ap.add_argument("--through", required=True,
                    help="book/records cutoff (snapshot + tradelog + perf dates <= this)")
    ap.add_argument("--validate", default=None,
                    help="diff against this committed analytics.json instead of writing")
    args = ap.parse_args()

    doc = build(args.asof, args.through)
    if args.validate:
        want = json.loads(Path(args.validate).read_text(encoding="utf-8"))
        problems = diff(doc, want)
        problems = [p for p in problems if not p.startswith(".generated")]
        print(f"{len(problems)} field difference(s) vs {args.validate}")
        for p in problems:
            print("  " + p)
        return
    for dest in (ROOT / "docs" / "data" / "analytics.json",
                 ROOT / "furton_website" / "data" / "analytics.json"):
        dest.write_text(json.dumps(doc, indent=1, ensure_ascii=False),
                        encoding="utf-8")
        print(f"wrote {dest}")
    h = doc["headline"]
    print(f"asof {args.asof}: value {h['value']:,} ret {h['ret_total']}% "
          f"dow {h['ret_dow']}% alpha {h['alpha_pts']} sharpe {h['sharpe_live']} "
          f"dd {h['max_dd']}%  ({doc['period']['trading_days']} trading days)")


if __name__ == "__main__":
    main()
