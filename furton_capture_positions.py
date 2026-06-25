#!/usr/bin/env python3
"""
Furton Research — Capture Actual Positions from a Schwab export (§7.3)
=====================================================================
The public furton_website/data/holdings.json records the committee's TARGET
weights. It does NOT record what actually filled: share counts and market
values. Those live at Schwab (the source of truth) and are needed to reconstruct
turnover and per-name attribution, exactly as the methodology paper's §7.3
commits ("for each position: ticker, share count, market value, portfolio
weight").

This tool ingests a Schwab "Positions" CSV export and writes a dated, private
snapshot:

    holdings/positions_YYYY-MM-DD.json      (project root — NEVER deploys)

containing, per position: ticker, shares, market_value, weight_pct (each
position's market value as a share of the total invested equity). It also
records cash and the account total when the export includes them, and reconciles
the tickers against the committee target file so you can spot any mismatch.

USAGE (PowerShell):
    $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    python furton_capture_positions.py "path\\to\\schwab_positions.csv"
    # optional: --date 2026-06-29   (else taken from the CSV header, else today)
    #           --force             (overwrite an existing snapshot for that date)

The parser is tolerant of Schwab's column layout: it locates the header row by
name and reads Symbol / Quantity / Market Value wherever they sit, skipping the
cash and "Account Total" footer rows.
"""

import sys, csv, json, re, argparse, datetime
from pathlib import Path

ROOT         = Path(__file__).resolve().parent
SNAPSHOT_DIR = ROOT / "holdings"
TARGET_FILE  = ROOT / "furton_website" / "data" / "holdings.json"

# Footer/non-position rows to skip (matched case-insensitively against Symbol).
SKIP_SYMBOLS = ("cash & cash investments", "cash and cash investments",
                "cash & money market", "account total", "total", "cash")


def _num(s):
    """Parse a Schwab money/quantity cell to float. Handles $, commas, %, (neg), '--'."""
    if s is None:
        return None
    t = str(s).strip().strip('"').replace(",", "").replace("$", "").replace("%", "").strip()
    if t in ("", "--", "N/A", "n/a"):
        return None
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def _find_header(rows):
    """Return (index, colmap) for the row that names Symbol + Market Value columns."""
    for i, row in enumerate(rows):
        low = [c.strip().strip('"').lower() for c in row]
        if "symbol" in low and any("market value" in c for c in low):
            colmap = {}
            for j, c in enumerate(low):
                if c == "symbol" and "symbol" not in colmap:
                    colmap["symbol"] = j
                elif (c in ("quantity", "qty") or "quantity" in c) and "qty" not in colmap:
                    colmap["qty"] = j
                elif "market value" in c and "mv" not in colmap:
                    colmap["mv"] = j
                elif c == "description" and "desc" not in colmap:
                    colmap["desc"] = j
                elif "% of account" in c and "pct" not in colmap:
                    colmap["pct"] = j
            if {"symbol", "qty", "mv"} <= colmap.keys():
                return i, colmap
    return None, None


def _date_from_header(rows):
    """Schwab exports often start with '...as of ... 2026/06/29' or similar."""
    blob = " ".join(c for row in rows[:5] for c in row)
    m = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", blob)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", blob)   # mm/dd/yyyy
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    return None


def parse_schwab(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise SystemExit(f"✗ {csv_path} is empty.")

    hdr_i, cols = _find_header(rows)
    if hdr_i is None:
        raise SystemExit("✗ Could not find a Schwab header row with Symbol + Market Value "
                         "columns. Is this the Positions export?")

    positions, cash_value, account_value = [], None, None
    for row in rows[hdr_i + 1:]:
        if not any(cell.strip() for cell in row):
            continue
        if cols["symbol"] >= len(row) or cols["mv"] >= len(row):
            continue
        sym = row[cols["symbol"]].strip().strip('"')
        mv  = _num(row[cols["mv"]])
        low = sym.lower()

        if low in ("cash & cash investments", "cash and cash investments",
                   "cash & money market", "cash"):
            if mv is not None:
                cash_value = (cash_value or 0) + mv
            continue
        if low in ("account total", "total") or low.startswith("account"):
            if mv is not None:
                account_value = mv
            continue
        if not sym or mv is None:
            continue
        qty = _num(row[cols["qty"]]) if cols["qty"] < len(row) else None
        positions.append({"ticker": sym.upper(), "shares": qty, "market_value": round(mv, 2)})

    if not positions:
        raise SystemExit("✗ Found the header but no position rows parsed. Check the CSV.")

    pos_mv = round(sum(p["market_value"] for p in positions), 2)
    for p in positions:
        p["weight_pct"] = round(p["market_value"] / pos_mv * 100, 2) if pos_mv else 0.0
    positions.sort(key=lambda p: p["weight_pct"], reverse=True)
    return positions, pos_mv, cash_value, account_value, rows


def reconcile(positions):
    """Compare captured tickers against the committee target file, if present."""
    if not TARGET_FILE.exists():
        return
    try:
        tgt = json.loads(TARGET_FILE.read_text(encoding="utf-8"))
        tgt_tk = {h["ticker"].upper() for h in tgt.get("holdings", [])}
    except Exception:
        return
    have = {p["ticker"] for p in positions}
    missing = sorted(tgt_tk - have)   # in target, not in account
    extra   = sorted(have - tgt_tk)   # in account, not in target
    print(f"\n  Reconciliation vs committee target ({TARGET_FILE.relative_to(ROOT)}):")
    print(f"    {len(have & tgt_tk)}/{len(tgt_tk)} target names present in the account")
    if missing:
        print(f"    ⚠ target names NOT in account: {', '.join(missing)}")
    if extra:
        print(f"    ⚠ account names NOT in target: {', '.join(extra)}")
    if not missing and not extra:
        print("    ✓ exact match")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # py3.14 console is cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to the Schwab Positions CSV export")
    ap.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default: from CSV header, else today)")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing snapshot for that date")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"✗ CSV not found: {csv_path}")

    positions, pos_mv, cash_value, account_value, rows = parse_schwab(csv_path)
    date_str = args.date or _date_from_header(rows) or datetime.date.today().isoformat()

    record = {
        "date": date_str,
        "source": "schwab_positions_export",
        "positions_market_value": pos_mv,
        "cash_value": round(cash_value, 2) if cash_value is not None else None,
        "account_value": round(account_value, 2) if account_value is not None
                         else (round(pos_mv + (cash_value or 0), 2)),
        "positions": positions,
    }

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    dest = SNAPSHOT_DIR / f"positions_{date_str}.json"
    if dest.exists() and not args.force:
        raise SystemExit(f"  • Snapshot already exists: holdings/{dest.name} "
                         f"(use --force to overwrite). Aborted.")
    dest.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"Captured {len(positions)} position(s) as of {date_str}")
    print(f"  positions market value: ${pos_mv:,.2f}"
          + (f"   cash: ${cash_value:,.2f}" if cash_value is not None else "")
          + (f"   account total: ${record['account_value']:,.2f}" if record['account_value'] is not None else ""))
    for p in positions:
        sh = f"{p['shares']:g}" if p['shares'] is not None else "?"
        print(f"    {p['ticker']:<6} {sh:>10} sh   ${p['market_value']:>10,.2f}   {p['weight_pct']:>6.2f}%")
    reconcile(positions)
    print(f"\n  ✓ Wrote holdings/{dest.name}  (private — never deploys)")


if __name__ == "__main__":
    main()
