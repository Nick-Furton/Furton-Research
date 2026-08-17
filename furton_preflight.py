"""Monday step-0: verify both data feeds are healthy BEFORE spending screen dollars.

Two checks, both of which have actually failed in production:
  1. Tiingo health via THREE spot symbols through the SAME code path the
     screener uses (furton_server.fetch_settled_close), asserting each returns
     a close dated within the freshness gate. Deliberately NOT all 30 names:
     Tiingo's free tier rate-limits around 50 symbol-requests per hour
     (observed live, HTTP 429, 2026-08-10), so a 30-name preflight plus the
     30-name screen in the same hour would strand the screen mid-run — the
     exact failure mode this script exists to prevent. Per-name coverage is
     already guaranteed by the screener's own hard gate, which fails any name
     loudly before archiving.
  2. Anthropic credit balance, via a 1-token Haiku ping. The 2026-08-10 screen
     died at name 20/30 on an empty credit balance, wasting a restart and
     archiving one phantom-quorum record before the gate caught it.

Usage:  py furton_preflight.py        (exit 0 = safe to screen, 1 = do not)
Cost:   ~$0.00002 (the Haiku ping); Tiingo calls are free-tier.
"""
import json
import sys
import time

import furton_server as fs

# First, middle, last of the roster alphabetically — three independent symbols
# so a single delisted/renamed ticker cannot fail the whole preflight.
SPOT_SYMBOLS = ["AAPL", "KO", "WMT"]


def check_tiingo():
    fs.TIINGO_KEY = fs.load_tiingo_key()
    print(f"Tiingo settled closes — {len(SPOT_SYMBOLS)} spot symbols, "
          f"{fs.MAX_CLOSE_AGE_DAYS}-day freshness gate "
          f"(3 of the hourly request budget; the screen needs 30)")
    failures = []
    t0 = time.time()
    for t in SPOT_SYMBOLS:
        try:
            close, asof = fs.fetch_settled_close(t)
            print(f"  {t:6} {close:10.2f}  {asof}")
        except RuntimeError as e:
            failures.append(t)
            print(f"  {t:6} FAILED — {e}")
        time.sleep(0.15)
    print(f"  ({int(time.time() - t0)}s)")
    return failures


def check_anthropic():
    fs.API_KEY = fs.load_api_key()
    resp, status = fs.call_anthropic({
        "model": fs.HAIKU_MODEL,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }, timeout=30, max_retries=1)
    if status == 200:
        print("Anthropic API: OK (1-token ping accepted — credits present)")
        return True
    try:
        msg = json.loads(resp).get("error", {}).get("message", "")[:160]
    except Exception:
        msg = resp.decode("utf-8", "replace")[:160]
    print(f"Anthropic API: FAILED — HTTP {status}: {msg}")
    return False


def main():
    tiingo_failures = check_tiingo()
    print()
    anthropic_ok = check_anthropic()

    print()
    if tiingo_failures or not anthropic_ok:
        print("PREFLIGHT FAILED — do NOT start the screen:")
        if tiingo_failures:
            print(f"  Tiingo failures: {', '.join(tiingo_failures)}")
        if not anthropic_ok:
            print("  Anthropic API not accepting calls (credits? key?)")
        return 1
    print("PREFLIGHT PASSED — both feeds healthy, safe to start the screen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
