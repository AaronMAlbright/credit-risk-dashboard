#!/usr/bin/env python
"""
Standalone CLI script for running the Macro Credit Risk alert engine.

Usage:
  python scripts/send_alerts.py [options]

Options:
  --dry-run          Build and print the email HTML without sending
  --min-level LEVEL  Minimum severity to report: INFO | WARNING | ALERT  (default: INFO)
  --state-path PATH  Override default state file path
  --show-html        Print the email HTML body to stdout

Environment variables (required for sending):
  ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASSWORD,
  ALERT_FROM_ADDR, ALERT_TO_ADDRS (comma-separated)

Schedule via cron (daily at 8 AM):
  0 8 * * * cd /path/to/credit-risk-dashboard && \
    /path/to/venv/bin/python scripts/send_alerts.py >> logs/alerts.log 2>&1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alert_engine import AlertConfig, config_from_env, run_alerts
from src.position_sizing import run_position_sizing

_DATA_PATH = Path("data/scored_macro_credit_data.csv")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Macro Credit Risk alert engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Format email but do not send")
    p.add_argument("--min-level", default="INFO",
                   choices=["INFO", "WARNING", "ALERT"],
                   help="Minimum alert severity to fire (default: INFO)")
    p.add_argument("--state-path", default=None,
                   help="Override alert state file path")
    p.add_argument("--show-html", action="store_true",
                   help="Print email HTML body to stdout")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print("Loading scored data...")
    try:
        df = pd.read_csv(_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Scored data not found at {_DATA_PATH}. "
              "Run the scoring pipeline first.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Failed to load scored data: {exc}", file=sys.stderr)
        return 1

    if df is None or df.empty:
        print("ERROR: No data returned from composite engine.", file=sys.stderr)
        return 1

    print(f"  {len(df)} rows loaded, latest: {str(df.iloc[-1].get('date',''))[:10]}")

    print("Loading position sizing...")
    try:
        sizing = run_position_sizing(df)
    except Exception as exc:
        print(f"  WARNING: Position sizing failed ({exc}); proceeding without blend.")
        sizing = None

    config: AlertConfig = config_from_env()
    config.dry_run  = args.dry_run or config.dry_run
    config.min_level = args.min_level

    state_path_kw = {}
    if args.state_path:
        state_path_kw["state_path"] = args.state_path

    print("Running alert checks...")
    result = run_alerts(df, config=config, sizing=sizing, **state_path_kw)

    alerts   = result["alerts"]
    current  = result["current"]
    previous = result["previous"]

    print(f"\nCurrent state:")
    print(f"  Decision  : {current.get('decision','—')}")
    print(f"  Composite : {current.get('composite','—'):.1f}")
    blend = current.get("blend")
    print(f"  Blend     : {f'{blend:.0%}' if blend is not None else '—'}")
    print(f"  Shock     : {current.get('shock_flag','—')}")
    print(f"  Data date : {current.get('last_date','—')}")

    if alerts:
        print(f"\n{len(alerts)} alert(s) fired:")
        for a in alerts:
            print(f"  [{a['level']:7s}] {a['trigger']:20s} {a['message']}")
    else:
        print("\nNo alerts fired.")

    if args.show_html and result["email_html"]:
        print("\n--- EMAIL HTML ---")
        print(result["email_html"])
        print("--- END ---")

    if result["email_html"]:
        if args.dry_run:
            print("\nDry-run: email not sent.")
        elif result["email_sent"]:
            print(f"\nEmail sent to: {', '.join(config.to_addrs)}")
        else:
            print("\nERROR: Email send failed.", file=sys.stderr)
            return 1

    if result["state_saved"]:
        print("State saved.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
