#!/usr/bin/env bash
# shellcheck shell=bash
# run_v2_forward_paper.sh - safe V2 forward-paper launcher + readiness report
#
# Defaults:
#   - Binance uses Futures testnet, never live.
#   - Deriv uses the configured demo/virtual token.
#   - A readiness report is generated after startup.
#
# Usage:
#   ./run_v2_forward_paper.sh
#   ./run_v2_forward_paper.sh --dry-run
#   ./run_v2_forward_paper.sh --report-only
#   ./run_v2_forward_paper.sh --binance-only
#   ./run_v2_forward_paper.sh --deriv-only
#
# Env overrides:
#   PORTFOLIO_CONFIG=v2_portfolio.wave5_quality13_pesfix.json
#   MODEL_SET=v2_wave5_extended
#   BUNDLE=bundle_intraday_core
#   REPORT_DELAY_SECS=5
#   FORWARD_MIN_DAYS=30
#   FORWARD_MAX_STALE_HOURS=26

set -euo pipefail

PORTFOLIO_CONFIG="${PORTFOLIO_CONFIG:-v2_portfolio.wave5_quality13_pesfix.json}"
MODEL_SET="${MODEL_SET:-v2_wave5_extended}"
BUNDLE="${BUNDLE:-bundle_intraday_core}"
REPORT_DELAY_SECS="${REPORT_DELAY_SECS:-5}"
FORWARD_MIN_DAYS="${FORWARD_MIN_DAYS:-30}"
FORWARD_MAX_STALE_HOURS="${FORWARD_MAX_STALE_HOURS:-26}"

RUN_START=1
RUN_REPORT=1
DRY_RUN_FLAG=""
SCOPE_FLAG=""

for arg in "$@"; do
  case "$arg" in
    --report-only)
      RUN_START=0
      ;;
    --start-only)
      RUN_REPORT=0
      ;;
    --dry-run)
      DRY_RUN_FLAG="--dry-run"
      ;;
    --binance-only|--deriv-only)
      SCOPE_FLAG="$arg"
      ;;
    --help|-h)
      sed -n '1,34p' "$0"
      exit 0
      ;;
    --live)
      echo "Refusing --live: forward-paper launcher is testnet/demo only." >&2
      exit 2
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p logs state tmp

echo "V2 forward-paper workflow"
echo "  portfolio_config=$PORTFOLIO_CONFIG"
echo "  model_set=$MODEL_SET"
echo "  bundle=$BUNDLE"
echo "  binance_mode=testnet"
echo "  deriv_mode=demo/virtual-token"
if [ -n "$DRY_RUN_FLAG" ]; then
  echo "  dry_run=true"
fi
echo ""

if [ "$RUN_START" -eq 1 ]; then
  ./start_v2_traders.sh --testnet $DRY_RUN_FLAG $SCOPE_FLAG
fi

if [ "$RUN_REPORT" -eq 1 ]; then
  if [ "$RUN_START" -eq 1 ] && [ "$REPORT_DELAY_SECS" != "0" ]; then
    sleep "$REPORT_DELAY_SECS"
  fi
  uv run python v2_forward_report.py \
    --min-days "$FORWARD_MIN_DAYS" \
    --max-stale-hours "$FORWARD_MAX_STALE_HOURS" \
    --output-json tmp/v2_forward_report.json \
    --output-md tmp/v2_forward_report.md
  echo ""
  echo "Report: tmp/v2_forward_report.md"
fi
