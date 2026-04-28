#!/usr/bin/env bash
# shellcheck shell=bash
# start_v2_traders.sh — launch both V2 live traders in the background
#
# Usage:
#   ./start_v2_traders.sh              # Binance testnet (BTC/ETH/SOL) + Deriv demo
#   ALLOW_REAL_MONEY=YES_I_UNDERSTAND ./start_v2_traders.sh --live
#   ./start_v2_traders.sh --deriv-only # Deriv only (no Binance)
#   ./start_v2_traders.sh --binance-only
#
# Stop: kill $(cat logs/v2_binance.pid) $(cat logs/v2_deriv.pid)

set -euo pipefail

PORTFOLIO_CONFIG="${PORTFOLIO_CONFIG:-v2_portfolio.wave5_quality13_pesfix.json}"
MODEL_SET="${MODEL_SET:-v2_wave5_extended}"
BUNDLE="${BUNDLE:-bundle_intraday_core}"

mkdir -p logs state

LIVE_FLAG="--testnet"
RUN_BINANCE=1
RUN_DERIV=1

for arg in "$@"; do
  case "$arg" in
    --testnet)       LIVE_FLAG="--testnet" ;;
    --live)          LIVE_FLAG="--live" ;;
    --binance-only)  RUN_DERIV=0 ;;
    --deriv-only)    RUN_BINANCE=0 ;;
  esac
done

echo "Starting V2 traders  config=$PORTFOLIO_CONFIG  model_set=$MODEL_SET"
echo "  Binance: ${LIVE_FLAG}"
echo ""

if [ "$RUN_BINANCE" -eq 1 ]; then
  nohup uv run v2_live_binance.py \
    --portfolio-config "$PORTFOLIO_CONFIG" \
    --model-set "$MODEL_SET" \
    --state-path state/v2_binance_live.json \
    $LIVE_FLAG \
    >> logs/v2_binance.log 2>&1 &
  echo $! > logs/v2_binance.pid
  echo "  Binance trader started  PID=$(cat logs/v2_binance.pid)"
fi

if [ "$RUN_DERIV" -eq 1 ]; then
  nohup uv run v2_live_deriv.py \
    --portfolio-config "$PORTFOLIO_CONFIG" \
    --model-set "$MODEL_SET" \
    --bundle "$BUNDLE" \
    --state-path state/v2_deriv_live.json \
    >> logs/v2_deriv.log 2>&1 &
  echo $! > logs/v2_deriv.pid
  echo "  Deriv trader started   PID=$(cat logs/v2_deriv.pid)"
fi

echo ""
echo "Logs: tail -f logs/v2_binance.log logs/v2_deriv.log"
echo "Stop: kill \$(cat logs/v2_binance.pid) \$(cat logs/v2_deriv.pid)"
