#!/usr/bin/env bash
# shellcheck shell=bash
# run_v2_dashboards.sh - visible Binance + Deriv V2 dashboards in tmux
#
# This is the operator view. It runs both live/dashboard programs in foreground
# tmux panes so their built-in TTY dashboards stay visible.
#
# Safety defaults:
#   - Binance is always testnet from this launcher.
#   - Deriv uses the configured demo/virtual token.
#   - --live is refused here; real capital must not be launched from dashboards.
#
# Usage:
#   ./run_v2_dashboards.sh
#   ./run_v2_dashboards.sh --dry-run
#   ./run_v2_dashboards.sh --session v2_forward
#   ./run_v2_dashboards.sh --no-attach

set -euo pipefail

PORTFOLIO_CONFIG="${PORTFOLIO_CONFIG:-v2_portfolio.wave5_quality13_pesfix.json}"
MODEL_SET="${MODEL_SET:-v2_wave5_extended}"
BUNDLE="${BUNDLE:-bundle_intraday_core}"
SESSION="v2_forward"
DRY_RUN_FLAG=""
ATTACH=1

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN_FLAG="--dry-run"
      ;;
    --no-attach)
      ATTACH=0
      ;;
    --session=*)
      SESSION="${arg#--session=}"
      ;;
    --session)
      echo "--session requires --session=name form" >&2
      exit 2
      ;;
    --help|-h)
      sed -n '1,24p' "$0"
      exit 0
      ;;
    --live)
      echo "Refusing --live: dashboard launcher is testnet/demo only." >&2
      exit 2
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for side-by-side dashboards." >&2
  echo "Install on macOS with: brew install tmux" >&2
  exit 127
fi

mkdir -p logs state tmp

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

BINANCE_CMD=(
  uv run v2_live_binance.py
  --portfolio-config "$PORTFOLIO_CONFIG"
  --model-set "$MODEL_SET"
  --state-path state/v2_binance_live.json
  --testnet
)
DERIV_CMD=(
  uv run v2_live_deriv.py
  --portfolio-config "$PORTFOLIO_CONFIG"
  --model-set "$MODEL_SET"
  --bundle "$BUNDLE"
  --state-path state/v2_deriv_live.json
)

if [ -n "$DRY_RUN_FLAG" ]; then
  BINANCE_CMD+=("$DRY_RUN_FLAG")
  DERIV_CMD+=("$DRY_RUN_FLAG")
fi

printf -v BINANCE_LINE "%q " "${BINANCE_CMD[@]}"
printf -v DERIV_LINE "%q " "${DERIV_CMD[@]}"

tmux new-session -d -s "$SESSION" -n dashboards
tmux send-keys -t "$SESSION:dashboards.0" "printf 'BINANCE TESTNET DASHBOARD\\n\\n'; $BINANCE_LINE" C-m
tmux split-window -h -t "$SESSION:dashboards.0"
tmux send-keys -t "$SESSION:dashboards.1" "printf 'DERIV DEMO DASHBOARD\\n\\n'; $DERIV_LINE" C-m
tmux select-layout -t "$SESSION:dashboards" even-horizontal >/dev/null
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

echo "Started tmux dashboard session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b then d"
echo "Stop: tmux kill-session -t $SESSION"

if [ "$ATTACH" -eq 1 ]; then
  tmux attach -t "$SESSION"
fi
