#!/usr/bin/env bash
# shellcheck shell=bash
# rotate_v2_logs.sh - archive current V2 operator logs before a clean forward run
#
# Usage:
#   ./rotate_v2_logs.sh
#   ./rotate_v2_logs.sh --stop
#
# --stop kills known background PID files and the v2_forward tmux session before
# rotation. Use it when starting a new clean forward-paper evidence window.

set -euo pipefail

STOP=0
for arg in "$@"; do
  case "$arg" in
    --stop) STOP=1 ;;
    --help|-h)
      sed -n '1,13p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p logs/archive

if [ "$STOP" -eq 1 ]; then
  for pidfile in logs/v2_binance.pid logs/v2_deriv.pid; do
    if [ -f "$pidfile" ]; then
      pid="$(cat "$pidfile" 2>/dev/null || true)"
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
      rm -f "$pidfile"
    fi
  done
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t v2_forward 2>/dev/null; then
    tmux kill-session -t v2_forward
  fi
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
dest="logs/archive/$stamp"
mkdir -p "$dest"

moved=0
for file in \
  logs/v2_binance.log \
  logs/v2_deriv.log \
  logs/v2_binance.console.log \
  logs/v2_deriv.console.log
do
  if [ -f "$file" ]; then
    mv "$file" "$dest/"
    moved=1
  fi
done

touch logs/v2_binance.log logs/v2_deriv.log

echo "Archived logs to: $dest"
if [ "$moved" -eq 0 ]; then
  echo "No existing logs were found; clean log files were created."
fi
