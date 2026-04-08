#!/usr/bin/env python3
"""
Single-cycle research loop controller.

Runs verifier -> backtest -> memory refresh -> optional OOS evaluation,
so the agent loop is driven by explicit infra rather than prompt discipline alone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analyze_results import effective_status, load_runs


ROOT = Path(__file__).resolve().parent


def _run(command: list[str], dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    pretty = " ".join(command)
    print(f"\n>>> {pretty}")
    if dry_run:
        return None
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _latest_run(results_path: Path):
    records = load_runs(results_path)
    return records[-1] if records else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one guided research cycle.")
    parser.add_argument("--timeframe", default="1h", choices=["15m", "1h", "4h"])
    parser.add_argument("--description", default="Guided research cycle")
    parser.add_argument("--label", default="", help="Explicit label for evaluate.py")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--strict-verify", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--evaluate-all", action="store_true", help="Run evaluate.py even when the backtest status is not KEEP")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    python_bin = sys.executable
    results_path = ROOT / "results.tsv"

    if not args.skip_verify:
        verify_cmd = [python_bin, "verify_harness.py"]
        if args.strict_verify:
            verify_cmd.append("--strict")
        _run(verify_cmd, args.dry_run)

    backtest_cmd = [python_bin, "backtest.py", "--timeframe", args.timeframe, "--description", args.description]
    _run(backtest_cmd, args.dry_run)

    if args.dry_run:
        if not args.skip_analyze:
            _run([python_bin, "analyze_results.py", "--update-memory"], True)
        if not args.skip_evaluate:
            label = args.label or "<latest-exp>"
            _run([python_bin, "evaluate.py", "--timeframe", args.timeframe, "--label", label], True)
        return 0

    latest = _latest_run(results_path)
    if latest is None:
        raise RuntimeError("results.tsv did not contain a latest experiment row after backtest")

    if not args.skip_analyze:
        _run([python_bin, "analyze_results.py", "--update-memory"], False)

    latest_effective_status = effective_status(latest)
    should_evaluate = not args.skip_evaluate and (
        args.evaluate_all or latest_effective_status in {"KEEP", "CANDIDATE"}
    )
    if should_evaluate:
        label = args.label or latest.commit
        _run([python_bin, "evaluate.py", "--timeframe", args.timeframe, "--label", label], False)
    elif not args.skip_evaluate:
        print(
            f"\nSkipping evaluate.py because latest effective status is {latest_effective_status}. "
            "Use --evaluate-all to force it."
        )

    print("\nCycle Summary")
    print(f"latest run:   {latest.commit}")
    print(f"status:       {latest.status}")
    print(f"effective:    {latest_effective_status}")
    print(f"score:        {latest.score:.3f}")
    print(f"description:  {latest.description}")
    print("memory file:  RESEARCH_MEMORY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
