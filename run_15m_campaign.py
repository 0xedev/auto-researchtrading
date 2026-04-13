#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.tsv"


@dataclass
class RunRow:
    commit: str
    score: float
    sharpe: float
    max_dd: float
    status: str
    description: str


def _latest_run() -> RunRow | None:
    if not RESULTS_PATH.exists():
        return None
    lines = [line.rstrip("\n") for line in RESULTS_PATH.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    commit, score, sharpe, max_dd, status, description = lines[-1].split("\t", 5)
    return RunRow(
        commit=commit,
        score=float(score),
        sharpe=float(sharpe),
        max_dd=float(max_dd),
        status=status,
        description=description,
    )


def _exp_num(label: str) -> int:
    return int(label.replace("exp", ""))


def _run(command: list[str], env: dict[str, str] | None = None) -> int:
    pretty = " ".join(command)
    print(f"\n>>> {pretty}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env)
    return completed.returncode


def _campaign_variants() -> list[tuple[str, dict[str, str]]]:
    gate_profiles = {
        "control": {
            "STRAT_15M_TREND_BULL": "0.45",
            "STRAT_15M_TREND_META": "0.25",
            "STRAT_15M_PUSH_BULL": "0.43",
            "STRAT_15M_PUSH_META": "0.32",
        },
        "push_relax": {
            "STRAT_15M_TREND_BULL": "0.45",
            "STRAT_15M_TREND_META": "0.25",
            "STRAT_15M_PUSH_BULL": "0.42",
            "STRAT_15M_PUSH_META": "0.31",
        },
        "hybrid": {
            "STRAT_15M_TREND_BULL": "0.46",
            "STRAT_15M_TREND_META": "0.27",
            "STRAT_15M_PUSH_BULL": "0.42",
            "STRAT_15M_PUSH_META": "0.31",
        },
    }
    weight_profiles = {
        "control": {
            "STRAT_15M_TREND_WEIGHT": "0.65",
            "STRAT_15M_PUSH_WEIGHT": "0.22",
        },
        "push_bias": {
            "STRAT_15M_TREND_WEIGHT": "0.55",
            "STRAT_15M_PUSH_WEIGHT": "0.26",
        },
    }
    allocators = ["push_priority", "ranked", "baseline"]
    max_positions = [4, 3]

    variants: list[tuple[str, dict[str, str]]] = []
    for allocator, max_pos, gate_name, weight_name in itertools.product(
        allocators,
        max_positions,
        gate_profiles,
        weight_profiles,
    ):
        if allocator == "push_priority" and max_pos == 4 and gate_name == "control" and weight_name == "control":
            continue
        env = {
            "STRAT_15M_ALLOCATOR": allocator,
            "STRAT_15M_MAX_POSITIONS": str(max_pos),
            **gate_profiles[gate_name],
            **weight_profiles[weight_name],
        }
        desc = f"apr08-vast4090 15m campaign {allocator} p{max_pos} {gate_name} {weight_name}"
        variants.append((desc, env))

    variants.append(
        (
            "apr08-vast4090 15m campaign trend_priority p4 hybrid push_bias",
            {
                "STRAT_15M_ALLOCATOR": "trend_priority",
                "STRAT_15M_MAX_POSITIONS": "4",
                **gate_profiles["hybrid"],
                **weight_profiles["push_bias"],
            },
        )
    )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a structured 15m campaign until a target experiment number.")
    parser.add_argument("--target-exp", type=int, default=380)
    parser.add_argument("--start-with-verify", action="store_true")
    parser.add_argument("--analyze-every", type=int, default=5)
    args = parser.parse_args()

    python_bin = sys.executable
    latest = _latest_run()
    if latest is None:
        raise RuntimeError("results.tsv is empty; cannot infer starting experiment number")

    if args.start_with_verify:
        code = _run([python_bin, "verify_harness.py"])
        if code != 0:
            return code

    current_exp = _exp_num(latest.commit)
    if current_exp >= args.target_exp:
        print(f"Target already reached at {latest.commit}", flush=True)
        return 0

    variants = _campaign_variants()
    needed = args.target_exp - current_exp
    chosen = variants[:needed]
    if len(chosen) < needed:
        raise RuntimeError(f"Need {needed} variants to reach exp{args.target_exp}, only have {len(chosen)}")

    for idx, (description, env_overrides) in enumerate(chosen, start=1):
        env = os.environ.copy()
        env.update(env_overrides)

        before = _latest_run()
        before_num = _exp_num(before.commit) if before else -1
        print(
            f"\n=== Campaign Run {idx}/{len(chosen)} | next after exp{before_num} | {description} ===",
            flush=True,
        )
        exit_code = _run([python_bin, "backtest.py", "--timeframe", "15m", "--description", description], env=env)
        if exit_code != 0:
            print(f"Backtest exited with code {exit_code} for {description}", flush=True)
            continue

        after = _latest_run()
        if after is None:
            raise RuntimeError("results.tsv became unreadable after backtest")

        after_num = _exp_num(after.commit)
        print(
            f"Recorded {after.commit} | score {after.score:.3f} | sharpe {after.sharpe:.3f} | dd {after.max_dd:.2f} | {after.description}",
            flush=True,
        )
        if after_num <= before_num:
            print("No new experiment row was recorded; stopping campaign to avoid duplicate work.", flush=True)
            break

        if args.analyze_every > 0 and (idx % args.analyze_every == 0 or after_num >= args.target_exp):
            _run([python_bin, "analyze_results.py", "--update-memory"])

        if after_num >= args.target_exp:
            print(f"Reached target at {after.commit}", flush=True)
            break

    final = _latest_run()
    if final is not None:
        print(
            f"\nCampaign complete at {final.commit} | score {final.score:.3f} | {final.description}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
