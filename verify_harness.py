#!/usr/bin/env python3
"""
Lightweight harness verifier for the current research workspace.

Checks core files, results schema, dependency drift, model reference drift,
and cached data availability without importing the full training/backtest stack.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_COLUMNS = ["commit", "score", "sharpe", "max_dd", "status", "description"]
CORE_FILES = [
    "strategy.py",
    "prepare.py",
    "backtest.py",
    "evaluate.py",
    "train_model.py",
    "program.md",
    "README.md",
    "POSITIONING.md",
    "CONTROL_BASELINE.md",
]
PREPARE_CONSTANTS = [
    "TIME_BUDGET",
    "INITIAL_CAPITAL",
    "MAKER_FEE",
    "TAKER_FEE",
    "SLIPPAGE_BPS",
    "TRAIN_START",
    "TRAIN_END",
    "VAL_START",
    "VAL_END",
    "TEST_END",
    "DATA_END",
    "ROBUST_START",
    "ROBUST_END",
]
CONTROL_MODEL_DIR = Path("models/exp256_active")
CONTROL_MODEL_FILES = [
    "lead_15m.xgb",
    "meta_15m.xgb",
    "lead_long_15m.xgb",
    "lead_short_15m.xgb",
    "meta_long_15m.xgb",
    "meta_short_15m.xgb",
    "lead_1h.xgb",
    "meta_1h.xgb",
    "lead_4h.xgb",
    "meta_4h.xgb",
]


@dataclass
class CheckResult:
    level: str
    message: str


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _extract_constant(text: str, name: str) -> str | None:
    match = re.search(rf"^{name}\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _add(results: list[CheckResult], level: str, message: str) -> None:
    results.append(CheckResult(level=level, message=message))


def _strip_quotes(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


def run_checks(strict: bool = False) -> tuple[list[CheckResult], int]:
    results: list[CheckResult] = []

    for rel_path in CORE_FILES:
        path = ROOT / rel_path
        if path.exists():
            _add(results, "PASS", f"Core file present: {rel_path}")
        else:
            _add(results, "FAIL", f"Missing core file: {rel_path}")

    results_path = ROOT / "results.tsv"
    if results_path.exists():
        header = results_path.read_text().splitlines()[0].split("\t")
        header = [item.strip() for item in header]
        if header == RESULT_COLUMNS:
            _add(results, "PASS", "results.tsv schema matches expected columns")
        else:
            _add(results, "FAIL", f"results.tsv header drift detected: {header}")
    else:
        _add(results, "WARN", "results.tsv is missing")

    prepare_text = _read_text(ROOT / "prepare.py")
    for name in PREPARE_CONSTANTS:
        value = _extract_constant(prepare_text, name)
        if value is None:
            _add(results, "FAIL", f"Could not extract `{name}` from prepare.py")
        else:
            _add(results, "PASS", f"{name} = {value}")

    data_end = _strip_quotes(_extract_constant(prepare_text, "DATA_END"))
    if data_end is None:
        _add(results, "WARN", "prepare.py does not define `DATA_END`, so downloads may not cover the OOS window")
    elif data_end < "2025-12-31":
        _add(results, "FAIL", f"`DATA_END` is {data_end}, which does not cover the documented 2025 OOS window")
    else:
        _add(results, "PASS", f"`DATA_END` covers the documented OOS window ({data_end})")

    pyproject_text = _read_text(ROOT / "pyproject.toml")
    source_imports_hmm = "from hmmlearn import hmm" in prepare_text or "from hmmlearn import hmm" in _read_text(ROOT / "strategy.py")
    if source_imports_hmm and "hmmlearn" not in pyproject_text:
        _add(results, "WARN", "Source imports `hmmlearn`, but pyproject.toml does not declare it")
    else:
        _add(results, "PASS", "pyproject dependency declarations cover visible model/runtime imports")

    strategy_text = _read_text(ROOT / "strategy.py")
    if 'AUTOTRADER_MODEL_SET", "exp256_active"' in strategy_text:
        _add(results, "PASS", "strategy.py default model pin is exp256_active")
    else:
        _add(results, "WARN", "strategy.py does not appear to pin the default model set to exp256_active")

    data_dir = Path.home() / ".cache" / "autotrader" / "data"
    if data_dir.exists():
        count_1h = len(list(data_dir.glob("*_1h.parquet")))
        count_15m = len(list(data_dir.glob("*_15m.parquet")))
        _add(results, "PASS", f"Data cache present: {count_1h} x 1h parquet files, {count_15m} x 15m parquet files")
        if count_1h == 0:
            _add(results, "WARN", "No 1h parquet files found in the data cache")
        if count_15m == 0:
            _add(results, "WARN", "No 15m parquet files found in the data cache")
        if (data_dir / "SP500_1h.parquet").exists():
            _add(results, "PASS", "SP500 cache present for beta/excess-return reporting")
        else:
            _add(results, "WARN", "SP500_1h.parquet is missing, so beta-to-SPX will rely on the daily FRED fallback")
    else:
        _add(results, "WARN", f"Data cache directory missing: {data_dir}")

    control_dir = ROOT / CONTROL_MODEL_DIR
    if control_dir.exists():
        _add(results, "PASS", f"Control model directory present: {CONTROL_MODEL_DIR}")
        for filename in CONTROL_MODEL_FILES:
            if (control_dir / filename).exists():
                _add(results, "PASS", f"Control artifact present: {CONTROL_MODEL_DIR / filename}")
            else:
                _add(results, "FAIL", f"Missing control artifact: {CONTROL_MODEL_DIR / filename}")
    else:
        _add(results, "FAIL", f"Missing control model directory: {CONTROL_MODEL_DIR}")

    for tf in ("15m", "1h", "4h"):
        lead_candidates = [ROOT / "models" / f"lead_{tf}.xgb", ROOT / "models" / f"lead_{tf}.json"]
        meta_candidates = [ROOT / "models" / f"meta_{tf}.xgb", ROOT / "models" / f"meta_{tf}.json"]
        if any(path.exists() for path in lead_candidates):
            _add(results, "PASS", f"Global lead model available for {tf}")
        else:
            _add(results, "FAIL", f"No global lead model found for {tf}")
        if any(path.exists() for path in meta_candidates):
            _add(results, "PASS", f"Global meta model available for {tf}")
        else:
            _add(results, "FAIL", f"No global meta model found for {tf}")

    active_dir = ROOT / "models" / "exp256_active"
    if active_dir.exists():
        specialist_count = len(list(active_dir.glob("*.xgb"))) + len(list(active_dir.glob("*.json")))
        _add(results, "PASS", f"Specialist model directory present with {specialist_count} artifacts")
    else:
        _add(results, "WARN", "Specialist model directory models/exp256_active is missing")

    fail_count = sum(1 for result in results if result.level == "FAIL")
    warn_count = sum(1 for result in results if result.level == "WARN")
    if strict and warn_count > 0:
        exit_code = 1
    elif fail_count > 0:
        exit_code = 1
    else:
        exit_code = 0
    return results, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the current research harness.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args()

    results, exit_code = run_checks(strict=args.strict)
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for result in results:
        counts[result.level] += 1
        symbol = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[result.level]
        print(f"{symbol} {result.message}")

    print("")
    print(
        "Summary: "
        f"{counts['PASS']} pass, "
        f"{counts['WARN']} warn, "
        f"{counts['FAIL']} fail"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
