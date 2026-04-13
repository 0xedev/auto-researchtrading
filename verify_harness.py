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

    data_dir = Path.home() / ".cache" / "autotrader" / "data"
    if data_dir.exists():
        count_1h = len(list(data_dir.glob("*_1h.parquet")))
        count_15m = len(list(data_dir.glob("*_15m.parquet")))
        _add(results, "PASS", f"Data cache present: {count_1h} x 1h parquet files, {count_15m} x 15m parquet files")
        if count_1h == 0:
            _add(results, "WARN", "No 1h parquet files found in the data cache")
        if count_15m == 0:
            _add(results, "WARN", "No 15m parquet files found in the data cache")
    else:
        _add(results, "WARN", f"Data cache directory missing: {data_dir}")

    exact_refs = [
        Path("models/exp256_active/macro_hmm.joblib"),
        Path("models/exp256_active/macro_scaler.joblib"),
    ]
    fallback_refs = {
        Path("models/exp256_active/macro_hmm.joblib"): Path("models/macro_hmm.joblib"),
        Path("models/exp256_active/macro_scaler.joblib"): Path("models/macro_scaler.joblib"),
    }
    for ref in exact_refs:
        exact_path = ROOT / ref
        fallback = ROOT / fallback_refs[ref]
        if exact_path.exists():
            _add(results, "PASS", f"Exact strategy reference exists: {ref}")
        elif fallback.exists():
            _add(results, "WARN", f"Missing exact strategy reference `{ref}`, but fallback artifact exists at `{fallback_refs[ref]}`")
        else:
            _add(results, "FAIL", f"Missing strategy reference `{ref}` and no obvious fallback artifact exists")

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
