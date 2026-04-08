#!/usr/bin/env python3
"""
Results analyzer for autoresearch-style experiment logs.

Parses results.tsv, summarizes keep/revert behavior, flags plateau risk,
and can refresh the generated block inside RESEARCH_MEMORY.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


RESULT_FIELDS = ("commit", "score", "sharpe", "max_dd", "status", "description")
MEMORY_BEGIN = "<!-- BEGIN AUTO SUMMARY -->"
MEMORY_END = "<!-- END AUTO SUMMARY -->"
STOPWORDS = {
    "auto",
    "iteration",
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "than",
    "that",
    "this",
    "keep",
    "revert",
    "new",
    "best",
    "raise",
    "lower",
    "tighten",
    "relax",
    "add",
    "first",
    "true",
    "only",
    "long",
    "short",
    "size",
    "sized",
    "scaled",
    "model",
    "models",
    "strategy",
    "gate",
    "entry",
    "entries",
    "signal",
    "signals",
}


@dataclass
class RunRecord:
    commit: str
    score: float
    sharpe: float
    max_dd: float
    status: str
    description: str


def _to_float(raw: str) -> float:
    try:
        return float(raw.strip())
    except Exception:
        return float("nan")


def load_runs(results_path: Path) -> list[RunRecord]:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    with results_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for row in reader:
            if not row:
                continue
            cleaned = {k.strip(): (v or "").strip() for k, v in row.items() if k is not None}
            if not cleaned.get("commit"):
                continue
            rows.append(
                RunRecord(
                    commit=cleaned.get("commit", ""),
                    score=_to_float(cleaned.get("score", "")),
                    sharpe=_to_float(cleaned.get("sharpe", "")),
                    max_dd=_to_float(cleaned.get("max_dd", "")),
                    status=cleaned.get("status", "").upper(),
                    description=cleaned.get("description", ""),
                )
            )
    return rows


def _looks_like_15m_candidate(record: RunRecord) -> bool:
    desc = record.description.lower()
    return (
        "15m" in desc
        and record.score > 0.0
        and record.sharpe > 0.0
        and record.max_dd < 10.0
    )


def effective_status(record: RunRecord) -> str:
    status = record.status.upper().strip() or "UNKNOWN"
    if status == "REVERT" and _looks_like_15m_candidate(record):
        return "CANDIDATE"
    return status


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def _commit_number(commit: str) -> int:
    match = re.search(r"(\d+)", commit)
    return int(match.group(1)) if match else -1


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"exp\d+[:\s-]*", " ", text.lower())
    tokens = re.findall(r"[a-z0-9_]+", text)
    kept = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) <= 2 and token not in {"1h", "4h"}:
            continue
        kept.append(token)
    return kept


def _top_tokens(records: Iterable[RunRecord], limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(_tokenize(record.description))
    return [token for token, _ in counter.most_common(limit)]


def _score_slope(records: list[RunRecord]) -> float:
    if len(records) < 2:
        return 0.0
    xs = list(range(len(records)))
    ys = [record.score for record in records]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return numer / denom


def summarize_runs(records: list[RunRecord], recent_window: int) -> dict:
    if not records:
        raise ValueError("No experiment rows found in results.tsv")

    recent = records[-recent_window:]
    keeps = [record for record in records if effective_status(record) == "KEEP"]
    candidates = [record for record in records if effective_status(record) == "CANDIDATE"]
    progress = [record for record in records if effective_status(record) in {"KEEP", "CANDIDATE"}]
    reverts = [record for record in records if effective_status(record) == "REVERT"]
    recent_keeps = [record for record in recent if effective_status(record) == "KEEP"]
    recent_candidates = [record for record in recent if effective_status(record) == "CANDIDATE"]
    recent_progress = [record for record in recent if effective_status(record) in {"KEEP", "CANDIDATE"}]
    recent_reverts = [record for record in recent if effective_status(record) == "REVERT"]

    best = max(records, key=lambda record: (record.score, _commit_number(record.commit)))
    last = records[-1]
    last_keep = max(keeps, key=lambda record: _commit_number(record.commit)) if keeps else None
    last_progress = max(progress, key=lambda record: _commit_number(record.commit)) if progress else None
    best_index = records.index(best)
    runs_since_best = len(records) - best_index - 1
    keep_rate = _pct(len(keeps), len(records))
    candidate_rate = _pct(len(candidates), len(records))
    progress_rate = _pct(len(progress), len(records))
    recent_keep_rate = _pct(len(recent_keeps), len(recent))
    recent_progress_rate = _pct(len(recent_progress), len(recent))
    recent_slope = _score_slope(recent)

    if runs_since_best >= recent_window and recent_progress_rate < 20.0:
        mode = "structural_change"
    elif recent_progress_rate < 35.0:
        mode = "explore"
    else:
        mode = "exploit"

    guidance = []
    if mode == "structural_change":
        guidance.append("Plateau detected. Prefer structural changes over threshold-only tuning.")
        guidance.append("Refresh rejected families before editing strategy.py again.")
    elif mode == "explore":
        guidance.append("Recent keep rate is weak. Broaden the search, but keep changes small and auditable.")
        guidance.append("Avoid repeating the most common revert-heavy themes without a new gating idea.")
    else:
        guidance.append("Recent keep rate is healthy enough to keep exploiting the current family.")
        guidance.append("Bias toward incremental improvements around the current baseline before widening scope.")

    if last.status == "REVERT":
        guidance.append("The latest run reverted. Read its description before the next mutation.")
    if recent_slope < 0:
        guidance.append("Recent score trend is negative. Verify that the loop is not overfitting to a dead branch.")

    summary = {
        "total_runs": len(records),
        "keep_count": len(keeps),
        "candidate_count": len(candidates),
        "progress_count": len(progress),
        "revert_count": len(reverts),
        "keep_rate_pct": round(keep_rate, 2),
        "candidate_rate_pct": round(candidate_rate, 2),
        "progress_rate_pct": round(progress_rate, 2),
        "recent_window": len(recent),
        "recent_keep_rate_pct": round(recent_keep_rate, 2),
        "recent_progress_rate_pct": round(recent_progress_rate, 2),
        "recent_score_slope": round(recent_slope, 4),
        "best_run": asdict(best),
        "last_run": asdict(last),
        "last_keep": asdict(last_keep) if last_keep else None,
        "last_progress": asdict(last_progress) if last_progress else None,
        "runs_since_best": runs_since_best,
        "mode": mode,
        "revert_heavy_tokens": _top_tokens(recent_reverts),
        "keep_heavy_tokens": _top_tokens(recent_progress),
        "guidance": guidance,
    }
    return summary


def render_summary(summary: dict) -> str:
    lines = [
        "Results Summary",
        f"total runs:        {summary['total_runs']}",
        f"keep rate:         {summary['keep_rate_pct']:.2f}%",
        f"candidate rate:    {summary['candidate_rate_pct']:.2f}%",
        f"progress rate:     {summary['progress_rate_pct']:.2f}%",
        f"recent keep rate:  {summary['recent_keep_rate_pct']:.2f}% (last {summary['recent_window']})",
        f"recent progress:   {summary['recent_progress_rate_pct']:.2f}% (last {summary['recent_window']})",
        f"runs since best:   {summary['runs_since_best']}",
        f"mode:              {summary['mode']}",
        f"best run:          {summary['best_run']['commit']} | score {summary['best_run']['score']:.3f} | {summary['best_run']['description']}",
        f"last run:          {summary['last_run']['commit']} | {summary['last_run']['status']} | score {summary['last_run']['score']:.3f} | {summary['last_run']['description']}",
    ]
    if summary["last_progress"]:
        lines.append(
            f"last progress:     {summary['last_progress']['commit']} | score {summary['last_progress']['score']:.3f} | {summary['last_progress']['description']}"
        )
    if summary["revert_heavy_tokens"]:
        lines.append(f"revert-heavy:      {', '.join(summary['revert_heavy_tokens'])}")
    if summary["keep_heavy_tokens"]:
        lines.append(f"keep-heavy:        {', '.join(summary['keep_heavy_tokens'])}")
    lines.append("guidance:")
    for item in summary["guidance"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def render_memory_block(summary: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    last_keep = summary["last_keep"]
    last_keep_line = (
        f"- Last KEEP: `{last_keep['commit']}` | score {last_keep['score']:.3f} | {last_keep['description']}"
        if last_keep
        else "- Last KEEP: none yet"
    )
    lines = [
        MEMORY_BEGIN,
        "## Auto Summary",
        f"- Updated: {now}",
        f"- Total logged runs: {summary['total_runs']}",
        f"- Best run: `{summary['best_run']['commit']}` | score {summary['best_run']['score']:.3f} | {summary['best_run']['description']}",
        f"- Last run: `{summary['last_run']['commit']}` | {summary['last_run']['status']} | score {summary['last_run']['score']:.3f} | {summary['last_run']['description']}",
        last_keep_line,
        (
            f"- Last promising run: `{summary['last_progress']['commit']}` | score {summary['last_progress']['score']:.3f} | {summary['last_progress']['description']}"
            if summary["last_progress"]
            else "- Last promising run: none yet"
        ),
        f"- Recent keep rate: {summary['recent_keep_rate_pct']:.2f}% over the last {summary['recent_window']} runs",
        f"- Recent progress rate: {summary['recent_progress_rate_pct']:.2f}% over the last {summary['recent_window']} runs",
        f"- Runs since best: {summary['runs_since_best']}",
        f"- Suggested mode: `{summary['mode']}`",
    ]
    if summary["revert_heavy_tokens"]:
        lines.append(f"- Revert-heavy themes: {', '.join(summary['revert_heavy_tokens'])}")
    if summary["keep_heavy_tokens"]:
        lines.append(f"- Keep-heavy themes: {', '.join(summary['keep_heavy_tokens'])}")
    lines.append("")
    lines.append("## Next Guidance")
    for item in summary["guidance"]:
        lines.append(f"- {item}")
    lines.append(MEMORY_END)
    return "\n".join(lines)


def update_memory_file(memory_path: Path, summary: dict) -> None:
    block = render_memory_block(summary)
    if memory_path.exists():
        text = memory_path.read_text()
    else:
        text = "# Research Memory\n\n"

    if MEMORY_BEGIN in text and MEMORY_END in text:
        pattern = re.compile(
            re.escape(MEMORY_BEGIN) + r".*?" + re.escape(MEMORY_END),
            flags=re.DOTALL,
        )
        updated = pattern.sub(block, text, count=1)
    else:
        suffix = "" if text.endswith("\n") else "\n"
        updated = text + suffix + "\n" + block + "\n"
    memory_path.write_text(updated)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze autoresearch experiment logs.")
    parser.add_argument("--results", default="results.tsv", help="Path to results.tsv")
    parser.add_argument("--recent-window", type=int, default=20, help="Window size for recent summary")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--update-memory",
        action="store_true",
        help="Refresh the generated auto-summary block in RESEARCH_MEMORY.md",
    )
    parser.add_argument(
        "--memory-file",
        default="RESEARCH_MEMORY.md",
        help="Memory file to refresh when --update-memory is used",
    )
    args = parser.parse_args()

    records = load_runs(Path(args.results))
    summary = summarize_runs(records, max(1, args.recent_window))

    if args.update_memory:
        update_memory_file(Path(args.memory_file), summary)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_summary(summary))
        if args.update_memory:
            print(f"\nUpdated {args.memory_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
