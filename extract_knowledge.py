"""
extract_knowledge.py — Extract trading insights from PDFs using Gemini Flash.

Reads PDFs from ./docs/tradebridge/, sends text to gemini-2.0-flash,
and saves structured insights to ./docs/knowledge.json.

Requires: GOOGLE_API_KEY in environment (or .env file).

Usage:
    uv run extract_knowledge.py              # process all PDFs
    uv run extract_knowledge.py --force      # re-extract even if cached
    uv run extract_knowledge.py --status     # show extraction status
    uv run extract_knowledge.py --summary    # print merged knowledge summary
"""

import os
import sys
import json
import time
import argparse
import textwrap

try:
    import pdfplumber
except ImportError:
    sys.exit("pdfplumber not found — run: uv sync")

try:
    import google.generativeai as genai
except ImportError:
    sys.exit("google-generativeai not found — run: uv sync")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs", "tradebridge")
OUT_FILE  = os.path.join(os.path.dirname(__file__), "docs", "knowledge.json")
MODEL     = "gemini-2.0-flash"
MAX_CHARS = 60_000   # truncate long PDFs before sending (stay within token limit)

EXTRACTION_PROMPT = textwrap.dedent("""
You are a quantitative trading researcher. Analyze this excerpt from a trading document
and extract actionable knowledge for an algorithmic crypto trading system.

Return ONLY valid JSON with this exact structure:
{
  "signals": [
    "string describing a specific entry or exit signal (e.g., 'Buy when RSI crosses above 30 after downtrend')"
  ],
  "indicators": [
    "string naming an indicator and how it is used (e.g., 'Bollinger Bands: enter on squeeze breakout')"
  ],
  "risk_rules": [
    "string describing a position sizing, stop-loss, or drawdown rule"
  ],
  "concepts": [
    "string describing a higher-level market concept or principle worth encoding"
  ],
  "crypto_relevance": "high | medium | low — how applicable this is to crypto/perpetuals trading"
}

Keep each item concise (1-2 sentences max). Only include items directly stated or strongly implied
by the text. If a category has nothing relevant, use an empty array [].

Document excerpt:
---
{text}
---
""").strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        # Try loading from .env manually (no python-dotenv dependency)
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not key:
        sys.exit("GOOGLE_API_KEY not found in environment or .env file.")
    return key


def _extract_text(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber. Returns empty string on failure."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        return ""


def _call_gemini(model: genai.GenerativeModel, text: str, filename: str) -> dict:
    """Call Gemini Flash and parse JSON response. Returns empty dict on failure."""
    truncated = text[:MAX_CHARS]
    prompt = EXTRACTION_PROMPT.replace("{text}", truncated)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    JSON parse error for {filename}: {e}")
        return {}
    except Exception as e:
        print(f"    Gemini error for {filename}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_all(force: bool = False) -> dict:
    """
    Process all PDFs in DOCS_DIR.
    Returns the full knowledge dict (loaded from / saved to OUT_FILE).
    """
    if not os.path.isdir(DOCS_DIR):
        sys.exit(f"Docs directory not found: {DOCS_DIR}\nRun: uv run fetch_docs.py")

    pdf_files = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf"))
    if not pdf_files:
        sys.exit(f"No PDFs found in {DOCS_DIR}\nRun: uv run fetch_docs.py")

    # Load existing knowledge cache
    knowledge = {}
    if os.path.exists(OUT_FILE) and not force:
        with open(OUT_FILE) as f:
            knowledge = json.load(f)

    api_key = _load_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)

    total = len(pdf_files)
    processed = 0
    skipped   = 0
    failed    = []

    for i, filename in enumerate(pdf_files, 1):
        if filename in knowledge and not force:
            print(f"  [{i:2d}/{total}] SKIP  {filename}")
            skipped += 1
            continue

        pdf_path = os.path.join(DOCS_DIR, filename)
        text = _extract_text(pdf_path)

        if not text.strip():
            print(f"  [{i:2d}/{total}] EMPTY {filename} — no extractable text")
            knowledge[filename] = {"signals": [], "indicators": [], "risk_rules": [], "concepts": [], "crypto_relevance": "low", "_empty": True}
            failed.append(filename)
            continue

        char_count = len(text)
        print(f"  [{i:2d}/{total}] OK    {filename} ({char_count:,} chars) → Gemini...")

        result = _call_gemini(model, text, filename)

        if result:
            knowledge[filename] = result
            n_items = sum(len(v) for v in result.values() if isinstance(v, list))
            print(f"           ✓ {n_items} items extracted (relevance: {result.get('crypto_relevance', '?')})")
            processed += 1
        else:
            knowledge[filename] = {"signals": [], "indicators": [], "risk_rules": [], "concepts": [], "crypto_relevance": "low", "_error": True}
            failed.append(filename)

        # Save incrementally so progress isn't lost on interruption
        os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
        with open(OUT_FILE, "w") as f:
            json.dump(knowledge, f, indent=2)

        time.sleep(0.5)  # polite rate limiting

    print()
    print("─" * 60)
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped} (already cached)")
    if failed:
        print(f"  Failed:    {len(failed)}")
        for fn in failed:
            print(f"    - {fn}")
    print(f"\n  Knowledge saved to: {OUT_FILE}")

    return knowledge


def print_status():
    """Show which PDFs have been extracted."""
    if not os.path.exists(OUT_FILE):
        print("No knowledge.json found. Run: uv run extract_knowledge.py")
        return

    with open(OUT_FILE) as f:
        knowledge = json.load(f)

    pdf_files = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")) if os.path.isdir(DOCS_DIR) else []

    print(f"\nKnowledge file: {OUT_FILE}")
    print(f"PDFs directory: {DOCS_DIR}")
    print("─" * 60)
    for filename in pdf_files:
        if filename in knowledge:
            entry = knowledge[filename]
            if entry.get("_error") or entry.get("_empty"):
                print(f"  ✗  {filename} (failed)")
            else:
                n = sum(len(v) for v in entry.values() if isinstance(v, list))
                rel = entry.get("crypto_relevance", "?")
                print(f"  ✓  {filename} ({n} items, {rel})")
        else:
            print(f"  ·  {filename} (not yet extracted)")
    print("─" * 60)
    print(f"  {len(knowledge)}/{len(pdf_files)} extracted\n")


def print_summary():
    """Print a merged summary of all high/medium relevance insights."""
    if not os.path.exists(OUT_FILE):
        print("No knowledge.json found. Run: uv run extract_knowledge.py")
        return

    with open(OUT_FILE) as f:
        knowledge = json.load(f)

    all_signals    = []
    all_indicators = []
    all_risk_rules = []
    all_concepts   = []

    for filename, entry in knowledge.items():
        if entry.get("_error") or entry.get("_empty"):
            continue
        rel = entry.get("crypto_relevance", "low")
        if rel not in ("high", "medium"):
            continue
        all_signals.extend(entry.get("signals", []))
        all_indicators.extend(entry.get("indicators", []))
        all_risk_rules.extend(entry.get("risk_rules", []))
        all_concepts.extend(entry.get("concepts", []))

    def _print_section(title, items):
        if not items:
            return
        print(f"\n### {title} ({len(items)})")
        for item in items:
            print(f"  - {item}")

    print("\n# Trading Knowledge Summary (high + medium crypto relevance)")
    print("=" * 60)
    _print_section("Signals", all_signals)
    _print_section("Indicators", all_indicators)
    _print_section("Risk Rules", all_risk_rules)
    _print_section("Concepts", all_concepts)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract trading insights from PDFs using Gemini Flash")
    parser.add_argument("--force",   action="store_true", help="Re-extract even if already cached")
    parser.add_argument("--status",  action="store_true", help="Show extraction status and exit")
    parser.add_argument("--summary", action="store_true", help="Print merged knowledge summary and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    if args.summary:
        print_summary()
        sys.exit(0)

    print(f"Extracting trading knowledge from PDFs in {DOCS_DIR}")
    print(f"Model: {MODEL}  |  Output: {OUT_FILE}")
    print()
    extract_all(force=args.force)
