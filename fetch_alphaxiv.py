"""
fetch_alphaxiv.py — Live research ingestion from arXiv API.

Searches for quant trading papers, extracts insights with Gemini, and merges
results into docs/knowledge.json — the same file that feeds feature engineering.

Uses arXiv REST API (free, no auth required) for automation.
For richer semantic search, alphaxiv MCP is registered in Claude Code and
can be used directly in sessions (browser OAuth — not scriptable).

Usage:
    uv run fetch_alphaxiv.py                  # discover + extract new papers
    uv run fetch_alphaxiv.py --search-only    # print paper list, no extraction
    uv run fetch_alphaxiv.py --max 20         # limit papers per query (default 5)
    uv run fetch_alphaxiv.py --force          # re-extract already-seen papers

Requires: GOOGLE_API_KEY for Gemini extraction (same as extract_knowledge.py)
"""

import os
import sys
import json
import time
import textwrap
import argparse
import xml.etree.ElementTree as ET

import requests

try:
    import google.generativeai as genai
    GENAI_OK = True
except ImportError:
    GENAI_OK = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_FILE     = os.path.join(os.path.dirname(__file__), "docs", "knowledge.json")
SEEN_FILE    = os.path.join(os.path.dirname(__file__), "docs", "seen_papers.json")
GEMINI_MODEL = "gemini-2.0-flash"
MAX_CHARS    = 50_000

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# Search queries targeting signals relevant to our 7-asset trading system.
# Each tuple: (query_string, search_mode)
# search_mode: "semantic" (alphaxiv embedding) | "keyword" (alphaxiv/arXiv text)
SEARCH_QUERIES = [
    ("cryptocurrency perpetuals momentum mean reversion hourly signals",            "semantic"),
    ("cross-asset macro signals equity gold oil cryptocurrency correlation",         "semantic"),
    ("Kelly criterion position sizing volatility targeting crypto",                  "semantic"),
    ("regime detection hidden Markov model financial time series crypto",            "semantic"),
    ("Sharpe ratio optimization deep learning trading agent",                        "semantic"),
    ("RSI MACD Bollinger Bands effectiveness crypto high frequency",                 "keyword"),
    ("transformer neural network financial time series prediction",                  "keyword"),
    ("funding rate perpetual futures basis trading cryptocurrency",                  "keyword"),
    ("drawdown control risk management algorithmic trading",                         "keyword"),
    ("cross-asset momentum spillover gold silver oil equity",                        "keyword"),
]

EXTRACTION_PROMPT = textwrap.dedent("""
You are a quantitative trading researcher. Analyze this excerpt from a trading or
finance research paper and extract actionable knowledge for an algorithmic trading
system that trades BTC, ETH, SOL, GOLD, SPX, OIL, and SILVER on 1-hour bars.

Return ONLY valid JSON with this exact structure:
{
  "signals": [
    "string describing a specific entry or exit signal"
  ],
  "indicators": [
    "string naming an indicator and how it is used"
  ],
  "risk_rules": [
    "string describing a position sizing, stop-loss, or drawdown rule"
  ],
  "concepts": [
    "string describing a higher-level market concept or principle worth encoding as a feature"
  ],
  "crypto_relevance": "high | medium | low"
}

Keep each item concise (1-2 sentences). Only include items directly stated or strongly
implied by the text. Empty array [] if nothing relevant for a category.

Document excerpt:
---
{text}
---
""").strip()


# ---------------------------------------------------------------------------
# Env / API key loading
# ---------------------------------------------------------------------------

def _load_env() -> dict:
    env = {}
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            m_export = line.strip()
            if m_export.startswith("export "):
                m_export = m_export[7:]
            idx = m_export.find("=")
            if idx > 0:
                k = m_export[:idx].strip()
                v = m_export[idx+1:].strip().strip('"').strip("'")
                env[k] = v
    return env


def _get_key(name: str, env: dict) -> str:
    return os.environ.get(name, env.get(name, ""))


# ---------------------------------------------------------------------------
# arXiv REST API fallback
# ---------------------------------------------------------------------------

def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv using the public Atom/REST API."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [arXiv] request failed: {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id_url = entry.find("atom:id", ns).text or ""
        arxiv_id = arxiv_id_url.split("/abs/")[-1].strip()
        title = (entry.find("atom:title", ns).text or "").replace("\n", " ").strip()
        abstract = (entry.find("atom:summary", ns).text or "").replace("\n", " ").strip()
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract[:2000],
        })
    return papers


def arxiv_fetch_text(arxiv_id: str) -> str:
    """Fetch paper abstract + HTML text from ar5iv (rendered arXiv HTML)."""
    clean_id = arxiv_id.split("v")[0]  # strip version suffix
    url = f"https://ar5iv.org/abs/{clean_id}"
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "autotrader-research/1.0"})
        if resp.status_code != 200:
            return ""
        # Strip HTML tags simply — good enough for Gemini
        import re
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text)
        return text[:MAX_CHARS]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Gemini extraction
# ---------------------------------------------------------------------------

def extract_insights(text: str, gemini_key: str) -> dict | None:
    if not GENAI_OK or not gemini_key:
        return None
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = EXTRACTION_PROMPT.format(text=text[:MAX_CHARS])
    try:
        resp = model.generate_content(prompt)
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"  [Gemini] extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# knowledge.json helpers
# ---------------------------------------------------------------------------

def load_knowledge() -> list:
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            return json.load(f)
    return []


def save_knowledge(records: list):
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(records, f, indent=2)


def load_seen() -> set:
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen(seen: set):
    os.makedirs(os.path.dirname(SEEN_FILE), exist_ok=True)
    with open(SEEN_FILE, "w") as f:
        json.dump(sorted(seen), f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch trading research from alphaxiv / arXiv")
    parser.add_argument("--search-only", action="store_true", help="Print papers without extracting")
    parser.add_argument("--max", type=int, default=5, help="Max papers per query (default 5)")
    parser.add_argument("--force", action="store_true", help="Re-extract already-seen papers")
    args = parser.parse_args()

    env = _load_env()
    gemini_key = _get_key("GOOGLE_API_KEY", env)

    if not gemini_key and not args.search_only:
        print("WARN: GOOGLE_API_KEY not set — will discover papers but skip extraction")

    print(f"Mode: arXiv REST API")
    print(f"Note: alphaxiv MCP is available in Claude Code sessions (claude mcp list)")
    print(f"Queries: {len(SEARCH_QUERIES)} | max per query: {args.max}")
    print()

    seen      = set() if args.force else load_seen()
    knowledge = load_knowledge()
    new_count = 0

    for query, search_mode in SEARCH_QUERIES:
        print(f"[Search] {query[:60]}...")
        try:
            papers = arxiv_search(query, args.max)
        except Exception as e:
            print(f"  [ERROR] search failed: {e}")
            continue

        print(f"  Found {len(papers)} papers")

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if not arxiv_id:
                continue
            if arxiv_id in seen and not args.force:
                print(f"  skip (seen): {arxiv_id}")
                continue

            title = paper.get("title", arxiv_id)
            print(f"  → {arxiv_id}: {title[:70]}")

            if args.search_only:
                seen.add(arxiv_id)
                continue

            # Fetch full text via ar5iv, fall back to abstract
            text = arxiv_fetch_text(arxiv_id)
            if not text:
                text = paper.get("abstract", "")

            if not text:
                print(f"    no text available, skipping")
                seen.add(arxiv_id)
                continue

            if not gemini_key:
                seen.add(arxiv_id)
                continue

            # Extract insights
            insights = extract_insights(text, gemini_key)
            if insights:
                record = {
                    "source": "alphaxiv" if client else "arxiv",
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "query": query,
                    **insights,
                }
                knowledge.append(record)
                new_count += 1
                cr = insights.get("crypto_relevance", "?")
                n_signals = len(insights.get("signals", []))
                n_concepts = len(insights.get("concepts", []))
                print(f"    extracted: relevance={cr}, signals={n_signals}, concepts={n_concepts}")
            else:
                print(f"    extraction failed")

            seen.add(arxiv_id)
            time.sleep(1.5)  # be polite to APIs

        time.sleep(0.5)

    # Save
    save_seen(seen)
    if new_count > 0:
        save_knowledge(knowledge)
        print(f"\nDone. Added {new_count} new papers. Total records: {len(knowledge)}")
        print(f"Saved to: {OUT_FILE}")
    else:
        print(f"\nDone. No new records (total: {len(knowledge)}).")

    if new_count > 0 and not args.search_only:
        print()
        print("Next steps:")
        print("  1. Review docs/knowledge.json for new signals/concepts")
        print("  2. Add high-value insights as features in features.py")
        print("  3. uv run train.py  (retrain with enriched features)")


if __name__ == "__main__":
    main()
