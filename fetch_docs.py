"""
fetch_docs.py — Download trading PDFs from tradebridge/DOCs on GitHub.

Downloads 56 PDFs to ./docs/tradebridge/ for use by extract_knowledge.py.
No API key required — raw GitHub content is public.

Usage:
    uv run fetch_docs.py            # download all PDFs
    uv run fetch_docs.py --force    # re-download even if cached
    uv run fetch_docs.py --status   # show what's already downloaded
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error

GITHUB_API_URL = "https://api.github.com/repos/0xedev/tradebridge/contents/DOCs"
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs", "tradebridge")

# Hardcoded file list (56 files from 0xedev/tradebridge/DOCs)
# Using raw GitHub URLs directly — no API key needed.
PDF_FILES = [
    "123system.pdf",
    "25_Rules_Of_Forex_Trading_Discipline.pdf",
    "A_Course_in_Miracles.pdf",
    "A_Six-Part_Study_Guide_to_Market_Profile.pdf",
    "Bollinger_Bandit_Trading_Strategy.pdf",
    "Calming_The_Mind.pdf",
    "Coders_Guru_Full_Course.pdf",
    "Core_Point_and_Figure_Chart_Patterns.pdf",
    "Dow_Mini-Value_Area.pdf",
    "Dynamic_Breakout_II_Strategy.pdf",
    "Eleven_Elliott_Wave_Patterns.pdf",
    "Evolving_Chart_Pattern_Sensitive_Neural_Network_Based_Forex_Trading_Agents.pdf",
    "Forex-Trading-For-Beginners-The-Ultimate-Guide.pdf",
    "Ghost_Trader_Trading_Strategy.pdf",
    "Heisenberg Uncertainty Principle and Economic Analogues of Basic Physical Quantities.pdf",
    "How_George_Soros_Knows_What_He_Knows.pdf",
    "King_Keltner_Trading_Strategy.pdf",
    "LBR_Scalp_setups.pdf",
    "MOD_free_chapter.pdf",
    "Macroeconomic_Implications_of_the_Beliefs_and_Behavior_of_Foreign_Exchange_Traders.pdf",
    "Market_Profile_Basics.pdf",
    "Money_Manager_Trading_Strategy.pdf",
    "New_Elliott_Wave_Rule_Achieve_Definitive_Wave_Counts.pdf",
    "Point-and-Figure-Charting-a-Computational-Methodology-and-Trading-Rule-Performance-in-the-SP500-Futures-Market.pdf",
    "Predictive_Power_of_Price_Patterns.pdf",
    "SFO_raschke_0803.pdf",
    "Super_Combo_Day_Trading_Strategy.pdf",
    "The string prediction models as an invariants of time series in forex market.pdf",
    "The_5_Steps_to_Becoming_a_Trader.pdf",
    "The_7_Deadly_Sins_of_Forex.pdf",
    "The_Sharpe_Ratio.pdf",
    "Thermostat_Trading_Strategy.pdf",
    "Tick.pdf",
    "Trading_as_a_Business.pdf",
    "Trend_Determination.pdf",
    "Trend_vs_no_trend.pdf",
    "Using Recurrent Neural Networks to Forecasting of Forex.pdf",
    "What_moves_the_currency_market.pdf",
    "a-new-interprtation-of-information-rate-kelly.pdf",
    "all_about_forex_market_in_usa.pdf",
    "bid-ask_spreads.pdf",
    "cci_manual.pdf",
    "dealership_market.pdf",
    "depth_volatility.pdf",
    "intro_to_market_profile.pdf",
    "jesse_livermore.pdf",
    "lifestyle.pdf",
    "lss_3_day_cycle_method.pdf",
    "market_turns.pdf",
    "nicktrader_on_no_price_trading.pdf",
    "order_driven_market.pdf",
    "picking_tops.pdf",
    "supply_demand.pdf",
    "the_interaction_between_the_frequency_of_market_quotes_spread_and_volatility_in_Forex.pdf",
    "trade_behavior.pdf",
]

RAW_BASE = "https://raw.githubusercontent.com/0xedev/tradebridge/main/DOCs"


def _url_encode_filename(filename: str) -> str:
    """URL-encode spaces and special chars in filename for the download URL."""
    import urllib.parse
    return urllib.parse.quote(filename, safe="._-/")


def download_pdfs(force: bool = False) -> dict:
    """
    Download all PDFs from tradebridge/DOCs to ./docs/tradebridge/.
    Returns: {"downloaded": int, "skipped": int, "failed": list}
    """
    os.makedirs(DOCS_DIR, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = []

    total = len(PDF_FILES)
    for i, filename in enumerate(PDF_FILES, 1):
        dest = os.path.join(DOCS_DIR, filename)

        if os.path.exists(dest) and not force:
            size_kb = os.path.getsize(dest) // 1024
            print(f"  [{i:2d}/{total}] SKIP  {filename} ({size_kb}KB cached)")
            skipped += 1
            continue

        encoded = _url_encode_filename(filename)
        url = f"{RAW_BASE}/{encoded}"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "autotrader-fetch/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

            with open(dest, "wb") as f:
                f.write(data)

            size_kb = len(data) // 1024
            print(f"  [{i:2d}/{total}] OK    {filename} ({size_kb}KB)")
            downloaded += 1

        except urllib.error.HTTPError as e:
            print(f"  [{i:2d}/{total}] FAIL  {filename} — HTTP {e.code}")
            failed.append(filename)
        except Exception as e:
            print(f"  [{i:2d}/{total}] FAIL  {filename} — {e}")
            failed.append(filename)

        time.sleep(0.1)  # be polite to GitHub CDN

    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def print_status():
    """Show which PDFs are already downloaded."""
    print(f"\nDocs directory: {DOCS_DIR}")
    print("─" * 60)
    total_size = 0
    present = 0
    for filename in PDF_FILES:
        dest = os.path.join(DOCS_DIR, filename)
        if os.path.exists(dest):
            size_kb = os.path.getsize(dest) // 1024
            total_size += size_kb
            print(f"  ✓  {filename} ({size_kb}KB)")
            present += 1
        else:
            print(f"  ✗  {filename}")
    print("─" * 60)
    print(f"  {present}/{len(PDF_FILES)} downloaded  ({total_size}KB total)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download tradebridge PDFs from GitHub")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--status", action="store_true", help="Show download status and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    print(f"Downloading {len(PDF_FILES)} PDFs → {DOCS_DIR}")
    print()
    result = download_pdfs(force=args.force)
    print()
    print("─" * 60)
    print(f"  Downloaded: {result['downloaded']}")
    print(f"  Skipped:    {result['skipped']} (already cached)")
    if result["failed"]:
        print(f"  Failed:     {len(result['failed'])}")
        for f in result["failed"]:
            print(f"    - {f}")
    print()
    print(f"PDFs saved to: {DOCS_DIR}")
    print("Next: uv run extract_knowledge.py")
