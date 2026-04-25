"""
main.py — Command-line interface for the credit memo agent.

Usage:
    python main.py --borrower data/sample_borrowers/healthy_company.json --scenario baseline
    python main.py --borrower data/sample_borrowers/stressed_company.json --scenario severe_stress
    python main.py --borrower data/sample_borrowers/distressed_company.json --scenario mild_recession --output memo.md

Scenarios: baseline, mild_recession, severe_stress
"""

import argparse
import json
import sys

from src.schemas import BorrowerFinancials, CreditRequest
from src.agents.graph import run_agent


def main():
    parser = argparse.ArgumentParser(
        description="Generate a credit memo for a borrower using the AI agent.",
    )
    parser.add_argument(
        "--borrower", "-b",
        required=True,
        help="Path to a borrower JSON file (see data/sample_borrowers/)",
    )
    parser.add_argument(
        "--scenario", "-s",
        default="baseline",
        choices=["baseline", "mild_recession", "severe_stress"],
        help="Macro scenario for stress testing (default: baseline)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional: file path to save the memo as Markdown",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print intermediate state for each node",
    )
    
    args = parser.parse_args()
    
    # Load borrower data
    print(f"Loading borrower data from {args.borrower}...")
    with open(args.borrower) as f:
        borrower_data = json.load(f)
    
    borrower = BorrowerFinancials(**borrower_data)
    request = CreditRequest(borrower=borrower, macro_scenario=args.scenario)
    
    print(f"Running agent for {borrower.borrower_name} under scenario '{args.scenario}'...")
    print("(This typically takes 15-30 seconds — most of it LLM generation.)\n")
    
    # Run the agent
    result = run_agent(request)
    
    # Check for errors
    if result.get("errors"):
        print("ERRORS occurred during execution:", file=sys.stderr)
        for err in result["errors"]:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        ratios = result["ratios"]
        pd_result = result["pd_result"]
        stressed = result["stressed_pd"]
        print("\n" + "=" * 60)
        print("INTERMEDIATE COMPUTATION:")
        print("=" * 60)
        print(f"Leverage:        {ratios.leverage_ratio:.2f}x ({ratios.leverage_assessment})")
        print(f"Coverage:        {ratios.interest_coverage:.2f}x ({ratios.coverage_assessment})")
        print(f"Current ratio:   {ratios.current_ratio:.2f} ({ratios.liquidity_assessment})")
        print(f"Baseline PD:     {pd_result.baseline_pd:.2%} ({pd_result.risk_rating})")
        print(f"Stressed PD:     {stressed.stressed_pd:.2%} ({stressed.stressed_rating})")
        print()
    
    # Output the memo
    memo_md = result["memo"].to_markdown()
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(memo_md)
        print(f"\nMemo saved to {args.output}")
    else:
        print("=" * 60)
        print(memo_md)


if __name__ == "__main__":
    main()
