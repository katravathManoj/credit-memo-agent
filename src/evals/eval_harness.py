"""
eval_harness.py — Tests the agent on a set of borrowers and reports metrics.

Run with:
    python -m src.evals.eval_harness

What it does:
- Generates 15 synthetic borrowers spanning healthy → distressed
- Runs the full agent on each
- Checks: did it produce a valid memo? Are numbers in the memo consistent
  with the underlying computed values?

This is what makes the project look "real" rather than "tutorial-grade".
Banks insist on evals before any model goes near production.
"""

import re
import json

from src.schemas import BorrowerFinancials, CreditRequest
from src.agents.graph import run_agent


# ============================================================================
# TEST CASES — synthetic borrowers spanning the credit spectrum
# ============================================================================

TEST_CASES = [
    # Healthy
    {"borrower_name": "Healthy Mfg A", "industry": "Manufacturing", "revenue": 1200, "ebitda": 240, "interest_expense": 18, "total_debt": 420, "cash_and_equivalents": 120, "current_assets": 380, "current_liabilities": 170, "total_equity": 700},
    {"borrower_name": "Healthy Tech B", "industry": "Technology", "revenue": 850, "ebitda": 220, "interest_expense": 8, "total_debt": 180, "cash_and_equivalents": 150, "current_assets": 280, "current_liabilities": 95, "total_equity": 580},
    {"borrower_name": "Healthy HC C", "industry": "Healthcare", "revenue": 950, "ebitda": 195, "interest_expense": 22, "total_debt": 480, "cash_and_equivalents": 80, "current_assets": 290, "current_liabilities": 140, "total_equity": 510},
    {"borrower_name": "Healthy Energy D", "industry": "Energy", "revenue": 2100, "ebitda": 850, "interest_expense": 65, "total_debt": 1300, "cash_and_equivalents": 220, "current_assets": 540, "current_liabilities": 280, "total_equity": 1850},
    {"borrower_name": "Healthy Fin E", "industry": "Financial Services", "revenue": 720, "ebitda": 220, "interest_expense": 35, "total_debt": 580, "cash_and_equivalents": 95, "current_assets": 240, "current_liabilities": 120, "total_equity": 480},
    
    # Moderate
    {"borrower_name": "Moderate Retail F", "industry": "Retail", "revenue": 1800, "ebitda": 145, "interest_expense": 38, "total_debt": 520, "cash_and_equivalents": 65, "current_assets": 380, "current_liabilities": 240, "total_equity": 380},
    {"borrower_name": "Moderate Mfg G", "industry": "Manufacturing", "revenue": 1500, "ebitda": 175, "interest_expense": 32, "total_debt": 580, "cash_and_equivalents": 70, "current_assets": 340, "current_liabilities": 220, "total_equity": 420},
    {"borrower_name": "Moderate RE H", "industry": "Real Estate", "revenue": 920, "ebitda": 510, "interest_expense": 165, "total_debt": 2800, "cash_and_equivalents": 120, "current_assets": 320, "current_liabilities": 240, "total_equity": 980},
    {"borrower_name": "Moderate Hosp I", "industry": "Hospitality", "revenue": 1100, "ebitda": 180, "interest_expense": 48, "total_debt": 720, "cash_and_equivalents": 55, "current_assets": 220, "current_liabilities": 180, "total_equity": 320},
    {"borrower_name": "Moderate Const J", "industry": "Construction", "revenue": 1850, "ebitda": 145, "interest_expense": 42, "total_debt": 480, "cash_and_equivalents": 85, "current_assets": 480, "current_liabilities": 320, "total_equity": 380},
    
    # Stressed
    {"borrower_name": "Stressed Retail K", "industry": "Retail", "revenue": 920, "ebitda": 38, "interest_expense": 42, "total_debt": 540, "cash_and_equivalents": 22, "current_assets": 220, "current_liabilities": 280, "total_equity": 95},
    {"borrower_name": "Stressed Hosp L", "industry": "Hospitality", "revenue": 480, "ebitda": 28, "interest_expense": 32, "total_debt": 380, "cash_and_equivalents": 8, "current_assets": 95, "current_liabilities": 145, "total_equity": 45},
    {"borrower_name": "Stressed Mfg M", "industry": "Manufacturing", "revenue": 720, "ebitda": 45, "interest_expense": 48, "total_debt": 480, "cash_and_equivalents": 18, "current_assets": 180, "current_liabilities": 220, "total_equity": 65},
    {"borrower_name": "Stressed Energy N", "industry": "Energy", "revenue": 1200, "ebitda": 95, "interest_expense": 145, "total_debt": 1800, "cash_and_equivalents": 35, "current_assets": 280, "current_liabilities": 320, "total_equity": 220},
    {"borrower_name": "Stressed Tech O", "industry": "Technology", "revenue": 380, "ebitda": -45, "interest_expense": 28, "total_debt": 320, "cash_and_equivalents": 25, "current_assets": 95, "current_liabilities": 110, "total_equity": 75},
]


def check_pd_in_memo(memo_text: str, expected_pd: float, tolerance: float = 0.005) -> bool:
    """
    Check that the PD percentage mentioned in the memo matches the computed value
    within tolerance. This catches LLM hallucinations of numbers.
    """
    # Find percentages like "5.43%" or "12%" in the memo
    pct_pattern = r"(\d+\.?\d*)%"
    matches = re.findall(pct_pattern, memo_text)
    
    if not matches:
        return False
    
    expected_pct = expected_pd * 100
    
    # The memo should mention a percentage close to the expected PD
    return any(abs(float(m) - expected_pct) <= (tolerance * 100) for m in matches)


def check_rating_in_memo(memo_text: str, expected_rating: str) -> bool:
    """Check that the expected rating appears in the memo text."""
    # Be careful about partial matches: 'A' could match many things
    # We check for the rating with surrounding word boundaries
    return bool(re.search(rf"\b{re.escape(expected_rating)}\b", memo_text))


def run_evals():
    print(f"Running eval harness on {len(TEST_CASES)} test cases...\n")
    
    results = {
        "total": 0,
        "succeeded": 0,
        "memo_complete": 0,
        "pd_consistent": 0,
        "rating_consistent": 0,
        "errors": [],
    }
    
    for i, case in enumerate(TEST_CASES, 1):
        results["total"] += 1
        print(f"  [{i}/{len(TEST_CASES)}] {case['borrower_name']}...", end=" ")
        
        try:
            borrower = BorrowerFinancials(**case)
            request = CreditRequest(borrower=borrower, macro_scenario="baseline")
            state = run_agent(request)
            
            if state.get("errors"):
                print("FAILED:", state["errors"])
                results["errors"].append((case["borrower_name"], state["errors"]))
                continue
            
            results["succeeded"] += 1
            
            memo = state["memo"]
            memo_text = memo.to_markdown()
            
            # Check 1: memo has all sections
            if all([memo.executive_summary, memo.financial_analysis, memo.peer_comparison,
                    memo.credit_risk_assessment, memo.stress_scenario_analysis, memo.recommendation]):
                results["memo_complete"] += 1
            
            # Check 2: PD consistency
            if check_pd_in_memo(memo_text, memo.pd_result.baseline_pd):
                results["pd_consistent"] += 1
            
            # Check 3: rating consistency
            if check_rating_in_memo(memo_text, memo.pd_result.risk_rating):
                results["rating_consistent"] += 1
            
            print(f"OK (PD: {memo.pd_result.baseline_pd:.2%}, Rating: {memo.pd_result.risk_rating})")
        
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results["errors"].append((case["borrower_name"], str(e)))
    
    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)
    print(f"Total cases:           {results['total']}")
    print(f"Successfully ran:      {results['succeeded']} ({100*results['succeeded']/results['total']:.0f}%)")
    print(f"Memo complete:         {results['memo_complete']} ({100*results['memo_complete']/results['total']:.0f}%)")
    print(f"PD consistent:         {results['pd_consistent']} ({100*results['pd_consistent']/results['total']:.0f}%)")
    print(f"Rating consistent:     {results['rating_consistent']} ({100*results['rating_consistent']/results['total']:.0f}%)")
    
    if results["errors"]:
        print("\nERRORS:")
        for name, err in results["errors"]:
            print(f"  - {name}: {err}")
    
    print()
    return results


if __name__ == "__main__":
    run_evals()
