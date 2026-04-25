"""
ratio_calculator.py — Computes credit ratios from borrower financials.

KEY ARCHITECTURAL DECISION:
This module deliberately does NOT use the LLM. All ratio computations are pure
Python arithmetic. Why?

1. LLMs are unreliable at math. They will sometimes return 4.2 when the answer
   is 4.18, or worse, completely wrong values for edge cases.
2. Reproducibility: same inputs → same outputs, every time. LLMs don't guarantee that.
3. Auditability: in a real bank, every number in a credit memo must be traceable
   to a specific calculation. Pure Python gives us that. LLM output doesn't.
4. Cost and speed: an arithmetic call costs $0 and takes microseconds. An LLM call
   costs money and takes seconds.


I split the agent into two layers — deterministic computation in Python, 
narrative synthesis in the LLM. Quantitative outputs come from code that I 
can audit and unit-test; the LLM only interprets and writes about those numbers.
"""

from src.schemas import BorrowerFinancials, CreditRatios


# ----- Threshold constants — these are simplified industry rules of thumb -----
# In a real bank, these would vary by industry and be calibrated from data.

LEVERAGE_THRESHOLDS = {
    "low": 2.0,         # < 2x is low leverage
    "moderate": 4.0,    # 2-4x is moderate
    "high": 6.0,        # 4-6x is high
    # > 6x is very_high
}

COVERAGE_THRESHOLDS = {
    "strong": 6.0,      # > 6x interest coverage is strong
    "adequate": 3.0,    # 3-6x is adequate
    "weak": 1.5,        # 1.5-3x is weak
    # < 1.5x is distressed
}

LIQUIDITY_THRESHOLDS = {
    "strong": 2.0,      # current ratio > 2 is strong
    "adequate": 1.2,    # 1.2-2 is adequate
    "weak": 1.0,        # 1-1.2 is weak
    # < 1.0 is distressed
}


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide without crashing on zero. Returns default (or a very large number for safety)."""
    if denominator == 0 or denominator is None:
        # For coverage ratios, zero denominator (e.g., zero interest expense) means
        # effectively infinite coverage. We use a large finite number for sanity.
        return 999.99 if numerator > 0 else default
    return numerator / denominator


def _classify_leverage(leverage: float) -> str:
    """Categorize leverage ratio into a qualitative bucket."""
    if leverage < LEVERAGE_THRESHOLDS["low"]:
        return "low"
    elif leverage < LEVERAGE_THRESHOLDS["moderate"]:
        return "moderate"
    elif leverage < LEVERAGE_THRESHOLDS["high"]:
        return "high"
    else:
        return "very_high"


def _classify_coverage(coverage: float) -> str:
    """Categorize interest coverage ratio."""
    if coverage >= COVERAGE_THRESHOLDS["strong"]:
        return "strong"
    elif coverage >= COVERAGE_THRESHOLDS["adequate"]:
        return "adequate"
    elif coverage >= COVERAGE_THRESHOLDS["weak"]:
        return "weak"
    else:
        return "distressed"


def _classify_liquidity(current_ratio: float) -> str:
    """Categorize liquidity based on current ratio."""
    if current_ratio >= LIQUIDITY_THRESHOLDS["strong"]:
        return "strong"
    elif current_ratio >= LIQUIDITY_THRESHOLDS["adequate"]:
        return "adequate"
    elif current_ratio >= LIQUIDITY_THRESHOLDS["weak"]:
        return "weak"
    else:
        return "distressed"


def compute_ratios(financials: BorrowerFinancials) -> CreditRatios:
    """
    Main entry point: takes borrower financials, returns computed ratios.
    
    This function is fully deterministic — same input always produces same output.
    No randomness, no LLM, no external calls. This is what makes it auditable.
    """
    
    # 1. Leverage = Total Debt / EBITDA
    # Industry interpretation: how many years of EBITDA would it take to repay all debt?
    # < 3x = conservative; 3-5x = moderate; > 5x = aggressive (LBO territory).
    leverage = _safe_divide(financials.total_debt, financials.ebitda)
    
    # 2. Interest Coverage = EBITDA / Interest Expense
    # How many times can we cover our interest payments? Higher = safer.
    # Banks often require >2x as a minimum loan covenant.
    coverage = _safe_divide(financials.ebitda, financials.interest_expense)
    
    # 3. Current Ratio = Current Assets / Current Liabilities
    # Can short-term assets cover short-term obligations? > 1 healthy.
    current_ratio = _safe_divide(financials.current_assets, financials.current_liabilities)
    
    # 4. Quick Ratio (or "acid test") — same as current ratio but excluding inventory.
    # We don't have inventory in our schema, so we approximate by subtracting an
    # estimate of inventory (~30% of current assets is a typical ballpark for 
    # non-financial companies — in a real system you'd have actual inventory).
    estimated_inventory = financials.current_assets * 0.3
    quick_ratio = _safe_divide(
        financials.current_assets - estimated_inventory,
        financials.current_liabilities,
    )
    
    # 5. Debt-to-Equity = Total Debt / Total Equity
    # Capital structure: how much of the company is financed by debt vs equity?
    debt_to_equity = _safe_divide(financials.total_debt, financials.total_equity)
    
    # 6. Cash Ratio = Cash / Current Liabilities
    # The most conservative liquidity measure — what if we had to pay everyone TODAY?
    cash_ratio = _safe_divide(
        financials.cash_and_equivalents,
        financials.current_liabilities,
    )
    
    # 7. EBITDA Margin = EBITDA / Revenue
    # Profitability: how much of every dollar of revenue becomes EBITDA?
    # 15-20% is healthy for most industries; varies a lot.
    ebitda_margin = _safe_divide(financials.ebitda, financials.revenue)
    
    # Now classify the assessments
    leverage_assessment = _classify_leverage(leverage)
    coverage_assessment = _classify_coverage(coverage)
    liquidity_assessment = _classify_liquidity(current_ratio)
    
    return CreditRatios(
        leverage_ratio=round(leverage, 2),
        interest_coverage=round(coverage, 2),
        current_ratio=round(current_ratio, 2),
        quick_ratio=round(quick_ratio, 2),
        debt_to_equity=round(debt_to_equity, 2),
        cash_ratio=round(cash_ratio, 2),
        ebitda_margin=round(ebitda_margin, 4),
        leverage_assessment=leverage_assessment,
        coverage_assessment=coverage_assessment,
        liquidity_assessment=liquidity_assessment,
    )

