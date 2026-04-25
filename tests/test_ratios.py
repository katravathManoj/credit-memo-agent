"""
Unit tests for ratio_calculator. Run with: pytest tests/

These tests verify our deterministic math is correct. They run without
the LLM, ChromaDB, or any network calls — fast and reliable.
"""

import pytest

from src.schemas import BorrowerFinancials
from src.tools.ratio_calculator import compute_ratios, _safe_divide


class TestSafeDivide:
    def test_normal_division(self):
        assert _safe_divide(10, 2) == 5.0
    
    def test_zero_denominator_with_positive_numerator(self):
        # Should return a "very large" value, not crash
        assert _safe_divide(10, 0) == 999.99
    
    def test_zero_denominator_with_zero_numerator(self):
        assert _safe_divide(0, 0) == 0.0


class TestComputeRatios:
    def make_borrower(self, **overrides) -> BorrowerFinancials:
        """Helper: build a borrower with defaults that can be overridden."""
        defaults = {
            "borrower_name": "Test Co",
            "industry": "Manufacturing",
            "revenue": 1000.0,
            "ebitda": 200.0,
            "interest_expense": 20.0,
            "total_debt": 400.0,
            "cash_and_equivalents": 100.0,
            "current_assets": 300.0,
            "current_liabilities": 150.0,
            "total_equity": 600.0,
        }
        defaults.update(overrides)
        return BorrowerFinancials(**defaults)
    
    def test_healthy_company_ratios(self):
        """A healthy company should have moderate leverage and strong coverage."""
        # Use values that fall cleanly within bands (avoid boundary ambiguity)
        borrower = self.make_borrower(
            ebitda=250.0,    # 400/250 = 1.6 leverage → "low" (< 2.0)
            current_assets=400.0,  # 400/150 = 2.67 → "strong" (> 2.0)
        )
        ratios = compute_ratios(borrower)
        
        # 400 / 250 = 1.6
        assert ratios.leverage_ratio == 1.6
        # 250 / 20 = 12.5 (above strong threshold of 6.0)
        assert ratios.interest_coverage == 12.5
        # 400 / 150 = 2.67
        assert ratios.current_ratio == 2.67
        
        # Assessments
        assert ratios.leverage_assessment == "low"
        assert ratios.coverage_assessment == "strong"
        assert ratios.liquidity_assessment == "strong"
    
    def test_distressed_company_ratios(self):
        borrower = self.make_borrower(
            ebitda=30.0,
            interest_expense=40.0,
            total_debt=600.0,
            current_assets=80.0,
            current_liabilities=100.0,
        )
        ratios = compute_ratios(borrower)
        
        assert ratios.leverage_ratio == 20.0
        assert ratios.interest_coverage == 0.75
        assert ratios.current_ratio == 0.8
        
        assert ratios.leverage_assessment == "very_high"
        assert ratios.coverage_assessment == "distressed"
        assert ratios.liquidity_assessment == "distressed"
    
    def test_zero_interest_handled_safely(self):
        """A company with no interest expense shouldn't crash."""
        borrower = self.make_borrower(interest_expense=0.0)
        ratios = compute_ratios(borrower)
        
        # Should be very high (effectively infinite coverage)
        assert ratios.interest_coverage > 100
    
    def test_negative_ebitda(self):
        """Negative EBITDA should still produce valid ratios (just very bad ones)."""
        borrower = self.make_borrower(ebitda=-50.0)
        ratios = compute_ratios(borrower)
        
        # Negative leverage doesn't make sense semantically but math should work
        assert ratios.leverage_ratio == -8.0
