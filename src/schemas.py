"""
schemas.py — The data shapes that flow through our agent.

LangGraph agents pass a "state" object between nodes. Each node reads from state
and writes back to state. To prevent bugs, we define the *exact* shape of state
using Pydantic models. This way, if a node tries to write `borrower.name` when
the field is actually `borrower.borrower_name`, Pydantic raises an error
immediately instead of letting a silent bug propagate.


"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# INPUT SCHEMAS — what the user provides
# ============================================================================

class BorrowerFinancials(BaseModel):
    """
    The raw financial data for a borrower.
    All amounts are in millions of USD unless otherwise noted.
    """
    borrower_name: str = Field(description="Legal name of the borrower")
    industry: str = Field(description="Industry sector, e.g., 'Manufacturing', 'Retail', 'Tech'")
    
    # Income statement
    revenue: float = Field(description="Annual revenue ($M)")
    ebitda: float = Field(description="Earnings before interest, taxes, depreciation, amortization ($M)")
    interest_expense: float = Field(description="Annual interest expense ($M)")
    
    # Balance sheet
    total_debt: float = Field(description="Total interest-bearing debt ($M)")
    cash_and_equivalents: float = Field(description="Cash and short-term investments ($M)")
    current_assets: float = Field(description="Current assets ($M)")
    current_liabilities: float = Field(description="Current liabilities ($M)")
    total_equity: float = Field(description="Shareholders' equity ($M)")
    
    # Optional context
    years_in_business: Optional[int] = Field(default=None, description="Years operating")
    public_or_private: Optional[str] = Field(default=None, description="'Public' or 'Private'")


class CreditRequest(BaseModel):
    """The full input to the agent."""
    borrower: BorrowerFinancials
    macro_scenario: str = Field(
        default="baseline",
        description="One of: 'baseline', 'mild_recession', 'severe_stress'"
    )


# ============================================================================
# INTERMEDIATE SCHEMAS — what each node produces
# ============================================================================

class CreditRatios(BaseModel):
    """Computed by the ratio calculator node. Pure math, no LLM."""
    leverage_ratio: float = Field(description="Total Debt / EBITDA. Lower is better. <3 typical.")
    interest_coverage: float = Field(description="EBITDA / Interest Expense. Higher is better. >3 typical.")
    current_ratio: float = Field(description="Current Assets / Current Liabilities. >1 healthy.")
    quick_ratio: float = Field(description="(Current Assets - Inventory) / Current Liabilities. ~1 healthy.")
    debt_to_equity: float = Field(description="Total Debt / Equity. <1 typical for non-financial.")
    cash_ratio: float = Field(description="Cash / Current Liabilities. >0.2 healthy.")
    ebitda_margin: float = Field(description="EBITDA / Revenue, expressed as a decimal (0.15 = 15%).")
    
    # Interpretive flags — set by ratio logic, not LLM
    leverage_assessment: str = Field(description="'low', 'moderate', 'high', or 'very_high'")
    coverage_assessment: str = Field(description="'strong', 'adequate', 'weak', or 'distressed'")
    liquidity_assessment: str = Field(description="'strong', 'adequate', 'weak', or 'distressed'")


class PeerComparison(BaseModel):
    """Output of RAG retrieval — comparable companies and their characteristics."""
    peer_documents: list[str] = Field(description="Raw text of the 3-5 most similar peer companies")
    peer_industries: list[str] = Field(description="Industries of retrieved peers")
    similarity_scores: list[float] = Field(description="Cosine similarity scores (0-1) for each peer")


class PDResult(BaseModel):
    """Output of PD scoring node."""
    baseline_pd: float = Field(description="Probability of default in next 12 months (0.0-1.0)")
    risk_rating: str = Field(description="Letter rating: AAA, AA, A, BBB, BB, B, CCC, CC, C, D")
    rating_explanation: str = Field(description="Brief reason for the rating")


class StressedPD(BaseModel):
    """Output of stress scenario node."""
    scenario: str = Field(description="Which scenario was applied")
    multiplier: float = Field(description="PD multiplier for this scenario (e.g., 2.5 for severe)")
    stressed_pd: float = Field(description="PD adjusted for the scenario")
    stressed_rating: str = Field(description="Rating after stress")


# ============================================================================
# OUTPUT SCHEMA — final memo
# ============================================================================

class CreditMemo(BaseModel):
    """The final output: a structured credit memo."""
    borrower_name: str
    memo_date: str
    macro_scenario: str
    
    executive_summary: str
    financial_analysis: str
    peer_comparison: str
    credit_risk_assessment: str
    stress_scenario_analysis: str
    recommendation: str
    
    # Quantitative facts (NOT generated by LLM — these come straight from compute nodes)
    key_ratios: CreditRatios
    pd_result: PDResult
    stressed_pd_result: StressedPD
    
    def to_markdown(self) -> str:
        """Render the memo as a nicely formatted Markdown document."""
        ratios = self.key_ratios
        return f"""# Credit Memo — {self.borrower_name}

**Date:** {self.memo_date}  
**Scenario:** {self.macro_scenario}  
**Risk Rating:** {self.pd_result.risk_rating}  
**Stressed Rating:** {self.stressed_pd_result.stressed_rating}

---

## Executive Summary
{self.executive_summary}

---

## Financial Analysis
{self.financial_analysis}

### Key Ratios
| Metric | Value | Assessment |
|---|---|---|
| Leverage (Debt/EBITDA) | {ratios.leverage_ratio:.2f}x | {ratios.leverage_assessment} |
| Interest Coverage | {ratios.interest_coverage:.2f}x | {ratios.coverage_assessment} |
| Current Ratio | {ratios.current_ratio:.2f} | {ratios.liquidity_assessment} |
| Quick Ratio | {ratios.quick_ratio:.2f} | — |
| Debt/Equity | {ratios.debt_to_equity:.2f} | — |
| Cash Ratio | {ratios.cash_ratio:.2f} | — |
| EBITDA Margin | {ratios.ebitda_margin:.1%} | — |

---

## Peer Comparison
{self.peer_comparison}

---

## Credit Risk Assessment
{self.credit_risk_assessment}

**Probability of Default (12-month):** {self.pd_result.baseline_pd:.2%}  
**Internal Rating:** {self.pd_result.risk_rating}

---

## Stress Scenario Analysis
{self.stress_scenario_analysis}

**Scenario applied:** {self.stressed_pd_result.scenario}  
**PD multiplier:** {self.stressed_pd_result.multiplier}x  
**Stressed PD:** {self.stressed_pd_result.stressed_pd:.2%}  
**Stressed Rating:** {self.stressed_pd_result.stressed_rating}

---

## Recommendation
{self.recommendation}

---

*Generated by Credit Memo Generator Agent. This is a portfolio project — not a production credit decision.*
"""


# ============================================================================
# AGENT STATE — the dictionary that LangGraph passes between nodes
# ============================================================================

from typing import TypedDict


class AgentState(TypedDict):
    """
    The shared state passed between LangGraph nodes.
    
    WHY TypedDict instead of Pydantic here? LangGraph is designed around
    TypedDict for state. It allows partial updates (a node returns a dict
    with only the fields it modifies, and LangGraph merges it in).
    
    Pydantic models are still used for validated data INSIDE this state.
    """
    # Inputs
    request: CreditRequest
    
    # Filled in by nodes as the agent runs
    ratios: Optional[CreditRatios]
    peers: Optional[PeerComparison]
    pd_result: Optional[PDResult]
    stressed_pd: Optional[StressedPD]
    
    # Final output
    memo: Optional[CreditMemo]
    
    # Errors (if any)
    errors: list[str]
