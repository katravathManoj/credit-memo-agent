"""
stress_adjuster.py — Adjusts PD based on macro stress scenario.

This is the "your BoA work in code" piece. In ILST and CCAR frameworks, banks
quantify how PDs shift under adverse macro conditions. The simplest form:
multiply baseline PD by a scenario-specific factor.

In real banks, the multipliers come from regression models trained on historical
data linking macro variables (GDP, unemployment, equity returns) to default rates.
We use simplified, illustrative factors here.
"""

from src.schemas import PDResult, StressedPD
from src.tools.pd_scorer import _pd_to_rating


# Scenario-specific PD multipliers.
# In a real CCAR/DFAST exercise, these would be derived from macro regression models.
SCENARIO_MULTIPLIERS = {
    "baseline": 1.0,
    "mild_recession": 1.8,    # ~80% increase in default rate
    "severe_stress": 3.0,     # ~3x increase, similar to 2008-style stress
}


def apply_stress_scenario(pd_result: PDResult, scenario: str) -> StressedPD:
    """
    Apply a macro stress scenario to a baseline PD.
    
    Returns the stressed PD (capped at 0.95 to avoid edge cases).
    """
    if scenario not in SCENARIO_MULTIPLIERS:
        # Default to baseline if unknown scenario passed
        scenario = "baseline"
    
    multiplier = SCENARIO_MULTIPLIERS[scenario]
    stressed_pd = min(pd_result.baseline_pd * multiplier, 0.95)
    
    stressed_rating, _ = _pd_to_rating(stressed_pd)
    
    return StressedPD(
        scenario=scenario,
        multiplier=multiplier,
        stressed_pd=stressed_pd,
        stressed_rating=stressed_rating,
    )
