"""
pd_scorer.py — Loads the trained PD model and scores a borrower.

This module is called by the agent's PD scoring node. It uses the model that
train_pd_model.py produced.
"""

import os
import pickle
import pandas as pd

from src.schemas import CreditRatios, PDResult


MODEL_PATH = "models/pd_model.pkl"


# PD-to-rating mapping (simplified S&P-style ratings).
# In a real bank, this is calibrated annually against actual default frequencies.
RATING_BANDS = [
    # (max_pd, rating, explanation)
    (0.001, "AAA", "Extremely strong capacity to meet obligations"),
    (0.005, "AA",  "Very strong capacity"),
    (0.015, "A",   "Strong capacity, somewhat susceptible to adverse conditions"),
    (0.04,  "BBB", "Adequate capacity, but more susceptible to adverse conditions"),
    (0.10,  "BB",  "Less vulnerable in near-term, but faces ongoing uncertainties"),
    (0.20,  "B",   "More vulnerable, but currently has capacity to meet obligations"),
    (0.40,  "CCC", "Currently vulnerable, depends on favorable conditions"),
    (0.60,  "CC",  "Highly vulnerable"),
    (0.85,  "C",   "Highly vulnerable, near default"),
    (1.01,  "D",   "Default or near-default"),
]


def _pd_to_rating(pd: float) -> tuple[str, str]:
    """Map a PD value to a letter rating and explanation."""
    for max_pd, rating, explanation in RATING_BANDS:
        if pd < max_pd:
            return rating, explanation
    return "D", "Default"  # fallback (shouldn't reach here)


_cached_model = None


def _load_model():
    """Load the model lazily and cache it."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"PD model not found at {MODEL_PATH}. "
                f"Run: python -m src.tools.train_pd_model"
            )
        with open(MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
    return _cached_model


def score_borrower(ratios: CreditRatios) -> PDResult:
    """
    Predict the 12-month probability of default for a borrower.
    
    Inputs: pre-computed ratios (from ratio_calculator).
    Output: PDResult with probability and risk rating.
    """
    model = _load_model()
    
    # Build feature vector in the EXACT same order the model was trained on.
    # If you change feature order, the model will produce nonsense.
    features = pd.DataFrame([{
        "leverage_ratio": ratios.leverage_ratio,
        "interest_coverage": ratios.interest_coverage,
        "current_ratio": ratios.current_ratio,
        "debt_to_equity": ratios.debt_to_equity,
        "ebitda_margin": ratios.ebitda_margin,
    }])
    
    # Predict probability of class 1 (defaulted)
    pd_value = float(model.predict_proba(features)[0, 1])
    
    # Map to rating
    rating, explanation = _pd_to_rating(pd_value)
    
    return PDResult(
        baseline_pd=pd_value,
        risk_rating=rating,
        rating_explanation=explanation,
    )


