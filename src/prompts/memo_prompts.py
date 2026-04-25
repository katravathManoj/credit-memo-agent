"""
memo_prompts.py — The prompts we send to the LLM to generate memo sections.

WHY ARE PROMPTS A SEPARATE FILE?
1. Prompts are the "soul" of an LLM application. Tweaking them is the main 
   way you improve output quality. Keeping them separate makes iteration easy.
2. A reviewer can see exactly what we're asking the LLM, without digging 
   through agent code.
3. Production teams version-control prompts as separate artifacts, just like 
   model weights.

ON PROMPT DESIGN:
We use a structured approach: clear role, explicit constraints, structured
output specification. The LLM ONLY narrates what we computed; it does NOT 
invent numbers.
"""


SYSTEM_PROMPT = """You are a senior credit analyst at a major US commercial bank with 15 years 
of experience writing credit memos for wholesale lending decisions.

Your role: synthesize quantitative analysis into a professional credit memo.

CRITICAL RULES — VIOLATING ANY OF THESE IS A FAILURE:
1. You may ONLY use numbers and facts that appear in the data provided to you.
   NEVER invent, estimate, or extrapolate numbers that aren't in the data.
2. Your tone is professional, balanced, and analytical. No marketing language.
3. You write in past/present tense, declarative sentences. No future speculation 
   beyond what the data supports.
4. When summarizing peer comparisons, refer to peers by their general profile 
   ("a similar mid-cap manufacturing peer..."), not invented names.
5. If the data is concerning, say so plainly. Do not soften credit risk.
6. If the data is strong, say so plainly. Do not artificially balance with 
   speculative concerns.

You will be given:
- Borrower's raw financial data
- Computed credit ratios with assessment categories
- Comparable peer company descriptions (from a database)
- Probability of default and risk rating
- Stress scenario results

You will produce specific sections of a credit memo as plain text (NOT markdown,
NOT JSON — just professional prose).
"""


# Each function below builds the user prompt for one memo section.
# We generate sections one at a time so each prompt is small and focused.
# (Alternative would be one giant prompt for the whole memo — possible but 
# harder to debug, and uses more tokens.)


def build_executive_summary_prompt(state: dict) -> str:
    """Build prompt for executive summary section."""
    request = state["request"]
    ratios = state["ratios"]
    pd_result = state["pd_result"]
    stressed = state["stressed_pd"]
    
    return f"""Write the EXECUTIVE SUMMARY section of a credit memo.

BORROWER: {request.borrower.borrower_name} ({request.borrower.industry})
Revenue: ${request.borrower.revenue:.0f}M | EBITDA: ${request.borrower.ebitda:.0f}M | Total Debt: ${request.borrower.total_debt:.0f}M

KEY METRICS:
- Leverage (Debt/EBITDA): {ratios.leverage_ratio:.2f}x — {ratios.leverage_assessment}
- Interest Coverage: {ratios.interest_coverage:.2f}x — {ratios.coverage_assessment}
- Liquidity (Current Ratio): {ratios.current_ratio:.2f} — {ratios.liquidity_assessment}
- 12-month PD: {pd_result.baseline_pd:.2%}
- Internal Rating: {pd_result.risk_rating}
- Stressed PD ({stressed.scenario}): {stressed.stressed_pd:.2%}, Rating: {stressed.stressed_rating}

INSTRUCTIONS:
- Write 3-5 sentences.
- State the borrower, industry, and overall credit position.
- Mention the rating and the most material strengths/weaknesses.
- Reference the stress scenario impact briefly.
- Keep it dense and analytical — no fluff.
- Output plain prose, no headers, no markdown.
"""


def build_financial_analysis_prompt(state: dict) -> str:
    """Build prompt for financial analysis section."""
    request = state["request"]
    ratios = state["ratios"]
    
    return f"""Write the FINANCIAL ANALYSIS section of a credit memo.

BORROWER FINANCIALS (in $M):
- Revenue: {request.borrower.revenue:.0f}
- EBITDA: {request.borrower.ebitda:.0f} (margin: {ratios.ebitda_margin:.1%})
- Interest Expense: {request.borrower.interest_expense:.0f}
- Total Debt: {request.borrower.total_debt:.0f}
- Cash: {request.borrower.cash_and_equivalents:.0f}
- Current Assets: {request.borrower.current_assets:.0f}
- Current Liabilities: {request.borrower.current_liabilities:.0f}
- Total Equity: {request.borrower.total_equity:.0f}

COMPUTED RATIOS:
- Leverage: {ratios.leverage_ratio:.2f}x ({ratios.leverage_assessment})
- Interest Coverage: {ratios.interest_coverage:.2f}x ({ratios.coverage_assessment})
- Current Ratio: {ratios.current_ratio:.2f} ({ratios.liquidity_assessment})
- Quick Ratio: {ratios.quick_ratio:.2f}
- Debt/Equity: {ratios.debt_to_equity:.2f}
- Cash Ratio: {ratios.cash_ratio:.2f}

INSTRUCTIONS:
- Write 2-3 paragraphs (5-8 sentences total).
- Discuss leverage profile, debt service capacity, and balance sheet liquidity.
- Reference SPECIFIC numbers from the data above; do not invent any others.
- Highlight any metrics that are notably strong or weak compared to typical bank 
  thresholds (e.g., leverage <3x is conservative; >5x is aggressive; coverage 
  >5x is strong; <2x is concerning; current ratio >1.5 is healthy).
- Output plain prose, no headers, no markdown, no bullet points.
"""


def build_peer_comparison_prompt(state: dict) -> str:
    """Build prompt for peer comparison section using RAG output."""
    request = state["request"]
    ratios = state["ratios"]
    peers = state["peers"]
    
    peer_section = "\n".join(
        f"- Peer {i+1} ({peers.peer_industries[i]}, similarity: {peers.similarity_scores[i]:.2f}):\n  {peers.peer_documents[i]}"
        for i in range(len(peers.peer_documents))
    )
    
    return f"""Write the PEER COMPARISON section of a credit memo.

BORROWER PROFILE:
- Industry: {request.borrower.industry}
- Revenue: ${request.borrower.revenue:.0f}M
- EBITDA Margin: {ratios.ebitda_margin:.1%}
- Leverage: {ratios.leverage_ratio:.2f}x

RETRIEVED PEER COMPANIES:
{peer_section}

INSTRUCTIONS:
- Write 2-3 paragraphs (4-6 sentences total).
- Compare the borrower's leverage, margins, and credit profile to the peer set.
- Refer to peers generically ("a comparable mid-market manufacturing peer", 
  "another company in the industry"), NOT by inventing names.
- Note where the borrower is in line with peers vs. where they're an outlier.
- Output plain prose, no headers, no markdown, no bullets.
"""


def build_credit_risk_assessment_prompt(state: dict) -> str:
    """Build prompt for credit risk assessment section."""
    pd_result = state["pd_result"]
    ratios = state["ratios"]
    
    return f"""Write the CREDIT RISK ASSESSMENT section of a credit memo.

QUANTITATIVE OUTPUT:
- 12-month Probability of Default: {pd_result.baseline_pd:.2%}
- Internal Risk Rating: {pd_result.risk_rating}
- Rating Definition: {pd_result.rating_explanation}

KEY DRIVERS OF THE RATING:
- Leverage: {ratios.leverage_ratio:.2f}x ({ratios.leverage_assessment})
- Interest Coverage: {ratios.interest_coverage:.2f}x ({ratios.coverage_assessment})
- Liquidity (Current Ratio): {ratios.current_ratio:.2f} ({ratios.liquidity_assessment})
- EBITDA Margin: {ratios.ebitda_margin:.1%}

INSTRUCTIONS:
- Write 2 paragraphs (4-6 sentences).
- State the rating clearly with the PD.
- Explain the main drivers of the rating — which ratios contributed positively 
  vs. negatively.
- Note any specific risk factors implied by the metric assessments.
- Output plain prose, no headers, no markdown.
"""


def build_stress_analysis_prompt(state: dict) -> str:
    """Build prompt for stress scenario analysis section."""
    pd_result = state["pd_result"]
    stressed = state["stressed_pd"]
    
    scenario_descriptions = {
        "baseline": "current macroeconomic conditions",
        "mild_recession": "a mild recession scenario, similar to a 1-2% GDP contraction",
        "severe_stress": "a severe stress scenario similar to 2008-09 or COVID-onset, with significant GDP contraction and elevated unemployment",
    }
    scenario_desc = scenario_descriptions.get(stressed.scenario, "an adverse scenario")
    
    return f"""Write the STRESS SCENARIO ANALYSIS section of a credit memo.

BASELINE:
- PD: {pd_result.baseline_pd:.2%}
- Rating: {pd_result.risk_rating}

STRESSED SCENARIO ({stressed.scenario}):
- Description: {scenario_desc}
- Multiplier applied: {stressed.multiplier}x baseline PD
- Stressed PD: {stressed.stressed_pd:.2%}
- Stressed Rating: {stressed.stressed_rating}

INSTRUCTIONS:
- Write 2-3 sentences.
- Describe what the scenario represents and what it means for this borrower.
- State how much the rating migrates (if at all).
- Note implications for the bank's loan loss reserves or risk-weighted capital.
- Output plain prose, no headers, no markdown.
"""


def build_recommendation_prompt(state: dict) -> str:
    """Build prompt for the final recommendation."""
    request = state["request"]
    ratios = state["ratios"]
    pd_result = state["pd_result"]
    stressed = state["stressed_pd"]
    
    return f"""Write the RECOMMENDATION section of a credit memo.

BORROWER: {request.borrower.borrower_name}
RATING: {pd_result.risk_rating} (baseline) / {stressed.stressed_rating} (stressed under {stressed.scenario})
KEY METRICS:
- Leverage: {ratios.leverage_ratio:.2f}x ({ratios.leverage_assessment})
- Coverage: {ratios.interest_coverage:.2f}x ({ratios.coverage_assessment})
- Liquidity: {ratios.current_ratio:.2f} ({ratios.liquidity_assessment})

INSTRUCTIONS:
- Write 3-4 sentences.
- Provide a clear recommendation: APPROVE / APPROVE WITH CONDITIONS / DECLINE / 
  REFER TO COMMITTEE.
- Justify the recommendation in 1-2 sentences referencing the metrics.
- If conditions are warranted, propose 1-2 specific covenants 
  (e.g., "maintain Debt/EBITDA below 4.0x", "minimum cash balance of $50M").
- Output plain prose, no headers, no markdown.
"""
