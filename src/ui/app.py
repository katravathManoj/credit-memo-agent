"""
app.py — Streamlit web UI for the credit memo agent.

Run with:
    streamlit run src/ui/app.py

Opens at http://localhost:8501
"""

import json
import os
import sys
from pathlib import Path

# Make project root importable when Streamlit runs this file directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.schemas import BorrowerFinancials, CreditRequest
from src.agents.graph import run_agent


# Page config
st.set_page_config(
    page_title="Credit Memo Agent",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Credit Memo Generator")
st.caption("Agentic AI system • LangGraph + Gemini 2.0 Flash • Synthetic data demo")

# Sidebar: explanation
with st.sidebar:
    st.markdown("### How it works")
    st.markdown("""
1. **Parse** — validates borrower input  
2. **Ratios** — computes credit ratios (Python, deterministic)  
3. **Peers** — RAG retrieval of comparable companies (ChromaDB)  
4. **PD Score** — runs trained logistic regression  
5. **Stress** — applies macro scenario adjustment  
6. **Memo** — Gemini synthesizes the narrative

**Architectural note:** numerical computation happens in Python; the LLM only writes prose. This is the same separation real banks use for explainable AI.
""")
    st.markdown("---")
    st.markdown("**Limitations:** The PD model is trained on synthetic data. This is a portfolio demonstration of agentic AI patterns, not a production credit decision system.")


# Two columns: input form, output
col_input, col_output = st.columns([1, 1.4])


with col_input:
    st.subheader("Borrower Input")
    
    # Sample loader
    sample_dir = "data/sample_borrowers"
    samples = ["(custom)"]
    if os.path.exists(sample_dir):
        samples += sorted([f for f in os.listdir(sample_dir) if f.endswith(".json")])
    
    selected_sample = st.selectbox("Load sample borrower:", samples, index=1 if len(samples) > 1 else 0)
    
    # Default values
    defaults = {
        "borrower_name": "Acme Manufacturing Co.",
        "industry": "Manufacturing",
        "revenue": 950.0,
        "ebitda": 175.0,
        "interest_expense": 18.0,
        "total_debt": 380.0,
        "cash_and_equivalents": 95.0,
        "current_assets": 320.0,
        "current_liabilities": 165.0,
        "total_equity": 540.0,
    }
    
    if selected_sample != "(custom)":
        with open(os.path.join(sample_dir, selected_sample)) as f:
            defaults.update(json.load(f))
    
    # Input fields
    borrower_name = st.text_input("Borrower Name", defaults["borrower_name"])
    industry = st.text_input("Industry", defaults["industry"])
    
    st.markdown("**Financials ($M):**")
    c1, c2 = st.columns(2)
    with c1:
        revenue = st.number_input("Revenue", min_value=0.0, value=float(defaults["revenue"]))
        ebitda = st.number_input("EBITDA", value=float(defaults["ebitda"]))
        interest_expense = st.number_input("Interest Expense", min_value=0.0, value=float(defaults["interest_expense"]))
        total_debt = st.number_input("Total Debt", min_value=0.0, value=float(defaults["total_debt"]))
    with c2:
        cash = st.number_input("Cash", min_value=0.0, value=float(defaults["cash_and_equivalents"]))
        current_assets = st.number_input("Current Assets", min_value=0.0, value=float(defaults["current_assets"]))
        current_liabilities = st.number_input("Current Liabilities", min_value=0.0, value=float(defaults["current_liabilities"]))
        total_equity = st.number_input("Total Equity", value=float(defaults["total_equity"]))
    
    scenario = st.selectbox(
        "Macro Scenario",
        ["baseline", "mild_recession", "severe_stress"],
        help="Stress scenario for PD adjustment. 'severe_stress' applies a 3x multiplier to baseline PD.",
    )
    
    generate = st.button("Generate Credit Memo", type="primary", use_container_width=True)


with col_output:
    st.subheader("Generated Memo")
    
    if generate:
        # Build request
        try:
            borrower = BorrowerFinancials(
                borrower_name=borrower_name,
                industry=industry,
                revenue=revenue,
                ebitda=ebitda,
                interest_expense=interest_expense,
                total_debt=total_debt,
                cash_and_equivalents=cash,
                current_assets=current_assets,
                current_liabilities=current_liabilities,
                total_equity=total_equity,
            )
            request = CreditRequest(borrower=borrower, macro_scenario=scenario)
        except Exception as e:
            st.error(f"Input validation error: {e}")
            st.stop()
        
        # Run agent with progress
        with st.status("Running agent...", expanded=True) as status:
            st.write("⏳ Computing ratios...")
            st.write("⏳ Retrieving peer companies (RAG)...")
            st.write("⏳ Scoring PD...")
            st.write("⏳ Applying stress scenario...")
            st.write("⏳ Generating memo with Gemini (6 LLM calls)...")
            
            try:
                result = run_agent(request)
                
                if result.get("errors"):
                    st.error("Errors during execution: " + "; ".join(result["errors"]))
                    st.stop()
                
                status.update(label="✅ Memo generated", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Agent failed: {e}")
                st.stop()
        
        # Display ratios as metrics first
        ratios = result["ratios"]
        pd_result = result["pd_result"]
        stressed = result["stressed_pd"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Leverage", f"{ratios.leverage_ratio:.2f}x", delta=ratios.leverage_assessment, delta_color="off")
        m2.metric("Coverage", f"{ratios.interest_coverage:.2f}x", delta=ratios.coverage_assessment, delta_color="off")
        m3.metric("Rating", pd_result.risk_rating, delta=f"PD {pd_result.baseline_pd:.1%}", delta_color="off")
        m4.metric("Stressed", stressed.stressed_rating, delta=f"PD {stressed.stressed_pd:.1%}", delta_color="off")
        
        st.markdown("---")
        
        # The memo itself
        memo_md = result["memo"].to_markdown()
        st.markdown(memo_md)
        
        # Download button
        st.download_button(
            "Download as Markdown",
            data=memo_md,
            file_name=f"credit_memo_{borrower_name.replace(' ', '_')}.md",
            mime="text/markdown",
        )
    else:
        st.info("Fill in the borrower data on the left and click **Generate Credit Memo**.")
