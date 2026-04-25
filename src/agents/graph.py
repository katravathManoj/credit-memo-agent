"""
graph.py — The LangGraph agent definition.

This is the heart of the project. It defines the nodes (functions that do work)
and wires them together into an executable graph.

THE FLOW:
   parse → ratios → peers → pd_score → stress → memo

Each step is a node. State (a TypedDict) flows between them.


"""

import os
import datetime
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.schemas import (
    AgentState,
    CreditMemo,
)
from src.tools.ratio_calculator import compute_ratios
from src.tools.peer_retriever import retrieve_peers
from src.tools.pd_scorer import score_borrower
from src.tools.stress_adjuster import apply_stress_scenario
from src.prompts.memo_prompts import (
    SYSTEM_PROMPT,
    build_executive_summary_prompt,
    build_financial_analysis_prompt,
    build_peer_comparison_prompt,
    build_credit_risk_assessment_prompt,
    build_stress_analysis_prompt,
    build_recommendation_prompt,
)


# Load .env so GOOGLE_API_KEY is available
load_dotenv()


# Cached LLM instance
_llm = None


def _get_llm():
    """Lazy-load the LLM. Cached after first call."""
    global _llm
    if _llm is None:
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError(
                "GROQ_API_KEY not found. Did you create a .env file? "
                "See docs/04_setup_guide.md."
            )
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,  # Low temp = more deterministic, important for finance
            max_tokens=2048,
        )
    return _llm


def _llm_call(user_prompt: str) -> str:
    """Send a prompt to Gemini, return the text response."""
    llm = _get_llm()
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ============================================================================
# NODES — each is a function that takes state and returns (partial) state
# ============================================================================
# LangGraph convention: a node returns a dict with the keys it modified.
# LangGraph merges that dict back into the state.


def parse_node(state: AgentState) -> dict:
    """
    Step 1: Validate input.
    
    The Pydantic model already validates structure when CreditRequest is built.
    Here we just confirm the input is present and add an empty errors list.
    """
    if state.get("request") is None:
        return {"errors": ["No credit request provided"]}
    return {"errors": []}


def ratios_node(state: AgentState) -> dict:
    """
    Step 2: Compute deterministic credit ratios. NO LLM call.
    """
    try:
        ratios = compute_ratios(state["request"].borrower)
        return {"ratios": ratios}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"ratios_node failed: {e}"]}


def peers_node(state: AgentState) -> dict:
    """
    Step 3: RAG — retrieve peer companies similar to the borrower.
    """
    try:
        peers = retrieve_peers(
            financials=state["request"].borrower,
            ratios=state["ratios"],
            n=4,
        )
        return {"peers": peers}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"peers_node failed: {e}"]}


def pd_score_node(state: AgentState) -> dict:
    """
    Step 4: Run the PD model on the computed ratios. NO LLM call.
    """
    try:
        pd_result = score_borrower(state["ratios"])
        return {"pd_result": pd_result}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"pd_score_node failed: {e}"]}


def stress_node(state: AgentState) -> dict:
    """
    Step 5: Apply macro stress scenario adjustment. NO LLM call.
    """
    try:
        stressed = apply_stress_scenario(
            pd_result=state["pd_result"],
            scenario=state["request"].macro_scenario,
        )
        return {"stressed_pd": stressed}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"stress_node failed: {e}"]}


def memo_node(state: AgentState) -> dict:
    """
    Step 6: Generate the credit memo using the LLM.
    
    We make 6 LLM calls — one per memo section. Why? Each prompt is small
    and focused, leading to better quality than one giant prompt. Trade-off:
    we use more tokens overall.
    
    In production, you'd async-batch these or use a single structured-output 
    call to write everything at once.
    """
    try:
        # Section 1: Executive Summary
        exec_summary = _llm_call(build_executive_summary_prompt(state))
        
        # Section 2: Financial Analysis
        financial_analysis = _llm_call(build_financial_analysis_prompt(state))
        
        # Section 3: Peer Comparison
        peer_comparison = _llm_call(build_peer_comparison_prompt(state))
        
        # Section 4: Credit Risk Assessment
        credit_assessment = _llm_call(build_credit_risk_assessment_prompt(state))
        
        # Section 5: Stress Analysis
        stress_analysis = _llm_call(build_stress_analysis_prompt(state))
        
        # Section 6: Recommendation
        recommendation = _llm_call(build_recommendation_prompt(state))
        
        # Assemble the final memo (numbers come straight from compute nodes — 
        # the LLM never invents them)
        memo = CreditMemo(
            borrower_name=state["request"].borrower.borrower_name,
            memo_date=datetime.date.today().isoformat(),
            macro_scenario=state["request"].macro_scenario,
            executive_summary=exec_summary,
            financial_analysis=financial_analysis,
            peer_comparison=peer_comparison,
            credit_risk_assessment=credit_assessment,
            stress_scenario_analysis=stress_analysis,
            recommendation=recommendation,
            key_ratios=state["ratios"],
            pd_result=state["pd_result"],
            stressed_pd_result=state["stressed_pd"],
        )
        
        return {"memo": memo}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"memo_node failed: {e}"]}


# ============================================================================
# GRAPH ASSEMBLY
# ============================================================================

def build_graph():
    """
    Wire the nodes into a LangGraph StateGraph and compile it.
    
    Flow: parse → ratios → peers → pd_score → stress → memo → END
    Linear flow. Each node has exactly one successor.
    
    Note: node names use a "_step" suffix to avoid collisions with state keys.
    LangGraph treats node names and state keys in the same namespace.
    """
    workflow = StateGraph(AgentState)
    
    # Register nodes (note _step suffix to avoid state-key collisions)
    workflow.add_node("parse_step", parse_node)
    workflow.add_node("ratios_step", ratios_node)
    workflow.add_node("peers_step", peers_node)
    workflow.add_node("pd_score_step", pd_score_node)
    workflow.add_node("stress_step", stress_node)
    workflow.add_node("memo_step", memo_node)
    
    # Set entry point
    workflow.set_entry_point("parse_step")
    
    # Wire the edges (linear flow)
    workflow.add_edge("parse_step", "ratios_step")
    workflow.add_edge("ratios_step", "peers_step")
    workflow.add_edge("peers_step", "pd_score_step")
    workflow.add_edge("pd_score_step", "stress_step")
    workflow.add_edge("stress_step", "memo_step")
    workflow.add_edge("memo_step", END)
    
    return workflow.compile()


# Build once, at module import. Cached for the lifetime of the process.
GRAPH = build_graph()


def run_agent(request) -> AgentState:
    """
    Run the full agent on a credit request and return final state.
    """
    initial_state: AgentState = {
        "request": request,
        "ratios": None,
        "peers": None,
        "pd_result": None,
        "stressed_pd": None,
        "memo": None,
        "errors": [],
    }
    final_state = GRAPH.invoke(initial_state)
    return final_state

