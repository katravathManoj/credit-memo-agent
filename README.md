# Credit Memo Generator Agent

> Agentic AI system for automated credit memo generation. **LangGraph + Groq + RAG + scikit-learn**.

This project demonstrates an end-to-end agentic AI pipeline applied to a real banking workflow: writing wholesale credit memos. Given a borrower's financials and a macro scenario, the agent computes credit ratios, retrieves comparable companies via RAG, runs a probability-of-default model, applies stress adjustments, and synthesizes a structured credit memo using an LLM.

According to McKinsey's 2025 research, US banks deploying agents for credit memos have seen **20-60% productivity gains** and **30% faster credit turnaround**. This project illustrates the engineering pattern behind that use case.

## Architecture

```
Input ──► Parse ──► Compute Ratios ──► RAG (Peers) ──► PD Score ──► Stress Test ──► LLM Memo ──► Output
                       │                  │              │              │              │
                       ▼                  ▼              ▼              ▼              ▼
                   Pure Python         ChromaDB +    sklearn         Python         llama-3.3-70b-versatile
                   (deterministic)     sentence-     logistic        (multiplier)    (synthesis only)
                                       transformers  regression
```

### Key design decision: structured math + LLM narration

Numerical computation (ratios, PD, stress adjustment) happens in deterministic Python. The LLM only synthesizes prose around the numbers we computed. This is the same separation banks use for explainable AI under model risk frameworks like SR 11-7.

## Quickstart

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate    # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Add your free Groq API key
cp .env.example .env
# edit .env and add your Groq_API_KEY(https://console.groq.com)

# 3. Verify setup
python tests/test_setup.py

# 4. Build the RAG corpus and train the PD model (one-time)
python -m src.tools.build_corpus
python -m src.tools.train_pd_model

# 5. Run the agent on a sample borrower
python main.py --borrower data/sample_borrowers/healthy_company.json --scenario baseline

# Or launch the Streamlit UI
streamlit run src/ui/app.py
```

## Sample output

The agent produces a Markdown credit memo with sections:
- Executive Summary
- Financial Analysis (with computed ratios table)
- Peer Comparison (RAG-retrieved)
- Credit Risk Assessment (with PD and rating)
- Stress Scenario Analysis
- Recommendation

Every number in the memo comes from deterministic computation — the LLM cannot invent figures.

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Agent framework | **LangGraph** | Graph-based orchestration, more auditable than ReAct loops — important for finance |
| LLM | **llama-3.3-70b-versatile** | Free tier, fast, structured output support |
| Embeddings | **sentence-transformers** (all-MiniLM-L6-v2) | Local, free, fast |
| Vector DB | **ChromaDB** (local persistent) | No infrastructure required for demo scale |
| ML | **scikit-learn** logistic regression | Industry standard for PD scoring; explainable |
| Validation | **Pydantic** | Type safety, runtime validation |
| UI | **Streamlit** | Fastest path to a clean demo interface |

## Repo structure

```
credit_memo_agent/
├── src/
│   ├── schemas.py                 ← Pydantic data models
│   ├── agents/graph.py            ← LangGraph definition
│   ├── tools/
│   │   ├── ratio_calculator.py    ← deterministic ratio math
│   │   ├── pd_scorer.py           ← logistic regression PD model
│   │   ├── stress_adjuster.py     ← scenario-based PD adjustment
│   │   ├── peer_retriever.py      ← RAG retrieval
│   │   ├── train_pd_model.py      ← trains the PD model (run once)
│   │   └── build_corpus.py        ← builds RAG corpus (run once)
│   ├── prompts/memo_prompts.py    ← LLM prompts
│   ├── evals/eval_harness.py      ← evaluation suite
│   └── ui/app.py                  ← Streamlit UI
├── data/sample_borrowers/         ← sample input JSONs
├── chroma_db/                     ← database
├── models/                        ← trained generated model
├── tests/                         ← unit tests + smoke test
├── main.py                        ← CLI entry point
├── requirements.txt
```

## Limitations

**What it is:** a faithful, working demonstration of the engineering pattern that banks are deploying for credit memo generation. The architecture (deterministic compute + LLM synthesis), the use of RAG for peer retrieval, and the stress-testing methodology are all genuinely representative of production patterns.

**What it isn't:** a production-grade credit risk system. Specifically:

1. **The PD model is trained on synthetic data.** It demonstrates the pattern of integrating an ML model into an agent pipeline, but its predictions have no real-world predictive validity. A production model would be trained on years of real loan performance data with full SR 11-7 model risk management.

2. **The RAG corpus uses 30 synthetic peer companies.** A real bank would ingest tens of thousands of company descriptions from sources like S&P Capital IQ or 10-K filings.

3. **The macro stress multipliers are illustrative.** Real CCAR/DFAST exercises derive PD shifts from regression models linking macro variables (GDP, unemployment, equity returns) to default rates.

4. **No human-in-the-loop or audit trail.** Production deployments would require explicit review steps and full decision logging.

## Future work

- Replace ChromaDB with a managed vector DB (Pinecone, Weaviate)
- Async LLM calls so the 6 memo sections generate in parallel
- Add LangSmith / Langfuse for tracing and observability
- Replace synthetic data with real Compustat or 10-K filings
- Build a production PD model with proper backtesting and validation
- Add output guardrails (regex-validate every number against source)


---

*Built as a portfolio project to demonstrate agentic AI engineering applied to wholesale credit risk workflows.*
