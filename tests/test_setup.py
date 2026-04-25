"""
test_setup.py — Smoke test that verifies the environment is set up correctly.

Run after installation:
    python tests/test_setup.py
"""

import sys


def check(name: str, fn):
    """Run a check function, print result, return success bool."""
    try:
        fn()
        print(f"✓ {name}")
        return True
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False


def check_python():
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ required, found {sys.version}")


def check_packages():
    import langgraph
    import langchain_groq
    import pydantic
    import sentence_transformers
    import chromadb
    import sklearn
    import streamlit


def check_env():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set. Create .env file.")
    if not os.getenv("GROQ_API_KEY").startswith("gsk_"):
        raise RuntimeError("GROQ_API_KEY doesn't look right (should start with 'gsk_')")


def check_groq():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke("Reply with exactly: OK")
    if "OK" not in response.content:
        raise RuntimeError(f"Unexpected response: {response.content}")


def check_chromadb():
    import chromadb
    client = chromadb.Client()
    coll = client.create_collection("smoke_test_temp")
    coll.add(ids=["a"], documents=["hello"])
    client.delete_collection("smoke_test_temp")


def check_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    vec = m.encode(["hello world"])
    if vec.shape != (1, 384):
        raise RuntimeError(f"Unexpected embedding shape: {vec.shape}")


def main():
    print("Running setup checks...\n")
    
    results = [
        check("Python version OK", check_python),
        check("Required packages installed", check_packages),
        check("Groq API key found", check_env),
        check("Groq API responding", check_groq),
        check("ChromaDB working", check_chromadb),
        check("sentence-transformers working", check_sentence_transformers),
    ]
    
    print()
    if all(results):
        print("All checks passed. You're ready to build.\n")
        print("Next steps:")
        print("  1. python -m src.tools.build_corpus    # Build the RAG corpus")
        print("  2. python -m src.tools.train_pd_model  # Train the PD model")
        print("  3. python main.py --borrower data/sample_borrowers/healthy_company.json  # Run agent")
        sys.exit(0)
    else:
        print("Some checks failed. Fix the issues above before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
