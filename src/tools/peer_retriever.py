"""
peer_retriever.py — RAG retrieval. Finds peer companies similar to the borrower.

This is the "R" in RAG. Given a borrower's industry and profile, we generate
a query string, embed it, and search ChromaDB for the closest peers.

The retrieved peer descriptions are passed to the LLM in the memo prompt.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from src.schemas import BorrowerFinancials, CreditRatios, PeerComparison


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "peer_companies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Cached objects to avoid re-loading on every request
_cached_embedder = None
_cached_collection = None


def _get_embedder():
    global _cached_embedder
    if _cached_embedder is None:
        _cached_embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _cached_embedder


def _get_collection():
    global _cached_collection
    if _cached_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            _cached_collection = client.get_collection(name=COLLECTION_NAME)
        except Exception:
            raise FileNotFoundError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found. "
                f"Run: python -m src.tools.build_corpus"
            )
    return _cached_collection


def _build_query(financials: BorrowerFinancials, ratios: CreditRatios) -> str:
    """
    Build a search query string from borrower data.
    
    The query should describe the borrower in terms similar to how peer
    documents are written, so semantic search finds good matches.
    """
    return (
        f"{financials.industry} company with revenue ${financials.revenue:.0f}M, "
        f"EBITDA margin {ratios.ebitda_margin:.0%}, "
        f"leverage {ratios.leverage_ratio:.1f}x, "
        f"{ratios.leverage_assessment} leverage and {ratios.coverage_assessment} coverage."
    )


def retrieve_peers(
    financials: BorrowerFinancials,
    ratios: CreditRatios,
    n: int = 4,
) -> PeerComparison:
    """
    Retrieve the top-n most similar peer companies.
    
    Returns a PeerComparison with:
    - peer_documents: the raw text of each peer
    - peer_industries: their industries  
    - similarity_scores: how close each peer is to the query (1.0 = identical)
    """
    embedder = _get_embedder()
    collection = _get_collection()
    
    query = _build_query(financials, ratios)
    
    query_embedding = embedder.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
    )
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    # ChromaDB returns L2 distances; convert to a 0-1 similarity score
    # Smaller distance = more similar. We invert and normalize roughly.
    similarities = [max(0.0, 1.0 - (d / 2.0)) for d in distances]
    
    return PeerComparison(
        peer_documents=documents,
        peer_industries=[m["industry"] for m in metadatas],
        similarity_scores=similarities,
    )

