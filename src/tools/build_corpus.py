"""
build_corpus.py — Builds the synthetic peer company corpus and indexes it in ChromaDB.

"""

import os
import json

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "peer_companies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================================
# SYNTHETIC PEER COMPANIES
# ============================================================================
# 30 fake companies across industries — Manufacturing, Retail, Tech, Healthcare,
# Energy, Finance, Real Estate, Hospitality. Each has a short description with
# representative metrics. The LLM uses these to write "comparable to..." text.

PEER_COMPANIES = [
    # --- Manufacturing ---
    {"id": "mfg_001", "industry": "Manufacturing",
     "text": "Industrial Tools Corp — Mid-size manufacturer of machine tools and industrial equipment. Annual revenue $850M, EBITDA margin 18%, leverage 2.5x. Strong cash generation, BBB-rated. Operates 12 plants across the US Midwest. Customer base is diversified across automotive, aerospace, and construction. Key risks: cyclical demand, raw material price exposure."},
    
    {"id": "mfg_002", "industry": "Manufacturing",
     "text": "Heavy Equipment Holdings — Large-cap manufacturer of construction and agricultural equipment. Revenue $4.2B, EBITDA margin 14%, leverage 3.8x. A-rated. Highly cyclical with construction spending. Geographic exposure across N. America, Europe, Asia."},
    
    {"id": "mfg_003", "industry": "Manufacturing",
     "text": "Specialty Chemical Manufacturing Co — Specialty chemicals producer with high margins. Revenue $1.1B, EBITDA margin 22%, leverage 4.2x post-acquisition. BBB-. Strong intellectual property moat in coatings and adhesives. Recent debt-funded acquisition raised leverage temporarily."},
    
    {"id": "mfg_004", "industry": "Manufacturing",
     "text": "Auto Parts Supplier Inc — Tier-1 automotive supplier serving major OEMs. Revenue $2.8B, EBITDA margin 9%, leverage 5.5x. BB-rated. Margin pressure from EV transition and OEM consolidation. Working capital intensive."},
    
    {"id": "mfg_005", "industry": "Manufacturing",
     "text": "Aerospace Components LLC — Specialized aerospace components manufacturer. Revenue $620M, EBITDA margin 19%, leverage 3.0x. BBB-rated. Defense and commercial aviation exposure. Long-term contracts provide revenue visibility but high R&D investment required."},

    # --- Retail ---
    {"id": "ret_001", "industry": "Retail",
     "text": "Specialty Apparel Retailer — Mid-market specialty retailer with 450 stores. Revenue $1.2B, EBITDA margin 11%, leverage 3.5x. BB+ rated. Faces pressure from e-commerce shift and lease obligations. Operates in mall-anchored locations."},
    
    {"id": "ret_002", "industry": "Retail",
     "text": "Big Box Grocery Holdings — Regional grocery chain with 320 stores. Revenue $8.5B, EBITDA margin 4%, leverage 4.0x. BBB-rated. Defensive consumer staples but thin margins. Heavy investment in private label and digital."},
    
    {"id": "ret_003", "industry": "Retail",
     "text": "Department Store Co — Legacy department store operator. Revenue $3.8B, EBITDA margin 5%, leverage 6.2x. B+ rated. Significant store closures over past 5 years; debt overhang from 2010s LBO. Turnaround risk elevated."},
    
    {"id": "ret_004", "industry": "Retail",
     "text": "Home Improvement Distributor — Wholesale home improvement distributor. Revenue $2.1B, EBITDA margin 8%, leverage 2.8x. BBB-rated. Strong relationships with contractors and home builders. Exposure to housing cycle."},

    # --- Technology ---
    {"id": "tech_001", "industry": "Technology",
     "text": "Enterprise Software Inc — SaaS provider for enterprise resource planning. Revenue $980M (90% recurring), EBITDA margin 25%, leverage 1.8x. A-rated. High net retention rate (115%). Strong moat in mid-market ERP."},
    
    {"id": "tech_002", "industry": "Technology",
     "text": "Cloud Infrastructure Provider — Mid-tier cloud hosting and infrastructure services. Revenue $1.5B, EBITDA margin 30%, leverage 2.2x. A-rated. Capital intensive but strong unit economics. Competition from hyperscalers."},
    
    {"id": "tech_003", "industry": "Technology",
     "text": "Cybersecurity Co — Cybersecurity software vendor focused on endpoint detection. Revenue $420M (95% recurring), EBITDA margin 18%, leverage 0.5x. BBB+ rated. Growing 30%+ annually. Investing heavily in R&D and sales."},
    
    {"id": "tech_004", "industry": "Technology",
     "text": "Hardware Components Maker — Semiconductor components manufacturer. Revenue $2.4B, EBITDA margin 22%, leverage 1.5x. A-rated. Heavy capex requirements for fab investments. Cyclical demand patterns."},

    # --- Healthcare ---
    {"id": "hc_001", "industry": "Healthcare",
     "text": "Specialty Pharma Holdings — Specialty pharmaceutical company. Revenue $1.6B, EBITDA margin 35%, leverage 3.2x. BBB+ rated. Patent cliff approaching for one major drug. Strong pipeline mitigates risk."},
    
    {"id": "hc_002", "industry": "Healthcare",
     "text": "Hospital System — Regional hospital system with 14 facilities. Revenue $3.8B, EBITDA margin 6%, leverage 5.8x. BB+ rated. Margin pressure from labor costs and Medicare reimbursement. Capital intensive."},
    
    {"id": "hc_003", "industry": "Healthcare",
     "text": "Medical Devices Co — Mid-cap medical device manufacturer. Revenue $890M, EBITDA margin 24%, leverage 2.5x. BBB+ rated. Diversified product portfolio in cardiovascular and orthopedics. Strong reimbursement positions."},
    
    {"id": "hc_004", "industry": "Healthcare",
     "text": "Outpatient Services Group — Multi-specialty ambulatory care provider. Revenue $1.1B, EBITDA margin 14%, leverage 4.5x. BB rated. Acquisition-driven growth strategy. Integration and PE sponsor dynamics."},

    # --- Energy ---
    {"id": "energy_001", "industry": "Energy",
     "text": "Independent Oil Producer — Mid-cap E&P company focused on Permian Basin. Revenue $2.8B, EBITDA margin 50%, leverage 1.5x. BBB-rated. Hedging program in place. Exposed to commodity price volatility."},
    
    {"id": "energy_002", "industry": "Energy",
     "text": "Renewable Energy Developer — Solar and wind project developer/operator. Revenue $1.4B, EBITDA margin 40%, leverage 5.5x (project finance). BBB-rated. Stable contracted cash flows under long-term PPAs."},
    
    {"id": "energy_003", "industry": "Energy",
     "text": "Natural Gas Pipeline Operator — Midstream pipeline operator. Revenue $3.2B, EBITDA margin 60%, leverage 4.5x. BBB+ rated. Take-or-pay contracts provide stability. Regulatory scrutiny increasing."},

    # --- Financial Services ---
    {"id": "fin_001", "industry": "Financial Services",
     "text": "Specialty Lender — Non-bank lender focused on small business and consumer loans. Revenue $750M, EBITDA margin 35%, leverage (debt-to-equity) 8x (typical for lenders). BBB-rated. Credit quality tightly monitored."},
    
    {"id": "fin_002", "industry": "Financial Services",
     "text": "Insurance Holdings — P&C insurance holding company. Revenue $4.2B, combined ratio 96%, leverage 1.2x. A-rated. Diversified across personal and commercial lines. Reserve adequacy carefully managed."},

    # --- Real Estate ---
    {"id": "re_001", "industry": "Real Estate",
     "text": "Industrial REIT — REIT focused on logistics and warehouse properties. Revenue $920M, EBITDA margin 65%, leverage 6.0x. BBB+ rated. Portfolio occupancy 96%, weighted average lease term 7 years. Strong tenant credit quality."},
    
    {"id": "re_002", "industry": "Real Estate",
     "text": "Office REIT Holdings — Office property REIT in major metros. Revenue $1.8B, EBITDA margin 55%, leverage 7.2x. BB+ rated. Significant office space recovery dependent on return-to-office trends."},
    
    {"id": "re_003", "industry": "Real Estate",
     "text": "Multifamily Residential — Apartment REIT in growth markets. Revenue $1.1B, EBITDA margin 60%, leverage 5.5x. BBB-rated. Geographic concentration in Sun Belt. Rent growth deceleration risk."},

    # --- Hospitality / Leisure ---
    {"id": "hosp_001", "industry": "Hospitality",
     "text": "Hotel Holdings Group — Mid-scale hotel franchisor and operator. Revenue $2.4B, EBITDA margin 25%, leverage 4.5x. BB+ rated. Asset-light model. Sensitive to travel demand cycles."},
    
    {"id": "hosp_002", "industry": "Hospitality",
     "text": "Restaurant Operating Co — Casual dining restaurant chain with 800 locations. Revenue $1.6B, EBITDA margin 11%, leverage 5.0x. B+ rated. Comparable sales recovery underway. Wage inflation pressuring margins."},

    # --- Distribution ---
    {"id": "dist_001", "industry": "Distribution",
     "text": "Industrial Distribution Co — National distributor of MRO supplies to manufacturers. Revenue $3.5B, EBITDA margin 9%, leverage 3.2x. BBB-rated. Diverse customer base and product breadth provide stability."},
    
    {"id": "dist_002", "industry": "Distribution",
     "text": "Food Service Distributor — Foodservice distribution to restaurants and institutions. Revenue $5.8B, EBITDA margin 5%, leverage 4.0x. BBB-rated. Thin margin business with scale advantages. Labor and fuel costs are key drivers."},

    # --- Construction ---
    {"id": "const_001", "industry": "Construction",
     "text": "Engineering & Construction Co — Public works contractor focused on transportation infrastructure. Revenue $2.1B, EBITDA margin 7%, leverage 2.5x. BBB-rated. Backlog visibility 18 months. Bonding capacity is a growth constraint."},
]


def build_corpus() -> None:
    """Build the corpus: encode all documents, store in ChromaDB."""
    print(f"Loading peer documents... {len(PEER_COMPANIES)} documents.")
    
    print(f"Loading embedding model ({EMBEDDING_MODEL})... ", end="", flush=True)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("done.")
    
    print("Generating embeddings... ", end="", flush=True)
    texts = [p["text"] for p in PEER_COMPANIES]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    print("done.")
    
    print("Storing in ChromaDB... ", end="", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Delete and recreate collection to ensure a clean state
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist, fine
    
    collection = client.create_collection(name=COLLECTION_NAME)
    
    collection.add(
        ids=[p["id"] for p in PEER_COMPANIES],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"industry": p["industry"]} for p in PEER_COMPANIES],
    )
    print("done.")
    
    print(f"\nReady. Corpus has {len(PEER_COMPANIES)} documents stored in {CHROMA_DIR}/")


if __name__ == "__main__":
    build_corpus()
