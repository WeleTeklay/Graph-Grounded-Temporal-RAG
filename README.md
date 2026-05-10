# Graph-Grounded Temporal RAG
### Contradiction-Resilient Question Answering Over Evolving Documents

> Bachelor Thesis — Mekelle Institute of Technology, Mekelle University (2026)

## The Problem
Standard RAG systems are temporally blind. When a document has been 
amended multiple times, they retrieve all versions equally and produce 
contradictory or outdated answers. This system fixes that.

## How It Works
1. **Parse** — extracts text from PDF/DOCX with metadata headers
2. **Chunk** — splits into 1,200-char overlapping segments (3,408 chunks)
3. **Embed** — converts chunks to 384-dim vectors (all-MiniLM-L6-v2)
4. **Graph** — Neo4j encodes SUPERSEDES relationships between versions
5. **Retrieve** — hybrid search (vector + BM25 + keyword) fused via RRF
6. **Rerank** — cross-encoder (ms-marco-MiniLM-L-6-v2) rescores top 30
7. **Answer** — Llama 3.2:3B generates grounded answer locally via Ollama

## Architecture
PDF/DOCX → Parser → Chunker → Embedder → ChromaDB
→ Entity Extractor → Neo4j Graph
User Question → Hybrid Search → Graph Resolve → Reranker → Llama 3.2 → Answer
## Tech Stack
| Component | Technology |
|---|---|
| LLM | Llama 3.2:3B (Ollama) |
| Embeddings | all-MiniLM-L6-v2 |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| Vector DB | ChromaDB |
| Graph DB | Neo4j |
| Sparse search | BM25Okapi (rank-bm25) |
| PDF parsing | PyMuPDF |
| Web UI | Streamlit |
| REST API | FastAPI |

## Setup

### Requirements
- Python 3.11
- Neo4j Desktop (running locally)
- Ollama with llama3.2:3b pulled

### Installation
```bash
git clone https://github.com/WeleTeklay/Graph-Grounded-Temporal-RAG
cd Graph-Grounded-Temporal-RAG
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configure
Copy `.env.example` to `.env` and fill in your Neo4j password:
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OLLAMA_MODEL=llama3.2:3b
### Run the pipeline
```bash
python src/01_parse_pdfs.py
python src/02_chunk_documents.py
python src/04_build_graph.py
python src/05_create_index.py
```

### Launch
```bash
# Web UI
streamlit run app.py

# REST API
python api.py
```

## Results
- 98% diagnostic pass rate on core components
- Average query response time: 3,481 ms on standard laptop
- Fully offline — no API keys, no internet required
- Works on any evolving document corpus (legal, medical, financial, HR)

## Authors
- Weldesemayat Teklay Gebre — github.com/WeleTeklay
- Gebregergs Mekonen — github.com/gere047

## Advisors
Assefa Tesfay (PhD) and Yaecob Girmay (MSc.) — Mekelle Institute of Technology
