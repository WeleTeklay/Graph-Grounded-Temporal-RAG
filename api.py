#api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import uuid
import pandas as pd
from pathlib import Path

from src.query_engine import QueryEngine
from src.ingest import ingest_single_document
from src.config import config
from src.logger import get_logger
import os
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

logger = get_logger(__name__)

app = FastAPI(
    title="LexTemporal AI API",
    description="Graph-Grounded Temporal RAG for Legal Document Intelligence",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = QueryEngine()


class QueryRequest(BaseModel):
    question: str
    target_date: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    graph_used: bool
    num_chunks_retrieved: int


class UploadResponse(BaseModel):
    status: str
    doc_id: str
    message: str


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "LexTemporal AI API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "vector_count": engine.collection.count(),
        "graph_enabled": engine.graph_enabled,
        "model": config.ollama.model
    }


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Answer a question using Graph-Grounded Temporal RAG."""
    try:
        result = engine.answer(request.question, request.target_date)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse, tags=["Document"])
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    effective_date: str = Form(...),
    supersedes: Optional[str] = Form(None)
):
    """Upload and process a new document."""
    try:
        doc_id = f"doc_{str(uuid.uuid4())[:8]}"
        
        # Save PDF
        pdf_path = config.RAW_PDFS_DIR / f"{doc_id}.pdf"
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)
        
        # Update manifest
        manifest_path = config.manifest_path
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
        else:
            manifest = pd.DataFrame(columns=['doc_id', 'doc_title', 'effective_date', 'supersedes_doc_id'])
        
        new_row = pd.DataFrame([{
            'doc_id': doc_id,
            'doc_title': title,
            'effective_date': effective_date,
            'supersedes_doc_id': supersedes
        }])
        
        manifest = pd.concat([manifest, new_row], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)
        
        # Ingest document
        success = ingest_single_document(doc_id, effective_date)
        
        if success:
            return UploadResponse(
                status="success",
                doc_id=doc_id,
                message=f"Document {title} processed successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest document")
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Document"])
async def list_documents():
    """List all documents in the system."""
    try:
        if config.manifest_path.exists():
            manifest = pd.read_csv(config.manifest_path)
            return manifest.to_dict(orient='records')
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["System"])
async def get_stats():
    """Get system statistics."""
    stats = {
        "vector_count": engine.collection.count(),
        "graph_enabled": engine.graph_enabled,
        "documents": 0
    }
    
    if config.manifest_path.exists():
        manifest = pd.read_csv(config.manifest_path)
        stats["documents"] = len(manifest)
    
    if engine.graph_enabled:
        try:
            with engine.neo4j_driver.session() as session:
                result = session.run("MATCH (n:Clause) RETURN count(n) as c")
                stats["graph_nodes"] = result.single()['c']
                result = session.run("MATCH ()-[r:SUPERSEDES]->() RETURN count(r) as c")
                stats["graph_relationships"] = result.single()['c']
        except:
            pass
    
    return stats


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)