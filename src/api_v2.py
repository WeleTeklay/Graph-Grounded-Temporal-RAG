#src/api_v2.py
"""
Production REST API with authentication, rate limiting, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import redis.asyncio as redis
import uuid
import jwt
from datetime import datetime, timedelta
import hashlib

from src.optimized_retriever import OptimizedQueryEngine
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Configuration
# ============================================================================
SECRET_KEY = config.api_secret_key or "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_PERIOD = 60  # seconds

# ============================================================================
# Models
# ============================================================================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    target_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$')
    stream: bool = False
    include_sources: bool = True

class QueryResponse(BaseModel):
    id: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    graph_used: bool
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    cached: bool
    timestamp: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class UserCredentials(BaseModel):
    username: str
    password: str

class APIKeyRequest(BaseModel):
    name: str
    expires_in_days: Optional[int] = 30

class APIKeyResponse(BaseModel):
    api_key: str
    name: str
    created_at: str
    expires_at: Optional[str]

# ============================================================================
# Authentication
# ============================================================================
security = HTTPBearer(auto_error=False)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_api_key(user_id: str, name: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create API key with prefix."""
    raw_key = hashlib.sha256(f"{user_id}:{uuid.uuid4()}:{datetime.utcnow()}".encode()).hexdigest()[:32]
    api_key = f"gr_{raw_key}"
    # Store in Redis with expiration
    return api_key

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> dict:
    """Verify JWT or API key."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication")
    
    token = credentials.credentials
    
    # Try API key first
    if token.startswith("gr_"):
        # Verify against Redis
        return {"type": "api_key", "user_id": "api_user"}
    
    # Try JWT
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return {"type": "jwt", "user_id": payload.get("sub")}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================================
# Application
# ============================================================================
app = FastAPI(
    title="LexTemporal AI API v2",
    description="Production Graph-Grounded Temporal RAG API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance (initialized at startup)
engine: Optional[OptimizedQueryEngine] = None

# ============================================================================
# Lifecycle
# ============================================================================
@app.on_event("startup")
async def startup():
    global engine
    # Initialize Redis
    redis_client = redis.from_url(config.redis_url, encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis_client)
    
    # Initialize engine
    engine = OptimizedQueryEngine()
    logger.info("API v2 started")

@app.on_event("shutdown")
async def shutdown():
    logger.info("API v2 stopped")

# ============================================================================
# Health & Metrics
# ============================================================================
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "neo4j": engine.graph_enabled if engine else False,
            "chromadb": bool(engine and engine.collection),
            "ollama": True
        }
    }

@app.get("/metrics")
async def metrics(auth: dict = Depends(verify_token)):
    """Prometheus-compatible metrics endpoint."""
    return {
        "cache_size": engine.get_cache_stats() if engine else {},
        "uptime_seconds": (datetime.utcnow() - startup_time).total_seconds()
    }

# ============================================================================
# Authentication Endpoints
# ============================================================================
@app.post("/auth/token", response_model=TokenResponse)
async def login(credentials: UserCredentials):
    """Get JWT access token."""
    # Validate credentials (replace with real auth)
    if credentials.username != "admin" or credentials.password != config.admin_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": credentials.username})
    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.post("/auth/api-key", response_model=APIKeyResponse)
async def create_api_key_endpoint(
    request: APIKeyRequest,
    auth: dict = Depends(verify_token)
):
    """Create new API key."""
    expires_delta = timedelta(days=request.expires_in_days) if request.expires_in_days else None
    api_key = create_api_key(auth["user_id"], request.name, expires_delta)
    
    return APIKeyResponse(
        api_key=api_key,
        name=request.name,
        created_at=datetime.utcnow().isoformat(),
        expires_at=(datetime.utcnow() + expires_delta).isoformat() if expires_delta else None
    )

# ============================================================================
# Query Endpoints
# ============================================================================
@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    auth: dict = Depends(verify_token),
    rate_limiter: None = Depends(RateLimiter(times=RATE_LIMIT_REQUESTS, seconds=RATE_LIMIT_PERIOD))
):
    """Answer a question using Graph-Grounded Temporal RAG."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        result = engine.answer(request.question, request.target_date)
        
        return QueryResponse(
            id=str(uuid.uuid4()),
            answer=result['answer'],
            confidence=result.get('confidence', 0.0),
            sources=result.get('sources', []) if request.include_sources else [],
            graph_used=result.get('graph_used', False),
            retrieval_time_ms=result.get('retrieval_time_ms', 0),
            generation_time_ms=result.get('generation_time_ms', 0),
            total_time_ms=result.get('total_time_ms', 0),
            cached=result.get('cached', False),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    auth: dict = Depends(verify_token)
):
    """Stream answer token by token."""
    from fastapi.responses import StreamingResponse
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    async def generate():
        for token in engine.answer_stream(request.question, request.target_date):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================================================
# Document Management
# ============================================================================
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = None,
    effective_date: str = None,
    supersedes: Optional[str] = None,
    auth: dict = Depends(verify_token)
):
    """Upload and process a new document."""
    from src.ingest import ingest_single_document
    import pandas as pd
    
    doc_id = f"doc_{str(uuid.uuid4())[:8]}"
    pdf_path = config.paths.raw_pdfs_dir / f"{doc_id}.pdf"
    
    content = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)
    
    # Update manifest
    manifest_path = config.paths.project_root / "document_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    
    new_row = pd.DataFrame([{
        'doc_id': doc_id,
        'doc_title': title or file.filename,
        'effective_date': effective_date or datetime.utcnow().strftime("%Y-%m-%d"),
        'supersedes_doc_id': supersedes
    }])
    manifest = pd.concat([manifest, new_row], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)
    
    # Ingest
    success = ingest_single_document(doc_id, effective_date)
    
    return {"status": "success" if success else "failed", "doc_id": doc_id}

@app.get("/documents")
async def list_documents(auth: dict = Depends(verify_token)):
    """List all documents."""
    import pandas as pd
    manifest_path = config.paths.project_root / "document_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        return df.to_dict(orient='records')
    return []

# ============================================================================
# Admin Endpoints
# ============================================================================
@app.post("/admin/cache/clear")
async def clear_cache(auth: dict = Depends(verify_token)):
    """Clear response cache."""
    if engine:
        engine.clear_cache()
        return {"status": "cache cleared"}
    return {"status": "engine not initialized"}

@app.get("/admin/stats")
async def get_stats(auth: dict = Depends(verify_token)):
    """Get detailed system statistics."""
    if not engine:
        return {"status": "engine not initialized"}
    
    return {
        "cache": engine.get_cache_stats(),
        "vector_count": engine.collection.count(),
        "graph_enabled": engine.graph_enabled,
        "model": config.ollama.model
    }

# ============================================================================
# Main
# ============================================================================
startup_time = datetime.utcnow()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api_v2:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        reload=False
    )