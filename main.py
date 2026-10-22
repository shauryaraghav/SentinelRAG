"""
main.py  –  Privacy-Preserving RAG System
==========================================
FastAPI application with:
  - REST endpoints for PII detection, document ingestion, and RAG querying
  - RBAC middleware using JWT tokens
  - Static HTML frontend serving
  - Comprehensive request/response logging
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Load env early ────────────────────────────────────────────────────────────
load_dotenv()

# ── Auth helpers ──────────────────────────────────────────────────────────────
try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    logging.warning("jose/passlib not installed – auth will use static demo tokens.")

# ── Local modules ─────────────────────────────────────────────────────────────
from pii_engine import get_pii_engine
from rag_engine import ROLE_PERMISSIONS, get_rag_engine, is_extraction_attack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Auth config
# ─────────────────────────────────────────────────────────────────────────────

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "privacy-rag-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours

if AUTH_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Demo users (in production, use a proper database)
DEMO_USERS: dict[str, dict] = {
    "admin":         {"password": "admin123",   "role": "admin"},
    "support_agent": {"password": "support123", "role": "support_agent"},
    "analyst":       {"password": "analyst123", "role": "analyst"},
    "guest":         {"password": "guest123",   "role": "guest"},
}

# Static API tokens for demo (bypasses JWT for quick testing)
STATIC_TOKENS: dict[str, str] = {
    "token-admin":   "admin",
    "token-support": "support_agent",
    "token-analyst": "analyst",
    "token-guest":   "guest",
}


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    if AUTH_AVAILABLE:
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return f"static-{data.get('sub', 'guest')}"


def decode_token(token: str) -> dict:
    # Check static tokens first (for demo)
    if token in STATIC_TOKENS:
        return {"sub": STATIC_TOKENS[token], "role": STATIC_TOKENS[token]}
    if AUTH_AVAILABLE:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            pass
    return {}


def get_current_role(request: Request) -> str:
    """Extract role from Authorization header or query param."""
    # Try Bearer token
    auth_header = request.headers.get("Authorization", "")
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    # Try query param fallback (for demo UI)
    if not token:
        token = request.query_params.get("token", "")

    if token:
        payload = decode_token(token)
        if payload:
            return payload.get("role") or payload.get("sub", "guest")

    # Default to guest
    return "guest"


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class PIIDetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Text to scan for PII")
    role: str = Field("guest", description="Caller role for masking level")

class PIIDetectResponse(BaseModel):
    original_text: str
    sanitized_text: str
    pii_found: bool
    entity_count: int
    entity_types: list[str]
    entities: list[dict]
    processing_time_ms: float

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1_000_000, description="Document text to ingest")
    source: str = Field("api_upload", description="Document source identifier")
    metadata: dict = Field(default_factory=dict, description="Optional metadata key-value pairs")

class IngestResponse(BaseModel):
    success: bool
    source: str
    total_chunks: int
    chunks_with_pii: int
    total_pii_entities: int
    entity_type_counts: dict
    chunk_ids: list[str]
    timestamp: str
    message: str

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5_000, description="Question to ask the RAG system")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")

class QueryResponse(BaseModel):
    answer: str
    blocked: bool
    role: str
    sources: list[dict]
    question_pii_found: Optional[bool] = None
    question_sanitized: Optional[str] = None
    context_chunks_used: Optional[int] = None
    processing_time_ms: float
    block_reason: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    role: str
    expires_in: int

class AttackTestRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2_000, description="Query to test for extraction attacks")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Privacy-Preserving RAG System",
    description=(
        "An end-to-end privacy-preserving RAG system with PII detection, "
        "sanitisation, RBAC, and attack guardrails."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS – adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ── Startup: eager-load engines ───────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Initialising PII engine…")
    get_pii_engine()
    logger.info("Initialising RAG engine…")
    get_rag_engine()
    logger.info("Privacy-Preserving RAG System ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/auth/login", response_model=LoginResponse, tags=["Auth"])
async def login(body: LoginRequest):
    """Authenticate and receive a JWT access token."""
    user = DEMO_USERS.get(body.username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if AUTH_AVAILABLE:
        # In production, compare hashed passwords
        if body.password != user["password"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    else:
        if body.password != user["password"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token({"sub": body.username, "role": user["role"]})
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.get("/api/auth/roles", tags=["Auth"])
async def list_roles():
    """List available roles and their permissions."""
    return {
        "roles": {
            role: {k: v for k, v in perms.items() if k != "mask_types"}
            for role, perms in ROLE_PERMISSIONS.items()
        },
        "demo_tokens": STATIC_TOKENS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PII endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/pii/detect", response_model=PIIDetectResponse, tags=["PII"])
async def detect_pii(body: PIIDetectRequest, request: Request):
    """
    Detect and optionally sanitise PII in the provided text.
    Returns the sanitised text and a list of all detected entities.
    """
    role = get_current_role(request)
    start = time.perf_counter()

    engine = get_pii_engine()
    result = engine.detect_and_sanitize(body.text)
    elapsed = (time.perf_counter() - start) * 1000

    # For non-admin roles, also mask the original text in the response
    original = body.text if role == "admin" else "[ hidden – insufficient privileges ]"

    return PIIDetectResponse(
        original_text=original,
        sanitized_text=result.sanitized_text,
        pii_found=result.pii_found,
        entity_count=result.entity_count,
        entity_types=result.entity_types,
        entities=result.entities if role == "admin" else [
            {k: v for k, v in e.items() if k not in ("text",)}
            for e in [ent.to_dict() for ent in result.entities]
        ],
        processing_time_ms=round(elapsed, 2),
    )


@app.post("/api/pii/batch-detect", tags=["PII"])
async def batch_detect_pii(request: Request, texts: list[str]):
    """Batch PII detection for multiple text inputs."""
    role = get_current_role(request)
    engine = get_pii_engine()
    results = []
    for text in texts[:50]:  # cap at 50 items
        r = engine.detect_and_sanitize(text)
        results.append({
            "sanitized_text": r.sanitized_text,
            "pii_found": r.pii_found,
            "entity_count": r.entity_count,
            "entity_types": r.entity_types,
        })
    return {"results": results, "count": len(results)}


# ─────────────────────────────────────────────────────────────────────────────
# RAG ingestion
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/rag/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest_document(body: IngestRequest, request: Request):
    """
    Ingest a document into the privacy-preserving RAG pipeline.
    - Detects and sanitises PII before embedding
    - Stores sanitised chunks in ChromaDB
    - Returns ingestion stats and PII audit
    """
    role = get_current_role(request)
    rag = get_rag_engine()

    result = rag.ingest(
        text=body.text,
        source=body.source,
        metadata=body.metadata,
        role=role,
    )

    if "error" in result:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=result["error"])

    return IngestResponse(
        success=True,
        source=result["source"],
        total_chunks=result["total_chunks"],
        chunks_with_pii=result["chunks_with_pii"],
        total_pii_entities=result["total_pii_entities"],
        entity_type_counts=result["entity_type_counts"],
        chunk_ids=result["chunk_ids"],
        timestamp=result["timestamp"],
        message=(
            f"Successfully ingested {result['total_chunks']} chunks. "
            f"PII sanitised in {result['chunks_with_pii']} chunks "
            f"({result['total_pii_entities']} entities removed)."
        ),
    )


@app.post("/api/rag/ingest/upload", tags=["RAG"])
async def ingest_upload(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form(default="file_upload"),
):
    """Upload a .txt file for ingestion into the RAG pipeline."""
    role = get_current_role(request)
    if file.content_type not in ("text/plain", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    rag = get_rag_engine()
    result = rag.ingest(text=text, source=source or file.filename or "upload", role=role)

    if "error" in result:
        raise HTTPException(status_code=403, detail=result["error"])

    return {"success": True, "filename": file.filename, **result}


# ─────────────────────────────────────────────────────────────────────────────
# RAG query
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/rag/query", response_model=QueryResponse, tags=["RAG"])
async def rag_query(body: QueryRequest, request: Request):
    """
    Query the privacy-preserving RAG system.
    - Applies RBAC based on the caller's role
    - Detects extraction attacks and blocks them
    - Returns LLM-generated answer with masked PII
    """
    role = get_current_role(request)
    start = time.perf_counter()

    rag = get_rag_engine()
    result = rag.query(question=body.question, role=role, top_k=body.top_k)

    elapsed = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=result["answer"],
        blocked=result.get("blocked", False),
        role=role,
        sources=result.get("sources", []),
        question_pii_found=result.get("question_pii_found"),
        question_sanitized=result.get("question_sanitized"),
        context_chunks_used=result.get("context_chunks_used"),
        processing_time_ms=round(elapsed, 2),
        block_reason=result.get("block_reason"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Attack testing
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/security/test-attack", tags=["Security"])
async def test_attack_pattern(body: AttackTestRequest):
    """
    Red-team utility: check whether a query would be flagged as an extraction attack.
    Returns detection result without actually running the query.
    """
    is_attack, reason = is_extraction_attack(body.query)
    pii_result = get_pii_engine().detect(body.query)
    return {
        "query": body.query,
        "is_extraction_attack": is_attack,
        "attack_reason": reason,
        "pii_in_query": len(pii_result) > 0,
        "pii_entities_in_query": [e.to_dict() for e in pii_result],
        "recommendation": (
            "BLOCK – this query would be rejected by the guardrail." if is_attack
            else "PASS – this query appears safe."
        ),
    }


@app.get("/api/security/attack-patterns", tags=["Security"])
async def list_attack_patterns():
    """Return the list of extraction-attack regex patterns used by the guardrail."""
    from rag_engine import _EXTRACTION_PATTERNS
    return {
        "patterns": _EXTRACTION_PATTERNS,
        "count": len(_EXTRACTION_PATTERNS),
        "description": "Regex patterns used to detect PII extraction and jailbreak attempts.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# System info
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/system/stats", tags=["System"])
async def system_stats(request: Request):
    """Return system statistics and health information."""
    role = get_current_role(request)
    rag = get_rag_engine()
    stats = rag.stats()

    # Redact audit log for non-admin roles
    if role != "admin":
        stats.pop("audit_log", None)

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "role": role,
        **stats,
    }


@app.get("/api/system/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ─────────────────────────────────────────────────────────────────────────────
# Serve the frontend
# ─────────────────────────────────────────────────────────────────────────────

_FRONTEND = Path(__file__).parent / "index.html"

@app.get("/", include_in_schema=False)
async def serve_frontend():
    if _FRONTEND.exists():
        return FileResponse(str(_FRONTEND), media_type="text/html")
    return JSONResponse({"message": "Frontend not found. Place index.html next to main.py."})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
