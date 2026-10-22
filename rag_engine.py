"""
rag_engine.py  –  Privacy-Preserving RAG System
================================================
Vector Storage & Retrieval-Augmented Generation Engine.

Pipeline
--------
  ingest(docs)
    → chunk text
    → PII sanitisation (pii_engine)
    → embed with sentence-transformers
    → persist in ChromaDB

  query(question, role)
    → PII-sanitise the question
    → embed & retrieve top-k chunks
    → RBAC guardrail (block extraction attacks)
    → build prompt
    → call LLM (Gemini / OpenAI / fallback)
    → apply role-aware masking on the answer
    → return answer + sources
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from pii_engine import get_pii_engine, PIIEngine

logger = logging.getLogger(__name__)

# ── Optional ChromaDB ─────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available.")

# ── Optional sentence-transformers ────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not available – embeddings disabled.")

# ── Optional Gemini ───────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Optional OpenAI ───────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

CHROMA_PERSIST_DIR = Path("./chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "privacy_rag"

# ─────────────────────────────────────────────────────────────────────────────
# RBAC configuration
# ─────────────────────────────────────────────────────────────────────────────

ROLE_PERMISSIONS: dict[str, dict] = {
    "admin": {
        "can_view_pii": True,
        "can_query": True,
        "can_ingest": True,
        "mask_answer": False,
        "max_results": 10,
    },
    "support_agent": {
        "can_view_pii": False,
        "can_query": True,
        "can_ingest": False,
        "mask_answer": True,
        "max_results": 5,
        "mask_types": {"PAN_CARD", "AADHAAR", "CREDIT_CARD", "BANK_ACCOUNT"},
    },
    "analyst": {
        "can_view_pii": False,
        "can_query": True,
        "can_ingest": False,
        "mask_answer": True,
        "max_results": 8,
        "mask_types": {"PAN_CARD", "AADHAAR", "CREDIT_CARD", "BANK_ACCOUNT",
                       "UPI_ID", "IFSC_CODE", "EMAIL_ADDRESS", "IN_PHONE"},
    },
    "guest": {
        "can_view_pii": False,
        "can_query": True,
        "can_ingest": False,
        "mask_answer": True,
        "max_results": 3,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Attack / extraction pattern detection
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_PATTERNS = [
    r"ignore\s+(?:previous|prior|above)\s+instructions?",
    r"reveal\s+(?:the\s+)?(?:original|raw|actual|real)\s+(?:data|text|document|content)",
    r"show\s+me\s+(?:the\s+)?(?:pii|personal|private|sensitive|hidden|masked)",
    r"what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:actual|real|original)\s+(?:phone|email|pan|aadhaar|upi|card)",
    r"bypass\s+(?:the\s+)?(?:filter|mask|privacy|security|redaction|guardrail)",
    r"pretend\s+(?:you\s+are|to\s+be)\s+(?:an?\s+)?admin",
    r"act\s+as\s+(?:an?\s+)?(?:admin|unrestricted|root|superuser)",
    r"jailbreak",
    r"print\s+(?:all\s+)?(?:pii|personal|private|hidden|sensitive)\s+data",
    r"list\s+(?:all\s+)?(?:phone\s+numbers?|emails?|pan\s+cards?|aadhaar)",
    r"extract\s+(?:all\s+)?(?:pii|personal|sensitive)",
    r"disregard\s+(?:the\s+)?(?:rules?|instructions?|guidelines?|privacy)",
    r"for\s+(?:testing|debug|development)\s+purposes?\s*,?\s*(?:show|reveal|print|display)",
    r"(?:as\s+)?(?:a\s+)?(?:developer|engineer|researcher|tester)\s*,?\s*(?:show|reveal|tell)",
]

_COMPILED_ATTACK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _EXTRACTION_PATTERNS]


def is_extraction_attack(query: str) -> tuple[bool, str]:
    """Return (is_attack, reason) for a given query string."""
    for pattern in _COMPILED_ATTACK_PATTERNS:
        if pattern.search(query):
            return True, f"Matched extraction pattern: {pattern.pattern[:60]}…"
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Text chunker
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by word boundary."""
    words = text.split()
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# RAGEngine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Main RAG engine that orchestrates ingestion, retrieval, and generation
    with PII guardrails and role-based access control.
    """

    def __init__(
        self,
        pii_engine: Optional[PIIEngine] = None,
        persist_dir: str = str(CHROMA_PERSIST_DIR),
        embed_model: str = EMBED_MODEL,
    ):
        self.pii = pii_engine or get_pii_engine()
        self.embed_model_name = embed_model
        self._embedder: Optional[SentenceTransformer] = None
        self._chroma_client = None
        self._collection = None
        self._ingested_docs: list[dict] = []  # in-memory audit log

        self._init_embedder()
        self._init_chroma(persist_dir)
        self._init_llm()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_embedder(self):
        if ST_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(self.embed_model_name)
                logger.info("Sentence-transformer loaded: %s", self.embed_model_name)
            except Exception as exc:
                logger.warning("Failed to load sentence-transformer: %s", exc)

    def _init_chroma(self, persist_dir: str):
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB unavailable – in-memory fallback will be used.")
            return
        try:
            CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB initialised. Collection: %s", COLLECTION_NAME)
        except Exception as exc:
            logger.error("ChromaDB init failed: %s", exc)
            self._chroma_client = None
            self._collection = None

    def _init_llm(self):
        self._llm_type = "none"
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if GEMINI_AVAILABLE and gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self._llm_type = "gemini"
                logger.info("LLM: Gemini 1.5 Flash")
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

        if self._llm_type == "none" and OPENAI_AVAILABLE and openai_key:
            try:
                self._openai_client = OpenAI(api_key=openai_key)
                self._llm_type = "openai"
                logger.info("LLM: OpenAI GPT-4o-mini")
            except Exception as exc:
                logger.warning("OpenAI init failed: %s", exc)

        if self._llm_type == "none":
            logger.warning("No LLM configured – will return context-only answers.")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        text: str,
        source: str = "unknown",
        metadata: Optional[dict] = None,
        role: str = "admin",
    ) -> dict:
        """
        Ingest a document:
          1. Chunk the text
          2. PII-sanitise each chunk
          3. Embed with sentence-transformers
          4. Store in ChromaDB
        """
        permissions = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["guest"])
        if not permissions.get("can_ingest"):
            return {"error": f"Role '{role}' does not have ingestion privileges."}

        chunks = chunk_text(text)
        results = {
            "source": source,
            "total_chunks": len(chunks),
            "chunks_with_pii": 0,
            "total_pii_entities": 0,
            "entity_type_counts": {},
            "chunk_ids": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        ids, documents, embeddings_list, metadatas = [], [], [], []

        for idx, chunk in enumerate(chunks):
            detection = self.pii.detect_and_sanitize(chunk)
            sanitized_chunk = detection.sanitized_text

            # Audit
            if detection.pii_found:
                results["chunks_with_pii"] += 1
                results["total_pii_entities"] += detection.entity_count
                for etype in detection.entity_types:
                    results["entity_type_counts"][etype] = (
                        results["entity_type_counts"].get(etype, 0) + 1
                    )

            chunk_id = f"{source}_{idx}_{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            documents.append(sanitized_chunk)

            chunk_meta = {
                "source": source,
                "chunk_index": idx,
                "pii_found": detection.pii_found,
                "pii_types": ",".join(detection.entity_types),
                **(metadata or {}),
            }
            metadatas.append(chunk_meta)

            if self._embedder:
                emb = self._embedder.encode(sanitized_chunk).tolist()
                embeddings_list.append(emb)

            results["chunk_ids"].append(chunk_id)

        # Persist to ChromaDB
        if self._collection:
            kwargs: dict = {"ids": ids, "documents": documents, "metadatas": metadatas}
            if embeddings_list:
                kwargs["embeddings"] = embeddings_list
            self._collection.upsert(**kwargs)

        # Audit log
        self._ingested_docs.append({
            "source": source,
            "chunks": len(chunks),
            "pii_entities_sanitised": results["total_pii_entities"],
            "timestamp": results["timestamp"],
        })

        return results

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Embed query and retrieve top-k similar chunks from ChromaDB."""
        if not self._collection:
            return []
        try:
            kwargs: dict = {"query_texts": [query], "n_results": min(top_k, self._collection.count() or 1)}
            if self._embedder:
                emb = self._embedder.encode(query).tolist()
                kwargs = {"query_embeddings": [emb], "n_results": kwargs["n_results"]}

            result = self._collection.query(**kwargs)
            chunks = []
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, distances):
                chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "relevance_score": round(1 - (dist or 0), 4),
                })
            return chunks
        except Exception as exc:
            logger.error("Retrieval error: %s", exc)
            return []

    # ── Query / Generation ────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        role: str = "guest",
        top_k: int = 5,
    ) -> dict:
        """
        Full RAG query pipeline with RBAC guardrails:
          1. Role permission check
          2. Extraction-attack detection
          3. Sanitise the question
          4. Retrieve relevant chunks
          5. Call LLM for a grounded answer
          6. Mask PII in the answer according to role
        """
        permissions = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["guest"])

        if not permissions.get("can_query"):
            return {
                "answer": "Access denied: your role does not permit queries.",
                "blocked": True,
                "role": role,
                "sources": [],
            }

        # ── Guardrail: attack detection ────────────────────────────────────────
        attack, reason = is_extraction_attack(question)
        if attack:
            logger.warning("Extraction attack detected from role=%s: %s", role, reason)
            return {
                "answer": (
                    "⚠️ Query blocked by Privacy Guardrail.\n\n"
                    "Your request appears to be attempting to extract raw PII or "
                    "bypass privacy controls. This action has been logged."
                ),
                "blocked": True,
                "block_reason": reason,
                "role": role,
                "sources": [],
            }

        # ── Sanitise the question itself ───────────────────────────────────────
        q_detection = self.pii.detect_and_sanitize(question)
        sanitized_question = q_detection.sanitized_text

        # ── Retrieve context ───────────────────────────────────────────────────
        max_results = permissions.get("max_results", top_k)
        chunks = self.retrieve(sanitized_question, top_k=max_results)

        if not chunks:
            return {
                "answer": (
                    "I could not find relevant information in the knowledge base. "
                    "Please ensure documents have been ingested first."
                ),
                "blocked": False,
                "role": role,
                "sources": [],
                "question_pii_found": q_detection.pii_found,
            }

        context = "\n\n---\n\n".join(c["text"] for c in chunks)

        # ── Generate answer ────────────────────────────────────────────────────
        raw_answer = self._generate(sanitized_question, context, role)

        # ── Role-aware answer masking ──────────────────────────────────────────
        if permissions.get("mask_answer"):
            final_answer = self.pii.mask_for_role(raw_answer, role)
        else:
            final_answer = raw_answer

        sources = [
            {
                "source": c["metadata"].get("source", "unknown"),
                "relevance": c["relevance_score"],
                "chunk_index": c["metadata"].get("chunk_index", 0),
                "pii_sanitised": c["metadata"].get("pii_found", False),
            }
            for c in chunks
        ]

        return {
            "answer": final_answer,
            "blocked": False,
            "role": role,
            "sources": sources,
            "question_pii_found": q_detection.pii_found,
            "question_sanitized": sanitized_question if q_detection.pii_found else None,
            "context_chunks_used": len(chunks),
        }

    # ── LLM generation ────────────────────────────────────────────────────────

    def _generate(self, question: str, context: str, role: str) -> str:
        role_note = {
            "admin":         "You are assisting an administrator with full data access.",
            "support_agent": "You are assisting a customer support agent. Do NOT reveal personal data.",
            "analyst":       "You are assisting a data analyst. Provide only aggregated insights.",
            "guest":         "You are assisting a guest with limited access.",
        }.get(role, "You are a helpful assistant.")

        system_prompt = textwrap.dedent(f"""
            You are a Privacy-Preserving RAG assistant.
            {role_note}

            IMPORTANT RULES:
            1. Base your answer ONLY on the provided context.
            2. NEVER reconstruct, guess, or infer original PII values from placeholders like <PAN_CARD_1>.
            3. If asked for raw personal data, politely decline and explain privacy policy.
            4. If the context does not contain the answer, say so honestly.
            5. Treat every <ENTITY_TYPE_N> placeholder as genuinely masked – never try to unmask it.
        """).strip()

        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        if self._llm_type == "gemini":
            return self._gemini_generate(prompt)
        elif self._llm_type == "openai":
            return self._openai_generate(prompt)
        else:
            return self._context_only_answer(question, context)

    def _gemini_generate(self, prompt: str) -> str:
        try:
            response = self._gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as exc:
            logger.error("Gemini generation error: %s", exc)
            return f"LLM generation failed: {exc}"

    def _openai_generate(self, prompt: str) -> str:
        try:
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("OpenAI generation error: %s", exc)
            return f"LLM generation failed: {exc}"

    def _context_only_answer(self, question: str, context: str) -> str:
        """Fallback when no LLM is configured: return relevant context snippet."""
        q_words = set(question.lower().split())
        best_chunk = ""
        best_overlap = -1
        for chunk in context.split("\n\n---\n\n"):
            overlap = len(q_words & set(chunk.lower().split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk
        if best_chunk:
            snippet = " ".join(best_chunk.split()[:80])
            return (
                f"[No LLM configured – showing relevant context]\n\n{snippet}…\n\n"
                "Set GEMINI_API_KEY or OPENAI_API_KEY in .env for full generation."
            )
        return "No relevant information found in the knowledge base."

    # ── Utility ───────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        doc_count = self._collection.count() if self._collection else 0
        return {
            "vector_store": "ChromaDB" if CHROMA_AVAILABLE else "unavailable",
            "embed_model": self.embed_model_name if ST_AVAILABLE else "unavailable",
            "llm_backend": self._llm_type,
            "total_chunks_stored": doc_count,
            "documents_ingested": len(self._ingested_docs),
            "audit_log": self._ingested_docs[-10:],  # last 10 entries
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_rag_instance: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine()
    return _rag_instance
