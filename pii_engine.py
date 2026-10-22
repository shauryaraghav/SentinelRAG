"""
pii_engine.py  –  Privacy-Preserving RAG System
=================================================
PII Detection & Sanitization Engine.

Layers:
  1. Custom regex patterns  → catches Indian-specific PII (PAN, UPI, Aadhaar)
  2. Microsoft Presidio     → NLP-powered recognition for phone, email, etc.
  3. spaCy NER              → fallback PERSON / ORG / GPE extraction

All detected entities are returned as a list of PIIEntity objects.
The `sanitize()` method replaces each span with a typed placeholder so the
original text can never be reconstructed from the sanitised version.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

# ── Optional Presidio import (graceful fallback) ───────────────────────────────
try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logging.warning("Presidio not available – using regex-only PII detection.")

# ── Optional spaCy import ─────────────────────────────────────────────────────
try:
    import spacy
    _NLP = spacy.load("en_core_web_lg")
    SPACY_AVAILABLE = True
except Exception:
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except Exception:
        SPACY_AVAILABLE = False
        logging.warning("spaCy model not available – NER layer disabled.")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PIIEntity:
    entity_type: str          # e.g. "PHONE_NUMBER", "PAN_CARD"
    text: str                  # original matched text
    start: int                 # character start in the original string
    end: int                   # character end in the original string
    score: float               # confidence [0, 1]
    source: str                # "regex" | "presidio" | "spacy"
    placeholder: str = field(init=False)

    def __post_init__(self):
        # e.g.  PAN_CARD  →  <PAN_CARD_1>  (uniqueness added by engine)
        self.placeholder = f"<{self.entity_type}>"

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "source": self.source,
            "placeholder": self.placeholder,
        }


@dataclass
class PIIDetectionResult:
    original_text: str
    sanitized_text: str
    entities: list[PIIEntity]
    pii_found: bool

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def entity_types(self) -> list[str]:
        return list({e.entity_type for e in self.entities})

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "sanitized_text": self.sanitized_text,
            "pii_found": self.pii_found,
            "entity_count": self.entity_count,
            "entity_types": self.entity_types,
            "entities": [e.to_dict() for e in self.entities],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Custom Regex Patterns  (Indian-specific PII + universal formats)
# ─────────────────────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, str, float]] = [
    # (entity_type, pattern, confidence)

    # PAN Card  e.g. ABCDE1234F
    ("PAN_CARD",       r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",                          0.95),

    # UPI ID    e.g. user@paytm, 9876543210@ybl
    ("UPI_ID",         r"\b[\w.\-]+@(?:paytm|upi|ybl|oksbi|okaxis|okhdfcbank|"
                       r"okicici|apl|ibl|axl|pingpay|naviaxis|freecharge|"
                       r"indus|jiomoney|airtel|barodampay|aubank|"
                       r"waicici|waaxis|barodampay|abfspay|"
                       r"icici|sbi|hdfc|axis|kotak|[a-z]{2,10})\b",             0.92),

    # Aadhaar  12-digit, optional spaces/dashes every 4 digits
    ("AADHAAR",        r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b",             0.90),

    # Indian mobile number
    ("IN_PHONE",       r"(?<!\d)(?:\+91[\s\-]?|0)?[6-9]\d{9}(?!\d)",          0.88),

    # Email address
    ("EMAIL_ADDRESS",  r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", 0.97),

    # Credit / Debit card  (13-19 digits)
    ("CREDIT_CARD",    r"\b(?:\d[ \-]?){13,19}\b",                             0.75),

    # IFSC code  e.g. HDFC0001234
    ("IFSC_CODE",      r"\b[A-Z]{4}0[A-Z0-9]{6}\b",                           0.90),

    # Generic Indian bank account (9-18 digits standing alone)
    ("BANK_ACCOUNT",   r"(?<!\d)\d{9,18}(?!\d)",                               0.55),

    # Passport number  e.g. A1234567
    ("PASSPORT",       r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b",           0.85),

    # IPv4
    ("IP_ADDRESS",     r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                        0.80),

    # Date of birth patterns
    ("DATE_OF_BIRTH",  r"\b(?:0?[1-9]|[12]\d|3[01])[\/\-\.](?:0?[1-9]|1[0-2])"
                       r"[\/\-\.](?:19|20)\d{2}\b",                            0.70),
]


# ─────────────────────────────────────────────────────────────────────────────
# Presidio custom recognisers for Indian PII
# ─────────────────────────────────────────────────────────────────────────────

def _build_presidio_recognisers() -> list:
    """Return extra Presidio PatternRecognizer objects for Indian PII."""
    if not PRESIDIO_AVAILABLE:
        return []

    pan_recogniser = PatternRecognizer(
        supported_entity="PAN_CARD",
        patterns=[Pattern(name="pan_card", regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", score=0.95)],
    )
    upi_recogniser = PatternRecognizer(
        supported_entity="UPI_ID",
        patterns=[Pattern(
            name="upi_id",
            regex=r"\b[\w.\-]+@(?:paytm|upi|ybl|oksbi|okaxis|okhdfcbank|okicici|"
                  r"apl|ibl|axl|pingpay|naviaxis|freecharge|indus|jiomoney|"
                  r"airtel|barodampay|aubank|waicici|waaxis|abfspay|"
                  r"icici|sbi|hdfc|axis|kotak|[a-z]{2,10})\b",
            score=0.92,
        )],
    )
    aadhaar_recogniser = PatternRecognizer(
        supported_entity="AADHAAR",
        patterns=[Pattern(name="aadhaar", regex=r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b", score=0.90)],
    )
    ifsc_recogniser = PatternRecognizer(
        supported_entity="IFSC_CODE",
        patterns=[Pattern(name="ifsc", regex=r"\b[A-Z]{4}0[A-Z0-9]{6}\b", score=0.90)],
    )
    return [pan_recogniser, upi_recogniser, aadhaar_recogniser, ifsc_recogniser]


# ─────────────────────────────────────────────────────────────────────────────
# PIIEngine
# ─────────────────────────────────────────────────────────────────────────────

class PIIEngine:
    """
    Three-layer PII detection and sanitisation engine.

    Usage
    -----
    engine = PIIEngine()
    result = engine.detect_and_sanitize(text)
    print(result.sanitized_text)
    print(result.entities)
    """

    # Entity types that should ALWAYS be redacted regardless of confidence
    _HIGH_RISK = {
        "PAN_CARD", "AADHAAR", "CREDIT_CARD", "BANK_ACCOUNT",
        "UPI_ID", "IFSC_CODE", "PASSPORT",
    }

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._compiled_patterns = [
            (etype, re.compile(pat, re.IGNORECASE if etype == "EMAIL_ADDRESS" else 0), score)
            for etype, pat, score in _PATTERNS
        ]

        # Presidio
        self._presidio_analyzer: Optional["AnalyzerEngine"] = None
        self._presidio_anonymizer: Optional["AnonymizerEngine"] = None
        if PRESIDIO_AVAILABLE:
            self._presidio_analyzer = AnalyzerEngine()
            for rec in _build_presidio_recognisers():
                self._presidio_analyzer.registry.add_recognizer(rec)
            self._presidio_anonymizer = AnonymizerEngine()
            logger.info("Presidio analyzer initialised.")
        else:
            logger.info("Presidio not available – regex-only mode.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, text: str) -> list[PIIEntity]:
        """Return all detected PII entities (deduplicated, sorted by position)."""
        entities: list[PIIEntity] = []
        entities.extend(self._regex_detect(text))
        if PRESIDIO_AVAILABLE:
            entities.extend(self._presidio_detect(text))
        if SPACY_AVAILABLE:
            entities.extend(self._spacy_detect(text))

        # Deduplicate: keep highest-confidence entity for overlapping spans
        entities = self._deduplicate(entities)
        # Filter by threshold (but always keep HIGH_RISK)
        entities = [
            e for e in entities
            if e.score >= self.confidence_threshold or e.entity_type in self._HIGH_RISK
        ]
        entities.sort(key=lambda e: e.start)
        return entities

    def sanitize(self, text: str, entities: Optional[list[PIIEntity]] = None) -> str:
        """
        Replace detected PII spans with typed placeholders.
        Entities are processed right-to-left so offsets stay valid.
        """
        if entities is None:
            entities = self.detect(text)
        result = text
        # Sort descending by start so replacements don't shift later offsets
        for i, entity in enumerate(sorted(entities, key=lambda e: e.start, reverse=True)):
            placeholder = f"<{entity.entity_type}_{i + 1}>"
            result = result[: entity.start] + placeholder + result[entity.end :]
        return result

    def detect_and_sanitize(self, text: str) -> PIIDetectionResult:
        """Detect PII, sanitise text, and return a full result object."""
        entities = self.detect(text)
        sanitized = self.sanitize(text, entities)
        return PIIDetectionResult(
            original_text=text,
            sanitized_text=sanitized,
            entities=entities,
            pii_found=len(entities) > 0,
        )

    def mask_for_role(self, text: str, role: str) -> str:
        """
        Apply role-aware masking:
          - "admin"         → full text (no masking)
          - "support_agent" → partial masking (show type placeholder)
          - "analyst"       → aggregate stats only, full mask
          - others          → full mask
        """
        if role == "admin":
            return text
        entities = self.detect(text)
        if not entities:
            return text
        if role == "support_agent":
            # Mask only HIGH_RISK entity types
            hr_entities = [e for e in entities if e.entity_type in self._HIGH_RISK]
            return self.sanitize(text, hr_entities)
        # Default: mask everything
        return self.sanitize(text, entities)

    # ── Layer 1: Regex ─────────────────────────────────────────────────────────

    def _regex_detect(self, text: str) -> list[PIIEntity]:
        found: list[PIIEntity] = []
        for entity_type, pattern, score in self._compiled_patterns:
            for m in pattern.finditer(text):
                # Skip very short BANK_ACCOUNT matches that are probably just numbers
                if entity_type == "BANK_ACCOUNT" and len(m.group()) < 10:
                    continue
                found.append(PIIEntity(
                    entity_type=entity_type,
                    text=m.group(),
                    start=m.start(),
                    end=m.end(),
                    score=score,
                    source="regex",
                ))
        return found

    # ── Layer 2: Presidio ──────────────────────────────────────────────────────

    def _presidio_detect(self, text: str) -> list[PIIEntity]:
        if not self._presidio_analyzer:
            return []
        try:
            results = self._presidio_analyzer.analyze(
                text=text,
                language="en",
                entities=[
                    "PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON",
                    "LOCATION", "URL", "CREDIT_CARD", "IBAN_CODE",
                    "IP_ADDRESS", "DATE_TIME", "NRP",
                    # custom
                    "PAN_CARD", "UPI_ID", "AADHAAR", "IFSC_CODE",
                ],
            )
            entities: list[PIIEntity] = []
            for r in results:
                entities.append(PIIEntity(
                    entity_type=r.entity_type,
                    text=text[r.start: r.end],
                    start=r.start,
                    end=r.end,
                    score=r.score,
                    source="presidio",
                ))
            return entities
        except Exception as exc:
            logger.warning("Presidio detection error: %s", exc)
            return []

    # ── Layer 3: spaCy NER ────────────────────────────────────────────────────

    def _spacy_detect(self, text: str) -> list[PIIEntity]:
        if not SPACY_AVAILABLE:
            return []
        doc = _NLP(text)
        label_map = {
            "PERSON": "PERSON",
            "ORG":    "ORGANIZATION",
            "GPE":    "LOCATION",
            "LOC":    "LOCATION",
            "DATE":   "DATE_TIME",
        }
        entities: list[PIIEntity] = []
        for ent in doc.ents:
            mapped = label_map.get(ent.label_)
            if mapped:
                entities.append(PIIEntity(
                    entity_type=mapped,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=0.70,
                    source="spacy",
                ))
        return entities

    # ── Deduplication ─────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(entities: list[PIIEntity]) -> list[PIIEntity]:
        """
        Remove overlapping spans, keeping the one with the highest confidence.
        O(n²) – fine for typical document chunk sizes.
        """
        entities = sorted(entities, key=lambda e: e.score, reverse=True)
        kept: list[PIIEntity] = []
        for candidate in entities:
            overlaps = any(
                not (candidate.end <= kept_e.start or candidate.start >= kept_e.end)
                for kept_e in kept
            )
            if not overlaps:
                kept.append(candidate)
        return kept


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_engine_instance: Optional[PIIEngine] = None


def get_pii_engine() -> PIIEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PIIEngine()
    return _engine_instance


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = (
        "Hello, my name is Priya Sharma. My PAN is ABCDE1234F and Aadhaar is "
        "2345 6789 0123. You can reach me at priya.sharma@example.com or call "
        "+91-9876543210. My UPI ID is priya@ybl and IFSC is HDFC0001234."
    )
    engine = PIIEngine()
    result = engine.detect_and_sanitize(sample)
    print("=== Detected PII ===")
    for e in result.entities:
        print(f"  [{e.source:8s}] {e.entity_type:20s} → '{e.text}'  (score={e.score:.2f})")
    print("\n=== Sanitised text ===")
    print(result.sanitized_text)
