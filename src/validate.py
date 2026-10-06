"""
Validation layer (the T in ETL): decides which generated clauses are kept.

Pure functions only: no API calls, no database. Input is GPT's raw text,
output is a ValidationResult that pipeline.py hands to db.py.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field

from src import config


# --- 1. Rejection reasons (MUST match the CHECK constraint in sql/schema.sql) --
REASON_INVALID_JSON = "invalid_json"
REASON_TOO_SHORT = "too_short"
REASON_WRONG_LABEL = "wrong_label"
REASON_DUPLICATE = "duplicate"


# --- 2. Result shapes ----------------------------------------------------------
@dataclass
class AcceptedClause:
    label: str
    text: str
    text_hash: str


@dataclass
class RejectedItem:
    raw_output: str
    reason: str


@dataclass
class ValidationResult:
    accepted: list[AcceptedClause] = field(default_factory=list)
    rejected: list[RejectedItem] = field(default_factory=list)


# --- 3. Text helpers -----------------------------------------------------------
def normalize_text(text: str) -> str:
    """Trim the ends and collapse every run of whitespace into one space."""
    return re.sub(r"\s+", " ", text).strip()


def compute_hash(text: str) -> str:
    """SHA-256 fingerprint of the normalized, lowercased text (64 hex chars)."""
    canonical = normalize_text(text).lower()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- 4. Checks -----------------------------------------------------------------
def parse_response(raw: str) -> list | None:
    """Return the list under "clauses", or None if the response is unusable."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("clauses"), list):
        return None
    return data["clauses"]


def check_clause(item, expected_label: str,
                 seen_hashes: set[str]) -> AcceptedClause | RejectedItem:
    """Run the four checks on one item, cheapest first.

    Side effect: an accepted clause's hash is added to seen_hashes.
    """
    raw = json.dumps(item, ensure_ascii=False)

    if not (isinstance(item, dict)
            and isinstance(item.get("label"), str)
            and isinstance(item.get("text"), str)):
        return RejectedItem(raw, REASON_INVALID_JSON)

    label = item["label"].strip()
    if label != expected_label or label not in config.LABELS:
        return RejectedItem(raw, REASON_WRONG_LABEL)

    text = normalize_text(item["text"])
    if len(text) < config.MIN_CLAUSE_CHARS:
        return RejectedItem(raw, REASON_TOO_SHORT)

    text_hash = compute_hash(text)
    if text_hash in seen_hashes:
        return RejectedItem(raw, REASON_DUPLICATE)

    seen_hashes.add(text_hash)
    return AcceptedClause(label, text, text_hash)


def validate_batch(raw: str, expected_label: str,
                   seen_hashes: set[str]) -> ValidationResult:
    """Validate one API response. The only function pipeline.py needs."""
    result = ValidationResult()

    clauses = parse_response(raw)
    if clauses is None:
        result.rejected.append(RejectedItem(raw, REASON_INVALID_JSON))
        return result

    for item in clauses:
        outcome = check_clause(item, expected_label, seen_hashes)
        if isinstance(outcome, AcceptedClause):
            result.accepted.append(outcome)
        else:
            result.rejected.append(outcome)
    return result