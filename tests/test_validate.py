"""
Unit tests for src/validate.py.

Run from the project root with:  python -m pytest -v
"""

import json
import re
from pathlib import Path

from src import config, validate
from src.validate import AcceptedClause, compute_hash, normalize_text, validate_batch

LABEL = "Termination"
GOOD_TEXT = "Either party may terminate this Agreement upon thirty days' written notice."
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


# --- Helpers ---------------------------------------------------------------------
def make_response(*items) -> str:
    """Wrap clause items in the JSON shape GPT is asked to return."""
    return json.dumps({"clauses": list(items)})


def clause(text: str = GOOD_TEXT, label: str = LABEL) -> dict:
    """One clause item; override only the field a test cares about."""
    return {"label": label, "text": text}


def reasons(result) -> list[str]:
    """Just the rejection reasons, in order."""
    return [r.reason for r in result.rejected]


def schema_check_values(column: str) -> set[str]:
    """Read the allowed values of a CHECK (column IN (...)) from schema.sql."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    match = re.search(rf"{column} IN \(([^)]*)\)", sql)
    assert match, f"CHECK constraint for '{column}' not found in schema.sql"
    return set(re.findall(r"'([^']+)'", match.group(1)))


# --- Text helpers ------------------------------------------------------------------
def test_normalize_collapses_whitespace():
    assert normalize_text("  Hello \n\t  World  ") == "Hello World"


def test_hash_ignores_case_and_spacing():
    assert compute_hash("Hello   World") == compute_hash("hello world")


def test_hash_is_64_hex_chars():
    assert re.fullmatch(r"[0-9a-f]{64}", compute_hash(GOOD_TEXT))


# --- Whole-response checks -----------------------------------------------------------
def test_not_json_rejects_whole_batch():
    result = validate_batch("this is not json", LABEL, set())
    assert result.accepted == []
    assert reasons(result) == [validate.REASON_INVALID_JSON]
    assert result.rejected[0].raw_output == "this is not json"


def test_missing_clauses_key_rejected():
    raw = json.dumps({"items": [clause()]})
    result = validate_batch(raw, LABEL, set())
    assert result.accepted == []
    assert reasons(result) == [validate.REASON_INVALID_JSON]


# --- Per-clause checks -------------------------------------------------------------
def test_valid_clause_accepted():
    result = validate_batch(make_response(clause()), LABEL, set())
    assert result.rejected == []
    assert result.accepted == [AcceptedClause(LABEL, GOOD_TEXT, compute_hash(GOOD_TEXT))]


def test_item_missing_text_is_invalid_json():
    raw = make_response({"label": LABEL}, clause())
    result = validate_batch(raw, LABEL, set())
    assert len(result.accepted) == 1
    assert reasons(result) == [validate.REASON_INVALID_JSON]


def test_wrong_label_rejected():
    result = validate_batch(make_response(clause(label="Notices")), LABEL, set())
    assert result.accepted == []
    assert reasons(result) == [validate.REASON_WRONG_LABEL]


def test_too_short_rejected():
    limit = config.MIN_CLAUSE_CHARS
    raw = make_response(clause(text="a" * (limit - 1)), clause(text="b" * limit))
    result = validate_batch(raw, LABEL, set())
    assert reasons(result) == [validate.REASON_TOO_SHORT]
    assert len(result.accepted) == 1


# --- Duplicate checks --------------------------------------------------------------
def test_duplicate_in_same_batch_rejected():
    raw = make_response(clause(), clause(text=GOOD_TEXT.upper()))
    result = validate_batch(raw, LABEL, set())
    assert len(result.accepted) == 1
    assert reasons(result) == [validate.REASON_DUPLICATE]


def test_duplicate_across_batches_rejected():
    seen = set()
    first = validate_batch(make_response(clause()), LABEL, seen)
    second = validate_batch(make_response(clause()), LABEL, seen)
    assert len(first.accepted) == 1
    assert second.accepted == []
    assert reasons(second) == [validate.REASON_DUPLICATE]
    assert seen == {compute_hash(GOOD_TEXT)}


def test_rejected_clause_does_not_record_hash():
    seen = set()
    raw = make_response(clause(label="Notices"), clause())
    result = validate_batch(raw, LABEL, seen)
    assert reasons(result) == [validate.REASON_WRONG_LABEL]
    assert len(result.accepted) == 1


# --- Realistic batch ---------------------------------------------------------------
def test_mixed_batch_counts():
    raw = make_response(
        clause(),
        clause(text="Too short."),
        clause(label="Confidentiality",
               text="The Receiving Party shall keep all Confidential Information secret."),
    )
    result = validate_batch(raw, LABEL, set())
    assert len(result.accepted) == 1
    assert reasons(result) == [validate.REASON_TOO_SHORT, validate.REASON_WRONG_LABEL]


# --- Contract with the database schema -----------------------------------------------
def test_reason_codes_match_schema():
    code_reasons = {
        validate.REASON_INVALID_JSON,
        validate.REASON_TOO_SHORT,
        validate.REASON_WRONG_LABEL,
        validate.REASON_DUPLICATE,
    }
    assert code_reasons == schema_check_values("reason")


def test_labels_match_schema():
    assert set(config.LABELS) == schema_check_values("label")