"""Tests for pipeline.py's pure helpers and its contract with the schema (offline, free)."""

import re
from pathlib import Path

from src import pipeline
from src.generate import estimate_cost_eur
from src.pipeline import budget_allows, worst_case_cost_eur
from src.prompts import build_recipes
from src import config

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


# --- Budget guard -----------------------------------------------------------------
def test_budget_allows_when_under_budget():
    assert budget_allows(0.50, 0.01, 1.00) is True


def test_budget_allows_exactly_at_budget():
    assert budget_allows(0.75, 0.25, 1.00) is True


def test_budget_blocks_when_next_call_would_exceed():
    assert budget_allows(0.995, 0.01, 1.00) is False


def test_worst_case_exceeds_a_real_call():
    recipe = build_recipes(1)[0]
    assert worst_case_cost_eur(recipe) > 0.00033  # smoke test: ~230 in / ~490 out tokens


def test_worst_case_covers_full_output_cap():
    recipe = build_recipes(1)[0]
    assert worst_case_cost_eur(recipe) >= estimate_cost_eur(0, config.MAX_COMPLETION_TOKENS)


# --- Contract with the database schema ----------------------------------------------
def test_statuses_match_schema():
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    match = re.search(r"status IN \(([^)]*)\)", sql)
    assert match, "CHECK constraint for 'status' not found in schema.sql"
    schema_statuses = set(re.findall(r"'([^']+)'", match.group(1)))
    code_statuses = {pipeline.STATUS_COMPLETED, pipeline.STATUS_BUDGET_STOPPED,
                     pipeline.STATUS_FAILED}
    assert code_statuses <= schema_statuses  # 'running' is the DB default, not set by code