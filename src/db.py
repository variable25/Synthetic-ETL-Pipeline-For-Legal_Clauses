"""
Database access layer: every SQL statement the pipeline needs lives here.

Functions never commit. The caller (pipeline.py) decides transaction
boundaries, so one API batch is saved all-or-nothing.
"""

import psycopg

from src import config


def get_connection() -> psycopg.Connection:
    """Open a new connection to the Postgres database defined in config."""
    return psycopg.connect(config.DATABASE_URL)


# --- generation_runs (the logbook) --------------------------------------------
def create_run(conn: psycopg.Connection, model: str, prompt_version: str,
               target_samples: int) -> int:
    """Insert a new run with status 'running' and return its run_id."""
    row = conn.execute(
        """
        INSERT INTO generation_runs (model, prompt_version, target_samples)
        VALUES (%s, %s, %s)
        RETURNING run_id
        """,
        (model, prompt_version, target_samples),
    ).fetchone()
    return row[0]


def add_batch_stats(conn: psycopg.Connection, run_id: int, prompt_tokens: int,
                    completion_tokens: int, cost_eur: float,
                    n_accepted: int, n_rejected: int) -> None:
    """Add one batch's token usage, cost and accept/reject counts to a run."""
    conn.execute(
        """
        UPDATE generation_runs
        SET prompt_tokens     = prompt_tokens + %s,
            completion_tokens = completion_tokens + %s,
            cost_eur          = cost_eur + %s,
            n_accepted        = n_accepted + %s,
            n_rejected        = n_rejected + %s
        WHERE run_id = %s
        """,
        (prompt_tokens, completion_tokens, cost_eur, n_accepted, n_rejected, run_id),
    )


def get_total_cost(conn: psycopg.Connection) -> float:
    """Total EUR spent across ALL runs. The cost guard compares this to the budget."""
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_eur), 0) FROM generation_runs"
    ).fetchone()
    return float(row[0])


def finish_run(conn: psycopg.Connection, run_id: int, status: str) -> None:
    """Mark a run as finished with its final status."""
    conn.execute(
        """
        UPDATE generation_runs
        SET status = %s, finished_at = now()
        WHERE run_id = %s
        """,
        (status, run_id),
    )


# --- synthetic_samples (the good shelf) ---------------------------------------
def insert_sample(conn: psycopg.Connection, run_id: int, label: str,
                  clause_text: str, text_hash: str, contract_type: str,
                  tone: str, length_bucket: str) -> bool:
    """Insert one accepted sample. Returns False if its hash already exists."""
    row = conn.execute(
        """
        INSERT INTO synthetic_samples
            (run_id, label, clause_text, text_hash, contract_type, tone, length_bucket)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (text_hash) DO NOTHING
        RETURNING sample_id
        """,
        (run_id, label, clause_text, text_hash, contract_type, tone, length_bucket),
    ).fetchone()
    return row is not None


def count_by_label(conn: psycopg.Connection) -> dict[str, int]:
    """Number of accepted samples per label, for checking class balance."""
    rows = conn.execute(
        """
        SELECT label, COUNT(*)
        FROM synthetic_samples
        GROUP BY label
        ORDER BY label
        """
    ).fetchall()
    return {label: count for label, count in rows}


# --- rejected_samples (the reject bin) ----------------------------------------
def insert_rejected(conn: psycopg.Connection, run_id: int, raw_output: str,
                    reason: str) -> None:
    """Store a rejected generation with the reason it failed validation."""
    conn.execute(
        """
        INSERT INTO rejected_samples (run_id, raw_output, reason)
        VALUES (%s, %s, %s)
        """,
        (run_id, raw_output, reason),
    )