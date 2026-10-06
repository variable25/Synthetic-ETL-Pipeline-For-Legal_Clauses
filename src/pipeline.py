"""
Pipeline orchestrator: runs Extract -> Transform -> Load for one generation run.

Hands out recipe cards, checks the budget before every API call and saves
each batch in its own transaction. Holds no SQL and no prompt text itself.

Run from the project root with:  python -m src.pipeline --samples 80
"""

import argparse
import logging
import math

import psycopg
from openai import OpenAI

from src import config, db
from src.generate import (GenerationResult, build_messages, estimate_cost_eur,
                          generate_batch, make_client)
from src.prompts import Recipe, build_recipes
from src.validate import REASON_DUPLICATE, ValidationResult, validate_batch

logger = logging.getLogger(__name__)


# --- 1. Run statuses (MUST match the CHECK constraint in sql/schema.sql) ---------
STATUS_COMPLETED = "completed"
STATUS_BUDGET_STOPPED = "budget_stopped"
STATUS_FAILED = "failed"

CHARS_PER_TOKEN_LOW = 3  # real average is ~4, so dividing by 3 overestimates tokens


# --- 2. Budget guard (pure) ------------------------------------------------------
def worst_case_cost_eur(recipe: Recipe) -> float:
    """Most the next call can cost: an overestimated prompt plus the full output cap."""
    prompt_chars = sum(len(message["content"]) for message in build_messages(recipe))
    prompt_tokens = math.ceil(prompt_chars / CHARS_PER_TOKEN_LOW)
    return estimate_cost_eur(prompt_tokens, config.MAX_COMPLETION_TOKENS)


def budget_allows(spent_eur: float, next_cost_eur: float, budget_eur: float) -> bool:
    """True if the next call cannot push total spending over the budget."""
    return spent_eur + next_cost_eur <= budget_eur


# --- 3. Load one batch -------------------------------------------------------------
def save_batch(conn: psycopg.Connection, run_id: int, result: GenerationResult,
               validated: ValidationResult) -> tuple[int, int]:
    """Write one validated batch and its stats. Does not commit.

    A clause can pass validation but still duplicate one from an EARLIER run;
    the database's unique hash catches it, and it is stored as rejected instead.
    Returns (n_accepted, n_rejected) as actually written.
    """
    recipe = result.recipe
    n_accepted = 0
    n_rejected = len(validated.rejected)

    for clause in validated.accepted:
        inserted = db.insert_sample(conn, run_id, clause.label, clause.text,
                                    clause.text_hash, recipe.contract_type,
                                    recipe.tone, recipe.length_bucket)
        if inserted:
            n_accepted += 1
        else:
            db.insert_rejected(conn, run_id, clause.text, REASON_DUPLICATE)
            n_rejected += 1

    for item in validated.rejected:
        db.insert_rejected(conn, run_id, item.raw_output, item.reason)

    db.add_batch_stats(conn, run_id, result.prompt_tokens, result.completion_tokens,
                       result.cost_eur, n_accepted, n_rejected)
    return n_accepted, n_rejected


# --- 4. Orchestration ---------------------------------------------------------------
def run_pipeline(client: OpenAI, conn: psycopg.Connection, target_samples: int) -> int:
    """Run one generation run end to end and return its run_id."""
    run_id = db.create_run(conn, config.LLM_MODEL, config.PROMPT_VERSION, target_samples)
    conn.commit()  # the logbook row survives even if a later batch crashes

    n_calls = math.ceil(target_samples / config.CLAUSES_PER_CALL)
    recipes = build_recipes(n_calls)
    seen_hashes: set[str] = set()
    logger.info("Run %d started: %d calls for ~%d samples", run_id, n_calls, target_samples)

    status = STATUS_COMPLETED
    try:
        for i, recipe in enumerate(recipes, start=1):
            spent = db.get_total_cost(conn)
            if not budget_allows(spent, worst_case_cost_eur(recipe), config.MAX_COST_EUR):
                logger.warning("Budget guard: EUR %.4f of %.2f spent, stopping before call %d",
                               spent, config.MAX_COST_EUR, i)
                status = STATUS_BUDGET_STOPPED
                break

            result = generate_batch(client, recipe)                              # E
            validated = validate_batch(result.raw_text, recipe.label, seen_hashes)  # T
            n_accepted, n_rejected = save_batch(conn, run_id, result, validated)    # L
            conn.commit()  # all-or-nothing per batch

            logger.info("[%d/%d] %-16s | +%d accepted, %d rejected | spent EUR %.4f",
                        i, n_calls, recipe.label, n_accepted, n_rejected,
                        spent + result.cost_eur)
    except BaseException:  # BaseException also catches Ctrl+C (KeyboardInterrupt)
        conn.rollback()
        db.finish_run(conn, run_id, STATUS_FAILED)
        conn.commit()
        logger.error("Run %d marked as '%s'", run_id, STATUS_FAILED)
        raise

    db.finish_run(conn, run_id, status)
    conn.commit()
    logger.info("Run %d finished with status '%s'", run_id, status)
    return run_id


# --- 5. Command line entry point ----------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic clauses into Postgres.")
    parser.add_argument("--samples", type=int, required=True,
                        help="target number of samples for this run")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be a positive number")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # the openai SDK's HTTP library; hides one log line per request

    client = make_client()
    with db.get_connection() as conn:
        run_pipeline(client, conn, args.samples)
        logger.info("Samples in database per label:")
        for label, count in db.count_by_label(conn).items():
            logger.info("  %-16s %d", label, count)


if __name__ == "__main__":
    main()