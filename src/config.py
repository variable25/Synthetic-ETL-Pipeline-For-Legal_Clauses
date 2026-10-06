"""
Central configuration for the Synthetic ETL pipeline.

Single source of truth: this is the ONLY module that reads .env.
Every other module imports its settings from here.
"""

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

# --- 1. Load .env into environment variables ---------------------------------
load_dotenv()


# --- 2. Helper ---------------------------------------------------------------
def _require(name: str) -> str:
    """Return an environment variable, or stop with a clear error if missing."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing environment variable: {name}. Check your .env file.")
    return value.strip()


# --- 3. Settings from .env (converted to the right types) --------------------
OPENAI_API_KEY = _require("OPENAI_API_KEY")
LLM_MODEL = _require("LLM_MODEL")
MAX_COST_EUR = float(_require("MAX_COST_EUR"))

POSTGRES_USER = _require("POSTGRES_USER")
POSTGRES_PASSWORD = _require("POSTGRES_PASSWORD")
POSTGRES_DB = _require("POSTGRES_DB")
POSTGRES_HOST = _require("POSTGRES_HOST")
POSTGRES_PORT = int(_require("POSTGRES_PORT"))

DATABASE_URL = (
    f"postgresql://{quote_plus(POSTGRES_USER)}:{quote_plus(POSTGRES_PASSWORD)}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


# --- 4. Label set (MUST match the CHECK constraint in sql/schema.sql) --------
LABELS = (
    "Governing Law",
    "Termination",
    "Confidentiality",
    "Indemnification",
    "Notices",
    "Severability",
    "Assignment",
    "Entire Agreement",
)


# --- 5. Recipe grid dials -----------------------------------------------------
CONTRACT_TYPES = (
    "non-disclosure agreement",
    "employment agreement",
    "software subscription agreement",
    "commercial lease",
    "supply agreement",
    "loan agreement",
    "consulting agreement",
    "partnership agreement",
)

# Short code (stored in DB) -> instruction text (sent to GPT)
TONES = {
    "formal": "formal, modern contract language as drafted by a corporate lawyer",
    "plain": "plain English that a non-lawyer can easily understand",
    "legalese": "traditional legalese using terms like 'hereinafter', 'notwithstanding' and 'whereas'",
}

LENGTH_BUCKETS = {
    "short": "exactly one sentence",
    "medium": "two to three sentences",
    "long": "four to six sentences",
}


# --- 6. Generation settings ---------------------------------------------------
CLAUSES_PER_CALL = 10
MIN_CLAUSE_CHARS = 40       # shorter than this -> rejected as "too_short"
PROMPT_VERSION = "v2"       # bump whenever the prompt text changes
RANDOM_SEED = 42            # makes the recipe-card shuffle reproducible
TEMPERATURE = 0.9           # creativity dial: higher = more variety, fewer duplicates
MAX_COMPLETION_TOKENS = 2500  # output cap per call: a cost seatbelt (~1,800 needed)
REQUEST_TIMEOUT_S = 60      # give up on one HTTP attempt after this many seconds
MAX_RETRIES = 5             # SDK retries on rate limits / server errors, with backoff


# --- 7. Pricing for the cost guard (verify on OpenAI's pricing page) ----------
# gpt-4o-mini list prices in USD per 1M tokens. Last verified: <fill in date>
PRICE_INPUT_PER_1M_USD = 0.15
PRICE_OUTPUT_PER_1M_USD = 0.60
USD_TO_EUR = 1.0            # deliberately conservative: overestimates EUR cost

# --- 8. File locations (absolute, so scripts work from any folder) ---------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LEDGAR_TEST_PATH = DATA_DIR / "ledgar_test.jsonl"