"""
Prompt templates and the recipe grid for synthetic clause generation.

Pure module: builds text and recipe lists only, never calls the API.
Any change to the wording here means bumping config.PROMPT_VERSION.
"""

import itertools
import random
from dataclasses import dataclass

from src import config


# --- 1. What each label means (so GPT doesn't mix up similar labels) -----------
# Each definition completes the sentence "a clause that ..."
LABEL_DEFINITIONS = {
    "Governing Law": "states which jurisdiction's laws govern the contract and its interpretation",
    "Termination": "explains when and how a party may end the contract and what happens afterwards",
    "Confidentiality": "obliges a party to keep the other party's non-public information secret and limits its use",
    "Indemnification": "requires one party to compensate the other for losses, damages or claims caused by specified events",
    "Notices": "specifies how formal communications between the parties must be delivered and when they take effect",
    "Severability": "states that if one provision is invalid or unenforceable, the rest of the contract stays in force",
    "Assignment": "controls whether a party may transfer its rights or obligations under the contract to someone else",
    "Entire Agreement": "declares that the written contract is the complete agreement and replaces all earlier discussions",
}


# --- 2. Standing instructions (MUST describe the shape validate.py expects) -----
SYSTEM_PROMPT = """You are an experienced contract lawyer who drafts realistic clauses for commercial contracts.

Rules:
- Write each clause exactly as it would appear inside a signed contract.
- Do NOT start a clause with a heading, title or clause number.
- Every clause must clearly be of the requested type and must not drift into other clause types.
- Make the clauses in one response clearly different from each other in wording and content.
- Use defined terms such as "the Company", "the Supplier" or "the Employee" instead of real names.

Respond with ONLY a JSON object in exactly this shape, with no other text:
{"clauses": [{"label": "<clause type>", "text": "<clause text>"}]}"""


# --- 3. One recipe card = one API call -----------------------------------------
@dataclass(frozen=True)
class Recipe:
    label: str
    contract_type: str
    tone: str           # key of config.TONES (stored in DB)
    length_bucket: str  # key of config.LENGTH_BUCKETS (stored in DB)


def build_recipes(n_calls: int, seed: int = config.RANDOM_SEED) -> list[Recipe]:
    """Return n_calls recipes; labels take turns so every class gets equal calls.

    Each label's (contract type, tone, length) combos are shuffled once and
    then used in order, wrapping around if a label needs more calls than combos.
    """
    rng = random.Random(seed)

    combos_per_label = {}
    for label in config.LABELS:
        combos = list(itertools.product(
            config.CONTRACT_TYPES, config.TONES, config.LENGTH_BUCKETS))
        rng.shuffle(combos)
        combos_per_label[label] = combos

    recipes = []
    for i in range(n_calls):
        label = config.LABELS[i % len(config.LABELS)]
        turn = i // len(config.LABELS)
        combos = combos_per_label[label]
        contract_type, tone, length_bucket = combos[turn % len(combos)]
        recipes.append(Recipe(label, contract_type, tone, length_bucket))
    return recipes


def build_user_prompt(recipe: Recipe, n_clauses: int = config.CLAUSES_PER_CALL) -> str:
    """Fill the request template for one recipe card."""
    return (
        f"Write {n_clauses} different {recipe.label} clauses for a {recipe.contract_type}.\n"
        f"Definition: {recipe.label} = a clause that {LABEL_DEFINITIONS[recipe.label]}.\n"
        f"Style: {config.TONES[recipe.tone]}.\n"
        f"Length: each clause must be {config.LENGTH_BUCKETS[recipe.length_bucket]}.\n"
        f'Set "label" to exactly "{recipe.label}" for every clause.'
    )