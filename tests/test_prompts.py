"""
Unit tests for src/prompts.py. No API calls: everything here is free.

Run from the project root with:  python -m pytest -v
"""

from collections import Counter

from src import config
from src.prompts import (
    LABEL_DEFINITIONS,
    SYSTEM_PROMPT,
    Recipe,
    build_recipes,
    build_user_prompt,
)

GRID_SIZE = (len(config.LABELS) * len(config.CONTRACT_TYPES)
             * len(config.TONES) * len(config.LENGTH_BUCKETS))


def label_spread(recipes) -> int:
    """Difference between the most and least frequent label."""
    counts = Counter(r.label for r in recipes)
    return max(counts.values()) - min(counts.values())


def test_every_label_has_a_definition():
    assert set(LABEL_DEFINITIONS) == set(config.LABELS)


def test_labels_are_balanced():
    recipes = build_recipes(500)
    assert {r.label for r in recipes} == set(config.LABELS)
    assert label_spread(recipes) <= 1


def test_balanced_even_if_run_stops_early():
    assert label_spread(build_recipes(500)[:37]) <= 1


def test_same_seed_gives_same_recipes():
    assert build_recipes(100, seed=1) == build_recipes(100, seed=1)


def test_different_seed_gives_different_recipes():
    assert build_recipes(100, seed=1) != build_recipes(100, seed=2)


def test_no_repeats_until_grid_used_up():
    recipes = build_recipes(GRID_SIZE)
    assert len(set(recipes)) == GRID_SIZE


def test_recipe_values_are_allowed_by_db():
    for r in build_recipes(GRID_SIZE + 50):
        assert r.contract_type in config.CONTRACT_TYPES
        assert r.tone in config.TONES
        assert r.length_bucket in config.LENGTH_BUCKETS


def test_user_prompt_contains_recipe_details():
    recipe = Recipe("Severability", "loan agreement", "plain", "short")
    prompt = build_user_prompt(recipe)
    assert "Severability" in prompt
    assert "loan agreement" in prompt
    assert config.TONES["plain"] in prompt
    assert config.LENGTH_BUCKETS["short"] in prompt
    assert str(config.CLAUSES_PER_CALL) in prompt


def test_system_prompt_mentions_json():
    # OpenAI's JSON mode rejects requests whose messages never mention "JSON"
    assert "JSON" in SYSTEM_PROMPT