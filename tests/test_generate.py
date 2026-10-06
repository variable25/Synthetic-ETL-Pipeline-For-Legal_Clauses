"""
Unit tests for src/generate.py. Uses a fake client: no internet, no API key, $0.

Run from the project root with:  python -m pytest -v
"""

import json
from types import SimpleNamespace

import pytest

from src import config
from src.generate import estimate_cost_eur, generate_batch
from src.prompts import SYSTEM_PROMPT, Recipe, build_user_prompt
from src.validate import validate_batch

RECIPE = Recipe("Termination", "commercial lease", "formal", "medium")
GOOD_REPLY = json.dumps({"clauses": [{
    "label": "Termination",
    "text": "Either party may terminate this Lease upon ninety days' written notice to the other party.",
}]})


class FakeClient:
    """Stands in for OpenAI: records the request and returns a canned reply."""

    def __init__(self, content, prompt_tokens=400, completion_tokens=700):
        self.last_request = None
        self._reply = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                                  completion_tokens=completion_tokens),
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_request = kwargs
        return self._reply


# --- Cost maths --------------------------------------------------------------------
def test_cost_of_zero_tokens_is_zero():
    assert estimate_cost_eur(0, 0) == 0


def test_cost_matches_price_list():
    expected = (config.PRICE_INPUT_PER_1M_USD + config.PRICE_OUTPUT_PER_1M_USD) * config.USD_TO_EUR
    assert estimate_cost_eur(1_000_000, 1_000_000) == pytest.approx(expected)


# --- Packing the reply -------------------------------------------------------------
def test_generate_batch_packs_reply():
    result = generate_batch(FakeClient(GOOD_REPLY, 400, 700), RECIPE)
    assert result.recipe == RECIPE
    assert result.raw_text == GOOD_REPLY
    assert (result.prompt_tokens, result.completion_tokens) == (400, 700)
    assert result.cost_eur == pytest.approx(estimate_cost_eur(400, 700))


def test_none_content_becomes_empty_string():
    result = generate_batch(FakeClient(None), RECIPE)
    assert result.raw_text == ""


# --- What we send ------------------------------------------------------------------
def test_request_uses_config_and_prompts():
    client = FakeClient(GOOD_REPLY)
    generate_batch(client, RECIPE)
    sent = client.last_request
    assert sent["model"] == config.LLM_MODEL
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert sent["messages"][1] == {"role": "user", "content": build_user_prompt(RECIPE)}
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["max_completion_tokens"] == config.MAX_COMPLETION_TOKENS


# --- E feeds T -----------------------------------------------------------------------
def test_generated_reply_flows_into_validator():
    result = generate_batch(FakeClient(GOOD_REPLY), RECIPE)
    validated = validate_batch(result.raw_text, result.recipe.label, set())
    assert len(validated.accepted) == 1
    assert validated.rejected == []