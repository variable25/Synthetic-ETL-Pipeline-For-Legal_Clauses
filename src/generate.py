"""
Extract step: sends one recipe card to the LLM and returns its raw reply.

Does not validate and does not touch the database: pipeline.py passes the
raw text to validate.py and the token counts to db.py.
"""

from dataclasses import dataclass

from openai import OpenAI

from src import config
from src.prompts import SYSTEM_PROMPT, Recipe, build_user_prompt


@dataclass
class GenerationResult:
    recipe: Recipe
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    cost_eur: float


def estimate_cost_eur(prompt_tokens: int, completion_tokens: int) -> float:
    """Convert token counts to euros using the list prices in config."""
    usd = (prompt_tokens * config.PRICE_INPUT_PER_1M_USD
           + completion_tokens * config.PRICE_OUTPUT_PER_1M_USD) / 1_000_000
    return usd * config.USD_TO_EUR


def make_client() -> OpenAI:
    """Create one OpenAI client; the SDK retries 429/5xx errors with backoff."""
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        timeout=config.REQUEST_TIMEOUT_S,
        max_retries=config.MAX_RETRIES,
    )


def build_messages(recipe: Recipe) -> list[dict]:
    """The conversation sent to the model: standing rules, then this request."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(recipe)},
    ]


def generate_batch(client: OpenAI, recipe: Recipe) -> GenerationResult:
    """Send one recipe to the model and pack its reply, tokens and cost."""
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=build_messages(recipe),
        response_format={"type": "json_object"},
        temperature=config.TEMPERATURE,
        max_completion_tokens=config.MAX_COMPLETION_TOKENS,
    )
    raw_text = response.choices[0].message.content or ""
    usage = response.usage
    return GenerationResult(
        recipe=recipe,
        raw_text=raw_text,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost_eur=estimate_cost_eur(usage.prompt_tokens, usage.completion_tokens),
    )