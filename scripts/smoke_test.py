"""
Smoke test: ONE real API call (~€0.0005), passed through the validator.
Nothing is written to the database.

Run from the project root with:  python -m scripts.smoke_test
"""

from src import config
from src.generate import generate_batch, make_client
from src.prompts import build_recipes
from src.validate import validate_batch


def main() -> None:
    recipe = build_recipes(1)[0]
    print(f"Model:  {config.LLM_MODEL}")
    print(f"Recipe: {recipe}\n")

    result = generate_batch(make_client(), recipe)
    validated = validate_batch(result.raw_text, recipe.label, set())

    print(f"--- Accepted ({len(validated.accepted)}) ---")
    for i, clause in enumerate(validated.accepted, start=1):
        print(f"{i:2}. {clause.text}\n")

    print(f"--- Rejected ({len(validated.rejected)}) ---")
    for item in validated.rejected:
        print(f"  [{item.reason}] {item.raw_output[:100]}")

    print("\n--- Cost ---")
    print(f"Tokens: {result.prompt_tokens} in / {result.completion_tokens} out")
    print(f"Cost:   EUR {result.cost_eur:.6f}")


if __name__ == "__main__":
    main()