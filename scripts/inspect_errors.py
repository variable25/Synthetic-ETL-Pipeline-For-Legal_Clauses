"""
Error inspection: print a few LEDGAR clauses the fine-tuned model got wrong.

Read-only: nothing is saved and nothing is tuned. Feeds the README's error analysis.

Run from the project root with:
    python -m scripts.inspect_errors
    python -m scripts.inspect_errors --true Notices --pred Indemnification
"""

import argparse
import random

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config
from src.evaluate import load_test_set
from src.score import MODEL_DIR, predict

SHOW_CHARS = 500        # long clauses are clipped when printed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", default="Notices", choices=config.LABELS)
    parser.add_argument("--pred", default="Termination", choices=config.LABELS)
    parser.add_argument("-n", type=int, default=5, help="how many examples to show")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

    texts, labels = load_test_set()
    predictions = predict(texts, tokenizer, model, device)

    errors = [text for text, label, pred in zip(texts, labels, predictions)
              if label == args.true and pred == args.pred]
    sample = random.Random(config.RANDOM_SEED).sample(errors, min(args.n, len(errors)))

    print(f"{len(errors)} clauses labelled {args.true!r} were predicted as {args.pred!r}. "
          f"Showing {len(sample)}:\n")
    for i, text in enumerate(sample, 1):
        clipped = text if len(text) <= SHOW_CHARS else text[:SHOW_CHARS] + " [...]"
        print(f"--- {i} ({len(text.split())} words)\n{clipped}\n")


if __name__ == "__main__":
    main()