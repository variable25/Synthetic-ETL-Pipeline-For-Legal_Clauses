"""
Final exam: the fine-tuned Legal-BERT is scored on the real LEDGAR clauses.

Uses the same functions as the zero-shot baseline (src/evaluate.py), so the
two results are directly comparable. Run once: nothing here is tuned on LEDGAR.

Run from the project root with:  python -m src.score
"""

import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config
from src.evaluate import compute_metrics, load_test_set, save_report
from src.train import MAX_TOKENS, MODEL_ID
from src.train import OUTPUT_DIR as MODEL_DIR

BATCH_SIZE = 32         # inference only: no gradients, so bigger batches fit
BASELINE_PATH = config.RESULTS_DIR / "baseline_metrics.json"


# --- 1. Pure helper ---------------------------------------------------------------------
def compare_reports(baseline: dict, finetuned: dict) -> list[tuple[str, float, float, float]]:
    """Per-label F1 side by side as (label, baseline, fine-tuned, change), macro-F1 last."""
    rows = [(label, baseline["per_class"][label]["f1"], finetuned["per_class"][label]["f1"])
            for label in config.LABELS]
    rows.append(("Macro-F1", baseline["macro_f1"], finetuned["macro_f1"]))
    return [(label, old, new, new - old) for label, old, new in rows]


# --- 2. Prediction --------------------------------------------------------------------
def count_truncated(texts: list[str], tokenizer) -> int:
    """How many clauses are longer than MAX_TOKENS and will be cut off."""
    return sum(len(ids) > MAX_TOKENS for ids in tokenizer(texts)["input_ids"])


@torch.inference_mode()
def predict(texts: list[str], tokenizer, model, device: str) -> list[str]:
    """Return one predicted label name per clause."""
    predictions = []
    for start in range(0, len(texts), BATCH_SIZE):
        inputs = tokenizer(texts[start:start + BATCH_SIZE], truncation=True,
                           max_length=MAX_TOKENS, padding=True,
                           return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            logits = model(**inputs).logits
        predictions += [model.config.id2label[i] for i in logits.argmax(dim=1).tolist()]
    return predictions


# --- 3. Entry point -------------------------------------------------------------------
def main() -> None:
    if not MODEL_DIR.exists():
        raise SystemExit(f"{MODEL_DIR} not found. Run: python -m src.train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

    texts, labels = load_test_set()
    n_truncated = count_truncated(texts, tokenizer)
    print(f"Fine-tuned Legal-BERT on {len(texts)} LEDGAR clauses ({device}); "
          f"{n_truncated} are longer than {MAX_TOKENS} tokens and get cut off")
    predictions = predict(texts, tokenizer, model, device)

    metrics = compute_metrics(labels, predictions)
    metrics["model"] = f"{MODEL_ID} fine-tuned on synthetic clauses"
    metrics["n_truncated"] = n_truncated
    save_report(metrics, "legal_bert")

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    print(f"\n{'label':<17} {'baseline':>8} {'fine-tuned':>10} {'change':>7}")
    for label, old, new, change in compare_reports(baseline, metrics):
        print(f"{label:<17} {old:>8.3f} {new:>10.3f} {change:>+7.3f}")


if __name__ == "__main__":
    main()