"""
Zero-shot baseline: an NLI model labels LEDGAR clauses without any training.

Each clause is paired with one hypothesis per label ("This is a Notices
clause."). The label whose hypothesis the model finds most entailed wins.

Run from the project root with:  python -m src.baseline
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config
from src.evaluate import compute_metrics, load_test_set, save_report

MODEL_ID = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This is a {} clause."  # fixed up front; never tuned on LEDGAR
CLAUSES_PER_BATCH = 8   # x 8 labels = 64 clause/hypothesis pairs per GPU pass
MAX_TOKENS = 512        # longer clauses are cut off; most LEDGAR clauses fit


# --- 1. Pure helpers --------------------------------------------------------------------
def build_pairs(texts: list[str]) -> tuple[list[str], list[str]]:
    """Pair every clause with every label's hypothesis, clause by clause."""
    hypotheses = [HYPOTHESIS_TEMPLATE.format(label) for label in config.LABELS]
    premises = [text for text in texts for _ in hypotheses]
    return premises, hypotheses * len(texts)


def entailment_index(model) -> int:
    """Which of the model's 3 outputs (contradiction/neutral/entailment) is 'entailment'."""
    for name, index in model.config.label2id.items():
        if name.lower().startswith("entail"):
            return index
    raise ValueError(f"No entailment label in {model.config.label2id}")


# --- 2. Prediction --------------------------------------------------------------------
@torch.inference_mode()
def predict(texts: list[str], tokenizer, model, device: str) -> list[str]:
    """Return one predicted label per clause."""
    entail = entailment_index(model)
    n_labels = len(config.LABELS)
    predictions = []

    for start in range(0, len(texts), CLAUSES_PER_BATCH):
        batch = texts[start:start + CLAUSES_PER_BATCH]
        premises, hypotheses = build_pairs(batch)
        inputs = tokenizer(premises, hypotheses, truncation="only_first",
                           max_length=MAX_TOKENS, padding=True,
                           return_tensors="pt").to(device)

        logits = model(**inputs).logits                         # (pairs, 3)
        scores = logits[:, entail].view(len(batch), n_labels)   # (clauses, labels)
        predictions += [config.LABELS[i] for i in scores.argmax(dim=1).tolist()]

        if (start // CLAUSES_PER_BATCH) % 50 == 0:
            print(f"  {start + len(batch)}/{len(texts)} clauses")
    return predictions


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    if device == "cuda":
        model = model.half()  # fp16: half the memory, faster, same answers in practice
    model = model.to(device).eval()

    texts, labels = load_test_set()
    print(f"Zero-shot {MODEL_ID} on {len(texts)} LEDGAR clauses ({device})")
    predictions = predict(texts, tokenizer, model, device)

    metrics = compute_metrics(labels, predictions)
    metrics["model"] = MODEL_ID
    metrics["hypothesis_template"] = HYPOTHESIS_TEMPLATE
    save_report(metrics, "baseline")


if __name__ == "__main__":
    main()