"""
Shared scoring: loads the LEDGAR test set and computes the metrics.

Every model (zero-shot baseline, fine-tuned Legal-BERT) is marked by these
same functions, so their numbers are directly comparable.
"""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src import config


# --- 1. The exam ---------------------------------------------------------------------
def load_test_set(path: Path = config.LEDGAR_TEST_PATH) -> tuple[list[str], list[str]]:
    """Read the LEDGAR JSONL file written by ledgar.py into (texts, labels)."""
    texts, labels = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


# --- 2. Marking (pure) -----------------------------------------------------------------
def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Macro-F1, accuracy, per-class scores and the confusion matrix.

    labels= is fixed to all 8 classes, so a class the model never predicts
    still counts in the average (with F1 = 0) instead of silently vanishing.
    """
    labels = list(config.LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1.mean()),  # macro = plain average, every class counts equally
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_samples": len(y_true),
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(s_f1),
                    "support": int(s)}
            for label, p, r, s_f1, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": {   # rows = true label, columns = predicted label
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }


def top_confusions(metrics: dict, n: int = 5) -> list[tuple[str, str, int]]:
    """The n most frequent mistakes as (true label, predicted label, count)."""
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    mistakes = [(labels[i], labels[j], count)
                for i, row in enumerate(matrix)
                for j, count in enumerate(row)
                if i != j and count > 0]
    return sorted(mistakes, key=lambda m: m[2], reverse=True)[:n]


# --- 3. Report ---------------------------------------------------------------------------
def save_report(metrics: dict, name: str) -> Path:
    """Save metrics to results/<name>_metrics.json and print a readable summary."""
    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / f"{name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n=== {name} ===")
    print(f"Macro-F1: {metrics['macro_f1']:.3f}   Accuracy: {metrics['accuracy']:.3f}"
          f"   (n = {metrics['n_samples']})\n")
    print(f"{'label':<17} {'prec':>6} {'rec':>6} {'f1':>6} {'n':>5}")
    for label, m in metrics["per_class"].items():
        print(f"{label:<17} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>5}")

    print("\nMost common mistakes (true -> predicted):")
    for true, pred, count in top_confusions(metrics):
        print(f"  {true} -> {pred}: {count}")
    print(f"\nSaved -> {path}")
    return path