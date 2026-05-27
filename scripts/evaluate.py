#!/usr/bin/env python3
"""
Model evaluation script — produces per-class F1, confusion matrix,
and a markdown report you can paste directly into your GitHub README.

Run: python scripts/evaluate.py
     python scripts/evaluate.py --data data/training_data.json --output reports/eval.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.ml.classifier import build_pipeline, preprocess

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def generate_markdown_report(
    report_dict: dict,
    cm: np.ndarray,
    classes: list,
    n_samples: int,
    cv_f1: float,
    cv_std: float,
) -> str:
    lines = [
        "## Model Evaluation Report\n",
        f"**Training samples:** {n_samples}  ",
        f"**CV F1-macro:** `{cv_f1:.4f} ± {cv_std:.4f}`  ",
        f"**Production ready:** {'✅ Yes' if cv_f1 >= 0.85 else '⚠️ Not yet — add more training data'}",
        "",
        "### Per-Class Performance\n",
        "| Category | Precision | Recall | F1-Score | Support |",
        "|---|---|---|---|---|",
    ]
    for cls in classes:
        m = report_dict.get(cls, {})
        lines.append(
            f"| {cls} | {m.get('precision', 0):.2f} | {m.get('recall', 0):.2f} "
            f"| {m.get('f1-score', 0):.2f} | {int(m.get('support', 0))} |"
        )

    lines += [
        "",
        "### Confusion Matrix\n",
        "Rows = actual, Columns = predicted\n",
        "| | " + " | ".join(f"**{c[:8]}**" for c in classes) + " |",
        "|---|" + "---|" * len(classes),
    ]
    for i, row in enumerate(cm):
        lines.append("| **" + classes[i][:8] + "** | " + " | ".join(str(v) for v in row) + " |")

    lines += [
        "",
        "> **Confidence threshold:** Predictions below `0.72` return `review_recommended: true`.",
        "> Tune this threshold based on your acceptable false-positive rate in production.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate invoice classifier")
    parser.add_argument("--data", type=Path, default=Path("data/training_data.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.md"))
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] Data not found: {args.data}")
        sys.exit(1)

    with open(args.data) as f:
        records = json.load(f)

    texts = [r["text"] for r in records]
    raw_labels = [r["category"] for r in records]

    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)
    classes = list(le.classes_)

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running 5-fold cross-validation...")
    y_pred = cross_val_predict(pipeline, texts, labels, cv=cv)

    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="f1_macro")

    report_dict = classification_report(labels, y_pred, target_names=classes, output_dict=True)
    report_str = classification_report(labels, y_pred, target_names=classes)
    cm = confusion_matrix(labels, y_pred)

    print("\n── Classification Report ──────────────────────────────")
    print(report_str)
    print(f"CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Production ready: {cv_scores.mean() >= 0.85}")
    print("────────────────────────────────────────────────────────\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    md = generate_markdown_report(report_dict, cm, classes, len(texts), cv_scores.mean(), cv_scores.std())
    args.output.write_text(md)
    print(f"Markdown report saved → {args.output}")


if __name__ == "__main__":
    main()
