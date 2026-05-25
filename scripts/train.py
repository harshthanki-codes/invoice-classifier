#!/usr/bin/env python3
"""
Standalone training script. Run from project root:

    python scripts/train.py
    python scripts/train.py --data data/custom_data.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.classifier import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Train the invoice expense classifier")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/training_data.json"),
        help="Path to JSON training data (default: data/training_data.json)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] Training data not found: {args.data}")
        sys.exit(1)

    print(f"Training on: {args.data}")
    metrics = train(args.data)

    print("\n── Training Complete ──────────────────────────")
    print(f"  Samples      : {metrics['num_samples']}")
    print(f"  Classes      : {', '.join(metrics['classes'])}")
    print(f"  CV F1 (macro): {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    print("  Model saved  : models/classifier.pkl")
    print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
