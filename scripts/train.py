#!/usr/bin/env python3
"""
Train the invoice expense classifier from the CLI.

Usage:
    python scripts/train.py
    python scripts/train.py --data data/custom_data.json
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.ml.classifier import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Train the invoice expense classifier")
    parser.add_argument("--data", type=Path, default=Path("data/training_data.json"))
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] Training data not found: {args.data}")
        sys.exit(1)

    print(f"Training on: {args.data}")
    m = train(args.data)

    print("\n── Training Complete ─────────────────────────────────")
    print(f"  Samples           : {m['num_samples']}")
    print(f"  Classes           : {', '.join(m['classes'])}")
    print(f"  CV F1 (macro)     : {m['cv_f1_mean']:.4f} ± {m['cv_f1_std']:.4f}")
    print(f"  Confidence thresh : {m['confidence_threshold']}")
    print(f"  Production ready  : {'✅ Yes' if m['production_ready'] else '⚠️  Not yet — add more data'}")
    print(f"  Model saved       : models/classifier.pkl")
    print("──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
