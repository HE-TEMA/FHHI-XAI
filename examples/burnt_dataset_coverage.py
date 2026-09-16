#!/usr/bin/env python
"""Report burnt mask coverage to choose a reference-bank threshold."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.burnt_dataset import BurntDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/Segmentation_Blaze")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    dataset = BurntDataset(root=args.root, split=args.split, require_burnt=False)
    records = []
    for item in dataset.files:
        label = dataset._load_label(item["label"])
        records.append({"name": item["name"], "coverage": float((label == 1).mean()),
                        "ignored": float((label == dataset.ignore_label).mean())})
    fractions = np.asarray([item["coverage"] for item in records])
    print(f"{args.split}: {len(records)} pairs; median burnt coverage {np.median(fractions):.2%}")
    for threshold in (0.0, 0.01, 0.03, 0.05, 0.10):
        print(f"  > {threshold:.0%}: {int((fractions > threshold).sum())} images")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
