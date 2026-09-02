"""
Per-image flood coverage analytics for General_Flood_v4 / TEMA_flood.

For every (image, mask) pair in the given list files, computes the fraction
of valid (non-ignore) pixels that are flood, plus raw pixel counts. Written
as a standalone script (not a notebook) since it just needs to run once over
the full dataset and dump a JSON — see examples/flood_v4_metrics.ipynb for
interactive analysis of the resulting file.

Usage:
    python examples/flood_dataset_coverage.py \
        --data-root data/General_Flood_v4 \
        --list PIDNet_brk2/data/lists/val_tema_9010.lst:val \
        --list PIDNet_brk2/data/lists/train_tema_9010.lst:train \
        --out examples/output/flood_v4_coverage.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.flood_dataset_metrics import FloodDataset

COLOR_LIST = FloodDataset.color_list  # [[0, 0, 0], [1, 1, 1]] -> background, flood


def iter_list_file(list_path):
    with open(list_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            yield parts[0], parts[1]


def compute_coverage(data_root: Path, list_path: Path, split: str):
    records = []
    for img_rel, mask_rel in iter_list_file(list_path):
        mask = np.array(Image.open(data_root / mask_rel).convert("RGB"))
        total = mask.shape[0] * mask.shape[1]

        is_bg = (mask == COLOR_LIST[0]).all(axis=2)
        is_flood = (mask == COLOR_LIST[1]).all(axis=2)
        n_flood = int(is_flood.sum())
        n_bg = int(is_bg.sum())
        n_ignore = total - n_flood - n_bg
        valid = n_flood + n_bg
        flood_frac = (n_flood / valid) if valid > 0 else float("nan")

        records.append({
            "split": split,
            "subset": Path(mask_rel).parts[1],  # e.g. TEMA_flood/brk_1/... -> "brk_1"
            "img": img_rel,
            "mask": mask_rel,
            "h": int(mask.shape[0]),
            "w": int(mask.shape[1]),
            "n_flood": n_flood,
            "n_bg": n_bg,
            "n_ignore": n_ignore,
            "flood_frac": flood_frac,
        })
    return records


def summarize(records):
    fracs = np.array([r["flood_frac"] for r in records if not np.isnan(r["flood_frac"])])
    bins = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0000001]
    labels = ["0%", "0-1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-100%"]
    counts, _ = np.histogram(fracs, bins=[-1e-9] + bins[1:])
    # first bucket "exactly 0%" handled separately for readability
    n_zero = int((fracs == 0).sum())
    n_nonzero_bins = np.histogram(fracs[fracs > 0], bins=bins[1:])[0] if (fracs > 0).any() else np.zeros(len(bins) - 2)

    print(f"n images: {len(records)}  (valid flood_frac: {len(fracs)})")
    print(f"mean flood coverage: {fracs.mean():.4f}  median: {np.median(fracs):.4f}")
    print(f"images with 0% flood: {n_zero} ({n_zero / len(fracs):.1%})")
    for label, n in zip(labels[1:], n_nonzero_bins):
        print(f"  {label:>8}: {int(n)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument(
        "--list", action="append", required=True,
        help="path:split, e.g. PIDNet_brk2/data/lists/val_tema_9010.lst:val",
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    all_records = []
    t0 = time.time()
    for spec in args.list:
        list_path, split = spec.rsplit(":", 1)
        recs = compute_coverage(args.data_root, Path(list_path), split)
        print(f"[{split}] {len(recs)} images in {time.time() - t0:.1f}s")
        all_records.extend(recs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_records, f)
    print(f"wrote {len(all_records)} records to {args.out}")

    summarize(all_records)


if __name__ == "__main__":
    main()
