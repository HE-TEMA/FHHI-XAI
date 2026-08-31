#!/usr/bin/env python
"""Render PCX explanation figures for several samples x several prototype counts.

Requires the CRP/PCX artifacts from `examples/build_crp_pcx_multilayer.py` for
the chosen layer.

Usage:

    python examples/render_explanations.py --dataset fire --layer 5 --prototypes 2 3 4 6 --n-samples 4
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from LCRP.models.pidnet import get_pidnet, infer_checkpoint_geometry

from examples.build_crp_pcx_multilayer import DATASETS, build_dataset
from src.plotpcx_gpu import plot_pcx_explanations_pidnet

LAYER_ALIASES = {
    "1": "layer1.0.conv1", "2": "layer2.0.conv1", "3": "layer3.0.conv1",
    "4": "layer4.0.conv1", "5": "layer5.0.conv1", "spp": "spp.scale1.3",
}

def resolve_layer(name: str) -> str:
    key = name.lower().replace("layer", "") if name.lower().startswith("layer") else name.lower()
    return LAYER_ALIASES.get(key, name)


def pick_samples(dataset, n: int) -> list:
    """Indices spread across the label's positive-pixel fraction."""
    fractions = np.array(
        [(dataset.load_sample(i)[1] == 1).mean() for i in range(len(dataset))],
        dtype=np.float32,
    )
    usable = np.flatnonzero(fractions > 0)
    if usable.size == 0:
        raise ValueError("No sample has a positive label.")
    order = usable[np.argsort(fractions[usable])]
    hi = int(0.98 * len(order))
    picks = np.linspace(0, max(hi - 1, 0), num=min(n, len(order))).astype(int)
    return [(int(order[p]), float(fractions[order[p]])) for p in picks]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="fire")
    parser.add_argument("--layer", default="5",
                        help="'5', 'layer5' or the full name; see LAYER_ALIASES")
    parser.add_argument("--prototypes", type=int, nargs="+", default=[2, 3, 4, 6],
                        help="one figure per value")
    parser.add_argument("--samples", type=int, nargs="+", default=None,
                        help="explicit dataset indices; overrides --n-samples")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-concepts", type=int, default=3)
    parser.add_argument("--n-refimgs", type=int, default=8)
    parser.add_argument("--outlier-percentile", type=float, default=1.0,
                        help="share of bank samples flagged as outliers (default 1)")
    parser.add_argument("--use-rf", action="store_true",
                        help="crop reference images to the concept's receptive field")
    parser.add_argument("--out-crp", type=Path, default=None)
    parser.add_argument("--out-pcx", type=Path, default=None)
    parser.add_argument("--ref-images", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--modality", choices=["rgb", "ir", "all"], default="rgb")
    parser.add_argument("--keep-empty", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for name in ("PIL", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("LCRP.utils.crp").setLevel(logging.CRITICAL)

    layer = resolve_layer(args.layer)
    spec = DATASETS[args.dataset]
    args.out_crp = args.out_crp or PROJECT_ROOT / f"output/crp/pidnet_{args.dataset}_sweep"
    args.out_pcx = args.out_pcx or PROJECT_ROOT / f"output/pcx/pidnet_{args.dataset}_sweep"
    args.ref_images = args.ref_images or PROJECT_ROOT / (
        f"output/ref_imgs_{args.dataset}_rf" if args.use_rf else f"output/ref_imgs_{args.dataset}")
    args.out_dir = args.out_dir or PROJECT_ROOT / f"output/figures/{args.dataset}_{layer}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bank = args.out_pcx / layer / "attributions.npy"
    if not bank.is_file():
        raise FileNotFoundError(
            f"{bank} missing. Run build_crp_pcx_multilayer.py for this layer first."
        )

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = str(PROJECT_ROOT / spec["checkpoint"])
    in_channels, classes = infer_checkpoint_geometry(checkpoint)
    model = get_pidnet(device=device, ckpt_path=checkpoint,
                       classes=classes, in_channels=in_channels).eval()

    extra = ({"modality": args.modality, "require_fire": not args.keep_empty}
             if args.dataset == "fire" else {})
    dataset = build_dataset(args.dataset, PROJECT_ROOT / spec["data_root"], **extra)

    n_rows = len(np.load(bank))
    if n_rows != len(dataset):
        raise ValueError(
            f"Bank has {n_rows} rows but the dataset has {len(dataset)}. The bank was "
            "built with different options; prototype images would be mismatched."
        )

    if args.samples:
        chosen = [(i, float((dataset.load_sample(i)[1] == 1).mean())) for i in args.samples]
    else:
        chosen = pick_samples(dataset, args.n_samples)

    print(f"dataset={args.dataset} layer={layer} images={len(dataset)}")
    print(f"samples: {[(i, f'{f:.4%}') for i, f in chosen]}")
    print(f"prototypes: {args.prototypes} | outlier percentile: {args.outlier_percentile}")

    index = []
    for sample_idx, fraction in chosen:
        image, _ = dataset[sample_idx]
        for k in args.prototypes:
            out = args.out_dir / f"sample{sample_idx:05d}_k{k}.png"
            fig = plot_pcx_explanations_pidnet(
                "pidnet", model, dataset, image_tensor=image, layer_name=layer,
                n_concepts=args.n_concepts, n_refimgs=args.n_refimgs,
                num_prototypes=k, ref_imgs_path=str(args.ref_images),
                output_dir_crp=str(args.out_crp), output_dir_pcx=str(args.out_pcx),
                device=device, precision="fp32",
                outlier_percentile=args.outlier_percentile, use_rf=args.use_rf,
            )
            fig.set_size_inches(18, 11)
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            index.append({
                "sample": sample_idx,
                "file": dataset.files[sample_idx],
                "label_fraction": round(fraction, 4),
                "num_prototypes": k,
                "concepts_used": int(getattr(fig, "_n_concepts_used", args.n_concepts)),
                "png": out.name,
            })
            print(f"  sample {sample_idx:5d} (Anteil {fraction:.4%})  k={k}  -> {out.name}")

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\n{len(index)} figures in {args.out_dir}")


if __name__ == "__main__":
    main()
