#!/usr/bin/env python
"""Build CRP statistics and PCX attribution banks for several layers in one pass.

Running the pipeline once per candidate layer wastes most of its time on
data loading and forward passes. `run_analysis` accepts a list of record layers,
and the attribution call does too, so both phases visit the dataset exactly once
regardless of how many layers are requested.

Example:

    python examples/build_crp_pcx_multilayer.py \
        --layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 \
                 layer4.0.conv1 layer5.0.conv1 spp.scale1.3
"""
import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from crp.concepts import ChannelConcept

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LCRP.models.pidnet import get_pidnet, infer_checkpoint_geometry
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, COMPOSITES
from src.glocal_analysis import run_analysis
from src.pidnet_bank import burnt_artifact_metadata

# All tasks use PIDNet and share the CRP/PCX machinery.
DATASETS = {
    "burnt": {
        "module": "src.datasets.burnt_dataset",
        "cls": "BurntDataset",
        "checkpoint": "models/pidnet_s_burnt.pt",
        "data_root": "data/Segmentation_Blaze",
        "class_id": 1,
    },
    "flood": {
        "module": "src.datasets.flood_dataset_crp",
        "cls": "FloodDataset",
        "checkpoint": "models/flood_model.pt",
        "data_root": "data/General_Flood_v3",
        "class_id": 1,
    },
    "fire": {
        "module": "src.datasets.fire_dataset",
        "cls": "FireDataset",
        "checkpoint": "models/firemodel_pidnet_multi.pth",
        "data_root": "data/FireSeg",
        "class_id": 1,
    },
}


def build_dataset(name: str, root, split: str = "train", **kwargs):
    import importlib
    spec = DATASETS[name]
    module = importlib.import_module(spec["module"])
    return getattr(module, spec["cls"])(root=str(root), split=split, **kwargs)


def add_burnt_arguments(parser):
    parser.add_argument("--min-burnt-coverage", type=float, default=0.0,
                        help="burnt pixel fraction; must match bank build and service")
    parser.add_argument("--burnt-channel-order", choices=["RGB", "BGR"], default="RGB")
    parser.add_argument("--burnt-size", type=int, nargs=2, default=[720, 1280],
                        metavar=("HEIGHT", "WIDTH"))


def burnt_dataset_options(args):
    return dict(require_burnt=not args.keep_empty,
                min_burnt_coverage=args.min_burnt_coverage,
                channel_order=args.burnt_channel_order, crop_size=tuple(args.burnt_size))


DEFAULT_LAYERS = [
    "layer1.0.conv1",
    "layer2.0.conv1",
    "layer3.0.conv1",
    "layer4.0.conv1",
    "layer5.0.conv1",
    "spp.scale1.3",
]


def build_pcx_banks(model, dataset, layers, class_id, device, out_pcx: Path):
    """One attribution per image, recording every layer, one bank per layer."""
    cc = ChannelConcept()
    attribution = ATTRIBUTORS["pidnet"](model)
    composite = COMPOSITES["pidnet"](canonizers=[CANONIZERS["pidnet"]()])

    rows = {layer: [] for layer in layers}
    skipped = {layer: 0 for layer in layers}
    sample_indices = {layer: [] for layer in layers}

    for index in tqdm(range(len(dataset)), desc="attribution banks", dynamic_ncols=True):
        image, _ = dataset[index]
        sample = image.unsqueeze(0).to(device).requires_grad_()
        try:
            attr = attribution(sample, [{"y": class_id}], composite,
                               record_layer=list(layers), init_rel=1)
            for layer in layers:
                relevance = attr.relevances.get(layer)
                if relevance is None:
                    skipped[layer] += 1
                    continue
                # An image with no pixel predicted as class_id yields all-zero
                # relevance; abs_norm then divides by zero and poisons the bank.
                row = cc.attribute(relevance, abs_norm=True).detach().cpu()
                if torch.isfinite(row).all():
                    rows[layer].append(row)
                    sample_indices[layer].append(index)
                else:
                    skipped[layer] += 1
            del attr
        except RuntimeError as exc:
            for layer in layers:
                skipped[layer] += 1
            print(f"  sample {index} failed: {exc}")
        finally:
            del sample
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    for layer in layers:
        if not rows[layer]:
            raise ValueError(f"{layer}: no usable attribution rows; bank not written")
        bank = torch.cat(rows[layer]).numpy()
        bank_dir = out_pcx / layer
        bank_dir.mkdir(parents=True, exist_ok=True)
        np.save(bank_dir / "attributions.npy", bank)
        np.save(bank_dir / "sample_indices.npy", np.asarray(sample_indices[layer], dtype=np.int64))
        print(f"  {layer}: bank {bank.shape}, skipped {skipped[layer]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="flood")
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-crp", type=Path, default=None)
    parser.add_argument("--out-pcx", type=Path, default=None)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--device", default=None, help="default: cuda:0 if available")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images")
    parser.add_argument("--modality", choices=["rgb", "ir", "all"], default="rgb",
                        help="fire only: which modality subset to analyse")
    parser.add_argument("--keep-empty", action="store_true",
                        help="fire/burnt: keep images with no target pixels. They "
                             "produce all-zero concept vectors and capture a whole "
                             "GMM component, so they are dropped by default.")
    parser.add_argument("--min-fire-coverage", type=float, default=0.0,
                        help="fire only: keep only images whose mask has more than "
                             "this fraction of fire pixels (e.g. 0.03 for 3%%). "
                             "Tightens the default >0 filter; ignored with --keep-empty.")
    parser.add_argument("--skip-crp", action="store_true")
    parser.add_argument("--skip-pcx", action="store_true")
    add_burnt_arguments(parser)
    args = parser.parse_args()

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("LCRP.utils.crp").setLevel(logging.CRITICAL)



    spec = DATASETS[args.dataset]
    checkpoint = args.checkpoint or Path(spec["checkpoint"])
    data_root = args.data_root or Path(spec["data_root"])
    class_id = spec["class_id"] if args.class_id is None else args.class_id
    args.out_crp = args.out_crp or Path(f"output/crp/pidnet_{args.dataset}_sweep")
    args.out_pcx = args.out_pcx or Path(f"output/pcx/pidnet_{args.dataset}_sweep")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    in_channels, classes = infer_checkpoint_geometry(str(checkpoint))
    model = get_pidnet(device=device, ckpt_path=str(checkpoint),
                       classes=classes, in_channels=in_channels).eval()

    fire_extra = ({"modality": args.modality, "require_fire": not args.keep_empty,
                   "min_fire_coverage": 0.0 if args.keep_empty else args.min_fire_coverage}
             if args.dataset == "fire" else {})
    dataset_options = burnt_dataset_options(args) if args.dataset == "burnt" else fire_extra
    dataset = build_dataset(args.dataset, data_root, **dataset_options)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        dataset.files = dataset.files[:args.limit]
    if args.dataset == "burnt" and (in_channels, classes, class_id) != (3, 2, 1):
        parser.error("Burnt requires a 3-channel, 2-class model with class-id 1")
    metadata = burnt_artifact_metadata(dataset, checkpoint) if args.dataset == "burnt" else None

    print(f"task={args.dataset} device={device} images={len(dataset)} "
          f"layers={len(args.layers)} in_channels={in_channels} classes={classes} "
          f"class_id={class_id}"
          + (f" modality={args.modality} require_fire={not args.keep_empty}"
             f" min_fire_coverage={fire_extra['min_fire_coverage']}" if fire_extra else ""))
    for layer in args.layers:
        print(f"  - {layer}")

    if not args.skip_crp:
        print("\n[1/2] CRP statistics")
        args.out_crp.mkdir(parents=True, exist_ok=True)
        if metadata is not None:
            (args.out_crp / "burnt_metadata.json").unlink(missing_ok=True)
        run_analysis(model_name="pidnet", model=model, dataset=dataset,
                     output_dir=str(args.out_crp), device=device,
                     class_id=class_id, record_layers=list(args.layers))
        if metadata is not None:
            (args.out_crp / "burnt_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"  written to {args.out_crp}")

    if not args.skip_pcx:
        print("\n[2/2] PCX attribution banks")
        if metadata is not None:
            for layer in args.layers:
                (args.out_pcx / layer / "burnt_metadata.json").unlink(missing_ok=True)
        build_pcx_banks(model, dataset, list(args.layers), class_id,
                        device, args.out_pcx)
        if metadata is not None:
            for layer in args.layers:
                (args.out_pcx / layer / "burnt_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"  written to {args.out_pcx}")

    print("\ndone")


if __name__ == "__main__":
    main()
