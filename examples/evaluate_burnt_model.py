#!/usr/bin/env python
"""Evaluate burnt segmentation and compare it with the supplied AUTH wrapper.

This script uses the colored Segmentation_Blaze masks for metrics. It also
loads BC_predictor from burnt_segmentation.zip and verifies that both inference
implementations produce the same native-resolution class map.
"""
import argparse
import importlib.util
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LCRP.models.pidnet import get_pidnet, infer_checkpoint_geometry
from src.datasets.burnt_dataset import BurntDataset


def _load_supplied_predictor(source_zip: Path, checkpoint: Path, temp_root: Path):
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(temp_root)
    source_root = temp_root / "burnt_segmentation" / "PIDNet_burnt"
    eval_path = source_root / "tools" / "eval.py"
    if not eval_path.is_file():
        raise FileNotFoundError(f"Supplied evaluator missing: {eval_path}")

    # eval.py imports its own top-level `models` package.
    sys.path.insert(0, str(source_root))
    try:
        spec = importlib.util.spec_from_file_location("supplied_burnt_eval", eval_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.BC_predictor(str(checkpoint))
    finally:
        sys.path.remove(str(source_root))


def _update_confusion(confusion, target, prediction, valid):
    encoded = 2 * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=4).reshape(2, 2)


def _metrics(confusion):
    tp = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = actual + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union != 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) != 0)
    return {
        "confusion_matrix_rows_actual_cols_predicted": confusion.tolist(),
        "pixel_accuracy": float(tp.sum() / confusion.sum()),
        "background": {"iou": float(iou[0]), "precision": float(precision[0]),
                       "recall": float(recall[0]), "f1": float(f1[0])},
        "burnt": {"iou": float(iou[1]), "precision": float(precision[1]),
                  "recall": float(recall[1]), "f1": float(f1[1])},
        "mean_iou": float(iou.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "models/pidnet_s_burnt.pt")
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT_ROOT / "data/Segmentation_Blaze")
    parser.add_argument("--source-zip", type=Path,
                        default=PROJECT_ROOT / "burnt_segmentation.zip")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "output/burnt_validation/full_test_metrics.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for name in ("PIL", "matplotlib", "h5py"):
        logging.getLogger(name).setLevel(logging.WARNING)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = BurntDataset(root=args.data_root, split="test", require_burnt=False)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        dataset.files = dataset.files[:args.limit]

    geometry = infer_checkpoint_geometry(str(args.checkpoint))
    if geometry != (3, 2):
        raise ValueError(f"Expected checkpoint geometry (3, 2), got {geometry}")
    model = get_pidnet(device=device, ckpt_path=str(args.checkpoint),
                       classes=2, in_channels=3).eval()

    confusion_ours = np.zeros((2, 2), dtype=np.int64)
    confusion_supplied = np.zeros((2, 2), dtype=np.int64)
    differing_pixels = 0
    compared_pixels = 0
    records = []

    with tempfile.TemporaryDirectory(prefix="burnt-source-") as temporary:
        supplied = _load_supplied_predictor(
            args.source_zip, args.checkpoint, Path(temporary))
        supplied.network.to(device).eval()

        for index in range(len(dataset)):
            rgb, target = dataset.load_sample(index)
            image = dataset.transform(rgb).unsqueeze(0).to(device)
            with torch.inference_mode():
                output = model(image)
                logits = output[1] if isinstance(output, (list, tuple)) else output
                ours_small = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            # BC_predictor expects an OpenCV BGR input and converts it to RGB.
            supplied_map = supplied.predictCv2(rgb[:, :, ::-1].copy())[:, :, 0]
            supplied_map = (supplied_map > 0).astype(np.uint8)
            ours_map = cv2.resize(ours_small, (rgb.shape[1], rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            if target.shape != rgb.shape[:2]:
                target = cv2.resize(target, (rgb.shape[1], rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

            valid = (target == 0) | (target == 1)
            _update_confusion(confusion_ours, target, ours_map, valid)
            _update_confusion(confusion_supplied, target, supplied_map, valid)
            differences = int((ours_map != supplied_map).sum())
            differing_pixels += differences
            compared_pixels += ours_map.size
            records.append({
                "index": index,
                "name": dataset.files[index]["name"],
                "valid_pixels": int(valid.sum()),
                "target_burnt_fraction": float((target[valid] == 1).mean()),
                "predicted_burnt_fraction": float((ours_map[valid] == 1).mean()),
                "implementation_differing_pixels": differences,
            })
            if (index + 1) % 20 == 0 or index + 1 == len(dataset):
                print(f"evaluated {index + 1}/{len(dataset)}", flush=True)

    report = {
        "samples": len(dataset),
        "device": device,
        "checkpoint_geometry": list(geometry),
        "preprocessing": dataset.bank_metadata(),
        "implementation_comparison": {
            "differing_pixels": differing_pixels,
            "compared_pixels": compared_pixels,
            "agreement": 1.0 - differing_pixels / compared_pixels,
        },
        "ours": _metrics(confusion_ours),
        "supplied_BC_predictor": _metrics(confusion_supplied),
        "per_image": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in
                      ("samples", "implementation_comparison", "ours",
                       "supplied_BC_predictor")}, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
