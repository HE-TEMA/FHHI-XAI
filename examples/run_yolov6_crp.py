#!/usr/bin/env python3
"""Run CRP for the two-class YOLOv6 checkpoint.

The script first finds dataset images for which the detector returns at least
one box. CRP is then run only on those images, because an empty post-NMS output
does not have a differentiable prediction target.
"""

import argparse
import contextlib
import gc
import io
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import box_iou


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LCRP.models import get_model
from src.datasets.detection_subset import DetectionSubset
from src.datasets.person_car_dataset import PersonCarDataset
from src.glocal_analysis import run_analysis
from src.letterbox_utils import YOLOv6TrainPreprocess
from src.yolo_class_mapping import resolve_display_class_names


MODEL_NAME = "yolov6s6"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_ckpt.pt"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "BRK" / "person_vehicle_detection"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "crp" / "yolo_person_car"


def select_device(requested):
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{index}, but only {torch.cuda.device_count()} CUDA "
                "device(s) are available."
            )
    return device


def load_model(checkpoint, device):
    model = get_model(
        model_name=MODEL_NAME,
        classes=2,
        ckpt_path=str(checkpoint),
        device=device,
        dtype=torch.float32,
    )
    return model.to(device).eval()


def _normalized_class_name(name):
    name = str(name).strip().lower()
    return "vehicle" if name in {"car", "vehicle"} else name


def detector_to_dataset_class_map(dataset, model_name=MODEL_NAME):
    """Map detector output IDs to annotation IDs by semantic class name."""

    detector_names = tuple(resolve_display_class_names(model_name, dataset))
    dataset_names = tuple(dataset.class_names)
    normalized_dataset_names = [
        _normalized_class_name(name) for name in dataset_names
    ]
    mapping = {}
    for detector_id, detector_name in enumerate(detector_names):
        normalized_name = _normalized_class_name(detector_name)
        matches = [
            dataset_id
            for dataset_id, dataset_name in enumerate(normalized_dataset_names)
            if dataset_name == normalized_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot map detector class {detector_id} ({detector_name!r}) "
                f"uniquely into dataset classes {dataset_names!r}."
            )
        mapping[detector_id] = matches[0]
    return mapping


def validate_dataset_labels(dataset, model, limit=None):
    """Validate labels using YOLOv6-compatible rules before CRP.

    Boxes crossing an image boundary are recoverable because YOLOv6 clips
    them. Zero-area boxes are reported and excluded later from IoU matching.
    """

    detector_names = tuple(resolve_display_class_names(MODEL_NAME, dataset))
    class_mapping = detector_to_dataset_class_map(dataset, MODEL_NAME)
    print(f"Detector class order: {detector_names}")
    print(f"Dataset class order:  {tuple(dataset.class_names)}")
    print(f"Detector-to-dataset class mapping: {class_mapping}")

    sample_count = len(dataset) if limit is None else min(limit, len(dataset))
    errors = []
    warnings = []
    for index in range(sample_count):
        image_file = dataset.image_files[index]
        label_file = dataset.label_files[index]
        if Path(image_file).stem != Path(label_file).stem:
            errors.append(
                f"index {index}: image/label stem mismatch "
                f"({image_file!r} vs {label_file!r})"
            )
            continue

        image_path = Path(dataset.image_dir) / image_file
        label_path = Path(dataset.label_dir) / label_file
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"index {index}: unreadable image {image_path}: {exc}")

        try:
            lines = label_path.read_text().splitlines()
        except Exception as exc:
            errors.append(f"index {index}: unreadable label {label_path}: {exc}")
            continue

        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                errors.append(
                    f"{label_path}:{line_number}: expected 5 fields, got {len(fields)}"
                )
                continue
            try:
                class_id = int(fields[0])
                x_center, y_center, width, height = map(float, fields[1:])
            except ValueError as exc:
                errors.append(f"{label_path}:{line_number}: invalid value: {exc}")
                continue

            if not 0 <= class_id < len(dataset.class_names):
                errors.append(
                    f"{label_path}:{line_number}: class {class_id} is outside "
                    f"[0, {len(dataset.class_names) - 1}]"
                )
            if width <= 0 or height <= 0:
                warnings.append(
                    f"{label_path}:{line_number}: zero-area box will be excluded"
                )
            if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
                errors.append(
                    f"{label_path}:{line_number}: normalized coordinates outside [0, 1]"
                )
            if (
                x_center - width / 2 < 0
                or x_center + width / 2 > 1
                or y_center - height / 2 < 0
                or y_center + height / 2 > 1
            ):
                warnings.append(
                    f"{label_path}:{line_number}: box crosses image boundary "
                    "and will be clipped like YOLOv6"
                )

    if errors:
        preview = "\n".join(errors[:50])
        remainder = len(errors) - min(50, len(errors))
        if remainder:
            preview += f"\n... and {remainder} additional error(s)"
        raise ValueError(f"Dataset validation failed:\n{preview}")

    print(f"Validated {sample_count} image/label pair(s): OK")
    if warnings:
        print(f"Recoverable annotation warning(s): {len(warnings)}")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} additional warning(s)")


def _ground_truth_in_model_coordinates(dataset, index, target_size=640):
    """Read labels and reproduce YOLOv6 training's exact box geometry."""

    image_path = Path(dataset.image_dir) / dataset.image_files[index]
    label_path = Path(dataset.label_dir) / dataset.label_files[index]
    with Image.open(image_path) as image:
        w0, h0 = image.size

    initial_ratio = target_size / max(h0, w0)
    resized_w = int(w0 * initial_ratio) if initial_ratio != 1 else w0
    resized_h = int(h0 * initial_ratio) if initial_ratio != 1 else h0

    letterbox_ratio = min(target_size / resized_h, target_size / resized_w)
    new_w = int(round(resized_w * letterbox_ratio))
    new_h = int(round(resized_h * letterbox_ratio))
    left = int(round((target_size - new_w) / 2 - 0.1))
    top = int(round((target_size - new_h) / 2 - 0.1))

    boxes, classes = [], []
    for line in label_path.read_text().splitlines():
        class_id, x_center, y_center, width, height = line.split()
        class_id = int(class_id)
        x_center, y_center, width, height = map(
            float, (x_center, y_center, width, height)
        )
        # A zero-area label cannot match a real predicted detection.
        if width <= 0 or height <= 0:
            continue
        scaled_w = resized_w * letterbox_ratio
        scaled_h = resized_h * letterbox_ratio
        box = [
            scaled_w * (x_center - width / 2) + left,
            scaled_h * (y_center - height / 2) + top,
            scaled_w * (x_center + width / 2) + left,
            scaled_h * (y_center + height / 2) + top,
        ]
        # Match TrainValDataset.__getitem__, which clips transformed boxes.
        box[0] = min(max(box[0], 0.0), target_size - 1e-3)
        box[1] = min(max(box[1], 0.0), target_size - 1e-3)
        box[2] = min(max(box[2], 0.0), target_size - 1e-3)
        box[3] = min(max(box[3], 0.0), target_size - 1e-3)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        boxes.append(box)
        classes.append(class_id)

    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(classes, dtype=torch.long),
    )


def find_ground_truth_matched_samples(
    model,
    dataset,
    device,
    limit=None,
    iou_threshold=0.5,
    cache_path=None,
    checkpoint_interval=100,
):
    """Keep rank-0 same-class predictions, with resumable scan checkpoints."""

    matched_indices = []
    matched_classes = {}
    diagnostics = []
    class_mapping = detector_to_dataset_class_map(dataset, MODEL_NAME)
    sample_count = len(dataset) if limit is None else min(limit, len(dataset))
    next_index = 0

    cache_path = Path(cache_path) if cache_path is not None else None
    expected_cache_config = {
        "schema_version": 2,
        "dataset_length": len(dataset),
        "sample_count": sample_count,
        "iou_threshold": float(iou_threshold),
        "class_mapping": {
            str(key): int(value) for key, value in class_mapping.items()
        },
    }

    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        actual_cache_config = {
            key: payload.get(key) for key in expected_cache_config
        }
        if actual_cache_config != expected_cache_config:
            raise ValueError(
                f"Validation cache configuration mismatch at {cache_path}. "
                f"Expected {expected_cache_config}, found {actual_cache_config}. "
                "Use a different cache path for this run."
            )
        next_index = int(payload.get("next_index", 0))
        matched_indices = [
            int(index) for index in payload.get("matched_indices", [])
        ]
        matched_classes = {
            int(index): tuple(int(class_id) for class_id in class_ids)
            for index, class_ids in payload.get("matched_classes", {}).items()
        }
        diagnostics = list(payload.get("diagnostics", []))
        print(
            f"Resuming validation scan at image {next_index}/{sample_count} "
            f"from {cache_path}"
        )

    def save_scan_checkpoint():
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **expected_cache_config,
            "next_index": next_index,
            "completed": next_index >= sample_count,
            "matched_indices": matched_indices,
            "matched_classes": {
                str(index): list(class_ids)
                for index, class_ids in matched_classes.items()
            },
            "diagnostics": diagnostics,
        }
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload))
        temporary_path.replace(cache_path)

    print(
        f"Checking images {next_index}..{sample_count - 1} for class-matched detections "
        f"with IoU >= {iou_threshold:.2f}..."
    )
    try:
        with torch.inference_mode():
            for index in range(next_index, sample_count):
                image, _ = dataset[index]
                with contextlib.redirect_stdout(io.StringIO()):
                    scores, boxes = model.predict_with_boxes(
                        image.unsqueeze(0).to(device, non_blocking=True)
                    )

                gt_boxes, gt_classes = _ground_truth_in_model_coordinates(
                    dataset, index
                )
                valid_classes = []
                if scores.numel() and boxes.shape[1] and gt_boxes.numel():
                    scores = scores[0].detach().cpu()
                    boxes = boxes[0].detach().cpu()
                    predicted_classes = scores.argmax(dim=1)
                    confidences = scores.max(dim=1).values

                    for class_id in torch.unique(predicted_classes).tolist():
                        dataset_class_id = class_mapping[int(class_id)]
                        pred_ids = torch.nonzero(
                            predicted_classes == class_id,
                            as_tuple=False,
                        ).flatten()
                        # This is precisely prediction_num=0 for the class.
                        selected_id = pred_ids[
                            confidences[pred_ids].argmax()
                        ]
                        gt_ids = torch.nonzero(
                            gt_classes == dataset_class_id,
                            as_tuple=False,
                        ).flatten()
                        if gt_ids.numel() == 0:
                            continue

                        ious = box_iou(
                            boxes[selected_id].reshape(1, 4),
                            gt_boxes[gt_ids],
                        )[0]
                        best_iou = float(ious.max())
                        diagnostics.append(
                            {
                                "dataset_index": index,
                                "class_id": int(class_id),
                                "dataset_class_id": int(dataset_class_id),
                                "confidence": float(confidences[selected_id]),
                                "iou": best_iou,
                            }
                        )
                        if best_iou >= iou_threshold:
                            valid_classes.append(int(class_id))

                if valid_classes:
                    matched_indices.append(index)
                    matched_classes[index] = tuple(valid_classes)

                next_index = index + 1
                if (
                    next_index % checkpoint_interval == 0
                    or next_index == sample_count
                ):
                    save_scan_checkpoint()
                    print(
                        f"\rChecked {next_index}/{sample_count}; "
                        f"matched images: {len(matched_indices)}",
                        end="",
                        flush=True,
                    )
    except KeyboardInterrupt:
        save_scan_checkpoint()
        print(
            f"\nScan interrupted; progress through image {next_index - 1} "
            f"was saved to {cache_path}."
        )
        raise

    print()
    return matched_indices, matched_classes, diagnostics


def find_detected_samples(model, dataset, device, limit=None):
    """Return detected indices and the classes YOLO predicted for each."""

    detected_indices = []
    predicted_classes = {}
    sample_count = len(dataset) if limit is None else min(limit, len(dataset))

    print(f"Scanning {sample_count} image(s) for valid detections...")
    with torch.inference_mode():
        for index in range(sample_count):
            image, _ = dataset[index]
            # The legacy YOLO wrapper prints the NMS tensor shape on every
            # forward pass; suppress that noise during the scan.
            with contextlib.redirect_stdout(io.StringIO()):
                scores, boxes = model.predict_with_boxes(
                    image.unsqueeze(0).to(device, non_blocking=True)
                )

            if boxes.ndim == 3 and boxes.shape[1] > 0 and scores.numel() > 0:
                detected_indices.append(index)
                predicted_classes[index] = tuple(
                    torch.unique(scores[0].argmax(dim=1)).cpu().tolist()
                )

            if (index + 1) % 100 == 0 or index + 1 == sample_count:
                print(
                    f"\rScanned {index + 1}/{sample_count}; "
                    f"valid images: {len(detected_indices)}",
                    end="",
                    flush=True,
                )

    print()
    return detected_indices, predicted_classes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device such as cuda:0, cuda:1, or cpu (default: cuda:0 if available).",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help="Optionally scan only the first N dataset images.",
    )
    parser.add_argument(
        "--no-canonizer",
        action="store_true",
        help="Disable the YOLOv6 canonizer. The canonizer is enabled by default.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum same-class IoU required before an image/class is admitted to CRP.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.getLogger("PIL").setLevel(logging.WARNING)

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if args.scan_limit is not None and args.scan_limit <= 0:
        raise ValueError("--scan-limit must be greater than zero.")
    if not 0 <= args.iou_threshold <= 1:
        raise ValueError("--iou-threshold must be between zero and one.")

    device = select_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = YOLOv6TrainPreprocess(
        target_size=640,
        stride=32,
        half=False,
    )
    dataset = PersonCarDataset(
        root_dir=str(dataset_root),
        split="train",
        transform=transform,
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset:    {dataset_root}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {device}")

    scan_model = load_model(checkpoint, device)
    validate_dataset_labels(
        dataset,
        scan_model,
        limit=args.scan_limit,
    )
    detected_indices, predicted_classes, diagnostics = (
        find_ground_truth_matched_samples(
            scan_model,
            dataset,
            device,
            limit=args.scan_limit,
            iou_threshold=args.iou_threshold,
        )
    )
    matched_detection_count = sum(
        len(class_ids) for class_ids in predicted_classes.values()
    )
    print(
        f"Accepted {matched_detection_count} class target(s) across "
        f"{len(detected_indices)} image(s)."
    )

    if not detected_indices:
        raise RuntimeError(
            "No rank-0 YOLO predictions matched a ground-truth box with the "
            f"same class and IoU >= {args.iou_threshold:.2f}; CRP was not run."
        )

    # Discard the inference-only instance and load a clean model so no hooks
    # from an earlier attribution attempt can be present.
    del scan_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = load_model(checkpoint, device)
    crp_dataset = DetectionSubset(
        dataset,
        detected_indices,
        predicted_classes,
    )
    print(f"Running CRP on {len(crp_dataset)} image(s) with detections.")

    run_analysis(
        model_name=MODEL_NAME,
        model=model,
        dataset=crp_dataset,
        output_dir=str(output_dir),
        device=device,
        use_canonizer=not args.no_canonizer,
    )


if __name__ == "__main__":
    main()
