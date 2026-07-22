import csv
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import torch


KPI_LOG_DIRNAME = "output/kpi_logs"
KPI_MINIO_PREFIX = "tfa02/KPI"
WINDOW_SIZE = 5
ALLOWED_GLOBAL_EXPLANATION_TIME_S = {
    "yolo": 160.0,
    "pidnet": 53.0025,
}
KPI_FIELDNAMES = [
    "timestamp",
    "model",
    "entity_type",
    "scope",
    "aggregation",
    "image",
    "window_size",
    "window_start",
    "window_end",
    "box_index",
    "num_boxes",
    "class_id",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "layer",
    "n_concepts",
    "n_refimgs",
    "prediction_time_s",
    "global_lcrp_time_s",
    "global_gmm_time_s",
    "global_total_time_s",
    "attribution_single_time_s",
    "allowed_time_for_global_explanations_s",
    "attribution_single_prediction_ratio",
    "allowed_global_total_ratio",
]
NUMERIC_FIELDS = [
    "prediction_time_s",
    "global_lcrp_time_s",
    "global_gmm_time_s",
    "global_total_time_s",
    "attribution_single_time_s",
    "allowed_time_for_global_explanations_s",
    "attribution_single_prediction_ratio",
    "allowed_global_total_ratio",
]


@dataclass
class TimedSection:
    elapsed_s: float = 0.0


def _coerce_device(device_like=None) -> Optional[torch.device]:
    if device_like is None:
        return None
    if isinstance(device_like, torch.device):
        return device_like
    return torch.device(device_like)


def sync_device(device_like=None) -> None:
    device = _coerce_device(device_like)
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed_section(device_like=None):
    timer = TimedSection()
    sync_device(device_like)
    start = time.perf_counter()
    try:
        yield timer
    finally:
        sync_device(device_like)
        timer.elapsed_s = time.perf_counter() - start


def get_log_dir(project_root: str) -> str:
    log_dir = os.path.join(project_root, KPI_LOG_DIRNAME)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def build_log_path(project_root: str, filename: str) -> str:
    return os.path.join(get_log_dir(project_root), filename)


def build_minio_object_name(local_path: str) -> str:
    base_name = os.path.basename(local_path)
    stem, _ = os.path.splitext(base_name)
    return f"{KPI_MINIO_PREFIX}/{stem}.txt"


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _float_or_zero(value) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _normalize_bbox(bbox) -> Dict[str, object]:
    if not bbox or len(bbox) != 4:
        return {
            "bbox_x1": "",
            "bbox_y1": "",
            "bbox_x2": "",
            "bbox_y2": "",
        }
    return {
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
    }


def build_kpi_record(
    *,
    model: str,
    entity_type: str,
    scope: str,
    aggregation: str,
    prediction_time_s: float,
    global_lcrp_time_s: float,
    global_gmm_time_s: Optional[float] = None,
    global_total_time_s: Optional[float] = None,
    attribution_single_time_s: Optional[float] = None,
    allowed_time_for_global_explanations_s: Optional[float] = None,
    image: str = "",
    window_size: str = "",
    window_start: str = "",
    window_end: str = "",
    box_index: str = "",
    num_boxes: str = "",
    class_id: str = "",
    confidence: str = "",
    bbox=None,
    layer: str = "",
    n_concepts: str = "",
    n_refimgs: str = "",
) -> Dict[str, object]:
    if global_total_time_s is None:
        global_total_time_s = global_lcrp_time_s
        if global_gmm_time_s is not None:
            global_total_time_s += global_gmm_time_s
    if global_gmm_time_s is None:
        global_gmm_time_s = max(global_total_time_s - global_lcrp_time_s, 0.0)
    if attribution_single_time_s is None:
        attribution_single_time_s = global_total_time_s
    if allowed_time_for_global_explanations_s is None:
        allowed_time_for_global_explanations_s = ALLOWED_GLOBAL_EXPLANATION_TIME_S.get(model, 0.0)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "entity_type": entity_type,
        "scope": scope,
        "aggregation": aggregation,
        "image": image,
        "window_size": window_size,
        "window_start": window_start,
        "window_end": window_end,
        "box_index": box_index,
        "num_boxes": num_boxes,
        "class_id": class_id,
        "confidence": confidence,
        "layer": layer,
        "n_concepts": n_concepts,
        "n_refimgs": n_refimgs,
        "prediction_time_s": float(prediction_time_s),
        "global_lcrp_time_s": float(global_lcrp_time_s),
        "global_gmm_time_s": float(global_gmm_time_s),
        "global_total_time_s": float(global_total_time_s),
        "attribution_single_time_s": float(attribution_single_time_s),
        "allowed_time_for_global_explanations_s": float(allowed_time_for_global_explanations_s),
    }
    record.update(_normalize_bbox(bbox))
    record["attribution_single_prediction_ratio"] = _safe_ratio(
        record["attribution_single_time_s"],
        record["prediction_time_s"],
    )
    record["allowed_global_total_ratio"] = _safe_ratio(
        record["allowed_time_for_global_explanations_s"],
        record["global_total_time_s"],
    )
    for field in KPI_FIELDNAMES:
        record.setdefault(field, "")
    return record


def append_kpi_record(log_path: str, record: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    with open(log_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KPI_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in KPI_FIELDNAMES})


def read_kpi_records(log_path: str) -> List[Dict[str, object]]:
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return []
    with open(log_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _get_last_recorded_window_end(avg_log_path: str) -> int:
    records = read_kpi_records(avg_log_path)
    if not records:
        return 0
    return max(int(record.get("window_end") or 0) for record in records)


def append_avg_window_record(
    source_log_path: str,
    avg_log_path: str,
    *,
    model: str,
    entity_type: str,
    scope: str,
    layer: str = "",
    window_size: int = WINDOW_SIZE,
) -> bool:
    records = read_kpi_records(source_log_path)
    if len(records) < window_size or len(records) % window_size != 0:
        return False

    window_end = len(records)
    if _get_last_recorded_window_end(avg_log_path) >= window_end:
        return False

    window = records[-window_size:]
    averaged = {
        field: sum(_float_or_zero(record.get(field)) for record in window) / window_size
        for field in NUMERIC_FIELDS
    }
    summary_record = build_kpi_record(
        model=model,
        entity_type=entity_type,
        scope=scope,
        aggregation=f"avg_{window_size}",
        image="",
        window_size=window_size,
        window_start=window_end - window_size + 1,
        window_end=window_end,
        prediction_time_s=averaged["prediction_time_s"],
        global_lcrp_time_s=averaged["global_lcrp_time_s"],
        global_gmm_time_s=averaged["global_gmm_time_s"],
        global_total_time_s=averaged["global_total_time_s"],
        attribution_single_time_s=averaged["attribution_single_time_s"],
        allowed_time_for_global_explanations_s=averaged["allowed_time_for_global_explanations_s"],
        layer=layer,
    )
    append_kpi_record(avg_log_path, summary_record)
    return True
