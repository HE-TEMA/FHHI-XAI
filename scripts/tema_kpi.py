"""Read the CSV-formatted KPI .txt files without importing torch/model code."""

import ast
import csv
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def constants(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = value
    return result


def download(destination):
    from minio import Minio
    config = constants(ROOT / "src/minio_client.py")
    client = Minio(
        os.environ.get("MINIO_ENDPOINT", config["MINIO_ENDPOINT"]),
        access_key=os.environ.get("MINIO_ACCESS_KEY", config["MINIO_ACCESS_KEY"]),
        secret_key=os.environ.get("MINIO_SECRET_KEY", config["MINIO_SECRET"]), secure=True,
    )
    bucket = os.environ.get("MINIO_BUCKET", config["FHHI_MINIO_BUCKET"])
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for obj in client.list_objects(bucket, prefix="tfa02/KPI/", recursive=True):
        if not obj.object_name.endswith((".txt", ".csv")):
            continue
        relative = Path(obj.object_name).relative_to("tfa02/KPI")
        if ".." in relative.parts:
            raise ValueError("Invalid object path")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, obj.object_name, str(target))
        count += 1
    if not count:
        raise ValueError("No KPI files found under tfa02/KPI/ in MinIO")
    return destination


def calculate(directory, output):
    config = constants(ROOT / "src/kpi_logging.py")
    fields = config["KPI_FIELDNAMES"]
    numeric = config["NUMERIC_FIELDS"]
    window_size = config["WINDOW_SIZE"]
    allowed = config["ALLOWED_GLOBAL_EXPLANATION_TIME_S"]
    groups = defaultdict(list)
    skipped_averages = 0
    for path in sorted(directory.rglob("*")):
        if path.suffix not in (".txt", ".csv") or not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"model", "prediction_time_s", "global_lcrp_time_s"}.issubset(reader.fieldnames):
                continue
            for number, record in enumerate(reader, 2):
                if record.get("aggregation", "raw") not in ("", "raw"):
                    skipped_averages += 1
                    continue
                try:
                    for key in numeric:
                        if record.get(key) not in (None, ""):
                            record[key] = float(record[key])
                            if not math.isfinite(record[key]) or record[key] < 0:
                                raise ValueError(f"invalid {key}")
                    for key in ("prediction_time_s", "global_lcrp_time_s"):
                        if record.get(key) in (None, ""):
                            raise ValueError(f"missing {key}")
                    if record.get("global_total_time_s") in (None, ""):
                        record["global_total_time_s"] = record["global_lcrp_time_s"] + float(record.get("global_gmm_time_s") or 0)
                    if record.get("global_gmm_time_s") in (None, ""):
                        record["global_gmm_time_s"] = max(record["global_total_time_s"] - record["global_lcrp_time_s"], 0)
                    if record.get("attribution_single_time_s") in (None, ""):
                        record["attribution_single_time_s"] = record["global_total_time_s"]
                    if record.get("allowed_time_for_global_explanations_s") in (None, ""):
                        record["allowed_time_for_global_explanations_s"] = allowed.get(record["model"], 0)
                    ratios(record)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
                key = (str(path.relative_to(directory)), *(record.get(k, "") for k in ("model", "entity_type", "scope", "layer")))
                groups[key].append(record)
    if not groups:
        raise ValueError(f"No raw KPI records found in {directory}. Console logs do not contain these CSV measurements.")
    output.mkdir(parents=True, exist_ok=True)
    windows, summary = [], []
    for key, records in groups.items():
        source, model, entity_type, scope, layer = key
        summary.append(dict(source=source, model=model, entity_type=entity_type, scope=scope, layer=layer,
                            count=len(records), statistics={field: {
                                "mean": statistics.mean(r[field] for r in records),
                                "min": min(r[field] for r in records), "max": max(r[field] for r in records)
                            } for field in numeric}))
        # Preserve source order and use complete, non-overlapping windows, like append_avg_window_record.
        for start in range(0, len(records) - window_size + 1, window_size):
            block = records[start:start + window_size]
            row = {field: "" for field in fields}
            row.update({field: statistics.mean(r[field] for r in block) for field in numeric})
            row.update(timestamp=block[-1].get("timestamp", ""), model=model, entity_type=entity_type,
                       scope=scope, layer=layer, aggregation=f"avg_{window_size}", window_size=window_size,
                       window_start=start + 1, window_end=start + window_size)
            # build_kpi_record recomputes ratios from averaged durations.
            ratios(row)
            windows.append({"source": source, **row})
    with (output / "kpi_avg5.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", *fields])
        writer.writeheader()
        writer.writerows(windows)
    report = dict(groups=summary, complete_windows=len(windows), skipped_precomputed_averages=skipped_averages,
                  notes=["Durations are in seconds; ratios are dimensionless.",
                         "Raw records and existing average rows are never mixed.",
                         "Groups stay separate by source file, model, entity type, scope and layer.",
                         "Average-window timestamps use the last input timestamp.",
                         "Incomplete final windows are excluded; full-summary statistics include all raw rows."])
    (output / "kpi_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Processed {sum(len(v) for v in groups.values())} raw KPI records; {len(windows)} complete windows. Results: {output}")
    return report


def ratios(record):
    prediction = record["prediction_time_s"]
    total = record["global_total_time_s"]
    record["attribution_single_prediction_ratio"] = record["attribution_single_time_s"] / prediction if prediction else 0.0
    record["allowed_global_total_ratio"] = record["allowed_time_for_global_explanations_s"] / total if total else 0.0
