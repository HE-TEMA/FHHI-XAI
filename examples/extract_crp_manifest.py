#!/usr/bin/env python3
"""Recover the exact CRP sample/class selection printed by the CRP notebook."""

import argparse
import ast
import json
import re
from pathlib import Path


ROW_PATTERN = re.compile(
    r"dataset=\s*(\d+)\s*\|\s*image=(.*?)\s*\|\s*classes=(\[[^\n]*\])"
)
CLASS_TO_ID = {"person": 0, "car": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    source_cell = None
    output_text = ""

    for cell_index, cell in enumerate(notebook["cells"]):
        text = "".join(
            fragment
            for output in cell.get("outputs", [])
            for fragment in output.get("text", [])
        )
        if "Accepted images: 8153" in text:
            source_cell = cell_index
            output_text = text
            break

    if source_cell is None:
        raise RuntimeError("Could not find the completed 8,153-image CRP scan output.")

    rows = ROW_PATTERN.findall(output_text)
    matched_indices = []
    matched_classes = {}
    selected_images = {}

    for index_text, image_name, classes_text in rows:
        index = int(index_text)
        class_names = ast.literal_eval(classes_text)
        try:
            class_ids = [CLASS_TO_ID[name] for name in class_names]
        except KeyError as error:
            raise RuntimeError(f"Unknown class name in CRP output: {error.args[0]}") from error

        matched_indices.append(index)
        matched_classes[str(index)] = class_ids
        selected_images[str(index)] = image_name.strip()

    target_count = sum(len(ids) for ids in matched_classes.values())
    if len(matched_indices) != 8153:
        raise RuntimeError(f"Expected 8,153 CRP images, extracted {len(matched_indices)}.")
    if len(set(matched_indices)) != len(matched_indices):
        raise RuntimeError("The extracted CRP indices contain duplicates.")
    if target_count != 12935:
        raise RuntimeError(f"Expected 12,935 CRP class targets, extracted {target_count}.")

    manifest = {
        "schema_version": 1,
        "selection_policy": "exactly_recovered_from_completed_crp_notebook_output",
        "source_notebook": str(args.notebook.resolve()),
        "source_cell_index_zero_based": source_cell,
        "checkpoint": "/home/heydari/FHHI-XAI-BRK/models/best_ckpt.pt",
        "dataset_root": (
            "/home/heydari/FHHI-XAI-BRK/data/BRK/person_vehicle_detection"
        ),
        "dataset_length": 9560,
        "crp_dataset_length": len(matched_indices),
        "class_target_count": target_count,
        "class_name_to_id_as_used_by_crp": CLASS_TO_ID,
        "matched_indices": matched_indices,
        "matched_classes": matched_classes,
        "selected_images": selected_images,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote exact CRP manifest: {args.output}")
    print(f"Images: {len(matched_indices)}")
    print(f"Class targets: {target_count}")
    print(f"Source notebook cell: {source_cell}")


if __name__ == "__main__":
    main()
